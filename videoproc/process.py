from mosaicking import preprocessing
from typing import Tuple
import configparser
import ast
import sys
import re
import numpy as np
import multiprocessing as mp
import queue
import argparse
import cv2
from videoproc import utils, io
import time


class Pipeline:
    def __init__(self, configuration_file: str = None):
        self._pipeline = tuple()
        if configuration_file is not None:
            self._pipeline = parse_configuration(configuration_file)._pipeline

    def __getitem__(self, item):
        return self._pipeline[item]

    def __len__(self):
        return len(self._pipeline)


def parse_configuration(configuration_file: str) -> (int, Tuple[Tuple]):
    """
    Given a configuration.ini file, construct a Pipeline object.
    Parameters
    ----------
    configuration_file: a .ini style configuration file with a ["PIPELINE"] section.

    Returns
    -------
    Pipeline instance.
    """
    paths = utils.find_files(configuration_file, ".ini")
    assert len(paths), "Could not find {}.".format(configuration_file)
    p = Pipeline()
    parser = configparser.ConfigParser()
    parser.read(str(paths[0]))
    funcs = {"resize": preprocessing.const_ar_scale,
             "contrast": preprocessing.fix_contrast,
             "color": preprocessing.fix_color,
             "light": preprocessing.fix_light,
             "enhance": preprocessing.enhance_detail,
             "rebalance": preprocessing.rebalance_color,
             "add": preprocessing.add_bias}
    expr = "([a-zA-Z]+)\d*"
    pipeline = []
    matcher = re.compile(expr)
    for key, val in parser["PIPELINE"].items():
        match_obj = matcher.match(key)
        if match_obj.groups() is not None:
            lookup = match_obj.groups()[0]
            if lookup in funcs:
                data = ast.literal_eval(val)
                assert isinstance(data, tuple), f"Parameter value for {key} must be a tuple, read as {data}."
                assert len(data), f"Tuple empty for parameter {key}."
                assert isinstance(data[0], bool), f"First element of tuple must be a boolean for parametery {key}."
                if data[0]:
                    pipeline.append((funcs[lookup], data[1:]))
            else:
                sys.stderr.write("Parse error: invalid method {}, should be one of {}.\n".format(
                    lookup,
                    ", ".join(funcs.keys())
                ))
        else:
            sys.stderr.write("Parse error: invalid PIPELINE parameter {}, should be one of {}.\n".format(
                key,
                ", ".join(funcs.keys())
            ))
    p._pipeline = tuple(pipeline)
    return p


class VideoProcessor(Pipeline):
    def __init__(self, configuration_file: str):
        super().__init__(configuration_file)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        for op, params in self._pipeline:
            img = op(img, *params)
        return img

    def get_scale(self) -> float:
        scale = 1.0
        for op, params in self._pipeline:
            if op == preprocessing.const_ar_scale:
                scale = scale * params[0]
        return scale


class VideoProcessorWorker(mp.Process, VideoProcessor):
    def __init__(self, configuration_file: str, input_queue: mp.Queue, output_queue: mp.Queue, shutdown_signal: mp.Event):
        self._input_queue = input_queue
        self._output_queue = output_queue
        self._shutdown_signal = shutdown_signal
        VideoProcessor.__init__(self, configuration_file)
        mp.Process.__init__(self)

    def run(self):
        while not self._shutdown_signal.is_set():
            try:
                frame_number, img = self._input_queue.get(True, 0.1)
            except queue.Empty:
                continue
            try:
                self._output_queue.put((frame_number, self(img)), True, 0.1)
            except queue.Full:
                continue
        print(self.name, " exited.")


class VideoProcessorManager:

    def __init__(self, configuration_file: str):
        parser = configparser.ConfigParser()
        parser.read(configuration_file)
        workers = int(parser.get("PERFORMANCE", "workers"))
        assert workers > 0, "Number of workers must be greater than 0."
        buffersize = int(parser.get("PERFORMANCE", "buffersize"))
        self._shutdown_signal = mp.Event()
        self._input_queue = mp.Queue(maxsize=buffersize)
        self._output_queue = mp.Queue(maxsize=buffersize)
        self._is_running = False
        self._workers = [VideoProcessorWorker(configuration_file,
                                              self._input_queue,
                                              self._output_queue,
                                              self._shutdown_signal) for _ in range(workers)]
        self.scale = self._workers[0].get_scale()

    def start(self):
        if self._is_running:
            sys.stderr.write("Workers already started!\n")
            return
        self._is_running = True
        [worker.start() for worker in self._workers]

    def stop(self):
        if not self._is_running:
            sys.stderr.write("Workers already stopped!\n")
            return
        self._shutdown_signal.set()
        time.sleep(1.0)
        self._is_running = False

    def release(self):
        if self._is_running:
            self.stop()

        while not self._input_queue.empty():
            self._input_queue.get()
        while not self._output_queue.empty():
            self._output_queue.get()
        self._input_queue.close()
        self._input_queue.join_thread()
        self._output_queue.close()
        self._output_queue.join_thread()
        [(worker.join(), print(f"{worker.name} quit.")) for worker in self._workers]

    def put(self, frame_number, img):
        self._input_queue.put((frame_number, img))

    def get_frame(self):
        try:
            return self._output_queue.get(True, 0.1)
        except queue.Empty:
            return (-1, None)

    def replace(self, frame_number, img):
        self._output_queue.put((frame_number, img))

    def get_queues(self):
        return self._input_queue, self._output_queue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply a modular pipeline to process videos.")
    parser.add_argument("inputs", nargs="+", type=str, help="Space separated list of paths to videos or folders containing videos to be processed. Can also handle URLs.")
    parser.add_argument("-t", "--types", nargs="+", type=str, default=[".mp4", ".mkv", ".avi", ".mov", ".webm"], help="Space separated list of video extensions to search through.")
    parser.add_argument("-r", "--recurse", action="store_true", help="Indicate to search directories for videos recusrively.")
    parser.add_argument("-c", "--configuration", type=str, required=True, help="Path to configuration.ini file containing processing configuration information.")
    parser.add_argument("-s", "--show", action="store_true", help="Indicate to display output image.")
    parser.add_argument("-w", "--write", action="store_true", help="Indicate to write video output to .mp4 file.")
    parser.add_argument("-p", "--prefix", type=str, default="processed_", help="prefix to add to name of input video file.")
    args = parser.parse_args()
    videos = utils.find_files(args.inputs, args.types, args.recurse)
    config_file = utils.find_files(args.configuration, ".ini")
    if not config_file:
        raise FileNotFoundError(f"Cannot find {args.configuration}.")
    for video in videos:
        print(f"Processing video: {video} ...")
        cap = cv2.VideoCapture(str(video))
        reader_manager = io.ReaderManager(config_file[0], cap)
        proc_manager = VideoProcessorManager(config_file[0])
        proc_input, proc_output = proc_manager.get_queues()
        if args.write:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * proc_manager.scale)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * proc_manager.scale)
            if str(video).startswith("http"):
                urlobj = utils.urlparse(str(video))
                output_path = utils.Path(args.prefix + utils.Path(urlobj.path).name).with_suffix(".mp4")
            else:
                output_path = video.with_name(args.prefix + str(video.name)).with_suffix(".mp4")
            writer = cv2.VideoWriter(str(output_path),
                                     cv2.VideoWriter_fourcc(*"mp4v"),
                                     cap.get(cv2.CAP_PROP_FPS),
                                     (width, height)
                                     )
            writer_manager = io.WriterManager(config_file[0], writer)
            write_queue = writer_manager.get_queue()
        loader = io.FrameSequencer(reader_manager.get_queue(), proc_input)
        ordered_output = queue.Queue()
        sequencer = io.FrameSequencer(proc_output, ordered_output)
        loader.start()
        sequencer.start()
        proc_manager.start()
        reader_manager.start()
        if args.write:
            writer_manager.start()
        if args.show:
            cv2.namedWindow(f"Video: {video.name}", cv2.WINDOW_NORMAL)
        counter = 0
        tstart = time.perf_counter()
        fps_ticks = 0
        fps = 0
        while counter < cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1:
            try:
                counter, img = ordered_output.get(True, 0.1)
                if args.write:
                    write_queue.put((counter, img.copy()), True, 0.1)
                textsize, baseline = cv2.getTextSize(f"{counter} / {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1}", cv2.FONT_HERSHEY_PLAIN, 4, 2)
                img = cv2.putText(img, f"{counter} / {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1}", (10, img.shape[0] - textsize[1]),
                                  cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 2)
                if (time.perf_counter() - tstart) > 1.0:
                    fps = counter - fps_ticks
                    tstart = time.perf_counter()
                    fps_ticks = counter
                textsize, baseline = cv2.getTextSize(f"{fps}", cv2.FONT_HERSHEY_PLAIN, 4, 2)
                img = cv2.putText(img, f"{fps}",
                                  (img.shape[1] - textsize[0] - 10, img.shape[0] - textsize[1]),
                                  cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 2)
                if args.show:
                    cv2.imshow(f"Video: {video.name}", img)
                    key = cv2.waitKey(1)
                    if key == 27:
                        break
            except queue.Empty:
                continue
            except queue.Full:
                continue
            except KeyboardInterrupt:
                break
        reader_manager.release()
        if args.show:
            cv2.destroyWindow(f"Video: {video.name}")
        cap.release()
        sequencer.release()
        loader.release()
        proc_manager.release()
        if args.write:
            writer_manager.release()
            writer.release()
        print(f"Processing video: {video} DONE.")