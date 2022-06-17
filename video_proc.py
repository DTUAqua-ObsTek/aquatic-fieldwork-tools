import argparse
import ast
import configparser
import re
import sys
from queue import SimpleQueue, Full, Empty
from threading import Thread, Lock
from typing import Tuple

import cv2
import numpy as np
from mosaicking import preprocessing

from utils import find_files


class VideoWorker(Thread):
    _queue = SimpleQueue()
    _mutex_lock = Lock()
    _writer = None
    counter = 0

    def __init__(self):
        self._is_running = False
        super().__init__(target=self._run)

    def start(self):
        assert isinstance(self._writer, cv2.VideoWriter), "Call VideoWriter.set_video_writer first!."
        if not self._is_running:
            self._is_running = True
            super().start()
        else:
            sys.stderr.write(f"{self.name} already running!\n")

    def stop(self):
        if self._is_running:
            self._is_running = False
        else:
            sys.stderr.write(f"{self.name} already stopped!\n")

    @staticmethod
    def write(packet: Tuple[int, np.ndarray]):
        VideoWorker._queue.put(packet)

    @classmethod
    def set_video_writer(cls, writer: cv2.VideoWriter):
        assert isinstance(writer,
                          cv2.VideoWriter), "Provide VideoWorker.set_video_writer with an instance of cv2.VideoWriter!"
        VideoWorker._writer = writer

    def _run(self):
        while self._is_running:
            try:
                # Get a lock
                if not VideoWorker._mutex_lock.acquire(True, 0.1):
                    continue
                # Pull a frame
                frame_number, frame = self._queue.get(True, 0.1)
                # VideoWorker.counter = frame_number
                if frame_number != VideoWorker.counter + 1:
                    self._queue.put((frame_number, frame), True, 0.1)
                    VideoWorker._mutex_lock.release()
                    continue
                else:
                    VideoWorker.counter = frame_number
            except (Full, Empty):
                VideoWorker._mutex_lock.release()
                continue
            # Write with the videowriter
            self._writer.write(frame)
            # Release the mutex
            VideoWorker._mutex_lock.release()


def parse_configuration(configuration_file: str) -> (int, Tuple[Tuple]):
    paths = find_files(configuration_file, ".ini")
    assert len(paths), "Could not find {}.".format(configuration_file)
    parser = configparser.ConfigParser()
    parser.read(str(paths[0]))
    workers = int(parser.get("PERFORMANCE", "workers"))
    assert workers > 0, "Number of workers must be greater than 0."
    funcs = {"resize": preprocessing.const_ar_scale,
             "contrast": preprocessing.fix_contrast,
             "color": preprocessing.fix_color,
             "light": preprocessing.fix_light,
             "enhance": preprocessing.enhance_detail,
             "rebalance": preprocessing.rebalance_color,
             "add": preprocessing.add_bias}
    expr = "([a-zA-Z]+)\d*"
    pipeline = []
    for key, val in parser["PIPELINE"].items():
        match_obj = re.match(expr, key)
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
    return workers, tuple(pipeline)


def main(args: argparse.Namespace):
    workers, pipeline = parse_configuration(args.configuration)
    paths = find_files(args.files, [".mp4", ".avi", ".mkv"], recursive=args.recursive)
    flag = True
    video_title = "Video (Press q to skip video, esc to exit program. Space to pause/unpause)"
    if args.show:
        cv2.namedWindow(video_title, cv2.WINDOW_NORMAL)
    print("CTRL+C to exit program.")
    is_playing = True
    for path in paths:
        try:
            if not flag:
                break
            cap = cv2.VideoCapture(str(path))
            output_path = path.with_name("colored_" + path.name)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            for op, params in pipeline:
                if op == preprocessing.const_ar_scale:
                    width = width * params[0]
                    height = height * params[0]
            writer = cv2.VideoWriter(str(output_path),
                                     cv2.VideoWriter_fourcc(*"mp4v"),
                                     cap.get(cv2.CAP_PROP_FPS),
                                     (int(width),
                                      int(height)))
            writer_workers = [VideoWorker() for _ in range(workers)]
            VideoWorker.set_video_writer(writer)
            [worker.start() for worker in writer_workers]
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            c = 0
            while c < frames:
                if is_playing:
                    c = c + 1
                    ret, frame = cap.read()
                    img = frame.copy()
                    if ret:
                        for op, params in pipeline:
                            if op == preprocessing.const_ar_scale:
                                frame = op(frame, *params)
                            img = op(img, *params)
                        VideoWorker.write((c, img))
                        if args.show:
                            cv2.imshow(video_title, np.concatenate((frame, img), axis=1))
                key = cv2.waitKey(1)
                if key == ord("q") or key == ord("Q"):
                    break
                elif key == 27:
                    flag = False
                    break
                elif key == 32:
                    is_playing = False if is_playing else True
        except KeyboardInterrupt:
            flag = False
            print("Exiting.")
        finally:
            [worker.stop() for worker in writer_workers]
            [worker.join() for worker in writer_workers]
            cap.release()
            writer.release()
            del cap, writer, writer_workers
    if args.show:
        cv2.destroyWindow("Video (Press q to skip video, esc to exit program.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("-r", "--recursive", action="store_true")
    parser.add_argument("-s", "--show", action="store_true", help="Show video comparison.")
    parser.add_argument("-c", "--configuration", type=str, required=True, help="Path to configuration file.")
    args = parser.parse_args()
    main(args)
