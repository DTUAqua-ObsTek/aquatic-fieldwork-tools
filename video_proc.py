import argparse
import configparser
import re
import sys
import ast

import cv2
from utils import find_files
from mosaicking import preprocessing
import numpy as np
from queue import SimpleQueue, Full, Empty
from threading import Thread, Lock
from typing import List, Tuple


class VideoWorker(Thread):
    _queue = SimpleQueue()
    _mutex_lock = Lock()

    def __init__(self, videowriter: cv2.VideoWriter):
        self._writer = videowriter
        self._is_running = False
        super().__init__(target=self._run)

    def start(self):
        if not self._is_running:
            self._is_running = True
            super().start()

    def stop(self):
        if self._is_running:
            self._is_running = False

    def write(self, frame: np.ndarray):
        self._queue.put((frame,))

    def _run(self):
        while self._is_running:
            try:
                # Get a lock
                if not self._mutex_lock.acquire(True, 0.5):
                    continue
                # Pull a frame
                frame = self._queue.get(True, 0.5)
            except Full:
                continue
            except Empty:
                continue
            # Write with the videowriter
            self._writer.write(frame[0])
            # Release the mutex
            self._mutex_lock.release()


def parse_configuration(configuration_file: str) -> (int,Tuple[Tuple]):
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
             "rebalance": preprocessing.rebalance_color}
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
    if args.show:
        cv2.namedWindow("Video (Press q to skip video, esc to exit program.", cv2.WINDOW_NORMAL)
    print("CTRL+C to exit program.")
    for path in paths:
        try:
            if not flag:
                break
            cap = cv2.VideoCapture(str(path))
            output_path = path.with_name("colored_"+path.name)
            writer = cv2.VideoWriter(str(output_path),
                                     cv2.VideoWriter_fourcc(*"mp4v"),
                                     cap.get(cv2.CAP_PROP_FPS),
                                     (  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
            workers = [VideoWorker(writer) for _ in range(workers)]
            [worker.start() for worker in workers]
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for pos in range(frames):
                ret, frame = cap.read()
                img = frame.copy()
                if ret:
                    for op, params in pipeline:
                        if isinstance(op, type(preprocessing.const_ar_scale)):
                            frame = op(frame, *params)
                        img = op(img, *params)
                    workers[0].write(img)
                    if args.show:
                        cv2.imshow("Video (Press q to skip video, esc to exit program.", np.concatenate((frame, img), axis=1))
                key = cv2.waitKey(1)
                if key == ord("q"):
                    break
                if key == 27:
                    flag = False
                    break
        except KeyboardInterrupt:
            flag = False
            print("Exiting.")
        finally:
            [worker.stop() for worker in workers]
            [worker.join() for worker in workers]
            cap.release()
            writer.release()
            del cap, writer, workers
    if args.show:
        cv2.destroyWindow("Video (Press q to skip video, esc to exit program.")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("-r", "--recursive", action="store_true")
    parser.add_argument("-s", "--show", action="store_true", help="Show video comparison.")
    parser.add_argument("-c", "--configuration", type=str, required=True, help="Path to configuration file.")
    args = parser.parse_args()
    main(args)
