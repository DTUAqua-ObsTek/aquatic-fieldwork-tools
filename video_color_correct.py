import argparse

import cv2

from utils import find_files
from cv2 import VideoCapture, VideoWriter
from mosaicking.preprocessing import fix_color
import numpy as np
import multiprocessing as mp
from queue import SimpleQueue, Full, Empty
from pathlib import Path
from threading import Thread


class VideoWorker(Thread):
    def __init__(self, path: Path, fps: float, width: int, height: int):
        self._writer = VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        self._queue = SimpleQueue()
        self._is_running = False
        super().__init__(target=self._run)

    def start(self):
        self._is_running = True
        super().start()

    def stop(self):
        self._is_running = False

    def write(self, frame: np.ndarray):
        self._queue.put((frame,))

    def _write(self, frame: np.ndarray):
        self._writer.write(frame)

    def _run(self):
        while self._is_running:
            try:
                frame = self._queue.get(True, 0.5)
                if frame is None:
                    break
            except Full:
                continue
            except Empty:
                continue
            self._writer.write(frame[0])
        self._writer.release()

def main(args: argparse.Namespace):
    paths = find_files(args.files, [".mp4", ".avi", ".mkv"], recursive=args.recursive)
    flag = True
    if args.show:
        cv2.namedWindow("Video (Press q to skip video, esc to exit program.", cv2.WINDOW_NORMAL)
    print("CTRL+C to exit program.")
    for path in paths:
        try:
            if not flag:
                break
            cap = VideoCapture(str(path))
            bro = VideoWorker(path.with_name("colored_"+path.name),
                              cap.get(cv2.CAP_PROP_FPS),
                              int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            bro.start()
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for pos in range(frames):
                ret, frame = cap.read()
                if ret:
                    img = fix_color(frame, args.percent)
                    bro.write(img)
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
            bro.stop()
            bro.join()
            cap.release()
            del cap

    if args.show:
        cv2.destroyWindow("Video (Press q to skip video, esc to exit program.")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("-r", "--recursive", action="store_true")
    parser.add_argument("-p", "--percent", type=float, default=0.8, help="Fractional percentage to rebalance.")
    parser.add_argument("-s", "--show", action="store_true", help="Show video comparison.")
    args = parser.parse_args()
    main(args)
