import cv2
import threading
import queue
import multiprocessing as mp
from typing import Union
from pathlib import Path
import configparser


class VideoReader(threading.Thread):
    _reader = None
    _lock = threading.Lock()
    _output_queue = queue.Queue()
    _shutdown_signal = threading.Event()

    def __init__(self, reader: cv2.VideoCapture = None, buffer: int = None):
        if reader is not None:
            VideoReader._reader = reader
        if buffer is not None:
            VideoReader._output_queue.maxsize = buffer
        threading.Thread.__init__(self)

    @staticmethod
    def stop():
        VideoReader._shutdown_signal.set()

    @classmethod
    def set_reader(cls, reader: cv2.VideoCapture):
        cls._reader = reader

    @staticmethod
    def get_img():
        try:
            return VideoReader._output_queue.get(True, 0.1)
        except queue.Empty:
            return (-1, None)

    def run(self):
        while not VideoReader._shutdown_signal.is_set():
            if self._lock.acquire(True, 0.1):
                ret, frame = self._reader.read()
                frame_number = int(self._reader.get(cv2.CAP_PROP_POS_FRAMES))
                if ret:
                    try:
                        VideoReader._output_queue.put((frame_number, frame), True, 0.1)
                    except queue.Full:
                        self._reader.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
                self._lock.release()


class FrameSequencer(threading.Thread):
    def __init__(self, input_queue: Union[mp.Queue, queue.Queue], output_queue: [mp.Queue, queue.Queue]):
        self._input_queue = input_queue
        self._output_queue = output_queue
        self._shutdown_event = threading.Event()
        self._current_frame = 1
        self._buffer = dict()
        self._lock = threading.Lock()
        threading.Thread.__init__(self)

    def stop(self):
        self._shutdown_event.set()

    def run(self):
        while not self._shutdown_event.is_set():
            if self._lock.acquire(True, 0.1):
                try:
                    frame_number, frame = self._input_queue.get(True, 0.1)
                    self._buffer.update({frame_number: frame})
                except queue.Empty:
                    pass
                if self._current_frame in self._buffer:
                    try:
                        self._output_queue.put((self._current_frame, self._buffer.pop(self._current_frame)))
                        self._current_frame = self._current_frame + 1
                    except queue.Full:
                        pass
                self._lock.release()

    def release(self):
        if not self._shutdown_event.is_set():
            self.stop()
        self.join()


class IOThreadManager:
    def __init__(self, configuration_file: Union[str, Path], reader: cv2.VideoCapture):
        parser = configparser.ConfigParser()
        parser.read(configuration_file)
        workers = int(parser.get("PERFORMANCE", "iothreads"))
        assert workers > 0, "Number of iothreads must be greater than 0."
        buffersize = int(parser.get("PERFORMANCE", "buffersize"))
        self._is_running = False
        self._workers = [VideoReader(reader, buffersize) for _ in range(workers)]

    def start(self):
        if self._is_running:
            print("Reader threads already running!")
            return
        [worker.start() for worker in self._workers]
        self._is_running = True

    def stop(self):
        if not self._is_running:
            print("Reader threads already stopped!")
            return
        VideoReader.stop()
        self._is_running = False

    def release(self):
        if self._is_running:
            self.stop()
        [(worker.join(), f"{worker.name} quit!") for worker in self._workers]

    @staticmethod
    def get_queue():
        return VideoReader._output_queue

