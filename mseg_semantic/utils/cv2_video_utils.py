#!/usr/bin/python3

"""
Python-based utilities to avoid blowing up the disk with images, as FFMPEG requires.
Inspired by Detectron2:
https://github.com/facebookresearch/detectron2/blob/bab413cdb822af6214f9b7f70a9b7a9505eb86c5/demo/demo.py
See OpenCV documentation for more details:
https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html
"""

import cv2
import numpy as np


class VideoWriter:
    """
    Lazy init, so that the user doesn't have to know width/height a priori.
    Our default codec is "mp4v", though you may prefer "x264", if available
    on your system
    """

    def __init__(self, output_fpath: str, fps: int = 30) -> None:
        """ """
        self.output_fpath = output_fpath
        self.fps = fps
        self.writer = None
        self.codec = "mp4v"

    def init_outf(self, height: int, width: int) -> None:
        """ """
        self.writer = cv2.VideoWriter(
            filename=self.output_fpath,
            # some installation of opencv may not support x264 (due to its license),
            # you can try other format (e.g. MPEG)
            fourcc=cv2.VideoWriter_fourcc(*self.codec),
            fps=float(self.fps),
            frameSize=(width, height),
            isColor=True,
        )

    def add_frame(self, rgb_frame: np.ndarray) -> None:
        """ """
        h, w, _ = rgb_frame.shape
        if self.writer is None:
            self.init_outf(height=h, width=w)
        bgr_frame = rgb_frame[:, :, ::-1]
        self.writer.write(bgr_frame)

    def complete(self) -> None:
        """ """
        self.writer.release()


class VideoReader:
    def __init__(self, video_fpath: str) -> None:
        """ """
        self.video = cv2.VideoCapture(video_fpath)
        width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.video.get(cv2.CAP_PROP_FPS)
        print(f"Video fps: {fps:.2f} @ {height}x{width} resolution.")
        self.num_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_frame(self) -> np.ndarray:
        """ """
        if self.video.isOpened():
            success, frame_bgr = self.video.read()
            if success:
                frame_bgr = np.array(frame_bgr)
                frame_rgb = frame_bgr[:, :, ::-1]
                return frame_rgb
            else:
                return None

    def complete(self) -> None:
        """ """
        self.video.release()
