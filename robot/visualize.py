import os
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


def save_frames_to_video(frames: Sequence[np.ndarray], video_path: Path | str, fps: int = 20):
  """
  Save a sequence of RGB images with shape (height, width, 3) to video.
  Color channels should be of type float32 and bounded in [0.0, 1.0].
  """
  video_writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter.fourcc(*'H264'), fps, tuple(reversed(frames[0].shape[:-1])))
  for frame in frames:
    video_writer.write(cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
  video_writer.release()
