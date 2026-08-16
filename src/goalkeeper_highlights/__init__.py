import os
os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = os.environ.get("GOALKEEPER_OPENCV_READ_ATTEMPTS", "65536")
__version__ = "0.13.27"
