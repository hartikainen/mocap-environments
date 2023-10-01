import pathlib
from typing import Iterable
import warnings

import numpy as np
import skvideo.io

Path = pathlib.Path


def save_video(
    save_path: Path,
    frames: Iterable[np.ndarray],
    input_fps: int | float = 60,
    output_fps: int | float = 60,
):
    """Write video `frames` into `save_path`.

    Note: This currently uses `skvideo` for writing the videos. `skvideo`
    expects `frames` to be in the form of a 4D numpy array with dimensions
    `(T, H, W, C)`, where `T` is the number of frames, `H` and `W` are the
    height and width of the video, and `C` is the number of channels. The frames
    should be in RGB format.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    inputdict = {"-r": str(input_fps)}

    if save_path.suffix == ".webm":
        outputdict = {
            "-vcodec": "libvpx",
            "-b": "1000000",
        }
    elif save_path.suffix == ".mp4":
        outputdict = {
            "-f": "mp4",
            "-pix_fmt": "yuv420p",  # "-pix_fmt=yuv420p" needed for osx https://github.com/scikit-video/scikit-video/issues/74
        }
    else:
        raise ValueError(
            f"{save_path=} format not recognized. Allowed suffices are: {{'.webm', '.mp4'}}"
        )

    outputdict = {"-r": str(output_fps), **outputdict}

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        skvideo.io.vwrite(
            save_path,
            frames,
            inputdict=inputdict,
            outputdict=outputdict,
        )
