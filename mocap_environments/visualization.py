"""Visualization functions."""

from typing import Callable, Optional, Sequence

import chex
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


def add_text_overlays_to_frames(
    frames: Sequence[np.ndarray],
    overlay_function: Callable[[int], str],
    color: npt.ArrayLike = (255, 0, 0),
    position: npt.ArrayLike = (0, 0),
    font: Optional[ImageFont.FreeTypeFont] = None,
):
    color = np.array(color).tolist()
    position = np.array(position).tolist()
    if font is None:
        font = ImageFont.truetype("DejaVuSansMono.ttf", 20)

    modified_frames = []
    for t, frame in enumerate(frames):
        image = Image.fromarray(frame)
        text_overlay_str = overlay_function(t)
        draw = ImageDraw.Draw(image)
        draw.text(position, text_overlay_str, font=font, fill=color)
        modified_frames.append(np.array(image))

    return modified_frames


def plot_action_pixels(
    action: np.ndarray,
    figsize: tuple[float | int, float | int],
    dpi: int | float,
) -> np.ndarray:
    action_dim = action.size

    figure = plt.figure(figsize=figsize, constrained_layout=True, dpi=dpi)
    canvas = FigureCanvasAgg(figure)
    figure.dpi = dpi

    gridspec = figure.add_gridspec(1, 1, hspace=0, wspace=0)
    gridspec.update(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    axis = gridspec.subplots(sharex=True, sharey=True)

    axis.grid("on", which="both", linestyle="--")
    axis.axhline(y=0, color="k")
    axis.set_ylim(-1, +1)
    axis.set_xlim(0, action_dim)
    axis.tick_params(labelbottom=False)
    axis.set_xticks(np.arange(action_dim))
    axis.set_xticklabels([])
    axis.set_yticklabels([])
    axis.tick_params(
        top=False,
        bottom=False,
        left=False,
        right=False,
        labelleft=False,
        labelbottom=False,
    )

    _ = axis.step(
        np.arange(action_dim + 1),
        np.concatenate([action, action[[-1]]]),
        color="k",
        where="post",
        linewidth=1,
    )

    figure.canvas.draw()

    pixels = np.array(figure.canvas.buffer_rgba())[..., :3]

    figure.clear()
    plt.close(figure)

    return pixels


def plot_actions_pixels(
    actions: Sequence[np.ndarray],
    figsize: tuple[float | int, float | int],
    dpi: float | int,
) -> Sequence[np.ndarray]:
    action_pixels = [plot_action_pixels(action, figsize, dpi) for action in actions]
    return action_pixels


def attach_action_pixels_to_frames(
    frames: Sequence[np.ndarray], action_pixels: Sequence[np.ndarray]
) -> Sequence[np.ndarray]:
    np.testing.assert_equal(frames[0].shape[-2:], action_pixels[0].shape[-2:])
    action_pixels_height = np.shape(action_pixels[0])[-3]
    new_frames = []
    for frame, action_pixels in zip(frames[:-1], action_pixels):
        new_frame = frame.copy()
        new_frame[:action_pixels_height] = action_pixels
        new_frames.append(new_frame)
    new_frames.append(frames[-1])
    return new_frames


def add_action_plots_to_frames(
    frames: Sequence[np.ndarray],
    actions: Sequence[np.ndarray],
    action_shape_ratio: tuple[float, float] = (1.0, 0.125),
) -> Sequence[np.ndarray]:
    np.testing.assert_equal(len(frames) - 1, len(actions))
    chex.assert_trees_all_equal_shapes(*frames)
    chex.assert_rank(frames[0], 3)
    chex.assert_trees_all_equal_shapes(*actions)
    chex.assert_rank(actions[0], 1)
    action_shape_ratio = np.array(action_shape_ratio)

    frame_figsize_pixels = np.array(frames[0].shape[0:2][::-1])
    dpi = 75
    action_pixels_figsize = frame_figsize_pixels * action_shape_ratio / dpi

    action_pixels = plot_actions_pixels(actions, action_pixels_figsize, dpi)
    frames = attach_action_pixels_to_frames(frames, action_pixels)
    return frames
