"""Visualization functions."""

from typing import Sequence

import chex
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt
import numpy as np


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

    pixels = np.frombuffer(figure.canvas.tostring_rgb(), dtype=np.uint8)
    pixels = pixels.reshape((*figure.canvas.get_width_height()[::-1], 3))

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
