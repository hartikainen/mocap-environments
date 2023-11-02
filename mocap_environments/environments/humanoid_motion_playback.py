"""Humanoid motion playback composer environment."""

# pylint: disable=duplicate-code

from typing import Literal, Optional

from dm_control import composer
from dm_control.locomotion import arenas
import numpy as np

from mocap_environments.tasks import playback as playback_task
from mocap_environments.walkers import simple_humanoid as simple_humanoid_walker


def load(
    *,
    walker_type: Literal[
        "SimpleHumanoid",
        "SimpleHumanoidPositionControlled",
    ],
    random_state: Optional[np.random.RandomState] = None,
    task_kwargs,
) -> composer.Environment:
    """Helper constructor for a mocap tracking environment."""

    if walker_type == "SimpleHumanoidPositionControlled":
        walker_class = simple_humanoid_walker.SimpleHumanoidPositionControlled
    elif walker_type == "SimpleHumanoid":
        walker_class = simple_humanoid_walker.SimpleHumanoid
    else:
        raise ValueError(f"{walker_type=}")

    arena = arenas.Floor(aesthetic="default")

    task = playback_task.PlaybackTask(
        walker=walker_class,
        arena=arena,
        **task_kwargs,
    )

    return composer.Environment(
        time_limit=float("inf"),
        task=task,
        random_state=random_state,
        strip_singleton_obs_buffer_dim=True,
    )
