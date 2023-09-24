"""Humanoid motion tracking composer environment."""

from typing import Literal, Optional

from dm_control import composer
from dm_control.locomotion import arenas
import numpy as np

from mocap_environments.tasks import tracking as tracking_task
from mocap_environments.walkers import simple_humanoid as simple_humanoid_walker
from mocap_environments.walkers import cmu_humanoid as cmu_humanoid_walker


def load(
    *,
    walker_type: Literal[
        "SimpleHumanoid",
        "SimpleHumanoidPositionControlled",
        "CMUHumanoid",
        "CMUHumanoidPositionControlled",
    ],
    random_state: Optional[np.random.RandomState] = None,
    task_kwargs,
):
    """Helper constructor for a mocap tracking environment."""

    if walker_type == "SimpleHumanoidPositionControlled":
        walker_class = simple_humanoid_walker.SimpleHumanoidPositionControlled
    elif walker_type == "SimpleHumanoid":
        walker_class = simple_humanoid_walker.SimpleHumanoid
    elif walker_type == "CMUHumanoidPositionControlled":
        walker_class = cmu_humanoid_walker.CMUHumanoidPositionControlled
    elif walker_type == "CMUHumanoid":
        walker_class = cmu_humanoid_walker.CMUHumanoid
    else:
        raise ValueError(f"{walker_type=}")

    arena = arenas.Floor(aesthetic="default")

    task = tracking_task.TrackingTask(
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
