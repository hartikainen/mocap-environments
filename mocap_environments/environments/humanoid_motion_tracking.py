"""Humanoid motion tracking composer environment."""

# pylint: disable=duplicate-code

import pathlib
from typing import Literal, Optional

from dm_control import composer
from dm_control.locomotion import arenas
import numpy as np

from mocap_environments.tasks import tracking as tracking_task
from mocap_environments.walkers import simple_humanoid as simple_humanoid_walker


Path = pathlib.Path


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
        mjpc_task_xml_file_path = (
            Path(__file__).parent.parent
            / "tasks"
            / "mjpc_tasks"
            / "simple_humanoid_motion_tracking.xml"
        )
    elif walker_type == "SimpleHumanoid":
        walker_class = simple_humanoid_walker.SimpleHumanoid
        mjpc_task_xml_file_path = (
            Path(__file__).parent.parent
            / "tasks"
            / "mjpc_tasks"
            / "simple_humanoid_motion_tracking.xml"
        )
    else:
        raise ValueError(f"{walker_type=}")

    arena = arenas.Floor(aesthetic="default")

    task = tracking_task.TrackingTask(
        walker=walker_class,
        arena=arena,
        mjpc_task_xml_file_path=mjpc_task_xml_file_path,
        **task_kwargs,
    )

    return composer.Environment(
        time_limit=float("inf"),
        task=task,
        random_state=random_state,
        strip_singleton_obs_buffer_dim=True,
    )
