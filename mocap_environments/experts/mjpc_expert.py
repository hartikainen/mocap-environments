"""A MuJoCo MPC planner expert."""

from typing import Any, Optional

from dm_control.rl import control
import dm_env
import numpy as np

import mujoco_mpc
import mujoco_mpc.agent

NestedArray = Any  # TODO(hartikainen)


class MJPCExpert:
    """Agent that plans using the MuJoCo MPC (MJPC) api."""

    def __init__(
        self,
        warm_start_steps: int = 0,
        warm_start_tolerance: float = float("inf"),
        mjpc_workers: Optional[int] = None,
    ):
        self.agent = None
        self.environment = None
        self._warm_start_steps = warm_start_steps
        self._warm_start_tolerance = warm_start_tolerance
        self._mjpc_workers = mjpc_workers

    def select_action(
        self, time_step: dm_env.TimeStep, environment: control.Environment
    ):
        """Select an action by using the MJPC planner.

        Args:
          time_step: unused
          environment: unused

        Returns:
          an action in the form of np array.
        """
        del time_step
        assert self.agent is not None
        assert self.environment is environment and self.environment is not None

        data = self.environment.physics.data._data  # pylint: disable=protected-access
        self.agent.set_state(
            time=data.time,
            qpos=data.qpos,
            qvel=data.qvel,
            act=data.act,
            mocap_pos=data.mocap_pos,
            mocap_quat=data.mocap_quat,
            userdata=data.userdata,
        )
        # TODO(hartikainen): Should perhaps allow multiple planning steps here
        # the same way as for the reset below.
        self.agent.planner_step()
        action = self.agent.get_action(data.time)
        return action

    def observe_first(
        self, time_step: dm_env.TimeStep, environment: control.Environment
    ):
        """Observe the first timestep and reset the agent for the episode."""
        del time_step

        if environment._n_sub_steps != 1:  # pylint: disable=protected-access
            raise ValueError(
                "MJPCExpert currently expects the physics and control timestep to "
                f"be equal. Got `{environment._n_sub_steps=}`"  # pylint: disable=protected-access
            )

        self.environment = environment

        # TODO(mjpc-dagger): might have to change this depending on the C++
        # implementation.
        task_id = "Humanoid Track"

        if self.agent is not None:
            self.agent.close()

        if self._mjpc_workers is not None:
            extra_flags = ["--mjpc_workers=4"]
        else:
            extra_flags = []

        self.agent = mujoco_mpc.agent.Agent(
            task_id,
            environment.physics.model._model,  # pylint: disable=protected-access
            extra_flags=extra_flags,
        )

        data = self.environment.physics.data._data  # pylint: disable=protected-access
        self.agent.set_state(
            time=data.time,
            qpos=data.qpos,
            qvel=data.qvel,
            act=data.act,
            mocap_pos=data.mocap_pos,
            mocap_quat=data.mocap_quat,
            userdata=data.userdata,
        )

        # The initial `ctrl` of zeros can be quite far off from the optimal and
        # thus we warm the agent up to get a reasonable starting point for the
        # planning. Not doing this may cause e.g. the agent to jerk weirdly when
        # using `act_dot` regularization because the initial `act` of zeros can
        # be very far from the desired `act`.
        action0 = self.agent.get_action(data.time)
        for _ in range(self._warm_start_steps):
            self.agent.planner_step()
            action1 = self.agent.get_action(data.time)

            if np.all(np.abs(action0 - action1) < self._warm_start_tolerance):
                break

            action0 = action1

    def observe(
        self,
        action: NestedArray,
        next_time_step: dm_env.TimeStep,
        environment: control.Environment,
    ):
        """Just checks that the environment is unchanged within the episode."""
        del action, next_time_step
        assert self.environment is environment

    def update(self, wait: bool = False):
        """No-op to conform the API."""
        del wait

    def __del__(self):
        if self.agent is not None:
            self.agent.close()
