"""Tests for the MJPC expert in humanoid tracking task."""

import pathlib
import unittest

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from mocap_environments.data import simple_humanoid_amass
from mocap_environments.environments import humanoid_motion_tracking

try:
    from . import mjpc_expert
except ImportError:
    mjpc_expert = None

Path = pathlib.Path

DATA_PATH = Path(simple_humanoid_amass.__file__).parent / "physics_data"


@unittest.skipIf(
    not DATA_PATH.exists(), f"Unable to load tracking data from '{DATA_PATH}'. Skipping"
)
class MJPCExpertTest(parameterized.TestCase):
    @unittest.skipIf(mjpc_expert is None, "unable to find `mjpc_expert`.")
    def test_step_environment(self):
        data_path = DATA_PATH

        motion_dataset = simple_humanoid_amass.load_dataset(
            data_path=data_path,
            mocap_id_filter_regex=r"^CMU/CMU/(?:108/108_13)_poses(?:\.npy|\.npz|\.xml)?$",
        )

        environment = humanoid_motion_tracking.load(
            walker_type="SimpleHumanoidPositionControlled",
            random_state=None,
            task_kwargs={
                "termination_threshold": 1.0,
                "motion_dataset": motion_dataset,
                "mocap_reference_steps": 0,
                "random_init_time_step": False,
            },
        )

        expert = mjpc_expert.MJPCExpert(
            warm_start_steps=100,
            warm_start_tolerance=1e-3,
        )

        time_steps = [environment.reset()]
        expert.observe_first(time_steps[-1], environment)

        while not (time_step_0 := time_steps[-1]).last():
            action = expert.select_action(time_step_0, environment)
            time_step_1 = environment.step(action)
            expert.observe(action, time_step_1, environment)
            time_steps.append(time_step_1)

        self.assertGreater(len(time_steps), 10)
        rewards = np.array([ts.reward for ts in time_steps[1:]])
        np.testing.assert_array_less(0.9, rewards)


if __name__ == "__main__":
    absltest.main()
