"""Tests for Simple Humanoid DAgger dataset."""

import pathlib
import time
import unittest

from absl.testing import absltest
from absl.testing import parameterized

from . import simple_humanoid_amass

Path = pathlib.Path

DATA_PATH = (Path(__file__).parent / "physics_data").expanduser()


@unittest.skipIf(
    not DATA_PATH.exists(), f"Unable to load tracking data from '{DATA_PATH}'. Skipping"
)
class SimpleHumanoidTrackingTest(parameterized.TestCase):
    def test_simple(self):
        data_path = DATA_PATH
        dataset = simple_humanoid_amass.load_dataset(
            data_path=data_path,
            mocap_id_filter_regex=r"^CMU/CMU/(?:.*/.*)_poses(?:\.npy|\.npz|\.xml)?$",
        )
        iterator = dataset.as_numpy_iterator()
        sequence = next(iterator)

        self.assertEqual(sequence["keyframes"].ndim, 3)
        num_timesteps, num_joints = sequence["keyframes"].shape[:2]
        self.assertIsInstance(sequence["mocap_id"], bytes)
        self.assertEqual(sequence["keyframes"].shape, (num_timesteps, num_joints, 3))
        self.assertEqual(
            sequence["qpos"].shape, (num_timesteps, sequence["qvel"].shape[-1] + 1)
        )
        self.assertEqual(
            sequence["qvel"].shape, (num_timesteps, sequence["qpos"].shape[-1] - 1)
        )

    def test_throughput(self):
        data_path = DATA_PATH
        num_iterations = 1000
        num_warmup_iterations = 5

        dataset = simple_humanoid_amass.load_dataset(
            data_path=data_path,
            mocap_id_filter_regex=r"^CMU/CMU/(?:.*/.*)_poses(?:\.npy|\.npz|\.xml)?$",
        ).repeat(num_iterations + num_warmup_iterations)
        iterator = dataset.as_numpy_iterator()

        for _ in range(num_warmup_iterations):
            _ = next(iterator)

        start = time.time()
        for _ in range(num_iterations):
            _ = next(iterator)
        end = time.time()

        duration = end - start

        # NOTE(hartikainen): 100 sequences per second here is actually quite
        # low but the github workflow tests fail with larger values. M1 Macbook
        # Air achieves ~500 sequences per second. Also note that this is run
        # without caching.
        desired_sequences_per_second = 100
        actual_sequences_per_second = num_iterations / duration

        self.assertLess(desired_sequences_per_second, actual_sequences_per_second)


if __name__ == "__main__":
    absltest.main()
