"""Tests for the tracking task."""

from absl.testing import absltest
from absl.testing import parameterized
from dm_control import composer
from dm_control import mjcf
from dm_control.locomotion import arenas
import numpy as np
from tensorflow import data as tf_data

from mocap_environments.environments import humanoid_motion_tracking
from mocap_environments.walkers import simple_humanoid
from mocap_environments.walkers import smpl_humanoid

from . import tracking


class TrackingTest(parameterized.TestCase):
    @parameterized.parameters(
        [
            simple_humanoid.SimpleHumanoid,
            simple_humanoid.SimpleHumanoidPositionControlled,
            smpl_humanoid.SMPLHumanoid,
            smpl_humanoid.SMPLHumanoidPositionControlled,
        ]
    )
    def test_reset_and_step_simple(self, walker_type):
        walker = walker_type()
        num_time_steps = 4
        num_joints = len(walker.observable_joints)
        num_tracking_sites = len(walker.mocap_tracking_sites)

        def get_joint_range(joint: mjcf.Element) -> np.array:
            if joint.range is not None:
                joint_range = joint.range
            elif joint.dclass.joint.range is not None:
                joint_range = joint.dclass.joint.range
            elif joint.dclass.parent.joint.range is not None:
                joint_range = joint.dclass.parent.joint.range
            else:
                raise ValueError(f"Joint range not found for `{joint.name}`")
            return joint_range

        joint_ranges = np.array(
            [get_joint_range(oj) for oj in walker.observable_joints]
        )
        joint_lows, joint_highs = joint_ranges[:, 0], joint_ranges[:, 1]

        qpos = np.random.uniform(
            low=np.concatenate([np.full(7, -1.0), joint_lows]),
            high=np.concatenate([np.full(7, +1.0), joint_highs]),
            size=(num_time_steps, num_joints + 7),
        )
        qpos[..., 3:7] = qpos[..., 3:7] / (
            np.linalg.norm(qpos[..., 3:7], axis=-1)[..., None]
        )
        qvel = np.random.uniform(
            low=np.full(num_joints + 6, -10.0),
            high=np.full(num_joints + 6, +10.0),
            size=(num_time_steps, num_joints + 6),
        )
        root_joint_index = next(
            (
                i
                for i, site in enumerate(walker.mocap_tracking_sites)
                if site.name == f"tracking[{walker.root_body.name}]"
            ),
            None,
        )
        if root_joint_index is None:
            raise RuntimeError(f"Root joint ({walker.root_body.name}) not found.")

        keyframes = np.random.uniform(
            low=np.concatenate(
                [
                    np.full(root_joint_index, -0.3),
                    np.full(1, -5.0),
                    np.full(num_tracking_sites - root_joint_index - 1, -0.3),
                ]
            )[..., None],
            high=np.concatenate(
                [
                    np.full(root_joint_index, +0.3),
                    np.full(1, +5.0),
                    np.full(num_tracking_sites - root_joint_index - 1, +0.3),
                ]
            )[..., None],
            size=(num_time_steps, num_tracking_sites, 3),
        )

        motion_dataset = tf_data.Dataset.from_tensors(
            {
                "qpos": qpos,
                "qvel": qvel,
                "keyframes": keyframes,
                "motion_id": "MOTION_ID",
            }
        )

        environment = humanoid_motion_tracking.load(
            walker_type=walker.__class__.__name__,
            random_state=np.random.RandomState(seed=0),
            task_kwargs={
                "termination_threshold": float("inf"),
                "motion_dataset": motion_dataset,
                "mocap_reference_steps": 0,
                "random_init_time_step": False,
            },
        )

        def verify_keyframes(environment, walker, expected_keyframe_global, time_step):
            expected_keyframes_local = walker.transform_vec_to_egocentric_frame(
                environment.physics,
                expected_keyframe_global
                - environment.physics.bind(walker.mocap_tracking_sites).xpos,
            )
            np.testing.assert_allclose(
                time_step.observation["walker/keyframes_local"],
                expected_keyframes_local,
                atol=1e-5,
            )

        time_steps = [environment.reset()]
        np.testing.assert_allclose(environment.physics.data.qpos, qpos[0], atol=1e-10)
        np.testing.assert_allclose(environment.physics.data.qvel, qvel[0], atol=1e-10)
        verify_keyframes(
            environment, environment.task._walker, keyframes[0], time_steps[0]
        )

        for i in range(1, num_time_steps):
            action = environment.action_spec().generate_value()
            time_steps.append(environment.step(action))
            verify_keyframes(
                environment, environment.task._walker, keyframes[i], time_steps[i]
            )


if __name__ == "__main__":
    absltest.main()
