"""Retargeting functions specific to `dm_control.lomotion.walkers`"""

from typing import Any, Optional

from dm_control import mjcf
from dm_control.locomotion.walkers import base as walkers_base
import mujoco
import numpy as np
import numpy.typing as npt

from mocap_environments.retargeting import inverse_kinematics as ik


def compute_inverse_kinematics_qpos_qvel(
    walker: walkers_base.Walker,
    physics: mjcf.Physics,
    keyframes: npt.ArrayLike,
    keyframe_fps: int | float,
    root_sites_names: tuple[str, str, str],
    root_keyframe_indices: tuple[int, int, int],
    root_joints_names: tuple[str],
    rest_sites_names: tuple[str, ...],
    rest_keyframe_indices: tuple[int, ...],
    rest_joints_names: tuple[str, ...],
    end_effector_sites_names: tuple[str, ...],
    end_effector_keyframe_indices: tuple[int, ...],
    end_effector_joints_names: tuple[str, ...],
    ik_kwargs: Optional[dict[str, Any]] = None,
    qvel_step_size: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Computes `walker` `physics` states to match poses in cartesian `keyframes`.

    This returns a (currently incomplete) physics states so as to minimize the cartesian
    site positions with the keyframe positions. This computation is done in two stages:
    First, we translate and rotate the root joint to match the three *root* sites with
    the three root keyframe positions. We then fix the root joint and find the positions
    of the *rest* of the joints so as to match the rest sites with the rest keyframe
    positions.

    Args:
      walker: A `base.Walker` instance.
      physics: A `mjcf.Physics` instance.
      root_sites_names: A sequence of strings specifying the names of the *root* target
        sites.
      root_keyframe_indices: A sequence of ints specifying the indices for `keyframes`,
        which match the site
      root_joints_names: A sequence of strings specifying the names of the *root* joints
        that should be controlled in order to translate and rotate the root in the first
        step.
      rest_sites_names: Similar to `root_site_names`, except for the rest of the joints
        for the second step.
      rest_keyframe_indices: Similar to `root_keyframe_indices`, except for the rest of
        the joints for the second step.
      rest_joints_names: Similar to `rest_joints_names`, except for the rest of the
        joints for the second step.
      ik_kwargs: arguments to be passed to `ik.qpos_from_site_pose`.
      qvel_step_size: Central differencing step size.
    Returns:
      A (currently incomplete) physics state for each keyframe.
    """
    del walker

    keyframes = np.array(keyframes)

    if ik_kwargs is None:
        ik_kwargs = dict(
            tol=1e-14,
            regularization_threshold=0.5,
            regularization_strength=1e-2,
            max_update_norm=2.0,
            progress_thresh=5000.0,
            max_steps=10_000,
            inplace=False,
            null_space_method=False,
        )

    ik_result_0 = ik.qpos_from_site_pose(
        physics=physics,
        sites_names=list(root_sites_names),
        target_pos=keyframes[0, list(root_keyframe_indices), ...],
        target_quat=None,
        joint_names=list(root_joints_names),
        **ik_kwargs,
    )

    with physics.reset_context():
        physics.named.data.qpos[:7] = ik_result_0.qpos[:7]

    qposes = []
    for keyframe in keyframes:
        ik_result_1 = ik.qpos_from_site_pose(
            physics=physics,
            sites_names=list(root_sites_names + rest_sites_names),
            target_pos=keyframe[list(root_keyframe_indices + rest_keyframe_indices)],
            target_quat=None,
            joint_names=list(root_joints_names + rest_joints_names),
            **ik_kwargs,
        )

        with physics.reset_context():
            physics.named.data.qpos[:] = ik_result_1.qpos

        ik_result_2 = ik.qpos_from_site_pose(
            physics=physics,
            sites_names=list(end_effector_sites_names),
            target_pos=keyframe[list(end_effector_keyframe_indices)],
            target_quat=None,
            joint_names=list(end_effector_joints_names),
            **ik_kwargs,
        )

        with physics.reset_context():
            physics.named.data.qpos[:] = ik_result_2.qpos

        qposes.append(physics.data.qpos.copy())

    qposes = np.stack(qposes)

    def differentiate_positions(
        qposes: npt.ArrayLike, max_step_size: int, keyframe_fps: float
    ) -> np.ndarray:
        qposes = np.asarray(qposes)
        num_qposes = qposes.shape[0]
        qvels = []

        qpos1_indices = np.maximum(np.arange(num_qposes) - max_step_size, 0)
        qpos2_indices = np.minimum(
            np.arange(num_qposes) + max_step_size, num_qposes - 1
        )
        np.testing.assert_equal(qpos1_indices.shape, qpos2_indices.shape)

        for qpos1_i, qpos2_i in zip(qpos1_indices, qpos2_indices):
            step_size = qpos2_i - qpos1_i
            dt = step_size / keyframe_fps

            qpos1 = qposes[qpos1_i, ...]
            qpos2 = qposes[qpos2_i, ...]

            qvel = np.empty_like(physics.data.qvel)
            mujoco.mj_differentiatePos(
                physics.model._model,
                qvel,
                dt=dt,
                qpos1=qpos1,
                qpos2=qpos2,
            )
            qvels.append(qvel)

        return np.stack(qvels)

    qvels = differentiate_positions(qposes, qvel_step_size, keyframe_fps)

    return qposes, qvels
