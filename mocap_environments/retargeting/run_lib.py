"""Functionality for retargeting scripts."""

import json
import pathlib
import re
from typing import Any, Callable, Literal, Optional

from absl import logging
from dm_control import composer
from dm_control import mjcf
from dm_control.locomotion.tasks.reference_pose import utils as reference_pose_utils
from dm_control.rl import control
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import tensorflow as tf
import tree

from mocap_environments import walkers
from mocap_environments.environments import humanoid_motion_tracking
from mocap_environments.experts import mjpc_expert
from mocap_environments.retargeting import dm_control_walker
from mocap_environments.retargeting import smpl
from mocap_environments.utils import video as video_utils
from mocap_environments.visualization import video as video_visualization

JnpArrayType = jnp.array
NpArrayType = np.ndarray
Path = pathlib.Path
WalkerEnum = Literal["SimpleHumanoidPositionController", "SimpleHumanoid"]
PhysicsEnvironment = control.Environment | composer.Environment

# pylint: disable=logging-fstring-interpolation


def convert_fps(
    keyframes: NpArrayType,
    source_fps: int | float,
    target_fps: int | float,
) -> NpArrayType:
    """Convert `keyframes` fps by linearly interpolating between frames."""
    fps_rate_float = target_fps / source_fps
    sequence_length = keyframes.shape[0]
    target_sequence_length = np.int32(np.ceil(sequence_length * fps_rate_float))

    # NOTE(hartikainen): Linearly interpolate the frames to achieve desired fps.
    frame_indices_float = np.linspace(
        0,
        sequence_length - 1,
        target_sequence_length,
    )
    frame_indices_floor = np.floor(frame_indices_float).astype("int")
    frame_indices_ceil = np.ceil(frame_indices_float).astype("int")

    ceil_frame_contribution = np.reshape(
        frame_indices_float - frame_indices_floor,
        (*frame_indices_float.shape, *[1] * (keyframes.ndim - 1)),
    )
    floor_frame_contribution = 1.0 - ceil_frame_contribution

    def take_interpolated_frames(frames):
        interpolated_frames = (
            floor_frame_contribution * frames[frame_indices_floor]
            + ceil_frame_contribution * frames[frame_indices_ceil]
        )
        return interpolated_frames

    keyframes = take_interpolated_frames(keyframes)

    return keyframes


def rollout_policy_and_save_video(
    policy,
    env_fn: Callable[[], PhysicsEnvironment],
    max_num_steps: int = 30_000,
    *,
    video_save_path: Path,
    output_fps: int | float = 60,
    render_kwargs: Optional[dict[str, Any]] = None,
):
    if render_kwargs is None:
        render_kwargs = {"width": 640, "height": 640, "camera_id": "walker/back"}

    environment = env_fn()

    actions = []
    time_steps = [environment.reset()]
    frames = [environment.physics.render(**render_kwargs)]
    mujoco_states = [
        {
            "qpos": environment.physics.data.qpos.copy(),
            "qvel": environment.physics.data.qvel.copy(),
        }
    ]

    assert time_steps[-1].reward is None, time_steps[-1]
    assert time_steps[-1].discount is None, time_steps[-1]

    policy.observe_first(time_steps[-1], environment)

    def get_mjpc_cost():
        weighted_cost_term_values = tree.map_structure(
            lambda cost, value: cost * value,
            policy.agent.get_cost_weights(),
            policy.agent.get_cost_term_values(),
        )
        return weighted_cost_term_values

    mjpc_costs = [get_mjpc_cost()]

    while (t := len(time_steps)) <= max_num_steps and not time_steps[-1].last():
        action = policy.select_action(time_steps[-1].observation, environment)
        actions.append(action)
        time_steps.append(environment.step(action))
        policy.agent.step()
        mujoco_states.append(
            {
                "qpos": environment.physics.data.qpos.copy(),
                "qvel": environment.physics.data.qvel.copy(),
            }
        )
        policy.observe(action, time_steps[-1], environment)
        mjpc_costs.append(get_mjpc_cost())
        frames.append(environment.physics.render(**render_kwargs))

    mocap_id = environment.task._motion_sequence["mocap_id"].decode()

    frames = video_visualization.add_action_plots_to_frames(frames, actions)
    frames = video_visualization.add_text_overlays_to_frames(
        frames,
        overlay_function=lambda t: mocap_id,
        position=(frames[0].shape[1] / 2, frames[0].shape[0] * 0.9),
        anchor="md",
    )

    Path(video_save_path).parent.mkdir(parents=True, exist_ok=True)
    video_save_path = video_save_path.with_stem(
        video_save_path.stem.format(mocap_id=mocap_id.replace("/", "+"))
    )

    video_utils.save_video(
        video_save_path,
        frames,
        input_fps=round(1.0 / environment.control_timestep(), 6),
        output_fps=output_fps,
    )

    num_time_steps = len(time_steps)
    assert num_time_steps == t, (num_time_steps, t)

    rewards = np.array(
        [ts.reward["tracking"] for ts in time_steps if ts.reward is not None]
    )

    return {
        "rewards": rewards,
        "num_time_steps": num_time_steps,
        "time_steps": time_steps,
        "mujoco_states": mujoco_states,
    }


def process_motion(
    keyframe_path: Path,
    keyframes_base_dir: Path,
    *,
    save_base_dir: Path,
    keyframe_fps: int | float = 60.0,
    override: bool = False,
    walker_type: WalkerEnum,
):
    keyframe_file = keyframes_base_dir / keyframe_path
    logging.info(f"Processing file '{keyframe_file}'")

    amass_id = str(keyframe_path.with_suffix(""))

    result_file = (
        save_base_dir / "data" / "valid" / amass_id.replace("/", "+")
    ).with_suffix(".npy")

    if not override and result_file.exists():
        logging.info(f"Output file ('{result_file}') already exists. Skipping.")
        return

    with keyframe_file.open("rb") as f:
        keyframes = np.load(f)

    # TODO(hartikainen): Make `output_fps` more flexible. The tracking environment
    # shouldn't care about the exact fps but rather automatically handle interpolation
    # between frames.
    output_fps = 60.0
    keyframes = convert_fps(keyframes, source_fps=keyframe_fps, target_fps=output_fps)

    if walker_type == "SimpleHumanoidPositionControlled":
        walker_class = walkers.SimpleHumanoidPositionControlled
    elif walker_type == "SimpleHumanoid":
        walker_class = walkers.SimpleHumanoid
    else:
        raise ValueError(f"{walker_type=}")

    empty_arena = composer.Arena()
    walker = reference_pose_utils.add_walker(
        walker_fn=walker_class, arena=empty_arena, name="walker"
    )

    physics = mjcf.Physics.from_xml_string(empty_arena.mjcf_model.to_xml_string())

    physics_to_kinematics_joint_name_map = dict(
        (
            ("pelvis", "pelvis"),
            ("head", "head"),
            ("ltoe", "left_foot"),
            ("rtoe", "right_foot"),
            ("lheel", "left_ankle"),
            ("rheel", "right_ankle"),
            ("lknee", "left_knee"),
            ("rknee", "right_knee"),
            ("lhand", "left_wrist"),
            ("rhand", "right_wrist"),
            ("lelbow", "left_elbow"),
            ("relbow", "right_elbow"),
            ("lshoulder", "left_shoulder"),
            ("rshoulder", "right_shoulder"),
            ("lhip", "left_hip"),
            ("rhip", "right_hip"),
        )
    )

    smpl_joint_names = tuple(smpl.SMPL_JOINT_NAMES)
    site_joint_name_re = re.compile(r"^tracking\[(\w+)\]$")
    site_smpl_indices = []
    # We need sites -> keyframes map for IK.
    for site_element in walker.mocap_tracking_sites:
        site_joint_name_match = re.match(site_joint_name_re, site_element.name)
        assert site_joint_name_match is not None
        site_joint_name = site_joint_name_match.group(1)
        assert site_joint_name in physics_to_kinematics_joint_name_map
        smpl_joint_name = physics_to_kinematics_joint_name_map[site_joint_name]
        site_smpl_index = smpl_joint_names.index(smpl_joint_name)
        site_smpl_indices.append(site_smpl_index)

    site_to_smpl_index_map = dict(
        zip(
            [f"walker/{s.name}" for s in walker.mocap_tracking_sites],
            site_smpl_indices,
        )
    )

    root_sites_names = (
        "walker/tracking[head]",
        "walker/tracking[lshoulder]",
        "walker/tracking[rshoulder]",
    )
    root_keyframe_indices = tuple(
        site_to_smpl_index_map[name] for name in root_sites_names
    )
    assert len(root_keyframe_indices) == 3, root_keyframe_indices
    rest_sites_names = (
        "walker/tracking[pelvis]",
        "walker/tracking[lhip]",
        "walker/tracking[rhip]",
    )
    rest_keyframe_indices = tuple(
        site_to_smpl_index_map[name] for name in rest_sites_names
    )

    end_effector_sites_names = (
        "walker/tracking[ltoe]",
        "walker/tracking[rtoe]",
        "walker/tracking[lheel]",
        "walker/tracking[rheel]",
        "walker/tracking[lknee]",
        "walker/tracking[rknee]",
        "walker/tracking[lhand]",
        "walker/tracking[rhand]",
        "walker/tracking[lelbow]",
        "walker/tracking[relbow]",
    )
    end_effector_keyframe_indices = tuple(
        site_to_smpl_index_map[name] for name in end_effector_sites_names
    )
    end_effector_joints_names = (
        "walker/hip_x_right",
        "walker/hip_z_right",
        "walker/hip_y_right",
        "walker/knee_right",
        "walker/ankle_y_right",
        "walker/ankle_x_right",
        "walker/hip_x_left",
        "walker/hip_z_left",
        "walker/hip_y_left",
        "walker/knee_left",
        "walker/ankle_y_left",
        "walker/ankle_x_left",
        "walker/shoulder1_right",
        "walker/shoulder2_right",
        "walker/elbow_right",
        "walker/shoulder1_left",
        "walker/shoulder2_left",
        "walker/elbow_left",
    )

    def translate_keyframes(
        keyframes: npt.ArrayLike,
        initial_translation: Optional[npt.ArrayLike] = None,
    ):
        keyframes = np.array(keyframes)
        if initial_translation is None:
            initial_translation = np.zeros(3, dtype=keyframes.dtype)
        initial_translation = np.array(initial_translation)

        pelvis_zs = keyframes[:, site_to_smpl_index_map[f"walker/tracking[pelvis]"], 2]
        ltoe_zs = keyframes[:, site_to_smpl_index_map[f"walker/tracking[ltoe]"], 2]
        rtoe_zs = keyframes[:, site_to_smpl_index_map[f"walker/tracking[rtoe]"], 2]
        min_toe_zs = np.minimum(ltoe_zs, rtoe_zs)
        pelvis_to_min_toe_cm = pelvis_zs - min_toe_zs
        standing_mask = (0.75 < pelvis_to_min_toe_cm) & (pelvis_to_min_toe_cm < 0.95)
        if standing_mask.any():
            min_z = min_toe_zs[standing_mask].min()
        else:
            min_z = keyframes[
                :,
                [
                    site_to_smpl_index_map[f"walker/tracking[{site_name}]"]
                    for site_name in [
                        "ltoe",
                        "lheel",
                        "lhand",
                        "rtoe",
                        "rheel",
                        "rhand",
                        "pelvis",
                    ]
                ],
                2,
            ].min()

        keyframes[..., 2] -= min_z

        # Set starting position to origin.
        keyframes[..., 0:2] -= keyframes[
            ...,
            0,
            site_to_smpl_index_map[f"walker/tracking[pelvis]"],
            0:2,
        ]

        keyframes[..., :, :, :] += initial_translation

        return keyframes

    if keyframes.shape[-2] == 24:
        keyframes = keyframes[..., 0:22, :]
    elif keyframes.shape[-2] == 22:
        pass
    else:
        raise ValueError(f"{keyframes.shape=}")

    keyframes = translate_keyframes(keyframes)
    velocity_t = min(keyframes.shape[0] - 1, 6)
    (
        qposes,
        qvels,
    ) = dm_control_walker.compute_inverse_kinematics_qpos_qvel(
        walker,
        physics,
        keyframes[[0, velocity_t], ...],
        keyframe_fps=keyframe_fps / velocity_t,
        root_sites_names=root_sites_names,
        root_keyframe_indices=root_keyframe_indices,
        root_joints_names=physics.named.data.qpos.axes.row.names[:1],
        rest_sites_names=rest_sites_names,
        rest_keyframe_indices=rest_keyframe_indices,
        rest_joints_names=physics.named.data.qpos.axes.row.names[1:],
        end_effector_sites_names=end_effector_sites_names,
        end_effector_keyframe_indices=end_effector_keyframe_indices,
        end_effector_joints_names=end_effector_joints_names,
    )

    qpos0, qvel0 = qposes[0], qvels[0]
    sequence_length = keyframes.shape[0]

    motion_dataset = tf.data.Dataset.from_tensors(
        {
            "keyframes": keyframes[..., site_smpl_indices, :],
            "qpos": np.concatenate(
                [
                    qpos0[None, ...],
                    np.zeros((sequence_length - 1, *qpos0.shape), qpos0.dtype),
                ],
                axis=0,
            ),
            "qvel": np.concatenate(
                [
                    qvel0[None, ...],
                    np.zeros((sequence_length - 1, *qvel0.shape), qvel0.dtype),
                ],
                axis=0,
            ),
            "mocap_id": amass_id,
        }
    )

    environment = humanoid_motion_tracking.load(
        walker_type=walker_type,
        random_state=np.random.RandomState(seed=1000),
        task_kwargs={
            "motion_dataset": motion_dataset.repeat(),
            "mocap_reference_steps": 0,
            "termination_threshold": 0.5,
            "random_init_time_step": False,
            # "mjpc_task_xml_file_path": None,
        },
    )

    physics_timestep = environment.physics.model.opt.timestep
    if not np.isclose(physics_timestep, 1.0 / output_fps):
        raise ValueError(f"{physics_timestep=} does not match the {output_fps=}.")

    video_save_path = (
        save_base_dir / "videos" / "valid" / f"{amass_id.replace('/', '+')}"
    ).with_suffix(".mp4")

    policy = mjpc_expert.MJPCExpert(
        warm_start_steps=10_000,
        warm_start_tolerance=1e-8,
        select_action_steps=1,
        select_action_tolerance=0.0,
        mjpc_workers=6,
        dtype=np.float32,
    )

    try:
        result = rollout_policy_and_save_video(
            policy=policy,
            env_fn=lambda: environment,
            max_num_steps=30_000,
            video_save_path=video_save_path,
            output_fps=output_fps,
            render_kwargs={"width": 640, "height": 640, "camera_id": "walker/back"},
        )
    finally:
        del policy

    result_info = {
        "mocap_id": amass_id,
        "rewards-q0.00": np.quantile(result["rewards"], 0.00),
        "rewards-q0.05": np.quantile(result["rewards"], 0.05),
        "rewards-q0.10": np.quantile(result["rewards"], 0.10),
        "rewards-q0.20": np.quantile(result["rewards"], 0.20),
        "rewards-q0.50": np.quantile(result["rewards"], 0.50),
        "rewards-q0.80": np.quantile(result["rewards"], 0.80),
        "rewards-q0.90": np.quantile(result["rewards"], 0.90),
        "rewards-q0.95": np.quantile(result["rewards"], 0.95),
        "rewards-q1.00": np.quantile(result["rewards"], 1.00),
        "rewards-mean": np.mean(result["rewards"]),
    }

    result.update(result_info)
    logging.info(f"{amass_id}: {json.dumps(result_info, indent=2, sort_keys=True)}")
    if result_info["rewards-q0.95"] < 0.9:
        logging.warning(f"Low reward for {amass_id=}!.")

    states = tree.map_structure(lambda *xs: np.stack(xs), *result["mujoco_states"])

    result_file.parent.mkdir(parents=True, exist_ok=True)
    with result_file.open("wb") as f:
        np.savez_compressed(
            f,
            qpos=states["qpos"],
            qvel=states["qvel"],
            keyframes=keyframes[..., site_smpl_indices, :],
            mocap_id=amass_id,
        )
