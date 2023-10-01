"""Functionality for retargeting scripts."""

import dataclasses
import json
import pathlib
import re
from typing import Any, Callable, Literal, Optional, Union

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
from mocap_environments.environments import humanoid_motion_playback
from mocap_environments.environments import humanoid_motion_tracking
from mocap_environments.experts import mjpc_expert
from mocap_environments.retargeting import dm_control_walker
from mocap_environments.retargeting import smpl
from mocap_environments.utils import video as video_utils
from mocap_environments.visualization import video as video_visualization

dataclass = dataclasses.dataclass
JnpArrayType = jnp.array
Path = pathlib.Path
WalkerEnum = Literal[
    "SimpleHumanoid",
    "SimpleHumanoidPositionControlled",
    "SMPLHumanoid",
    "SMPLHumanoidPositionControlled",
]
PhysicsEnvironment = control.Environment | composer.Environment

_DEFAULT_RENDER_KWARGS = {"width": 640, "height": 640, "camera_id": "walker/back"}


@dataclass
class RetargetingConfig:
    # walker_type: WalkerEnum
    site_to_data_index_map: dict[str, int]
    site_data_indices: tuple[int]
    root_sites_names: tuple[str]
    root_keyframe_indices: tuple[int]
    root_joints_names: tuple[str]
    rest_sites_names: tuple[str]
    rest_keyframe_indices: tuple[int]
    rest_joints_names: tuple[str]


def convert_fps(
    keyframes: npt.ArrayLike,
    source_fps: int | float,
    target_fps: int | float,
) -> np.ndarray:
    """Convert `keyframes` fps by linearly interpolating between frames."""
    keyframes = np.asarray(keyframes)

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
        render_kwargs = _DEFAULT_RENDER_KWARGS

    environment = env_fn()

    actions = []
    rewards = []
    discounts = []
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

    t = 0
    while (t := t + 1) <= max_num_steps and not time_steps[-1].last():
        action = policy.select_action(time_steps[-1].observation, environment)
        actions.append(action)
        time_steps.append(environment.step(action))
        rewards.append(time_steps[-1].reward)
        discounts.append(time_steps[-1].discount)
        for _ in range(environment.task.physics_steps_per_control_step):
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

    mocap_id = environment.task._motion_sequence[
        "mocap_id"
    ].decode()  # pylint: disable=protected-access

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

    rewards = tree.map_structure(lambda *rs: np.stack(rs), *rewards)
    actions = np.stack(actions)

    return {
        "rewards": rewards,
        "discounts": discounts,
        "num_time_steps": t,
        "time_steps": time_steps,
        "mujoco_states": mujoco_states,
        "actions": actions,
    }


def playback_motion(
    qposes: npt.ArrayLike,
    qvels: npt.ArrayLike,
    keyframes: npt.ArrayLike,
    render_kwargs: Optional[dict[str, Any]] = None,
    *,
    walker_type: WalkerEnum,
):
    qposes = np.asarray(qposes)
    qvels = np.asarray(qvels)
    keyframes = np.asarray(keyframes)

    if render_kwargs is None:
        render_kwargs = _DEFAULT_RENDER_KWARGS

    motion_dataset = tf.data.Dataset.from_tensors(
        {
            "keyframes": keyframes,
            "qpos": qposes,
            "qvel": qvels,
            "mocap_id": "dummy",
        }
    )

    environment = humanoid_motion_playback.load(
        walker_type=walker_type,
        random_state=np.random.RandomState(seed=1000),
        task_kwargs={
            "motion_dataset": motion_dataset.repeat(),
            "mocap_reference_steps": 0,
            "termination_threshold": float("inf"),
            "random_init_time_step": False,
            "control_timestep": 1.0 / 60.0,
        },
    )

    time_steps = [environment.reset()]
    frames = [environment.physics.render(**render_kwargs)]

    while not time_steps[-1].last():
        action = environment.action_spec().generate_value()
        time_steps.append(environment.step(action))
        frames.append(environment.physics.render(**render_kwargs))

    frames = np.asarray(frames)
    fps = round(1.0 / environment.control_timestep(), 6)
    return frames, fps


def rollout_policy(
    policy,
    env_fn: Callable[[], PhysicsEnvironment],
    max_num_steps: int = 30_000,
):
    environment = env_fn()

    actions = []
    time_steps = [environment.reset()]
    rewards = []
    discounts = []
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

    t = 0
    while (t := t + 1) <= max_num_steps and not time_steps[-1].last():
        action = policy.select_action(time_steps[-1].observation, environment)
        actions.append(action)
        time_steps.append(environment.step(action))
        rewards.append(time_steps[-1].reward)
        discounts.append(time_steps[-1].discount)
        for _ in range(environment.task.physics_steps_per_control_step):
            policy.agent.step()
        mujoco_states.append(
            {
                "qpos": environment.physics.data.qpos.copy(),
                "qvel": environment.physics.data.qvel.copy(),
            }
        )
        policy.observe(action, time_steps[-1], environment)
        mjpc_costs.append(get_mjpc_cost())

    mujoco_states = tree.map_structure(lambda *xs: np.stack(xs), *mujoco_states)
    mjpc_costs = tree.map_structure(lambda *xs: np.stack(xs), *mjpc_costs)
    actions = np.stack(actions)
    rewards = tree.map_structure(lambda *rs: np.stack(rs), *rewards)

    return {
        "rewards": rewards,
        "discounts": discounts,
        "num_time_steps": t,
        "time_steps": time_steps,
        "mujoco_states": mujoco_states,
        "actions": actions,
        "mjpc_costs": mjpc_costs,
    }


def create_retargeting_config(
    walker: Union[
        walkers.SimpleHumanoid,
        walkers.SimpleHumanoidPositionControlled,
        walkers.SMPLHumanoid,
        walkers.SMPLHumanoidPositionControlled,
    ],
):
    walker_name = walker._mjcf_root.model

    if isinstance(
        walker, (walkers.SimpleHumanoid, walkers.SimpleHumanoidPositionControlled)
    ):
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
        root_sites_names = (
            f"{walker_name}/tracking[head]",
            f"{walker_name}/tracking[rshoulder]",
            f"{walker_name}/tracking[lshoulder]",
        )
        rest_sites_names = (
            # f'{walker_name}/tracking[head]',
            f'{walker_name}/tracking[pelvis]',
            f'{walker_name}/tracking[rhip]',
            f'{walker_name}/tracking[rknee]',
            f'{walker_name}/tracking[rheel]',
            f'{walker_name}/tracking[rtoe]',
            f'{walker_name}/tracking[lhip]',
            f'{walker_name}/tracking[lknee]',
            f'{walker_name}/tracking[lheel]',
            f'{walker_name}/tracking[ltoe]',
            # f'{walker_name}/tracking[rshoulder]',
            f'{walker_name}/tracking[relbow]',
            f'{walker_name}/tracking[rhand]',
            # f'{walker_name}/tracking[lshoulder]',
            f'{walker_name}/tracking[lelbow]',
            f'{walker_name}/tracking[lhand]',
        )
        root_joints_names = (f"{walker_name}/",)
        rest_joints_names = tuple(
            f"{walker_name}/{j.name}" for j in walker.mocap_joints
        )
        data_joint_names = tuple(smpl.SMPL_JOINT_NAMES)

    elif isinstance(
        walker, (walkers.SMPLHumanoid, walkers.SMPLHumanoidPositionControlled)
    ):
        physics_to_kinematics_joint_name_map = dict(
            (
                ("Pelvis", "Pelvis"),
                ("L_Hip", "L_Hip"),
                ("L_Knee", "L_Knee"),
                ("L_Ankle", "L_Ankle"),
                ("L_Toe", "L_Toe"),
                ("R_Hip", "R_Hip"),
                ("R_Knee", "R_Knee"),
                ("R_Ankle", "R_Ankle"),
                ("R_Toe", "R_Toe"),
                ("Torso", "Torso"),
                ("Spine", "Spine"),
                ("Chest", "Chest"),
                ("Neck", "Neck"),
                ("Head", "Head"),
                ("L_Thorax", "L_Thorax"),
                ("L_Shoulder", "L_Shoulder"),
                ("L_Elbow", "L_Elbow"),
                ("L_Wrist", "L_Wrist"),
                ("L_Hand", "L_Hand"),
                ("R_Thorax", "R_Thorax"),
                ("R_Shoulder", "R_Shoulder"),
                ("R_Elbow", "R_Elbow"),
                ("R_Wrist", "R_Wrist"),
                ("R_Hand", "R_Hand"),
            )
        )
        root_sites_names = (
            f"{walker_name}/tracking[Pelvis]",
            f"{walker_name}/tracking[L_Hip]",
            f"{walker_name}/tracking[R_Hip]",
        )
        rest_sites_names = (
            # f"{walker_name}/tracking[Pelvis]",
            # f"{walker_name}/tracking[L_Hip]",
            # f"{walker_name}/tracking[R_Hip]",
            f"{walker_name}/tracking[L_Knee]",
            f"{walker_name}/tracking[R_Knee]",
            f"{walker_name}/tracking[L_Ankle]",
            f"{walker_name}/tracking[R_Ankle]",
            f"{walker_name}/tracking[L_Toe]",
            f"{walker_name}/tracking[R_Toe]",
            f"{walker_name}/tracking[Torso]",
            f"{walker_name}/tracking[Spine]",
            f"{walker_name}/tracking[Chest]",
            f"{walker_name}/tracking[Neck]",
            f"{walker_name}/tracking[Head]",
            f"{walker_name}/tracking[L_Thorax]",
            f"{walker_name}/tracking[R_Thorax]",
            f"{walker_name}/tracking[L_Shoulder]",
            f"{walker_name}/tracking[R_Shoulder]",
            f"{walker_name}/tracking[L_Elbow]",
            f"{walker_name}/tracking[R_Elbow]",
            f"{walker_name}/tracking[L_Wrist]",
            f"{walker_name}/tracking[R_Wrist]",
            f"{walker_name}/tracking[L_Hand]",
            f"{walker_name}/tracking[R_Hand]",
        )
        root_joints_names = (f"{walker_name}/",)
        rest_joints_names = tuple(
            f"{walker_name}/{j.name}" for j in walker.mocap_joints
        )

        # The reason this uses custom `data_joint_names` is that the data comes from
        # PHC codebase and is ordered unlike AMASS SMPL data.
        # fmt: off
        data_joint_names = (
            "Pelvis",
            "L_Hip", "L_Knee", "L_Ankle", "L_Toe",
            "R_Hip", "R_Knee", "R_Ankle", "R_Toe",
            "Torso", "Spine", "Chest", "Neck", "Head",
            "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand",
            "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand",
        )
        # fmt: on

    else:
        raise ValueError(f"{walker=}")

    site_joint_name_re = re.compile(r"^tracking\[(\w+)\]$")
    site_data_indices = []
    # We need sites -> keyframes map for IK.
    for site_element in walker.mocap_tracking_sites:
        site_joint_name_match = re.match(site_joint_name_re, site_element.name)
        assert site_joint_name_match is not None
        site_joint_name = site_joint_name_match.group(1)
        assert site_joint_name in physics_to_kinematics_joint_name_map
        data_joint_name = physics_to_kinematics_joint_name_map[site_joint_name]
        site_data_index = data_joint_names.index(data_joint_name)
        site_data_indices.append(site_data_index)

    site_data_indices = tuple(site_data_indices)

    site_to_data_index_map = dict(
        zip(
            [f"{walker_name}/{s.name}" for s in walker.mocap_tracking_sites],
            site_data_indices,
        )
    )

    root_keyframe_indices = tuple(
        site_to_data_index_map[name] for name in root_sites_names
    )
    assert len(root_keyframe_indices) == 3, root_keyframe_indices
    rest_keyframe_indices = tuple(
        site_to_data_index_map[name] for name in rest_sites_names
    )

    return RetargetingConfig(
        site_to_data_index_map=site_to_data_index_map,
        site_data_indices=site_data_indices,
        root_sites_names=root_sites_names,
        root_keyframe_indices=root_keyframe_indices,
        root_joints_names=root_joints_names,
        rest_sites_names=rest_sites_names,
        rest_keyframe_indices=rest_keyframe_indices,
        rest_joints_names=rest_joints_names,
    )


def process_motion(
    keyframe_path: Path,
    keyframes_base_dir: Path,
    *,
    save_base_dir: Path,
    keyframe_fps: int | float = 60.0,
    output_fps: int | float = 60.0,
    override: bool = False,
    walker_type: WalkerEnum,
):
    """Process a single motion file.

    TODO(hartikainen): Make handling of `output_fps` more flexible. The tracking
    environment shouldn't care about the exact fps but rather automatically handle
    interpolation between frames.
    """
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

    assert isinstance(keyframes, np.ndarray), type(keyframes)

    keyframes = convert_fps(keyframes, source_fps=keyframe_fps, target_fps=output_fps)
    keyframe_fps = output_fps

    if walker_type == "SimpleHumanoid":
        walker_class = walkers.SimpleHumanoid
    elif walker_type == "SimpleHumanoidPositionControlled":
        walker_class = walkers.SimpleHumanoidPositionControlled
    elif walker_type == "SMPLHumanoid":
        walker_class = walkers.SMPLHumanoid
    elif walker_type == "SMPLHumanoidPositionControlled":
        walker_class = walkers.SMPLHumanoidPositionControlled
    else:
        raise ValueError(f"{walker_type=}")

    empty_arena = composer.Arena()
    walker = reference_pose_utils.add_walker(
        walker_fn=walker_class, arena=empty_arena, name="walker"
    )

    physics = mjcf.Physics.from_xml_string(empty_arena.mjcf_model.to_xml_string())

    retargeting_config = create_retargeting_config(walker)

    def translate_keyframes(
        keyframes: npt.ArrayLike,
        initial_translation: Optional[npt.ArrayLike] = None,
    ):
        keyframes = np.array(keyframes)
        if initial_translation is None:
            initial_translation = np.zeros(3, dtype=keyframes.dtype)
        initial_translation = np.array(initial_translation)

        zs_flat = keyframes[..., 2].flatten()
        z_offset = np.quantile(zs_flat, 0.0)  # Find the minimum value.
        keyframes[..., 2] -= z_offset

        keyframe_root_index = retargeting_config.site_to_data_index_map[
            retargeting_config.root_sites_names[0]
        ]
        # Set starting position to origin.
        keyframes[..., 0:2] -= keyframes[
            ...,
            0,
            keyframe_root_index,
            0:2,
        ]

        keyframes[..., :, :, :] += initial_translation

        return keyframes

    if keyframes.shape[-2] != 24:
        raise ValueError(
            "This code assumes that the data is from AMASS and contains exactly 24 joints."
        )

    keyframes = translate_keyframes(keyframes)
    velocity_t = min(keyframes.shape[0] - 1, 6)
    (
        qposes,
        qvels,
    ) = dm_control_walker.compute_inverse_kinematics_qpos_qvel(
        physics,
        keyframes[[0, velocity_t], ...],
        keyframe_fps=keyframe_fps / velocity_t,
        root_sites_names=retargeting_config.root_sites_names,
        root_keyframe_indices=retargeting_config.root_keyframe_indices,
        root_joints_names=retargeting_config.root_joints_names,
        rest_sites_names=retargeting_config.rest_sites_names,
        rest_keyframe_indices=retargeting_config.rest_keyframe_indices,
        rest_joints_names=retargeting_config.rest_joints_names,
        qvel_step_size=1,
    )

    sequence_length = keyframes.shape[0]

    motion_dataset = tf.data.Dataset.from_tensors(
        {
            "keyframes": keyframes[..., retargeting_config.site_data_indices, :],
            "qpos": np.concatenate(
                [
                    qposes,
                    np.zeros((sequence_length - qposes.shape[0], *qposes[0].shape), qposes[0].dtype),
                ],
                axis=0,
            ),
            "qvel": np.concatenate(
                [
                    qvels,
                    np.zeros((sequence_length - qvels.shape[0], *qvels[0].shape), qvels[0].dtype),
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
            "termination_threshold": float("inf"),
            "random_init_time_step": False,
            # "control_timestep": 1.0 / 60.0,
        },
    )

    physics_timestep = environment.physics.model.opt.timestep
    if not np.isclose(physics_timestep, 1.0 / output_fps):
        raise ValueError(f"{physics_timestep=} does not match the {output_fps=}.")

    video_save_path = (
        save_base_dir / "videos" / "valid" / f"{amass_id.replace('/', '+')}"
    ).with_suffix(".mp4")

    if isinstance(walker, walkers.SimpleHumanoid):
        mjpc_task_id = "Humanoid Track"
    elif isinstance(walker, walkers.SMPLHumanoid):
        mjpc_task_id = "SMPLHumanoid Track"
    else:
        raise ValueError(f"{walker=}")

    policy = mjpc_expert.MJPCExpert(
        task_id=mjpc_task_id,
        warm_start_steps=10_000,
        warm_start_tolerance=1e-8,
        select_action_steps=100,
        select_action_tolerance=1e-2,
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
            render_kwargs=_DEFAULT_RENDER_KWARGS,
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
            keyframes=keyframes[..., retargeting_config.site_data_indices, :],
            mocap_id=amass_id,
        )


def retarget_motion(
    keyframes: npt.ArrayLike,
    keyframe_fps: int | float = 60.0,
    *,
    walker_type: WalkerEnum,
):
    # TODO(hartikainen): Make `output_fps` more flexible. The tracking environment
    # shouldn't care about the exact fps but rather automatically handle interpolation
    # between frames.
    output_fps = 60.0
    keyframes = convert_fps(keyframes, source_fps=keyframe_fps, target_fps=output_fps)

    if walker_type == "SimpleHumanoid":
        walker_class = walkers.SimpleHumanoid
    elif walker_type == "SimpleHumanoidPositionControlled":
        walker_class = walkers.SimpleHumanoidPositionControlled
    elif walker_type == "SMPLHumanoid":
        walker_class = walkers.SMPLHumanoid
    elif walker_type == "SMPLHumanoidPositionControlled":
        walker_class = walkers.SMPLHumanoidPositionControlled
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

        pelvis_zs = keyframes[:, site_to_smpl_index_map["walker/tracking[pelvis]"], 2]
        ltoe_zs = keyframes[:, site_to_smpl_index_map["walker/tracking[ltoe]"], 2]
        rtoe_zs = keyframes[:, site_to_smpl_index_map["walker/tracking[rtoe]"], 2]
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
            site_to_smpl_index_map["walker/tracking[pelvis]"],
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

    (
        qposes,
        qvels,
    ) = dm_control_walker.compute_inverse_kinematics_qpos_qvel(
        physics,
        keyframes,
        keyframe_fps=keyframe_fps,
        root_sites_names=root_sites_names,
        root_keyframe_indices=root_keyframe_indices,
        root_joints_names=physics.named.data.qpos.axes.row.names[:1],
        rest_sites_names=rest_sites_names,
        rest_keyframe_indices=rest_keyframe_indices,
        rest_joints_names=physics.named.data.qpos.axes.row.names[1:],
        end_effector_sites_names=end_effector_sites_names,
        end_effector_keyframe_indices=end_effector_keyframe_indices,
        end_effector_joints_names=end_effector_joints_names,
        qvel_step_size=4,
        ik_kwargs=dict(
            tol=1e-2,
            regularization_threshold=0.3,
            regularization_strength=1e-2,
            max_update_norm=1.0,
            progress_thresh=300.0,
            max_steps=10_000,
            inplace=False,
            null_space_method=False,
        ),
    )

    return qposes, qvels, keyframes[..., site_smpl_indices, :]


def track_motion(
    keyframes: np.ndarray,
    qposes: np.ndarray,
    qvels: np.ndarray,
    walker_type: WalkerEnum,
):
    """Given `keyframes` and `q{pose,vel}s` track motion with MJPCExpert.

    MJPCExpert implements a tracking residual, which can be a function of keyframes or
    `q{pose,vel}s`. If the residual doesn't use `q{pose,vel}s`, then we only need
    `q{pos,vel}` for the initial state when resetting the walker.
    """
    motion_dataset = tf.data.Dataset.from_tensors(
        {
            "keyframes": keyframes,
            "qpos": qposes,
            "qvel": qvels,
            "mocap_id": "NULL",
        }
    )

    environment = humanoid_motion_tracking.load(
        walker_type=walker_type,
        random_state=np.random.RandomState(seed=1000),
        task_kwargs={
            "motion_dataset": motion_dataset.repeat(),
            "mocap_reference_steps": 0,
            "termination_threshold": float("inf"),
            "random_init_time_step": False,
            # "control_timestep": 1.0 / 60.0,
        },
    )

    if walker_type in {"SimpleHumanoid", "SimpleHumanoidPositionControlled"}:
        mjpc_task_id = "Humanoid Track"
    elif walker_type in {"SMPLHumanoid", "SMPLHumanoidPositionControlled"}:
        mjpc_task_id = "SMPLHumanoid Track"
    else:
        raise ValueError(f"{walker_type=}")

    policy = mjpc_expert.MJPCExpert(
        task_id=mjpc_task_id,
        warm_start_steps=10_000,
        warm_start_tolerance=1e-8,
        select_action_steps=100,
        select_action_tolerance=1e-2,
        mjpc_workers=6,
        dtype=np.float32,
    )

    try:
        result = rollout_policy(
            policy=policy,
            env_fn=lambda: environment,
            max_num_steps=30_000,
        )
    finally:
        del policy

    percentiles = [0, 5, 10, 20, 50, 80, 95, 100]
    result_info = {
        "rewards-mean": np.mean(result["rewards"]),
    } | {f"rewards-p{p:d}": np.percentile(result["rewards"], p) for p in percentiles}

    result.update(result_info)

    return result
