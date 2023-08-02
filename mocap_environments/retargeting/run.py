r"""Functions for processing AMASS keyframes into mujoco humanoid tracking dataset.

```sh
python ./run.py \
  --keyframes_base_dir="/tmp/keyframes/" \
  --save_path="/tmp/physics_dataset" \
  --file_filter_regex="^CMU/CMU/(02/02_04|108/108_13)_poses.npy$" \
  --fps="60.0" \
  --override=False
```
"""

import functools
import multiprocessing.pool
import os
import pathlib
import re
import sqlite3
import sys
from typing import Any, Callable, Literal, Optional
import uuid

from absl import app
from absl import flags
from absl import logging
from dm_control import composer
from dm_control import mjcf
from dm_control.locomotion.tasks.reference_pose import utils as reference_pose_utils
from dm_control.locomotion.walkers import base as walkers_base
from dm_control.rl import control
import jax.numpy as jnp
import mujoco
import numpy as np
import numpy.typing as npt
import ray
import tensorflow as tf
import tree

from mocap_environments import visualization
from mocap_environments import walkers
from mocap_environments.environments import humanoid_motion_tracking
from mocap_environments.experts import mjpc_expert
from mocap_environments.retargeting import smpl
from mocap_environments.utils import video as video_utils

logging.set_verbosity(logging.INFO)

JnpArrayType = jnp.array
NpArrayType = np.ndarray
Path = pathlib.Path
WalkerEnum = Literal["SimpleHumanoidPositionController", "SimpleHumanoid"]

sys.path.append(str(Path(__file__).parent))
import inverse_kinematics as ik

# pylint: disable=logging-fstring-interpolation

_KEYFRAMES_BASE_DIR_FLAG = flags.DEFINE_string(
    "keyframes_base_dir",
    None,
    (
        "Path to the keyframe data directory. This directory corresponds to the "
        "`save_path`-flag in `pipeline_keyframes_only.py`."
    ),
)
_DEBUG_FLAG = flags.DEFINE_bool("debug", False, "Run as debug-friendly.")
_FILE_FILTER_REGEX_FLAG = flags.DEFINE_string(
    "file_filter_regex",
    None,
    (
        "Regular expression filter for the sequences. The dataset will only "
        "include those sequences whose filename passes the `re.match` test. For "
        r"example: '^CMU/CMU/(90/90_19|02/02_04)_poses.npz$'."
    ),
)

_SAVE_PATH_FLAG = flags.DEFINE_string(
    "save_path",
    None,
    (
        "Path to the save the physics dataset to. The processed results will be saved "
        "relative to the `save_path` similarly as `amass_paths` are read relatively."
    ),
)
_RESULT_DATABASE_PATH_FLAG = flags.DEFINE_string(
    "result_database_path",
    None,
    "(Optional) SQLite database path to the save the diagnostic information to.",
)
_RUN_ID_FLAG = flags.DEFINE_string("run_id", None, "Identifier for the run.")
_FPS_FLAG = flags.DEFINE_float("fps", 30.0, "Target output FPS.")
_NUM_ROLLOUTS_FLAG = flags.DEFINE_integer(
    "num_rollouts", 5, "Number of expert rollouts per mocap sequence."
)
_WALKER_TYPE_FLAG = flags.DEFINE_enum(
    "walker_type",
    "SimpleHumanoidPositionControlled",
    ("SimpleHumanoid", "SimpleHumanoidPositionControlled"),
    "Walker type.",
)
_OVERRIDE_FLAG = flags.DEFINE_boolean(
    "override", False, "Whether to override existing files."
)


def convert_fps(
    keyframes: NpArrayType,
    source_fps: int | float,
    target_fps: int | float,
) -> NpArrayType:
    """Convert `keyframes` fps by linearly interpolating between frames."""

    logging.info(f"{source_fps=}, {target_fps=}")

    if not target_fps <= source_fps:
        raise NotImplementedError(
            "`convert_fps`-function doesn't yet support increasing fps."
        )

    fps_rate_float = source_fps / target_fps
    sequence_length = keyframes.shape[0]

    # NOTE(hartikainen): Linearly interpolate the frames to achieve desired fps.
    frame_indices_float = np.linspace(
        0,
        sequence_length - 1,
        np.int32(np.ceil(sequence_length / fps_rate_float)),
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
    ik_kwargs: Optional[dict[str, Any]] = None,
):
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
      rest_joints_names:  Similar to `rest_joints_names`, except for the rest of the
        joints for the second step.
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

    qposes = []
    for keyframe in keyframes:
        ik_result_0 = ik.qpos_from_site_pose(
            physics=physics,
            sites_names=list(root_sites_names),
            target_pos=keyframe[list(root_keyframe_indices)],
            target_quat=None,
            joint_names=list(root_joints_names),
            **ik_kwargs,
        )

        with physics.reset_context():
            physics.named.data.qpos[:7] = ik_result_0.qpos[:7]

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

        qposes.append(physics.data.qpos.copy())

    qposes = np.stack(qposes)

    def differentiate_position(
        qpos1: npt.ArrayLike, qpos2: npt.ArrayLike
    ) -> npt.ArrayLike:
        qvel = np.empty_like(physics.data.qvel)
        mujoco.mj_differentiatePos(
            physics.model._model,
            qvel,
            dt=1.0 / keyframe_fps,
            qpos1=qpos1,
            qpos2=qpos2,
        )
        return qvel

    qvels = np.stack(
        [
            differentiate_position(qpos1, qpos2)
            for qpos1, qpos2 in zip(qposes[:-1], qposes[1:])
        ]
        + [np.zeros(physics.data.qvel.shape)]
        # + [np.full(physics.data.qvel.shape, np.nan)]
    )

    return qposes, qvels


def rollout_policy_and_save_video(
    policy,
    env_fn: Callable[[], control.Environment],
    max_num_steps: int = 30_000,
    *,
    video_save_path: Path,
    output_fps: int | float = 60,
    render_kwargs: Optional[dict[str, Any]] = None,
):
    if render_kwargs is None:
        render_kwargs = {"width": 640, "height": 640, "camera_id": 1}

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
        frames.append(environment.physics.render(**render_kwargs))

    frames = visualization.add_action_plots_to_frames(frames, actions)

    Path(video_save_path).parent.mkdir(parents=True, exist_ok=True)

    mocap_id = environment.task._motion_sequence["mocap_id"].decode()
    video_save_path = video_save_path.with_stem(
        video_save_path.stem.format(mocap_id=mocap_id)
    )

    video_utils.save_video(
        video_save_path,
        frames,
        input_fps=round(1.0 / environment.control_timestep(), 6),
        output_fps=output_fps,
    )

    num_time_steps = len(time_steps)
    assert num_time_steps == t, (num_time_steps, t)

    return_ = sum(ts.reward["tracking"] for ts in time_steps if ts.reward is not None)
    return_per_step = return_ / num_time_steps

    rewards = [ts.reward["tracking"] for ts in time_steps if ts.reward is not None]
    return {
        "rewards": rewards,
        "return": return_,
        "return_per_step": return_per_step,
        "num_time_steps": num_time_steps,
        "time_steps": time_steps,
        "mujoco_states": mujoco_states,
    }


def process_motion(
    keyframe_path: Path,
    keyframes_base_dir: Path,
    save_base_dir: Path,
    run_id: str,
    result_database_path: Path,
    fps: int | float = 60.0,
    num_rollouts: int = 5,
    override: bool = False,
    *,
    walker_type: WalkerEnum,
):
    keyframe_file = keyframes_base_dir / keyframe_path
    logging.info(f"Processing file '{keyframe_file}'")

    result_file = (save_base_dir / keyframe_path).with_suffix(".npy")
    if not override and result_file.exists():
        logging.info(f"Output file ('{result_file}') already exists. Skipping.")
        return

    with keyframe_file.open("rb") as f:
        keyframes = np.load(f)

    keyframe_id = str(keyframe_path.with_suffix(""))

    def translate_keyframes(
        keyframes: npt.ArrayLike, initial_translation: Optional[npt.ArrayLike] = None
    ):
        keyframes = np.array(keyframes)
        if initial_translation is None:
            initial_translation = np.zeros(3, dtype=keyframes.dtype)
        initial_translation = np.array(initial_translation)

        # Get rid of positive or negative offsets in motions.
        min_z = np.min(keyframes[..., -1])
        keyframes[..., -1] -= min_z

        # Set starting position to origin.
        keyframes[..., 0:2] -= keyframes[..., 0, 0, 0:2]

        keyframes[..., :, :, :] += initial_translation
        return keyframes

    # original_keyframes = keyframes.copy()
    keyframes = translate_keyframes(keyframes)

    logging.info(f"{keyframes.shape=}")

    if walker_type == "SimpleHumanoidPositionControlled":
        walker_class = walkers.SimpleHumanoidPositionControlled
    elif walker_type == "SimpleHumanoid":
        walker_class = walkers.SimpleHumanoid
    else:
        raise ValueError(f"{walker_type=}")

    walker = walker_class()
    empty_arena = composer.Arena()
    walker = reference_pose_utils.add_walker(
        walker_fn=lambda name: walker, arena=empty_arena, name="walker"
    )

    # mjcf_root = empty_arena.attach(walker)
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
        site_joint_name = re.match(site_joint_name_re, site_element.name).group(1)
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
    rest_sites_names = (
        "walker/tracking[pelvis]",
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
        "walker/tracking[lhip]",
        "walker/tracking[rhip]",
    )
    rest_keyframe_indices = tuple(
        site_to_smpl_index_map[name] for name in rest_sites_names
    )

    qposes, qvels = compute_inverse_kinematics_qpos_qvel(
        walker,
        physics,
        keyframes[:2, ...],
        keyframe_fps=fps,
        root_sites_names=root_sites_names,
        root_keyframe_indices=root_keyframe_indices,
        root_joints_names=physics.named.data.qpos.axes.row.names[:1],
        rest_sites_names=rest_sites_names,
        rest_keyframe_indices=rest_keyframe_indices,
        rest_joints_names=physics.named.data.qpos.axes.row.names[1:],
    )

    qpos0, qvel0 = qposes[0], qvels[0]

    sequence_length = keyframes.shape[0]

    motion_dataset = tf.data.Dataset.from_tensors(
        {
            "keyframes": keyframes[..., site_smpl_indices, :],
            "qpos": np.concatenate(
                [
                    qpos0[None, ...],
                    # np.full(
                    #     (sequence_length - 1, *qpos0.shape), np.nan, dtype=qpos0.dtype
                    # ),
                    np.zeros((sequence_length - 1, *qpos0.shape), qpos0.dtype),
                ],
                axis=0,
            ),
            "qvel": np.concatenate(
                [
                    qvel0[None, ...],
                    # np.full(
                    #     (sequence_length - 1, *qvel0.shape), np.nan, dtype=qvel0.dtype
                    # ),
                    np.zeros((sequence_length - 1, *qvel0.shape), qvel0.dtype),
                ],
                axis=0,
            ),
            "mocap_id": keyframe_id,
        }
    )

    environment = humanoid_motion_tracking.load(
        walker_type=walker_type,
        random_state=np.random.RandomState(seed=0),
        task_kwargs={
            "motion_dataset": motion_dataset.repeat(),
            "mocap_reference_steps": 0,
            "termination_threshold": 1.0,
            "random_init_time_step": False,
        },
    )

    policy = mjpc_expert.MJPCExpert(
        warm_start_steps=10_000,
        warm_start_tolerance=1e-8,
        select_action_steps=1,
        select_action_tolerance=0.0,
        mjpc_workers=6,
        dtype=np.float32,
    )

    def dump_result_to_database(result: dict[str, Any], database_path: Path):
        assert database_path.exists(), database_path
        connection = sqlite3.connect(str(database_path))
        cursor = connection.cursor()
        to_dump_keys = (
            "mocap_id",
            "rollout_id",
            "run_id",
            "num_time_steps",
            "rewards_min",
            "rewards_max",
            "rewards_mean",
            "rewards_median",
            "return",
            "success",
        )

        cursor.execute(
            (
                f"INSERT INTO my_table({', '.join(to_dump_keys)}) "
                f"VALUES ({', '.join('?' * len(to_dump_keys))})"
            ),
            [result[key] for key in to_dump_keys],
        )

        connection.commit()
        connection.close()

    def is_valid_result(result):
        minimum_satisfied = 0.85 < result["rewards_min"]
        median_satisfied = 0.925 < result["rewards_median"]
        terminated_early = result["time_steps"][-1].discount == 0

        return minimum_satisfied and median_satisfied and not terminated_early

    results = []
    for i in range(num_rollouts):
        rollout_id = str(uuid.uuid4()).split("-")[0]
        logging.info(f"{i=}, {rollout_id=}")

        video_save_path = (
            (save_base_dir / "videos" / keyframe_path)
            .with_stem(f"{keyframe_path.stem}-{i}-{rollout_id}")
            .with_suffix(".mp4")
        )

        result = rollout_policy_and_save_video(
            policy=policy,
            env_fn=lambda: environment,
            max_num_steps=10_000,
            video_save_path=video_save_path,
            output_fps=fps,
            render_kwargs={"width": 640, "height": 640, "camera_id": 1},
        )

        result["rewards"] = np.array(result["rewards"])

        result.update(
            {
                "mocap_id": keyframe_id,
                "rollout_id": rollout_id,
                "run_id": run_id,
                "rewards_min": result["rewards"].min(),
                "rewards_max": result["rewards"].max(),
                "rewards_mean": result["rewards"].mean(),
                "rewards_median": np.median(result["rewards"]),
            }
        )
        result["success"] = is_valid_result(result)
        results.append(result)

        if not result["success"]:
            video_save_path_invalid = (
                (save_base_dir / "videos" / "invalid" / keyframe_path)
                .with_stem(f"{keyframe_path.stem}-{i}-{rollout_id}")
                .with_suffix(".mp4")
            )
            video_save_path_invalid.parent.mkdir(parents=True, exist_ok=True)
            video_save_path.rename(video_save_path_invalid)

        if result_database_path is not None:
            dump_result_to_database(result, result_database_path)

    valid_results = [result for result in results if result["success"]]
    if not valid_results:
        logging.info(f"No valid results for {keyframe_id=}!.")
        return

    best_result = max(valid_results, key=lambda r: r["rewards_min"])
    best_states = tree.map_structure(
        lambda *xs: np.stack(xs), *best_result["mujoco_states"]
    )

    result_file.parent.mkdir(parents=True, exist_ok=True)
    with result_file.open("wb") as f:
        np.savez_compressed(
            f,
            qpos=best_states["qpos"],
            qvel=best_states["qvel"],
            keyframes=keyframes[..., site_smpl_indices, :],
            mocap_id=keyframe_id,
        )

    del policy


def _process_motion(process_i: int, *args, **kwargs):
    # cuda_visible_devices_str = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    # if cuda_visible_devices_str.strip() != "":
    #     gpu_ids_str = cuda_visible_devices_str.split(",")
    #     gpu_ids = [int(x.strip()) for x in gpu_ids_str]
    #     this_visible_device = gpu_ids[process_i % len(gpu_ids)]
    #     os.environ["CUDA_VISIBLE_DEVICES"] = str(this_visible_device)

    print(f"{multiprocessing.current_process().name=}")
    print(f"{process_i=}")
    print(f'{os.environ["CUDA_VISIBLE_DEVICES"]=}')
    import dm_control
    import tensorflow as tf

    return process_motion(*args, **kwargs)


def main(argv):
    del argv

    file_filter_regex = _FILE_FILTER_REGEX_FLAG.value
    keyframes_base_dir = Path(_KEYFRAMES_BASE_DIR_FLAG.value).expanduser()
    if _RESULT_DATABASE_PATH_FLAG.value is not None:
        result_database_path = Path(_RESULT_DATABASE_PATH_FLAG.value).expanduser()
    else:
        result_database_path = None
    override = _OVERRIDE_FLAG.value
    walker_type = _WALKER_TYPE_FLAG.value
    num_rollouts = _NUM_ROLLOUTS_FLAG.value
    fps = _FPS_FLAG.value
    save_base_dir = Path(_SAVE_PATH_FLAG.value).expanduser()
    run_id = _RUN_ID_FLAG.value
    debug = _DEBUG_FLAG.value

    keyframe_data_paths = tuple(
        x.relative_to(keyframes_base_dir) for x in keyframes_base_dir.rglob("**/*.npy")
    )
    if file_filter_regex is not None:
        file_pattern = re.compile(file_filter_regex)
        keyframe_data_paths = tuple(
            x for x in keyframe_data_paths if re.match(file_pattern, str(x))
        )

    if debug:
        num_rollouts = 1

    process_motion_partial = functools.partial(
        _process_motion,
        keyframes_base_dir=keyframes_base_dir,
        save_base_dir=save_base_dir,
        run_id=run_id,
        result_database_path=result_database_path,
        num_rollouts=num_rollouts,
        fps=fps,
        override=override,
        walker_type=walker_type,
    )

    @ray.remote(num_cpus=2, num_gpus=0.19)
    def process_motion_partial_ray(*x):
        return _process_motion(
            *x,
            keyframes_base_dir=keyframes_base_dir,
            save_base_dir=save_base_dir,
            run_id=run_id,
            result_database_path=result_database_path,
            num_rollouts=num_rollouts,
            fps=fps,
            override=override,
            walker_type=walker_type,
        )

    if debug:
        # pool_size = 1
        # pool_class = multiprocessing.pool.ThreadPool
        keyframe_data_paths = [
            next(k for k in keyframe_data_paths if "108_13" in str(k))
        ]
        process_motion_partial(0, keyframe_data_paths[0])
        return
    else:
        pool_size = 1
        pool_class = multiprocessing.Pool
        # pool_class = multiprocessing.pool.ThreadPool

    results_partial = [
        process_motion_partial_ray.remote(i, p)
        for i, p in enumerate(keyframe_data_paths)
    ]

    results = ray.get(results_partial)

    # with pool_class(pool_size) as p:
    #     p.starmap(process_motion_partial, enumerate(keyframe_data_paths))


if __name__ == "__main__":
    flags.mark_flags_as_required(
        (
            _KEYFRAMES_BASE_DIR_FLAG,
            _SAVE_PATH_FLAG,
            _RUN_ID_FLAG,
        )
    )
    app.run(main)
