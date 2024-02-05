"""Module documentation."""

import concurrent.futures
import functools
import pathlib
import re
from typing import Any, Callable, Literal, Optional

from absl import app
from absl import flags
from absl import logging
import dm_env
import mediapy
import numpy as np
import numpy.typing as npt
import tensorflow as tf
import tree

_DEFAULT_RENDER_KWARGS = {"width": 720, "height": 540, "camera_id": "walker/back"}

Path = pathlib.Path
WalkerEnum = Literal[
    "SimpleHumanoid",
    "SimpleHumanoidPositionControlled",
    "SMPLHumanoid",
    "SMPLHumanoidPositionControlled",
]

logging.set_verbosity(logging.INFO)

_WALKER_TYPE_FLAG = flags.DEFINE_enum(
    "walker_type",
    "SimpleHumanoidPositionControlled",
    (
        "SimpleHumanoid",
        "SimpleHumanoidPositionControlled",
        "SMPLHumanoid",
        "SMPLHumanoidPositionControlled",
    ),
    "Walker type.",
)
_MOTION_BASE_PATH_FLAG = flags.DEFINE_string(
    "motion_base_path",
    None,
    "Path to the motion base directory.",
)
_MAX_WORKERS_FLAG = flags.DEFINE_integer(
    "max_workers",
    16,
    "Maximum number of process pool workers, i.e. jobs to run in parallel.",
)
_DEBUG_FLAG = flags.DEFINE_bool("debug", False, "Run as debug-friendly.")
_OVERRIDE_FLAG = flags.DEFINE_bool(
    "override",
    False,
    "Override existing videos.",
)
_FILE_FILTER_REGEX_FLAG = flags.DEFINE_string(
    "file_filter_regex",
    None,
    (
        "Regular expression filter for the sequences. The dataset will only "
        "include those sequences whose filename passes the `re.match` test. For "
        r"example: '^CMU/CMU/(90/90_19|02/02_04)_poses.npz$'."
    ),
)
_VIDEO_FPS_FLAG = flags.DEFINE_float(
    "fps", 60.0, "Output video FPS. Must evenly divide `environment.control_timestep()`"
)

flags.mark_flags_as_required(
    (
        _MOTION_BASE_PATH_FLAG,
        _WALKER_TYPE_FLAG,
    )
)


def render_tracking_motion(
    index: int,
    motion_file_path: Path,
    output_path: Path,
    override: bool = False,
    *,
    walker_type: WalkerEnum,
    video_fps: float | int,
):
    video_save_path = (output_path / motion_file_path.stem).with_suffix(".mp4")
    if not override and video_save_path.exists():
        return None

    from dm_control import mjcf
    import mujoco

    from mocap_environments.environments import humanoid_motion_playback

    print(f"{index=}, {motion_file_path=}")

    with motion_file_path.open("rb") as f:
        motion = dict(np.load(f))

    print(f"{motion.keys()=}")
    assert motion.keys() == {"qpos", "qvel", "keyframes", "mocap_id"}, motion.keys()

    qpos = motion["qpos"]
    qvel = motion["qvel"]
    keyframes = motion["keyframes"]
    motion_id = motion["mocap_id"].item()
    # qpos[..., 3:7] = qpos[..., [6, 3, 4, 5]]
    print(f"{motion_id=}, {qpos.shape=}, {qvel.shape=}, {keyframes.shape=}")

    # mocap_body_element = self.root_entity.root_body.add(
    #     "body",
    #     name=f"mocap[{site_joint_name}]",
    #     mocap="true",
    # )
    # mocap_site_element = mocap_body_element.add(
    #     "site",
    #     name=f"mocap[{site_joint_name}]",
    #     type="sphere",
    #     size=str(site_sizes.get(site_joint_name, 0.03)),
    #     rgba=" ".join(map(str, site_colors.get(site_joint_name, (1, 0, 0, 1)))),
    #     **{"class": "mocap_site"},
    # )

    np.testing.assert_equal(qpos.shape[0], qvel.shape[0])
    # np.testing.assert_equal(qpos.shape[0], keyframes.shape[0] - 1)
    np.testing.assert_equal(qpos.shape[0], keyframes.shape[0])

    motion_dataset = tf.data.Dataset.from_tensors(
        {
            "keyframes": keyframes,
            "qpos": qpos,
            "qvel": qvel,
            "mocap_id": motion_id,
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
            # "mjpc_task_xml_file_path": None,
            "control_timestep": 1.0 / video_fps,
            # "control_timestep": 1.0 / output_fps,
        },
    )

    # breakpoint(); pass

    # num_mocap_joints = keyframes.shape[-2]
    # for i in range(num_mocap_joints):
    #     mocap_body_element = mjcf_model.root.worldbody.add(
    #         "body",
    #         name=f"mocap[{i}]",
    #         mocap="true",
    #     )
    #     mocap_site_element = mocap_body_element.add(
    #         "site",
    #         name=f"mocap[{i}]",
    #         type="sphere",
    #         size="0.03",
    #         rgba="1 0 0 1",
    #         group=2,
    #     )

    np.testing.assert_equal(qpos.shape[-1], environment.physics.model.nq)
    np.testing.assert_equal(qvel.shape[-1], environment.physics.model.nv)

    render_kwargs = _DEFAULT_RENDER_KWARGS.copy()

    def dummy_policy(time_step: dm_env.TimeStep) -> np.ndarray:
        del time_step
        return environment.action_spec().generate_value()

    with mediapy.VideoWriter(
        video_save_path,
        shape=(render_kwargs["height"], render_kwargs["width"]),
        fps=1 / environment.control_timestep(),
    ) as video_writer:
        time_step = environment.reset()
        while not time_step.last():
            frame = environment.physics.render(**render_kwargs)
            video_writer.add_image(frame)
            time_step = environment.step(dummy_policy(time_step))
        frame = environment.physics.render(**render_kwargs)
        video_writer.add_image(frame)

    return None


def main(argv):
    motion_base_path = Path(_MOTION_BASE_PATH_FLAG.value).expanduser()
    max_workers = _MAX_WORKERS_FLAG.value
    debug = _DEBUG_FLAG.value
    override = _OVERRIDE_FLAG.value
    file_filter_regex = _FILE_FILTER_REGEX_FLAG.value
    walker_type = _WALKER_TYPE_FLAG.value
    video_fps = _VIDEO_FPS_FLAG.value

    print(f"{motion_base_path=}")

    motion_file_paths = list(motion_base_path.rglob("**/*.npy"))

    if file_filter_regex is not None:
        file_pattern = re.compile(file_filter_regex)
        motion_file_paths = tuple(
            x for x in motion_file_paths if re.match(file_pattern, str(x))
        )

    # all_qposes = [
    #     np.load(motion_file_path.with_suffix(".npz"))["qpos"]
    #     for motion_file_path in motion_file_paths
    # ]

    # all_keyframes = [
    #     np.load(motion_file_path.with_suffix(".npy"))
    #     for motion_file_path in motion_file_paths
    # ]

    print(f"{len(motion_file_paths)=}")

    output_path = motion_base_path / "videos"
    output_path.mkdir(exist_ok=True, parents=True)

    if debug:
        return render_tracking_motion(
            0,
            motion_file_paths[0],
            output_path=output_path,
            override=override,
            walker_type=walker_type,
            video_fps=video_fps,
        )

    # motion_file_paths = motion_file_paths[:140]

    with concurrent.futures.ProcessPoolExecutor(max_workers) as executor:
        futures = executor.map(
            functools.partial(
                render_tracking_motion,
                output_path=output_path,
                override=override,
                walker_type=walker_type,
                video_fps=video_fps,
            ),
            range(len(motion_file_paths[0:])),
            motion_file_paths[0:],
        )
        results = list(futures)


if __name__ == "__main__":
    app.run(main)
