r"""Functions for tracking keyframe sequences with mujoco humanoid.

```sh
python ./run_retarget_for_dagger.py \
  --motions_base_dir="/tmp/keyframes" \
  --save_path="/tmp/run_track_keyframes" \
  --file_filter_regex="^.*(02_04_poses|87_01_poses)\.npy$" \
  --fps="60.0" \
  --override=true \
  --debug=true
```
"""

import concurrent.futures
import functools
import pathlib
import re
from typing import Literal

from absl import app
from absl import flags
from absl import logging
import jax.numpy as jnp
import numpy as np
import pandas as pd

from mocap_environments.utils import git as git_utils
from mocap_environments.utils import video as video_utils
from mocap_environments.visualization import video as video_visualization

JnpArrayType = jnp.array
NpArrayType = np.ndarray
Path = pathlib.Path
WalkerEnum = Literal[
    "SimpleHumanoid",
    "SimpleHumanoidPositionControlled",
    "SMPLHumanoid",
    "SMPLHumanoidPositionControlled",
]


_MOTIONS_BASE_DIR_FLAG = flags.DEFINE_string(
    "motions_base_dir",
    None,
    (
        "Path to the keyframe motions base directory containing keyframe sequences in "
        "`.npy` format."
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
_FPS_FLAG = flags.DEFINE_float(
    "fps", 60.0, "Input keyframe FPS. E.g. 20 for MDM outputs."
)
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
_OVERRIDE_FLAG = flags.DEFINE_boolean(
    "override", False, "Whether to override existing files."
)
_MAX_WORKERS_FLAG = flags.DEFINE_integer(
    "max_workers",
    38,
    "Maximum number of process pool workers, i.e. jobs to run in parallel.",
)


def process_motion(
    process_i: int,
    keyframe_path: Path,
    keyframes_base_dir: Path,
    save_base_dir: Path,
    keyframe_fps: int | float,
    override: bool,
    walker_type: WalkerEnum,
):
    logging.set_verbosity(logging.ERROR)

    # NOTE(hartikainen): Import here because otherwise mujoco rendering with EGL
    # will fail due to invalid EGL context.
    from mocap_environments.retargeting import run_lib

    keyframe_file = keyframes_base_dir / keyframe_path
    logging.info(f"Processing file '{keyframe_file}', '{process_i=}'")

    amass_id = str(keyframe_path.with_suffix(""))

    result_file = (
        save_base_dir / "data" / "valid" / amass_id.replace("/", "+")
    ).with_suffix(".npy")

    if not override and result_file.exists():
        logging.info(f"Output file ('{result_file}') already exists. Skipping.")
        return

    with keyframe_file.open("rb") as f:
        keyframes = np.load(f)

    try:
        qposes, qvels, keyframes = run_lib.retarget_motion(
            keyframes, keyframe_fps=keyframe_fps, walker_type=walker_type
        )
    except Exception:
        logging.error(
            f"`retarget_motion` failed for keyframe_path='{str(keyframe_path)}'"
        )
        raise

    try:
        tracking_result = run_lib.track_motion(
            keyframes=keyframes,
            qposes=qposes,
            qvels=qvels,
            walker_type=walker_type,
        )
    except Exception:
        logging.error(f"`track_motion` failed for keyframe_path='{str(keyframe_path)}'")
        raise

    if tracking_result["mujoco_states"]["qpos"].shape[0] != keyframes.shape[0]:
        logging.info(
            f"`track_motion` terminated early for keyframe_path='{str(keyframe_path)}'"
        )

    try:
        frames, frames_fps = run_lib.playback_motion(
            qposes=tracking_result["mujoco_states"]["qpos"],
            qvels=tracking_result["mujoco_states"]["qvel"],
            keyframes=keyframes,
            walker_type=walker_type,
        )
    except Exception:
        logging.error(
            f"`playback_motion` failed for keyframe_path='{str(keyframe_path)}'"
        )
        raise

    actions = tracking_result["actions"]
    frames = video_visualization.add_action_plots_to_frames(frames, actions)
    frames = video_visualization.add_text_overlays_to_frames(
        frames,
        overlay_function=lambda t: amass_id,
        position=(frames[0].shape[1] / 2, frames[0].shape[0] * 0.9),
        anchor="md",
    )
    cost_data_frame = pd.DataFrame(tracking_result["mjpc_costs"]).sort_index(axis=1)
    cost_qpos_qvel_columns_index = cost_data_frame.columns.str.contains(
        "q(?:pos|vel)", regex=True
    )
    cost_qpos_qvel_columns = cost_data_frame.columns[cost_qpos_qvel_columns_index]
    cost_rest_columns = cost_data_frame.columns[~cost_qpos_qvel_columns_index]
    cost_qpos_qvel_data_frame = cost_data_frame[cost_qpos_qvel_columns]
    cost_rest_data_frame = cost_data_frame[cost_rest_columns]
    frames = video_visualization.add_text_overlays_to_frames(
        frames,
        overlay_function=lambda t: (
            cost_qpos_qvel_data_frame.iloc[t].to_string(
                float_format=lambda x: f"{x:+08.4f}"
            )
        ),
        position=(0.0 * frames[0].shape[1], frames[0].shape[0] * 0.125),
    )
    frames = video_visualization.add_text_overlays_to_frames(
        frames,
        overlay_function=lambda t: (
            cost_rest_data_frame.iloc[t].to_string(float_format=lambda x: f"{x:+08.4f}")
        ),
        position=(0.55 * frames[0].shape[1], frames[0].shape[0] * 0.125),
    )

    video_save_path = (
        save_base_dir / "videos" / "valid" / f"{amass_id.replace('/', '+')}"
    ).with_suffix(".mp4")

    Path(video_save_path).parent.mkdir(parents=True, exist_ok=True)

    video_utils.save_video(
        video_save_path,
        frames,
        input_fps=frames_fps,
        output_fps=frames_fps,
    )

    result_file.parent.mkdir(parents=True, exist_ok=True)
    with result_file.open("wb") as f:
        np.savez_compressed(
            f,
            qpos=tracking_result["mujoco_states"]["qpos"],
            qvel=tracking_result["mujoco_states"]["qvel"],
            keyframes=keyframes,
            mocap_id=amass_id,
        )


def main(argv):
    del argv

    file_filter_regex = _FILE_FILTER_REGEX_FLAG.value
    motions_base_dir = Path(_MOTIONS_BASE_DIR_FLAG.value).expanduser()
    override = _OVERRIDE_FLAG.value
    walker_type = _WALKER_TYPE_FLAG.value
    fps = _FPS_FLAG.value
    save_base_dir = Path(_SAVE_PATH_FLAG.value).expanduser()
    debug = _DEBUG_FLAG.value
    max_workers = _MAX_WORKERS_FLAG.value
    keyframe_data_paths = tuple(
        x.relative_to(motions_base_dir) for x in motions_base_dir.rglob("**/*.npy")
    )
    if file_filter_regex is not None:
        file_pattern = re.compile(file_filter_regex)
        keyframe_data_paths = tuple(
            x for x in keyframe_data_paths if re.match(file_pattern, str(x))
        )

    process_motion_partial = functools.partial(
        process_motion,
        keyframes_base_dir=motions_base_dir,
        save_base_dir=save_base_dir,
        keyframe_fps=fps,
        override=override,
        walker_type=walker_type,
    )

    git_utils.save_git_info(
        Path(__file__).parent.parent.parent, save_base_dir / "git-info"
    )

    if debug:
        keyframe_data_paths = keyframe_data_paths[:1]
        iterables = map(
            process_motion_partial,
            range(len(keyframe_data_paths)),
            keyframe_data_paths,
        )
        results = list(iterables)
        del results
        return

    with concurrent.futures.ProcessPoolExecutor(max_workers) as executor:
        futures = executor.map(
            process_motion_partial, range(len(keyframe_data_paths)), keyframe_data_paths
        )
        results = list(futures)
        del results
        return


if __name__ == "__main__":
    flags.mark_flags_as_required(
        (
            _MOTIONS_BASE_DIR_FLAG,
            _SAVE_PATH_FLAG,
        )
    )
    app.run(main)
