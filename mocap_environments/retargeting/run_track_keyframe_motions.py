r"""Functions for tracking keyframe sequences with mujoco humanoid.

```sh
python ./run_track_keyframe_motions.py \
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

from absl import app
from absl import flags
from absl import logging
import jax.numpy as jnp
import numpy as np

from mocap_environments.utils import git as git_utils

logging.set_verbosity(logging.INFO)

JnpArrayType = jnp.array
NpArrayType = np.ndarray
Path = pathlib.Path


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
    ("SimpleHumanoid", "SimpleHumanoidPositionControlled"),
    "Walker type.",
)
_OVERRIDE_FLAG = flags.DEFINE_boolean(
    "override", False, "Whether to override existing files."
)
_MAX_WORKERS_FLAG = flags.DEFINE_integer(
    "max_workers",
    10,
    "Maximum number of process pool workers, i.e. jobs to run in parallel.",
)


def process_motion(process_i: int, keyframe_path: Path, *args, **kwargs):
    # NOTE(hartikainen): Import here because otherwise mujoco rendering with EGL
    # will fail due to invalid EGL context.
    from mocap_environments.retargeting import run_lib

    logging.info(f"{process_i=}")

    try:
        return run_lib.process_motion(keyframe_path, *args, **kwargs)
    except Exception:
        logging.error(f"Failed to process keyframe_path='{str(keyframe_path)}'")
        raise


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
