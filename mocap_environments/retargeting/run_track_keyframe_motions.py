r"""Functions for tracking keyframe sequences with mujoco humanoid.

```sh
python ./run_track_keyframe_motions.py \
  --motions_base_dir="/tmp/keyframes" \
  --save_path="/tmp/run_track_keyframes" \
  --file_filter_regex="^.*(02_04_poses|87_01_poses)\.npy$" \
  --keyframe_fps="120.0" \
  --output_fps="60.0" \
  --override=true \
  --debug=true
```
"""

import concurrent.futures
import functools
import itertools
import pathlib
import random
import re
import sqlite3
import time
from typing import Any, Optional

from absl import app
from absl import flags
from absl import logging
import jax.numpy as jnp
import numpy as np
import pandas as pd
import tqdm

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
_KEYFRAME_FPS_FLAG = flags.DEFINE_float(
    "keyframe_fps", 60.0, "Input keyframe FPS. E.g. 20 for MDM outputs."
)
_OUTPUT_FPS_FLAG = flags.DEFINE_float("output_fps", 60.0, "Output FPS.")
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
    10,
    "Maximum number of process pool workers, i.e. jobs to run in parallel.",
)


def process_motion(
    process_i: int,
    keyframe_path: Path,
    save_base_dir: Path,
    config: Optional[dict] = None,
    *args,
    **kwargs,
):
    # NOTE(hartikainen): Import here because otherwise mujoco rendering with EGL
    # will fail due to invalid EGL context.
    from mocap_environments.retargeting import run_lib

    assert config is None, config

    logging.info(f"{process_i=}")

    try:
        start_time = time.time()

        result = run_lib.process_motion(
            keyframe_path, *args, save_base_dir=save_base_dir, **kwargs
        )
        end_time = time.time()

        result.setdefault("times", {})["process_motion"] = end_time - start_time

        database_path = save_base_dir / "results.sqlite"
        database_connection = sqlite3.connect(database_path, timeout=180)
        df = pd.json_normalize(result, sep="/")
        df.to_sql("results", database_connection, if_exists="append", index=False)
        del result
        return True
    except Exception as e:
        logging.error(
            f"Failed to process keyframe_path='{str(keyframe_path)}'.", exc_info=e
        )
        return False


def runner_adapter(partial, args):
    i, (keyframe_path, config) = args
    return partial(i, keyframe_path, config=config)


def main(argv):
    del argv

    file_filter_regex = _FILE_FILTER_REGEX_FLAG.value
    motions_base_dir = Path(_MOTIONS_BASE_DIR_FLAG.value).expanduser()
    override = _OVERRIDE_FLAG.value
    walker_type = _WALKER_TYPE_FLAG.value
    keyframe_fps = _KEYFRAME_FPS_FLAG.value
    output_fps = _OUTPUT_FPS_FLAG.value
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

    existing_results_path = save_base_dir / "results.sqlite"
    if existing_results_path.exists():
        connection = sqlite3.connect(existing_results_path)
        df = pd.read_sql("SELECT * FROM results", connection)

        if override:
            df = df[
                ~df["mocap-id"].str.contains(
                    "|".join([x.stem for x in keyframe_data_paths])
                )
            ]
            # Override existing results.
            df.to_sql("results", connection, if_exists="replace", index=False)

        existing_results = set(f"{x.rsplit('-', 1)[0]}" for x in df["mocap-id"])
        keyframe_data_paths = tuple(
            x
            for x in keyframe_data_paths
            if str(x.with_suffix("")) not in existing_results
        )

    process_motion_partial = functools.partial(
        process_motion,
        keyframes_base_dir=motions_base_dir,
        save_base_dir=save_base_dir,
        keyframe_fps=keyframe_fps,
        output_fps=output_fps,
        override=override,
        walker_type=walker_type,
    )

    git_info_directory_path = save_base_dir / "git-info"
    if git_info_directory_path.exists():
        next_i = len(list(save_base_dir.glob("git-info*")))
        git_info_directory_path = save_base_dir / f"git-info-{next_i}"

    git_utils.save_git_info(
        Path(__file__).parent.parent.parent, git_info_directory_path
    )

    configs = [None]

    keyframe_data_paths = list(keyframe_data_paths)
    random.shuffle(keyframe_data_paths)

    runner_adapter_ = functools.partial(runner_adapter, process_motion_partial)
    if debug:
        keyframe_data_paths = keyframe_data_paths[:1]
        iterables = map(
            runner_adapter_,
            enumerate(itertools.product(keyframe_data_paths, configs)),
        )
        results = list(iterables)
    else:
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers) as executor:
            futures = executor.map(
                runner_adapter_,
                enumerate(itertools.product(keyframe_data_paths, configs)),
            )

            results = list(
                tqdm.tqdm(
                    futures,
                    total=len(list(itertools.product(keyframe_data_paths, configs))),
                )
            )

    if not all(results):
        print("Some runs failed.")
        breakpoint()
        pass
        pass
    else:
        print("All runs finished.")


if __name__ == "__main__":
    flags.mark_flags_as_required(
        (
            _MOTIONS_BASE_DIR_FLAG,
            _SAVE_PATH_FLAG,
        )
    )
    app.run(main)
