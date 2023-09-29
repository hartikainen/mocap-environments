"""Functions for loading simple humaonid AMASS tracking dataset."""

import pathlib
import re
from typing import Optional

import numpy as np
import numpy.typing as npt
import tensorflow as tf

Path = pathlib.Path


def preprocess_motion(x):
    # # TODO(hartikainen): Make this configurable and more transparent. These
    # # values depend on the joint ordering in the the AMASS dataset.
    # # fmt: off
    # smpl_to_simple_humanoid_indices = [
    #     15, 0, 2, 5, 8, 11, 1, 4, 7, 10, 17, 19, 21, 16, 18, 20,
    # ]

    # # fmt: on
    # num_joints = tf.shape(x["keyframes"])[-2]
    # if tf.equal(num_joints, 24):
    #     x["keyframes"] = tf.gather(
    #         x["keyframes"], smpl_to_simple_humanoid_indices, axis=-2
    #     )

    num_joints = tf.shape(x["keyframes"])[-2]
    tf.debugging.assert_equal(num_joints, 16)

    return x


def read_motion_file(
    motion_file: tf.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the motion file from the `npy`-serialized `motion_file`."""
    motion_file_path = Path(motion_file.numpy().decode())
    with motion_file_path.open("rb") as f:
        motion = np.load(f)
        motion = dict(motion)
        # expected_keys = {
        #     "qpos",
        #     "qvel",
        #     "act",
        #     "act_dot",
        #     "ctrl",
        #     "keyframes",
        #     "mocap_id",
        # }
        # if set(motion.keys()) != expected_keys:
        #     raise ValueError(
        #         f"Something wrong with the motion '{str(motion_file_path)}'. Expected motion to "
        #         f"contain keys '{expected_keys}' but got '{set(motion.keys())}'."
        #     )
        motion = {key: motion[key] for key in {"keyframes", "qpos", "qvel", "mocap_id"}}

    np.testing.assert_equal(motion["keyframes"].ndim, 3)
    num_timesteps, num_joints = motion["keyframes"].shape[:2]
    np.testing.assert_equal(motion["mocap_id"].shape, ())
    np.testing.assert_equal(motion["keyframes"].shape, (num_timesteps, num_joints, 3))
    np.testing.assert_equal(
        motion["qpos"].shape, (num_timesteps, motion["qvel"].shape[-1] + 1)
    )
    np.testing.assert_equal(
        motion["qvel"].shape, (num_timesteps, motion["qpos"].shape[-1] - 1)
    )

    result = (
        motion["mocap_id"],
        motion["keyframes"],
        motion["qpos"],
        motion["qvel"],
    )
    return result


def load_dataset(
    data_path: Path,
    mocap_id_filter_regex: Optional[str] = None,
    shuffle_files: bool = False,
) -> tf.data.Dataset:
    """Load a motion dataset from `data_path` in `tf.data.Dataset` form."""
    # TODO(mjpc-dagger): Might have to change this `glob` based on the file library.
    motion_file_paths = list(data_path.rglob("*.npy"))

    if mocap_id_filter_regex is not None:
        mocap_id_pattern = re.compile(mocap_id_filter_regex)
        motion_file_paths = [
            x
            for x in motion_file_paths
            if re.match(mocap_id_pattern, str(x.relative_to(data_path)))
        ]

    dataset = tf.data.Dataset.from_tensor_slices(list(map(str, motion_file_paths)))
    if shuffle_files:
        dataset = dataset.shuffle(dataset.cardinality())
    # NOTE(hartikainen): `tf.py_function` doesn't allow `dict` outputs so we map
    # to tuples and unpack them in the subsequent `map`.
    dataset = dataset.map(
        lambda x: tf.py_function(
            read_motion_file,
            [x],
            Tout=[
                # "mocap_id": tf.string,
                tf.TensorSpec(shape=(), dtype=tf.string),
                # "keyframes": tf.float32,
                tf.TensorSpec(shape=(None, 16, 3), dtype=tf.float32),
                # "qpos": tf.float32,
                tf.TensorSpec(shape=(None, 28), dtype=tf.float32),
                # "qvel": tf.float32,
                tf.TensorSpec(shape=(None, 27), dtype=tf.float32),
            ],
        )
    )
    dataset = dataset.map(
        lambda *x: {
            "mocap_id": x[0],
            "keyframes": x[1],
            "qpos": x[2],
            "qvel": x[3],
        }
    )
    dataset = dataset.map(preprocess_motion)
    return dataset
