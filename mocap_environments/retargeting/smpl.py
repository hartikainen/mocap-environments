"""Constants and data classes for SMPL data."""

import dataclasses

import numpy as np

dataclass = dataclasses.dataclass


@dataclass
class Skeleton:
    joint_names: np.ndarray
    joint_indices: np.ndarray
    joint_parents: np.ndarray
    joints_left: np.ndarray
    joints_right: np.ndarray


SMPL_H_JOINT_NAMES = np.array(
    (
        "pelvis",
        "left_hip",
        "right_hip",
        "spine1",
        "left_knee",
        "right_knee",
        "spine2",
        "left_ankle",
        "right_ankle",
        "spine3",
        "left_foot",
        "right_foot",
        "neck",
        "left_collar",
        "right_collar",
        "head",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_index1",
        "left_index2",
        "left_index3",
        "left_middle1",
        "left_middle2",
        "left_middle3",
        "left_pinky1",
        "left_pinky2",
        "left_pinky3",
        "left_ring1",
        "left_ring2",
        "left_ring3",
        "left_thumb1",
        "left_thumb2",
        "left_thumb3",
        "right_index1",
        "right_index2",
        "right_index3",
        "right_middle1",
        "right_middle2",
        "right_middle3",
        "right_pinky1",
        "right_pinky2",
        "right_pinky3",
        "right_ring1",
        "right_ring2",
        "right_ring3",
        "right_thumb1",
        "right_thumb2",
        "right_thumb3",
        "nose",
        "right_eye",
        "left_eye",
        "right_ear",
        "left_ear",
        "left_big_toe",
        "left_small_toe",
        "left_heel",
        "right_big_toe",
        "right_small_toe",
        "right_heel",
        "left_thumb",
        "left_index",
        "left_middle",
        "left_ring",
        "left_pinky",
        "right_thumb",
        "right_index",
        "right_middle",
        "right_ring",
        "right_pinky",
    )
)

SMPL_H_NUM_JOINTS = len(SMPL_H_JOINT_NAMES)
SMPL_H_JOINT_INDEX = np.arange(SMPL_H_NUM_JOINTS)
SMPL_H_JOINT_PARENTS = np.array(
    (
        -1,
        0,
        0,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        9,
        9,
        12,
        13,
        14,
        16,
        17,
        18,
        19,
        20,
        22,
        23,
        20,
        25,
        26,
        20,
        28,
        29,
        20,
        31,
        32,
        20,
        34,
        35,
        21,
        37,
        38,
        21,
        40,
        41,
        21,
        43,
        44,
        21,
        46,
        47,
        21,
        49,
        50,
    ),
    dtype=np.int32,
)
SMPL_H_JOINT_INDEX_LEFT = SMPL_H_JOINT_INDEX[
    np.char.startswith(SMPL_H_JOINT_NAMES, "left_")
]
SMPL_H_JOINT_INDEX_RIGHT = SMPL_H_JOINT_INDEX[
    np.char.startswith(SMPL_H_JOINT_NAMES, "right_")
]

np.testing.assert_array_equal(
    np.char.replace(SMPL_H_JOINT_NAMES[SMPL_H_JOINT_INDEX_LEFT], "left_", ""),
    np.char.replace(SMPL_H_JOINT_NAMES[SMPL_H_JOINT_INDEX_RIGHT], "right_", ""),
)

SMPL_NUM_JOINTS = 24
SMPL_JOINT_INDEX = np.arange(SMPL_NUM_JOINTS)
SMPL_JOINT_NAMES = np.concatenate(
    [SMPL_H_JOINT_NAMES[: SMPL_NUM_JOINTS - 2].copy(), ["left_hand", "right_hand"]]
)
SMPL_JOINT_PARENTS = np.array(
    (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21),
    dtype=np.int32,
)
np.testing.assert_equal(SMPL_NUM_JOINTS, SMPL_JOINT_INDEX.size)
np.testing.assert_equal(SMPL_NUM_JOINTS, SMPL_JOINT_NAMES.size)
np.testing.assert_equal(SMPL_NUM_JOINTS, SMPL_JOINT_PARENTS.size)

SMPL_JOINT_INDEX_LEFT = SMPL_JOINT_INDEX[np.char.startswith(SMPL_JOINT_NAMES, "left_")]
SMPL_JOINT_INDEX_RIGHT = SMPL_JOINT_INDEX[
    np.char.startswith(SMPL_JOINT_NAMES, "right_")
]

np.testing.assert_array_equal(
    np.char.replace(SMPL_JOINT_NAMES[SMPL_JOINT_INDEX_LEFT], "left_", ""),
    np.char.replace(SMPL_JOINT_NAMES[SMPL_JOINT_INDEX_RIGHT], "right_", ""),
)

SMPL_H_SKELETON = Skeleton(
    joint_names=SMPL_H_JOINT_NAMES,
    joint_indices=SMPL_H_JOINT_INDEX,
    joint_parents=SMPL_H_JOINT_PARENTS,
    joints_left=SMPL_H_JOINT_INDEX_LEFT,
    joints_right=SMPL_H_JOINT_INDEX_RIGHT,
)

SMPL_SKELETON = Skeleton(
    joint_names=SMPL_JOINT_NAMES,
    joint_indices=SMPL_JOINT_INDEX,
    joint_parents=SMPL_JOINT_PARENTS,
    joints_left=SMPL_JOINT_INDEX_LEFT,
    joints_right=SMPL_JOINT_INDEX_RIGHT,
)

# TODO(hartikainen): Not sure if indexing SMPL_H to create SMPL is kosher.
SMPL_H_TO_SMPL_JOINT_INDEX = np.concatenate(
    [
        *[
            SMPL_H_JOINT_INDEX[SMPL_H_JOINT_NAMES == joint_name]
            for joint_name in SMPL_JOINT_NAMES[:-2]
        ],
        SMPL_H_JOINT_INDEX[SMPL_H_JOINT_NAMES == "left_middle"],
        SMPL_H_JOINT_INDEX[SMPL_H_JOINT_NAMES == "right_middle"],
    ]
)

np.testing.assert_equal(SMPL_NUM_JOINTS, SMPL_H_TO_SMPL_JOINT_INDEX.size)
