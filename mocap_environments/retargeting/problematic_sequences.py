"""Functions for detecting problematic AMASS/BABEL sequences."""

import dataclasses
import pathlib
import re

import numpy as np

from mocap_environments.retargeting import babel

Path = pathlib.Path
BabelSample = babel.Sample

PROBLEMATIC_SEQUENCES_INDEX = None


@dataclasses.dataclass
class MotionSample:
    joints_xyz: np.ndarray
    fps: float | int
    sequence_length: float | int


def get_all_indexes():
    global PROBLEMATIC_SEQUENCES_INDEX

    if PROBLEMATIC_SEQUENCES_INDEX is not None:
        return PROBLEMATIC_SEQUENCES_INDEX

    index_directory_path = Path(__file__).parent / "physically_implausible_sequences"
    all_index_files = sorted(index_directory_path.rglob("*-index.txt"))

    result = {}
    for index_file in all_index_files:
        result[index_file.stem] = set(index_file.read_text().splitlines())

    PROBLEMATIC_SEQUENCES_INDEX = result

    return PROBLEMATIC_SEQUENCES_INDEX


def get_index(key: str):
    return get_all_indexes()[key]


def in_index(amass_id: str, index_key: str):
    problematic_sequences_index = get_index(index_key)
    return amass_id in problematic_sequences_index


@dataclasses.dataclass
class ProblematicSequenceDiagnostics:
    possibly_on_platform: bool
    possibly_on_treadmill: bool
    too_short_sequence: bool
    action_category_suggests_object: bool
    in_chair_index: bool
    in_external_force_index: bool
    in_hang_index: bool
    in_lean_index: bool
    in_platform_index: bool
    in_skate_index: bool
    in_soft_ground_index: bool
    in_swim_index: bool
    in_treadmill_index: bool
    in_manually_verified_physically_plausible_index: bool

    @property
    def is_problematic(self):
        return not self.in_manually_verified_physically_plausible_index and any(
            (
                self.possibly_on_platform,
                self.possibly_on_treadmill,
                self.too_short_sequence,
                self.action_category_suggests_object,
                self.in_chair_index,
                self.in_external_force_index,
                self.in_hang_index,
                self.in_lean_index,
                self.in_platform_index,
                self.in_skate_index,
                self.in_soft_ground_index,
                self.in_swim_index,
                self.in_treadmill_index,
            )
        )

    @property
    def is_manually_verified_and_problematic(self):
        return not self.in_manually_verified_physically_plausible_index and any(
            (
                self.too_short_sequence,
                self.in_chair_index,
                self.in_external_force_index,
                self.in_hang_index,
                self.in_lean_index,
                self.in_platform_index,
                self.in_skate_index,
                self.in_soft_ground_index,
                self.in_swim_index,
                self.in_treadmill_index,
            )
        )

    @property
    def is_manually_verified_and_unproblematic(self):
        return self.in_manually_verified_physically_plausible_index


def is_very_likely_on_platform(sample: MotionSample):
    fps = sample.fps
    joints_xyz = sample.joints_xyz
    joints_xyz[..., -1] -= np.min(joints_xyz[..., -1])

    height_threshold_m = 0.15
    time_threshold_seconds = 2.0
    time_threshold_frames = min(
        int(np.ceil(fps * time_threshold_seconds)), joints_xyz.shape[0]
    )

    window_min = np.min(
        np.lib.stride_tricks.sliding_window_view(
            joints_xyz[..., -1], time_threshold_frames, axis=0
        ),
        axis=tuple(np.arange(1, joints_xyz.ndim)),
    )

    np.testing.assert_equal(
        window_min.shape, (joints_xyz.shape[0] - time_threshold_frames + 1,)
    )

    window_min_max = np.max(window_min)

    # Explicitly cast to python built-in `bool`.
    very_likely_on_platform = bool(height_threshold_m < window_min_max)

    return very_likely_on_platform


def is_possibly_on_platform(sample: MotionSample):
    """Check if keyframe sample suggests a platform with somewhat loose tolerance.

    This function, with `height_threshold_m = 0.5` and `time_threshold_seconds = 1.0`
    returns a correct value for all AMASS sequences for which there exists a BABEL
    rendering, i.e. when BABEL sample exists.

    `True` for 187/15653 and `False` for 15466/15653.

    """
    fps = sample.fps
    joints_xyz = sample.joints_xyz
    joints_xyz[..., -1] -= np.min(joints_xyz[..., -1])

    height_threshold_m = 0.075
    time_threshold_seconds = 1.0
    time_threshold_frames = min(
        int(np.ceil(fps * time_threshold_seconds)), joints_xyz.shape[0]
    )

    window_min = np.min(
        np.lib.stride_tricks.sliding_window_view(
            joints_xyz[..., -1], time_threshold_frames, axis=0
        ),
        axis=tuple(np.arange(1, joints_xyz.ndim)),
    )

    np.testing.assert_equal(
        window_min.shape, (joints_xyz.shape[0] - time_threshold_frames + 1,)
    )

    window_min_max = np.max(window_min)

    # Explicitly cast to python built-in `bool`.
    very_likely_on_platform = bool(height_threshold_m < window_min_max)

    return very_likely_on_platform


def is_too_short_sequence(sample: MotionSample):
    return sample.sequence_length < 2


def action_category_suggests_object(sample: BabelSample):
    object_action_categories = {
        "place something",
        "interact with/use object",
        "grasp object",
        "move something",
        "action with ball",
        "clean something",
        "take/pick something up",
        "lift something",
        "open something",
        "give something",
        "knock",
        "touch object",
        "close something",
        "drink",
        "play catch",
        "interact with rope",
        "catch",
        "fill",
        "tie",
        "eat",
        "press something",
        "operate interface",
        "hang",
        "remove",
        "lose",
        "grind",
    }

    all_labels = sample.seq_ann.labels + (
        sample.frame_ann.labels if sample.frame_ann is not None else []
    )
    object_action_category_exists = any(
        object_action_categories & set(label.act_cat)
        for label in all_labels
        if label.act_cat is not None
    )

    return object_action_category_exists


def possibly_on_treadmill(motion_sample: MotionSample, babel_sample: BabelSample):
    frame_labels = (
        set.union(
            *(  # pylint: disable=long-ternary
                {*(label.act_cat or ()), label.proc_label, label.raw_label}
                for label in babel_sample.frame_ann.labels
            )
        )
        if babel_sample.frame_ann is not None
        else set()
    ) - {None}
    sequence_labels = set.union(
        *(
            {*(label.act_cat or ()), label.proc_label, label.raw_label}
            for label in babel_sample.seq_ann.labels
        )
    ) - {None}
    all_labels = frame_labels | sequence_labels

    move_regex = re.compile("walk|run|jog|tread", re.IGNORECASE)

    if not any(map(move_regex.search, all_labels)):
        return False

    root_xyz = motion_sample.joints_xyz[..., 0, :]
    root_xy = root_xyz[..., 0:2]
    root_from_origin = np.linalg.norm(root_xy - root_xy[0], ord=2, axis=-1)

    return all(root_from_origin < 0.25)


def is_problematic_sample(
    amass_id: str, motion_sample: MotionSample, babel_sample: BabelSample
):
    diagnostics = ProblematicSequenceDiagnostics(
        possibly_on_platform=is_possibly_on_platform(motion_sample),
        possibly_on_treadmill=(
            possibly_on_treadmill(motion_sample, babel_sample)
            if babel_sample is not None
            else False
        ),
        too_short_sequence=is_too_short_sequence(motion_sample),
        action_category_suggests_object=(
            action_category_suggests_object(babel_sample)
            if babel_sample is not None
            else False
        ),
        in_chair_index=in_index(amass_id, "chair-index"),
        in_external_force_index=in_index(amass_id, "external-force-index"),
        in_hang_index=in_index(amass_id, "hang-index"),
        in_lean_index=in_index(amass_id, "lean-index"),
        in_platform_index=in_index(amass_id, "platform-index"),
        in_skate_index=in_index(amass_id, "skate-index"),
        in_soft_ground_index=in_index(amass_id, "soft-ground-index"),
        in_swim_index=in_index(amass_id, "swim-index"),
        in_treadmill_index=in_index(amass_id, "treadmill-index"),
        in_manually_verified_physically_plausible_index=in_index(
            amass_id, "manually-verified-physically-plausible-index"
        ),
    )

    result = diagnostics.is_problematic

    return result, diagnostics
