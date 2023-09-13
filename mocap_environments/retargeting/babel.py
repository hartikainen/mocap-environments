"""Functions for manipulating BABEL [1] data.

[1] Punnakkal, Abhinanda R., et al. "BABEL: Bodies, action and behavior with
    english labels." Proceedings of the IEEE/CVF Conference on Computer Vision
    and Pattern Recognition. 2021.
"""

import copy
import dataclasses
import json
import pathlib
from typing import Dict, Iterable, List, Optional, Sequence, Union
import urllib

import tree

dataclass = dataclasses.dataclass
Path = pathlib.Path
URLType = urllib.parse.ParseResult

ALL_SPLITS = frozenset(
    {
        "extra_train.json",
        "extra_val.json",
        "test.json",
        "train.json",
        "val.json",
    }
)

# Splits with language labels.
LANGUAGE_LABELED_DENSE_SPLITS = frozenset({"train.json", "val.json", "test.json"})
EXTRA_SPLITS = frozenset({x for x in ALL_SPLITS if x.startswith("extra_")})


@dataclass
class Label:
    """Babel annotation label."""

    act_cat: List[str]
    proc_label: str
    raw_label: str
    seg_id: str


@dataclass
class SequenceLabel(Label):
    """Babel annotation for the full sequence."""

    ...


@dataclass
class FrameLabel(Label):
    """Babel annotation for the individual frames in the sequence."""

    end_t: float
    start_t: float


@dataclass
class Annotation:
    """Babel annotation, possibly containing multiple sequential labels."""

    anntr_id: str
    babel_lid: str
    mul_act: bool


@dataclass
class FrameAnnotation(Annotation):
    labels: List[FrameLabel]

    @classmethod
    def from_dict(cls, value: Dict):
        value["labels"] = type(value["labels"])(
            FrameLabel(**x) for x in value["labels"]
        )
        return cls(**value)


@dataclass
class SequenceAnnotation(Annotation):
    labels: List[SequenceLabel]

    @classmethod
    def from_dict(cls, value: Dict):
        value["labels"] = type(value["labels"])(
            SequenceLabel(**x) for x in value["labels"]
        )  # pytype: disable=wrong-arg-types
        return cls(**value)


@dataclass
class Sample:
    """Container for a single Babel sample."""

    babel_sid: int
    dur: float
    feat_p: Path
    frame_ann: Optional[FrameAnnotation]
    seq_ann: SequenceAnnotation
    url: URLType

    @classmethod
    def from_dict(cls, value: Dict):
        # Convert types if needed
        value["url"] = urllib.parse.urlparse(value["url"])
        value["feat_p"] = Path(value["feat_p"])

        if "frame_anns" in value:
            assert "frame_ann" not in value
            value["frame_ann"] = value.pop("frame_anns")

        if "seq_anns" in value:
            assert "seq_ann" not in value
            value["seq_ann"] = value.pop("seq_anns")

        if value["frame_ann"] is not None:
            if isinstance(value["frame_ann"], Sequence):
                # TODO(hartikainen): Use tail of `value["frame_ann"]`
                value["frame_ann"] = FrameAnnotation.from_dict(value["frame_ann"][0])
            elif isinstance(value["frame_ann"], dict):
                value["frame_ann"] = FrameAnnotation.from_dict(value["frame_ann"])
            else:
                raise ValueError(f"{value['frame_ann']=}")

        if isinstance(value["seq_ann"], Sequence):
            # TODO(hartikainen): Use tail of `value["seq_ann"]`
            value["seq_ann"] = SequenceAnnotation.from_dict(value["seq_ann"][0])
        elif isinstance(value["seq_ann"], dict):
            value["seq_ann"] = SequenceAnnotation.from_dict(value["seq_ann"])
        else:
            raise ValueError(f"{value['seq_ann']=}")

        return cls(**value)

    def get_all_labels(self):
        sequence_labels = self.seq_ann.labels
        frame_labels = self.frame_ann.labels if self.frame_ann is not None else []
        all_labels = {
            x
            for label in sequence_labels + frame_labels
            for x in {
                *(label.act_cat or []),
                label.proc_label,
                label.raw_label,
            }
        } - {None}
        return all_labels


def fix_sample_labels(babel_sample: Sample) -> Sample:
    """Return processed language labels for the given `babel_sample`.

    If the sample's frame annotations (`frame_ann`) don't exist, augment the
    sequence annotation (`seq_ann`) with timestamps and use that as frame
    annotation instead. Then create dense frame labels -- i.e. a ragged series of
    shape `(T, X)` where `T` is the sequence length and `X` is a variable-length
    depending on how many labels the given timestep included -- and return
    both the sparse dense and dense labels. In the downstream tasks, it's likely
    that only the dense labels are needed.
    """
    babel_sample = copy.deepcopy(babel_sample)
    assert babel_sample.seq_ann.labels is not None, babel_sample

    # TODO(hartikainen): Fix broken samples. E.g. `CMU/CMU/31/31_09_poses.npz`

    if babel_sample.frame_ann is None:
        frame_labels_raw = dataclasses.asdict(babel_sample.seq_ann)
        for frame_label in frame_labels_raw["labels"]:
            frame_label["start_t"] = 0.0
            frame_label["end_t"] = babel_sample.dur

        babel_sample.frame_ann = FrameAnnotation.from_dict(frame_labels_raw)

    return babel_sample


def verify_babel_directory(path: Path, expected_files=ALL_SPLITS):
    """Verify that the `path` contains expeceted babel files."""
    assert path.is_dir(), path

    for expected_filename in expected_files:
        expected_path = path / expected_filename
        if not expected_path.exists():
            raise ValueError(
                f"Expected to find '{expected_filename}' from '{path}', but could"
                " not. Make sure that the correct data is downloaded from"
                " 'https://babel.is.tue.mpg.de/data.html'. The folder should contain"
                f" the following files: {expected_files}"
            )
        assert expected_path.is_file()


def load_dataset_split(path: Path) -> Dict[str, Sample]:
    """Load single babel split, e.g. 'train'."""
    path = path.expanduser()

    assert path.exists(), path

    with path.open("rt") as f:
        json_data = json.load(f)

    data = {key: Sample.from_dict(value) for key, value in json_data.items()}

    return data


def load_dataset(
    path: Path | str,
    include_splits: Iterable[str] = LANGUAGE_LABELED_DENSE_SPLITS,
    fix_labels: bool = True,
) -> Dict[str, Dict[str, Sample]]:
    """Load the full babel dataset, i.e. all the splits excluding extras."""
    path = Path(path).expanduser()
    verify_babel_directory(path, expected_files=include_splits)

    data = {
        split_name: load_dataset_split(path / split_name)
        for split_name in include_splits
    }

    if fix_labels:
        data = tree.map_structure(fix_sample_labels, data)

    return data
