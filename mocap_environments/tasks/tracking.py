"""Motion tracking `dm_control.composer.Task`."""

import dataclasses
import pathlib
import re
from typing import Callable, Optional

from dm_control import composer
from dm_control import mjcf
from dm_control.composer.observation import observable as base_observable
from dm_control.locomotion.tasks.reference_pose import utils
from dm_control.locomotion.walkers import base
from dm_control.mujoco.wrapper import mjbindings
import dm_env
import numpy as np
from tensorflow import data as tf_data
import tree

dataclass = dataclasses.dataclass
mjlib = mjbindings.mjlib
Path = pathlib.Path

DEFAULT_MJPC_TASK_XML_FILE_PATH = (
    Path(__file__).parent / "motion_tracking_mjpc_task.xml"
)


@dataclass
class VisualizationConfig:
    """Configuration for motion capture rollout visualization."""

    resolution: tuple[int, int] = (960, 960)
    kinematic_marker_colors: dict[
        str, tuple[float, float, float, float]
    ] = dataclasses.field(default_factory=dict)
    kinematic_marker_sizes: dict[str, float] = dataclasses.field(default_factory=dict)


def add_keyframe_elements(keyframe_element, keyframes, qposes, qvels):
    for x in keyframe_element.all_children():
        assert x.tag == "key", x
        x.remove()

    for i, key_data in enumerate(keyframes, 0):
        mpos_str = " ".join(map(str, key_data.flatten()))
        key_element = keyframe_element.insert("key", i, mpos=mpos_str)
        if i == 0:
            key_element.set_attributes(name="home")
        if i < len(qposes):
            qpos_str = " ".join(map(str, qposes[i].flatten()))
            key_element.set_attributes(qpos=qpos_str)
        if i < len(qvels):
            qvel_str = " ".join(map(str, qvels[i].flatten()))
            key_element.set_attributes(qvel=qvel_str)

    return keyframe_element


class TrackingTask(composer.Task):
    """Task for motion tracking.

    In this task, the `walker`'s goal is to track the motion keyframe sequences from the
    `motion_dataset`. We assume that the `keyframes` in `motion_dataset` samples are
    ordered according to the walker's `mocap_tracking_sites`. That is, for each joint
    index `j in range(J)`, for `np.shape(sample["keyframes"]) == (T, J, 3)`, the
    `sample["keyframes"][T, j, :]` corresponds to `mocap_tracking_sites[j]`.
    TODO(hartikainen): Should make this configurable and more transparent.
    """

    def __init__(
        self,
        walker: Callable[[], base.Walker],
        arena: composer.Arena,
        motion_dataset: tf_data.Dataset,
        mocap_reference_steps: tuple[int, ...] | int = 0,
        visualization_config: Optional[VisualizationConfig] = None,
        physics_timestep: Optional[int | float] = None,
        termination_threshold: float = float("inf"),
        random_init_time_step: bool = False,
        mjpc_task_xml_file_path: Optional[Path] = None,
    ):
        self._termination_threshold = termination_threshold
        self._should_terminate = False
        self._arena = arena
        self._motion_dataset = motion_dataset
        self._motion_dataset_iterator = motion_dataset.as_numpy_iterator()
        self._mocap_reference_steps = np.array(mocap_reference_steps)
        self._random_init_time_step = random_init_time_step

        if mjpc_task_xml_file_path is None:
            mjpc_task_xml_file_path = DEFAULT_MJPC_TASK_XML_FILE_PATH

        mjpc_task_element = mjcf.parser.from_file(mjpc_task_xml_file_path)

        if physics_timestep is not None:
            np.testing.assert_equal(mjpc_task_element.option.timestep, physics_timestep)

        physics_timestep = mjpc_task_element.option.timestep
        agent_timestep = mjpc_task_element.custom.find("numeric", "agent_timestep").data
        np.testing.assert_equal(agent_timestep, physics_timestep)

        if hasattr(mjpc_task_element.option, "integrator"):
            model_integrator = mjpc_task_element.option.integrator
            if model_integrator == "implicitfast":
                agent_integrator = mjpc_task_element.custom.find(
                    "numeric", "agent_integrator"
                ).data
                assert agent_integrator == 3, agent_integrator

        self.root_entity._mjcf_root.option.timestep = mjpc_task_element.option.timestep

        self.root_entity._mjcf_root.include_copy(mjpc_task_element)

        self._walker = utils.add_walker(walker, self._arena)
        if visualization_config is None:
            visualization_config = VisualizationConfig()
        self.visualization_config = visualization_config

        offwidth, offheight = visualization_config.resolution
        # Use `getattr` because "global" is a reserved keyword and cannot be used
        # as an identifier.
        getattr(self.root_entity._mjcf_root.visual, "global").offwidth = offwidth
        getattr(self.root_entity._mjcf_root.visual, "global").offheight = offheight

        self._walker.mjcf_model.find("material", "self").rgba = np.array(
            [*self._walker.mjcf_model.find("material", "self").rgba[:3], 1.0]
        )

        # For whatever reason, `take(1)` hangs here on macos.
        # self._motion_sequence = next(motion_dataset.take(1).as_numpy_iterator())
        self._motion_sequence = next(motion_dataset.as_numpy_iterator())

        site_colors = visualization_config.kinematic_marker_colors
        site_sizes = visualization_config.kinematic_marker_sizes
        site_joint_name_re = re.compile(r"^tracking\[(\w+)\]$")
        self.mocap_sites = []
        for site_element in self._walker.mocap_tracking_sites:
            site_joint_name = re.match(site_joint_name_re, site_element.name).group(1)
            mocap_body_element = self.root_entity.root_body.add(
                "body",
                name=f"mocap[{site_joint_name}]",
                mocap="true",
            )
            mocap_site_element = mocap_body_element.add(
                "site",
                name=f"mocap[{site_joint_name}]",
                type="sphere",
                size=str(site_sizes.get(site_joint_name, 0.03)),
                rgba=" ".join(map(str, site_colors.get(site_joint_name, (0, 0, 1, 1)))),
                **{"class": "mocap_site"},
            )
            self.mocap_sites.append(mocap_site_element)

            self._walker.mjcf_model.sensor.insert(
                "framepos",
                -1,
                name=f"tracking_pos[{site_joint_name}]",
                objtype="site",
                objname=f"tracking[{site_joint_name}]",
            )

            self._walker.mjcf_model.sensor.insert(
                "framelinvel",
                -1,
                name=f"tracking_linvel[{site_joint_name}]",
                objtype="site",
                objname=f"tracking[{site_joint_name}]",
            )

        self._time_step = 0
        self._init_time_step = 0
        self._end_mocap = False

        control_timestep = physics_timestep
        self.set_timesteps(
            physics_timestep=physics_timestep, control_timestep=control_timestep
        )

        self._walker.observables.add_observable(
            "keyframes_local", base_observable.Generic(self.get_keyframes_local)
        )

        self._walker.observables.add_observable(
            "time_step", base_observable.Generic(self.get_time_step)
        )

        enabled_observables_names = [
            "body_height",
            "joints_pos",
            "joints_vel",
            "sensors_velocimeter",
            "sensors_gyro",
            "end_effectors_pos",
            "world_zaxis",
            "sensors_touch",
            "sensors_torque",
            "actuator_activation",
            "keyframes_local",
            "prev_action",
            "time_step",
        ]

        for key in enabled_observables_names:
            observable = self._walker.observables.get_observable(key)
            observable.enabled = True

    @property
    def root_entity(self):
        return self._arena

    def initialize_episode_mjcf(self, random_state: np.random.RandomState):
        del random_state
        self._motion_sequence = next(self._motion_dataset_iterator)

        keyframes = self._motion_sequence["keyframes"]
        qposes = self._motion_sequence["qpos"]
        qvels = self._motion_sequence["qvel"]
        if self._random_init_time_step:
            self._time_step = np.random.randint(keyframes.shape[0] - 1)
        else:
            self._time_step = 0

        # NOTE(hartikainen): Temporary sanity check. This should never happen unless
        # there's a bug somewhere.
        assert self._time_step < keyframes.shape[0] - 1, (
            self._time_step,
            keyframes.shape[0],
            self._motion_sequence["mocap_id"].decode(),
        )

        self._init_time_step = self._time_step

        keyframe_element = self.root_entity.mjcf_model.keyframe
        # Note how the times are currently inconsistent between here and the MJPC task.
        # Our `_time_step` can be non-zero, whereas the MJPC residual's `data->time`,
        # which is used for the residual computation, always starts from zero.
        add_keyframe_elements(
            keyframe_element,
            keyframes[self._time_step :],
            qposes[self._time_step :],
            qvels[self._time_step :],
        )

    def initialize_episode(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ):
        """Reset time and set the walker."""
        self._should_terminate = False

        # Set the walker at the beginning of the clip.
        self._set_walker(physics)
        self._set_mocap_data(physics)
        return super().initialize_episode(physics, random_state)

    def _set_walker(self, physics: mjcf.Physics):
        qpos = self._motion_sequence["qpos"][self._time_step]
        qvel = self._motion_sequence["qvel"][self._time_step]

        if np.any(np.isnan(qpos)) and np.any(np.isnan(qvel)):
            raise ValueError(
                "Unable to set walker position. The `q{pos,vel}` in the motion "
                f"sequence contains `nan`s for `{self._time_step=}`."
            )

        np.testing.assert_equal(physics.data.qpos.shape, qpos.shape)
        np.testing.assert_equal(physics.data.qvel.shape, qvel.shape)

        timestep_features = {
            "position": qpos[0:3],
            "quaternion": qpos[3:7],
            "joints": qpos[7:None],
            "velocity": qvel[0:3],
            "angular_velocity": qvel[3:6],
            "joints_velocity": qvel[6:None],
        }

        utils.set_walker_from_features(physics, self._walker, timestep_features)
        mjlib.mj_kinematics(  # pylint: disable=no-member
            physics.model.ptr, physics.data.ptr
        )

    def _set_mocap_data(self, physics):
        kinematic_data = self._motion_sequence["keyframes"]
        mocap_features = tree.map_structure(
            lambda x: x[self._time_step], kinematic_data
        )

        physics.bind(self.mocap_sites).pos = mocap_features
        mjlib.mj_kinematics(  # pylint: disable=no-member
            physics.model.ptr, physics.data.ptr
        )

    def before_step(
        self,
        physics: mjcf.Physics,
        action: np.ndarray,
        random_state: np.random.RandomState,
    ):
        # pylint: disable=useless-parent-delegation
        return super().before_step(physics, action, random_state)

    def after_step(self, physics: mjcf.Physics, random_state: np.random.RandomState):
        kinematic_data = self._motion_sequence["keyframes"]
        # self._time_step = (self._time_step + 1) % kinematic_data.shape[0]
        self._time_step = self._time_step + 1
        # NOTE(hartikainen): Temporary sanity check. This should never happen unless
        # there's a bug somewhere else.
        assert self._time_step < kinematic_data.shape[0], (
            self._time_step,
            kinematic_data.shape[0],
            self._motion_sequence["mocap_id"].decode(),
        )
        self._set_mocap_data(physics)
        # Set the `_end_mocap` flag to `True` if the mocap sequence has reached its
        # end, based on the sequence length.
        self._end_mocap = self._time_step == kinematic_data.shape[0] - 1
        mocap_tracking_distances = self._compute_mocap_tracking_distances_global(physics)

        termination_indices = []
        for mocap_termination_body_name in self._walker.mocap_termination_body_names:
            mocap_index = next(
                i
                for i, s in enumerate(self.mocap_sites)
                if s.name == f"mocap[{mocap_termination_body_name}]"
            )
            termination_indices.append(mocap_index)

        termination_distances = mocap_tracking_distances[termination_indices]
        self._should_terminate = (
            (self._termination_threshold < termination_distances).any().item()
        )

        return super().after_step(physics, random_state)

    def get_time_step(self, physics: mjcf.Physics) -> np.ndarray:
        """Return the current discrete episode time step."""
        del physics
        return np.array([self._time_step])

    def get_keyframes_local(self, physics: mjcf.Physics) -> np.ndarray:
        """Observation of the reference bodies relative to walker in local frame."""
        keyframes = self._motion_sequence["keyframes"]
        num_timesteps = keyframes.shape[0]
        keyframe_future_index = np.minimum(
            self._time_step + self._mocap_reference_steps, num_timesteps - 1
        )

        future_keyframes = keyframes[keyframe_future_index]
        walker_sites_xpos = physics.bind(
            self._walker.mocap_tracking_sites
        ).xpos  # pytype: disable=attribute-error

        return self._walker.transform_vec_to_egocentric_frame(
            physics, future_keyframes - walker_sites_xpos
        )

    def get_keyframes_global(self, physics: mjcf.Physics) -> np.ndarray:
        """Observation of the reference bodies in global frame."""
        del physics
        keyframes = self._motion_sequence["keyframes"]
        num_timesteps = keyframes.shape[0]
        keyframe_future_index = np.minimum(
            self._time_step + self._mocap_reference_steps, num_timesteps - 1
        )
        future_keyframes = keyframes[keyframe_future_index]

        return future_keyframes

    def get_discount(self, physics) -> float:
        del physics
        if self._should_terminate:
            discount = 0.0
        else:
            discount = 1.0
        return discount

    def should_terminate_episode(self, physics) -> bool:
        del physics
        return self._should_terminate or self._end_mocap

    def _compute_mocap_tracking_distances_local(self, physics) -> np.ndarray:
        mocap_tracking_sites = physics.bind(self._walker.mocap_tracking_sites)
        mocap_tracking_sites_xpos = mocap_tracking_sites.xpos

        root_body_name = self._walker.root_body.name

        mocap_tracking_root_xpos = physics.bind(
            next(
                x for x in self._walker.mocap_tracking_sites if root_body_name in x.name
            )
        ).xpos

        mocap_sites = physics.bind(self.mocap_sites)
        mocap_sites_xpos = mocap_sites.xpos
        mocap_sites_root_xpos = physics.bind(
            next(x for x in self.mocap_sites if root_body_name in x.name)
        ).xpos

        distances = np.linalg.norm(
            (
                (mocap_sites_xpos - mocap_sites_root_xpos)
                - (mocap_tracking_sites_xpos - mocap_tracking_root_xpos)
            ),
            ord=2,
            axis=-1,
        )
        return distances

    def _compute_mocap_tracking_distances_global(self, physics) -> np.ndarray:
        mocap_tracking_sites = physics.bind(self._walker.mocap_tracking_sites)
        mocap_tracking_sites_xpos = mocap_tracking_sites.xpos

        mocap_sites = physics.bind(self.mocap_sites)
        mocap_sites_xpos = mocap_sites.xpos

        distances = np.linalg.norm(
            mocap_sites_xpos - mocap_tracking_sites_xpos,
            ord=2,
            axis=-1,
        )
        return distances

    def get_reward(self, physics) -> dict[str, float]:
        mocap_tracking_distances_local = self._compute_mocap_tracking_distances_local(
            physics
        )
        mocap_tracking_distances_global = self._compute_mocap_tracking_distances_global(
            physics
        )
        if self._termination_threshold < float("inf"):
            max_distance = self._termination_threshold
        else:
            max_distance = 1.0  # 1 meter

        tracking_reward = (
            1.0
            - np.minimum(mocap_tracking_distances_local.mean(), max_distance)
            / max_distance
        )

        max_episode_length = (
            self._motion_sequence["keyframes"].shape[0] - self._init_time_step
        )
        normalized_tracking_reward = tracking_reward / (max_episode_length - 1)
        normalized_step_reward = 1.0 / (max_episode_length - 1)

        def compute_tracking_metrics(distances):
            return {
                "mean": np.mean(distances),
                "q0.00": np.quantile(distances, 0.00),
                "q0.05": np.quantile(distances, 0.05),
                "q0.10": np.quantile(distances, 0.10),
                "q0.20": np.quantile(distances, 0.20),
                "q0.50": np.quantile(distances, 0.50),
                "q0.80": np.quantile(distances, 0.80),
                "q0.90": np.quantile(distances, 0.90),
                "q0.95": np.quantile(distances, 0.95),
                "q1.00": np.quantile(distances, 1.00),
            }

        mocap_tracking_error_metrics = {
            **{
                f"mocap_tracking_distances_local-{k}": v
                for k, v in compute_tracking_metrics(
                    mocap_tracking_distances_local
                ).items()
            },
            **{
                f"mocap_tracking_distances_global-{k}": v
                for k, v in compute_tracking_metrics(
                    mocap_tracking_distances_global
                ).items()
            },
        }
        reward = {
            "tracking": tracking_reward,
            "step": 1.0,
            "normalized/tracking": normalized_tracking_reward,
            "normalized/step": normalized_step_reward,
            **mocap_tracking_error_metrics,
        }
        return reward

    def get_reward_spec(self) -> dict[str, dm_env.specs.Array]:
        return {
            "tracking": dm_env.specs.Array(
                shape=(),
                dtype=np.float64,
                name="reward/tracking",
            ),
            "step": dm_env.specs.Array(
                shape=(),
                dtype=np.float64,
                name="reward/step",
            ),
            "normalized/tracking": dm_env.specs.Array(
                shape=(),
                dtype=np.float64,
                name="reward/normalized/tracking",
            ),
            "normalized/step": dm_env.specs.Array(
                shape=(),
                dtype=np.float64,
                name="reward/normalized/step",
            ),
        }
