"""A simple humanoid walker."""

import collections
import pathlib
import re

from dm_control import composer
from dm_control import mjcf
from dm_control.composer.observation import observable
from dm_control.locomotion.walkers import base
from dm_control.locomotion.walkers import legacy_base
from dm_control.locomotion.walkers import scaled_actuators
from dm_control.utils import transformations
import numpy as np

_XML_PATH = pathlib.Path(__file__).parent / "simple_humanoid.xml"

_MOCAP_JOINTS = (
    "abdomen_z",
    "abdomen_y",
    "abdomen_x",
    "hip_x_right",
    "hip_z_right",
    "hip_y_right",
    "knee_right",
    "ankle_y_right",
    "ankle_x_right",
    "hip_x_left",
    "hip_z_left",
    "hip_y_left",
    "knee_left",
    "ankle_y_left",
    "ankle_x_left",
    "shoulder1_right",
    "shoulder2_right",
    "elbow_right",
    "shoulder1_left",
    "shoulder2_left",
    "elbow_left",
)


PositionActuatorParams = collections.namedtuple(
    "PositionActuatorParams", ["name", "forcerange", "kp", "damping"]
)

# fmt: off
# # NOTE(hartikainen): These values are set somewhat arbitrarily based on those from
# `CMUHumanoid`'s.
_POSITION_ACTUATORS = [
    PositionActuatorParams("abdomen_y", [-180, 180], 180, 20),
    PositionActuatorParams("abdomen_z", [-200, 200], 200, 20),
    PositionActuatorParams("abdomen_x", [-300, 300], 300, 15),
    PositionActuatorParams("hip_x_right", [-300, 300], 300, 15),
    PositionActuatorParams("hip_z_right", [-200, 200], 200, 10),
    PositionActuatorParams("hip_y_right", [-200, 200], 200, 10),
    PositionActuatorParams("knee_right", [-160, 160], 160, 8),

    PositionActuatorParams("ankle_x_right", [-50, 50], 50, 3),
    PositionActuatorParams("ankle_y_right", [-120, 120], 120, 6),
    PositionActuatorParams("hip_x_left", [-300, 300], 300, 15),
    PositionActuatorParams("hip_z_left", [-200, 200], 200, 10),
    PositionActuatorParams("hip_y_left", [-200, 200], 200, 10),
    PositionActuatorParams("knee_left", [-160, 160], 160, 8),

    PositionActuatorParams("ankle_x_left", [-50, 50], 50, 3),
    PositionActuatorParams("ankle_y_left", [-120, 120], 120, 6),
    PositionActuatorParams("shoulder1_right", [-120, 120], 120, 6),
    PositionActuatorParams("shoulder2_right", [-120, 120], 120, 6),
    PositionActuatorParams("elbow_right", [-90, 90], 90, 5),
    PositionActuatorParams("shoulder1_left", [-120, 120], 120, 6),
    PositionActuatorParams("shoulder2_left", [-120, 120], 120, 6),
    PositionActuatorParams("elbow_left", [-90, 90], 90, 5),
]
# fmt: on


_UPRIGHT_POS = (0.0, 0.0, 1.286)
_UPRIGHT_QUAT = (0.859, 1.0, 1.0, 0.859)

# Height of head above which the humanoid is considered standing.
_STAND_HEIGHT = 1.4
_TORSO_HEIGHT = 0.9


class SimpleHumanoid(legacy_base.Walker):
    """A simple humanoid walker."""

    def __init__(self, *args, **kwargs):
        self._prev_action = None
        super().__init__(*args, **kwargs)

    def _build(self, initializer=None, name="walker"):
        self._mjcf_root = mjcf.from_path(str(_XML_PATH))
        if name:
            self._mjcf_root.model = name

        super()._build(initializer=initializer)

        # Capture previous action applied to the walker through `apply_action`.
        self._prev_action = np.zeros(
            shape=self.action_spec.shape, dtype=self.action_spec.dtype
        )

    def initialize_episode(self, physics, random_state):
        self._prev_action = np.zeros(
            shape=self.action_spec.shape, dtype=self.action_spec.dtype
        )
        return super().initialize_episode(physics, random_state)

    def apply_action(self, physics, action, random_state):
        """Apply action to walker's actuators."""
        self._prev_action[:] = action
        return super().apply_action(physics, action, random_state)

    @property
    def prev_action(self):
        return self._prev_action

    def _build_observables(self):
        return SimpleHumanoidObservables(self)

    @property
    def mocap_joint_order(self):
        return _MOCAP_JOINTS

    @composer.cached_property
    def mocap_joints(self):
        mocap_joints = tuple(
            self._mjcf_root.find("joint", name) for name in self.mocap_joint_order
        )
        return mocap_joints

    @property
    def upright_pose(self):
        return base.WalkerPose(xpos=_UPRIGHT_POS, xquat=_UPRIGHT_QUAT)

    @property
    def mjcf_model(self):
        return self._mjcf_root

    @property
    def actuators(self):
        return tuple(self._mjcf_root.find_all("actuator"))

    @composer.cached_property
    def root_body(self):
        return self._mjcf_root.find("body", "pelvis")

    @composer.cached_property
    def head(self):
        return self._mjcf_root.find("body", "head")

    @composer.cached_property
    def ground_contact_geoms(self):
        return tuple(
            self._mjcf_root.find("body", "foot_right").find_all("geom")
            + self._mjcf_root.find("body", "foot_left").find_all("geom")
        )

    @composer.cached_property
    def standing_height(self):
        return _STAND_HEIGHT

    @composer.cached_property
    def end_effectors(self):
        return (
            self._mjcf_root.find("body", "hand_right"),
            self._mjcf_root.find("body", "hand_left"),
            self._mjcf_root.find("body", "foot_right"),
            self._mjcf_root.find("body", "foot_left"),
        )

    @composer.cached_property
    def observable_joints(self):
        return tuple(
            actuator.joint for actuator in self.actuators if actuator.joint is not None
        )

    @composer.cached_property
    def egocentric_camera(self):
        return self._mjcf_root.find("camera", "egocentric")

    @composer.cached_property
    def mocap_tracking_bodies(self):
        """Collection of bodies for mocap tracking."""
        # remove root body
        root_body = self._mjcf_root.find("body", "root")
        return tuple(b for b in self._mjcf_root.find_all("body") if b != root_body)

    @composer.cached_property
    def mocap_tracking_sites(self):
        """Collection of bodies for mocap tracking."""
        # remove root body
        all_sites = self._mjcf_root.find_all("site")

        return tuple(
            site for site in all_sites if re.match(r"tracking\[\w+\]", site.name)
        )


class SimpleHumanoidPositionControlled(SimpleHumanoid):
    """A position-controlled simple humanoid with control range scaled to [-1, 1]."""

    def _build(self, initializer=None, name="walker"):
        super()._build(initializer=initializer, name=name)

        position_actuators = _POSITION_ACTUATORS
        self._mjcf_root.default.general.forcelimited = "true"

        self._mjcf_root.actuator.motor.clear()
        for actuator_params in position_actuators:
            associated_joint = self._mjcf_root.find("joint", actuator_params.name)
            if hasattr(actuator_params, "damping"):
                associated_joint.damping = actuator_params.damping

            if associated_joint.range is not None:
                associated_joint_range = associated_joint.range
            elif associated_joint.dclass.joint.range is not None:
                associated_joint_range = associated_joint.dclass.joint.range
            elif associated_joint.dclass.parent.joint.range is not None:
                associated_joint_range = associated_joint.dclass.parent.joint.range
            else:
                raise ValueError(
                    f"No matching joint range found for joint {associated_joint.name}."
                )

            _ = scaled_actuators.add_position_actuator(
                name=actuator_params.name,
                target=associated_joint,
                kp=actuator_params.kp,
                qposrange=associated_joint_range,
                ctrlrange=(-1, 1),
                forcerange=actuator_params.forcerange,
                dyntype="filter",
                dynprm=[0.030],
            )

    def initialize_episode(self, physics, random_state):
        joint_names = [x.name for x in self.mocap_joints]
        joint_to_actuator_index = [
            joint_names.index(actuator.name) for actuator in self.actuators
        ]
        act_qpos = physics.data.qpos[7:][joint_to_actuator_index]
        act_ranges = (
            physics.bind(self.mocap_joints).range  # pytype: disable=attribute-error
        )[joint_to_actuator_index]

        # scale to `[0, 1]`
        act = (act_qpos - act_ranges[:, 0]) / np.ptp(act_ranges, axis=-1)

        # scale to `ctrlrange`
        ctrlrange = physics.named.model.actuator_ctrlrange
        act = ctrlrange[:, 0] + np.ptp(ctrlrange, axis=-1) * act

        error_tolerance = 0
        np.testing.assert_array_less(ctrlrange[:, 0] - error_tolerance, act)
        np.testing.assert_array_less(act, ctrlrange[:, 1] + error_tolerance)
        act = np.clip(act, ctrlrange[:, 0], ctrlrange[:, 1])

        physics.data.act[:] = act

        return super().initialize_episode(physics, random_state)


class SimpleHumanoidObservables(legacy_base.WalkerObservables):
    """Observables for `SimpleHumanoid`."""

    @composer.observable
    def bodies_quats(self):
        """Orientations of the bodies as quaternions, in the egocentric frame."""

        def bodies_orientations_in_egocentric_frame(physics):
            """Compute relative orientation of the bodies."""
            # Get the bodies
            bodies = self._entity.mocap_tracking_bodies
            # Get the quaternions of all the bodies & root in the global frame
            bodies_xquat = physics.bind(bodies).xquat
            root_xquat = physics.bind(self._entity.root_body).xquat
            # Compute the relative quaternion of the bodies in the root frame
            bodies_quat_diff = transformations.quat_diff(
                np.tile(root_xquat, len(bodies)).reshape(-1, 4), bodies_xquat
            )  # q1^-1 * q2
            return np.reshape(bodies_quat_diff, -1)

        return observable.Generic(bodies_orientations_in_egocentric_frame)

    @composer.observable
    def bodies_pos(self):
        """Position of bodies relative to root, in the egocentric frame."""

        def bodies_pos_in_egocentric_frame(physics):
            """Compute relative orientation of the bodies."""
            # Get the bodies
            bodies = self._entity.mocap_tracking_bodies
            bodies_xpos = physics.bind(bodies).xpos
            root_xpos = physics.bind(self._entity.root_body).xpos
            # Compute the relative position of the bodies in the root frame
            root_xmat = np.reshape(physics.bind(self._entity.root_body).xmat, (3, 3))
            return np.reshape(np.dot(bodies_xpos - root_xpos, root_xmat), -1)

        return observable.Generic(bodies_pos_in_egocentric_frame)

    @composer.observable
    def mocap_pos(self):
        """Position of bodies relative to root, in the egocentric frame."""

        def bodies_pos_in_egocentric_frame(physics):
            """Compute relative orientation of the bodies."""
            # Get the bodies
            bodies = self._entity.mocap_tracking_bodies
            bodies_xpos = physics.bind(bodies).xpos
            root_xpos = physics.bind(self._entity.root_body).xpos
            # Compute the relative position of the bodies in the root frame
            root_xmat = np.reshape(physics.bind(self._entity.root_body).xmat, (3, 3))
            return np.reshape(np.dot(bodies_xpos - root_xpos, root_xmat), -1)

        return observable.Generic(bodies_pos_in_egocentric_frame)

    @composer.observable
    def head_height(self):
        return observable.MJCFFeature("xpos", self._entity.head)[2]

    @composer.observable
    def actuator_activation(self):
        return observable.MJCFFeature(
            "act", self._entity.mjcf_model.find_all("actuator")
        )

    @composer.observable
    def appendages_pos(self):
        """Equivalent to `end_effectors_pos` with the head's position appended."""

        def relative_pos_in_egocentric_frame(physics):
            end_effectors_with_head = (*self._entity.end_effectors, self._entity.head)
            end_effector = physics.bind(end_effectors_with_head).xpos
            torso = physics.bind(self._entity.root_body).xpos
            xmat = np.reshape(physics.bind(self._entity.root_body).xmat, (3, 3))
            return np.reshape(np.dot(end_effector - torso, xmat), -1)

        return observable.Generic(relative_pos_in_egocentric_frame)

    @property
    def proprioception(self):
        return [
            *super().proprioception,
            self.appendages_pos,
            self.bodies_quats,
            self.bodies_pos,
            self.actuator_activation,
        ]
