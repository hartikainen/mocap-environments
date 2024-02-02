"""A smpl humanoid walker."""

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

_XML_PATH = pathlib.Path(__file__).parent / "smpl_humanoid.xml"

PositionActuatorParams = collections.namedtuple(
    "PositionActuatorParams", ["name", "forcerange", "kp", "damping"]
)

# fmt: off
# # NOTE(hartikainen): These values are set somewhat arbitrarily based on those from
# `CMUHumanoid`'s.
_POSITION_ACTUATORS = [
    PositionActuatorParams("L_Hip_x", [-250, +250], 250, None),
    PositionActuatorParams("L_Hip_y", [-250, +250], 250, None),
    PositionActuatorParams("L_Hip_z", [-250, +250], 250, None),
    PositionActuatorParams("L_Knee_x", [-250, +250], 250, None),
    PositionActuatorParams("L_Knee_y", [-250, +250], 250, None),
    PositionActuatorParams("L_Knee_z", [-250, +250], 250, None),
    PositionActuatorParams("L_Ankle_x", [-250, +250], 250, None),
    PositionActuatorParams("L_Ankle_y", [-250, +250], 250, None),
    PositionActuatorParams("L_Ankle_z", [-250, +250], 250, None),
    # PositionActuatorParams("L_Toe_x", [-250, +250], 250, None),
    PositionActuatorParams("L_Toe_y", [-250, +250], 250, None),
    # PositionActuatorParams("L_Toe_z", [-250, +250], 250, None),
    PositionActuatorParams("R_Hip_x", [-250, +250], 250, None),
    PositionActuatorParams("R_Hip_y", [-250, +250], 250, None),
    PositionActuatorParams("R_Hip_z", [-250, +250], 250, None),
    PositionActuatorParams("R_Knee_x", [-250, +250], 250, None),
    PositionActuatorParams("R_Knee_y", [-250, +250], 250, None),
    PositionActuatorParams("R_Knee_z", [-250, +250], 250, None),
    PositionActuatorParams("R_Ankle_x", [-250, +250], 250, None),
    PositionActuatorParams("R_Ankle_y", [-250, +250], 250, None),
    PositionActuatorParams("R_Ankle_z", [-250, +250], 250, None),
    # PositionActuatorParams("R_Toe_x", [-250, +250], 250, None),
    PositionActuatorParams("R_Toe_y", [-250, +250], 250, None),
    # PositionActuatorParams("R_Toe_z", [-250, +250], 250, None),
    PositionActuatorParams("Torso_x", [-250, +250], 250, None),
    PositionActuatorParams("Torso_y", [-250, +250], 250, None),
    PositionActuatorParams("Torso_z", [-250, +250], 250, None),
    PositionActuatorParams("Spine_x", [-250, +250], 250, None),
    PositionActuatorParams("Spine_y", [-250, +250], 250, None),
    PositionActuatorParams("Spine_z", [-250, +250], 250, None),
    PositionActuatorParams("Chest_x", [-250, +250], 250, None),
    PositionActuatorParams("Chest_y", [-250, +250], 250, None),
    PositionActuatorParams("Chest_z", [-250, +250], 250, None),
    # PositionActuatorParams("Neck_x", [-250, +250], 250, None),
    PositionActuatorParams("Neck_y", [-250, +250], 250, None),
    PositionActuatorParams("Neck_z", [-250, +250], 250, None),
    # PositionActuatorParams("Head_x", [-250, +250], 250, None),
    PositionActuatorParams("Head_y", [-250, +250], 250, None),
    PositionActuatorParams("Head_z", [-250, +250], 250, None),
    PositionActuatorParams("L_Thorax_x", [-250, +250], 250, None),
    # PositionActuatorParams("L_Thorax_y", [-250, +250], 250, None),
    PositionActuatorParams("L_Thorax_z", [-250, +250], 250, None),
    PositionActuatorParams("L_Shoulder_x", [-250, +250], 250, None),
    PositionActuatorParams("L_Shoulder_y", [-250, +250], 250, None),
    PositionActuatorParams("L_Shoulder_z", [-250, +250], 250, None),
    # PositionActuatorParams("L_Elbow_x", [-250, +250], 250, None),
    # PositionActuatorParams("L_Elbow_y", [-250, +250], 250, None),
    PositionActuatorParams("L_Elbow_z", [-250, +250], 250, None),
    PositionActuatorParams("L_Wrist_x", [-250, +250], 250, None),
    # PositionActuatorParams("L_Wrist_y", [-250, +250], 250, None),
    PositionActuatorParams("L_Wrist_z", [-250, +250], 250, None),
    # PositionActuatorParams("L_Hand_x", [-250, +250], 250, None),
    # PositionActuatorParams("L_Hand_y", [-250, +250], 250, None),
    # PositionActuatorParams("L_Hand_z", [-250, +250], 250, None),
    PositionActuatorParams("R_Thorax_x", [-250, +250], 250, None),
    # PositionActuatorParams("R_Thorax_y", [-250, +250], 250, None),
    PositionActuatorParams("R_Thorax_z", [-250, +250], 250, None),
    PositionActuatorParams("R_Shoulder_x", [-250, +250], 250, None),
    PositionActuatorParams("R_Shoulder_y", [-250, +250], 250, None),
    PositionActuatorParams("R_Shoulder_z", [-250, +250], 250, None),
    # PositionActuatorParams("R_Elbow_x", [-250, +250], 250, None),
    # PositionActuatorParams("R_Elbow_y", [-250, +250], 250, None),
    PositionActuatorParams("R_Elbow_z", [-250, +250], 250, None),
    PositionActuatorParams("R_Wrist_x", [-250, +250], 250, None),
    # PositionActuatorParams("R_Wrist_y", [-250, +250], 250, None),
    PositionActuatorParams("R_Wrist_z", [-250, +250], 250, None),
    # PositionActuatorParams("R_Hand_x", [-250, +250], 250, None),
    # PositionActuatorParams("R_Hand_y", [-250, +250], 250, None),
    # PositionActuatorParams("R_Hand_z", [-250, +250], 250, None),
]
# fmt: on


_MOCAP_JOINTS = tuple(
    actuator.name for actuator in _POSITION_ACTUATORS
    # f"{name}_{direction}"
    # for name in (
    #     "L_Hip",
    #     "L_Knee",
    #     "L_Ankle",
    #     "L_Toe",
    #     "R_Hip",
    #     "R_Knee",
    #     "R_Ankle",
    #     "R_Toe",
    #     "Torso",
    #     "Spine",
    #     "Chest",
    #     "Neck",
    #     "Head",
    #     "L_Thorax",
    #     "L_Shoulder",
    #     "L_Elbow",
    #     "L_Wrist",
    #     "L_Hand",
    #     "R_Thorax",
    #     "R_Shoulder",
    #     "R_Elbow",
    #     "R_Wrist",
    #     "R_Hand",
    # )
    # for direction in ("x", "y", "z")
)


_UPRIGHT_POS = (0.0, 0.0, 0.93288)
_UPRIGHT_QUAT = (0.859, 1.0, 1.0, 0.859)  # TODO(hartikainen)

# Height of head above which the humanoid is considered standing.
_STAND_HEIGHT = 1.4  # TODO(hartikainen)
_TORSO_HEIGHT = 0.9  # TODO(hartikainen)


class SMPLHumanoid(legacy_base.Walker):
    """A smpl humanoid walker."""

    def __init__(self, *args, **kwargs):
        self._prev_action = None
        super().__init__(*args, **kwargs)

    def _build(self, initializer=None, name="walker"):
        self._mjcf_root = mjcf.from_path(str(_XML_PATH))
        if name:
            self._mjcf_root.model = name

        for body_element in self._mjcf_root.find_all("body"):
            tracking_site_name = f"tracking[{body_element.name}]"
            body_element.insert(
                "site",
                0,
                name=tracking_site_name,
                pos=[0, 0, 0],
                **{"class": "tracking_site"},
            )

        self._mjcf_root.asset.insert(
            "material", 0, name="self", rgba=[0.7, 0.5, 0.3, 1]
        )

        tracking_site_default = self._mjcf_root.default.insert(
            "default", -1, **{"class": "tracking_site"}
        )
        mocap_site_default = self._mjcf_root.default.insert(
            "default", -1, **{"class": "mocap_site"}
        )

        tracking_site_default.site.type = "sphere"
        tracking_site_default.site.size = "0.027"
        tracking_site_default.site.rgba = [1, 0, 0, 1]
        tracking_site_default.site.group = 3

        mocap_site_default.site.type = "sphere"
        mocap_site_default.site.size = "0.027"
        mocap_site_default.site.rgba = [0, 0, 1, 1]
        mocap_site_default.site.group = 2

        super()._build(initializer=initializer)

        # Capture previous action applied to the walker through `apply_action`.
        self._prev_action = np.zeros(
            shape=self.action_spec.shape, dtype=self.action_spec.dtype
        )

    def initialize_episode(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ):
        self._prev_action = np.zeros(
            shape=self.action_spec.shape, dtype=self.action_spec.dtype
        )
        return super().initialize_episode(physics, random_state)

    def before_step(self, physics: mjcf.Physics, random_state: np.random.RandomState):
        """Store the previous action."""
        assert self.prev_action is not None
        self.prev_action[:] = physics.data.ctrl
        return super().before_step(physics, random_state)

    @property
    def prev_action(self):
        return self._prev_action

    def _build_observables(self):
        return SmplHumanoidObservables(self)

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
        return self._mjcf_root.find("body", "Pelvis")

    @composer.cached_property
    def head(self):
        return self._mjcf_root.find("body", "Head")

    @composer.cached_property
    def ground_contact_geoms(self):
        return tuple(
            self._mjcf_root.find("body", "R_Ankle").find_all("geom")
            + self._mjcf_root.find("body", "L_Ankle").find_all("geom")
        )

    @composer.cached_property
    def standing_height(self):
        return _STAND_HEIGHT

    @composer.cached_property
    def end_effectors(self):
        return (
            self._mjcf_root.find("body", "R_Hand"),
            self._mjcf_root.find("body", "L_Hand"),
            self._mjcf_root.find("body", "R_Toe"),
            self._mjcf_root.find("body", "L_Toe"),
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
        return tuple(b for b in self._mjcf_root.find_all("body"))

    @composer.cached_property
    def mocap_tracking_sites(self):
        """Collection of bodies for mocap tracking."""
        # remove root body
        all_sites = self._mjcf_root.find_all("site")

        return tuple(
            site for site in all_sites if re.match(r"tracking\[\w+\]", site.name)
        )

    @composer.cached_property
    def mocap_termination_body_names(self):
        return ("Pelvis", "L_Shoulder", "R_Shoulder")


class SMPLHumanoidPositionControlled(SMPLHumanoid):
    """A position-controlled simple humanoid with control range scaled to [-1, 1]."""

    def _build(self, initializer=None, name="walker"):
        super()._build(initializer=initializer, name=name)

        position_actuators = _POSITION_ACTUATORS
        self._mjcf_root.default.general.forcelimited = "true"

        self._mjcf_root.actuator.motor.clear()

        for actuator_params in position_actuators:
            associated_joint = self._mjcf_root.find("joint", actuator_params.name)

            if hasattr(actuator_params, "damping") and actuator_params.damping is not None:
                associated_joint.damping = actuator_params.damping
            # else:
            #     if not hasattr(associated_joint, "damping"):
            #         raise ValueError(f"{associated_joint=} has not damping.")
            #     assert (
            #         associated_joint.damping is not None
            #     ), associated_joint.name

            # associated_joint.damping /= 40
            # associated_joint.stiffness /= 40
            # breakpoint(); pass

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

    def initialize_episode(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ):
        act_row_names = physics.named.data.act.axes.row.names

        act_qpos = physics.named.data.qpos[act_row_names]
        act_ranges = physics.named.model.jnt_range[act_row_names]

        # scale to `[0, 1]`
        act = (act_qpos - act_ranges[:, 0]) / np.ptp(act_ranges, axis=-1)

        # scale to `ctrlrange`
        ctrlrange = physics.named.model.actuator_ctrlrange[act_row_names]
        act = ctrlrange[:, 0] + np.ptp(ctrlrange, axis=-1) * act

        # error_tolerance = 0
        # np.testing.assert_array_less(ctrlrange[:, 0] - error_tolerance, act)
        # np.testing.assert_array_less(act, ctrlrange[:, 1] + error_tolerance)
        # act = np.clip(act, ctrlrange[:, 0], ctrlrange[:, 1])

        physics.named.data.act[act_row_names] = act

        return super().initialize_episode(physics, random_state)


class SmplHumanoidObservables(legacy_base.WalkerObservables):
    """Observables for `SmplHumanoid`."""

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
