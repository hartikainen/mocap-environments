"""A CMU humanoid walker."""

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

_XML_PATH = pathlib.Path(__file__).parent / "cmu_humanoid.xml"

_MOCAP_JOINTS = (
    "lfemurrz", "lfemurry", "lfemurrx", "ltibiarx", "lfootrz", "lfootrx",
    "ltoesrx", "rfemurrz", "rfemurry", "rfemurrx", "rtibiarx", "rfootrz",
    "rfootrx", "rtoesrx", "lowerbackrz", "lowerbackry", "lowerbackrx",
    "upperbackrz", "upperbackry", "upperbackrx", "thoraxrz", "thoraxry",
    "thoraxrx", "lowerneckrz", "lowerneckry", "lowerneckrx", "upperneckrz",
    "upperneckry", "upperneckrx", "headrz", "headry", "headrx", "lclaviclerz",
    "lclaviclery", "lhumerusrz", "lhumerusry", "lhumerusrx", "lradiusrx",
    "lwristry", "lhandrz", "lhandrx", "lfingersrx", "lthumbrz", "lthumbrx",
    "rclaviclerz", "rclaviclery", "rhumerusrz", "rhumerusry", "rhumerusrx",
    "rradiusrx", "rwristry", "rhandrz", "rhandrx", "rfingersrx", "rthumbrz",
    "rthumbrx",
)

PositionActuatorParams = collections.namedtuple(
    "PositionActuatorParams", ["name", "forcerange", "kp", "damping"]
)

# fmt: off
_POSITION_ACTUATORS = [
    PositionActuatorParams("headrx",      [-40,   40 ], 40 , 2 ),
    PositionActuatorParams("headry",      [-40,   40 ], 40 , 2 ),
    PositionActuatorParams("headrz",      [-40,   40 ], 40 , 2 ),
    PositionActuatorParams("lclaviclery", [-80,   80 ], 80 , 20),
    PositionActuatorParams("lclaviclerz", [-80,   80 ], 80 , 20),
    PositionActuatorParams("lfemurrx",    [-300,  300], 300, 15),
    PositionActuatorParams("lfemurry",    [-200,  200], 200, 10),
    PositionActuatorParams("lfemurrz",    [-200,  200], 200, 10),
    PositionActuatorParams("lfingersrx",  [-20,   20 ], 20 , 1 ),
    PositionActuatorParams("lfootrx",     [-120,  120], 120, 6 ),
    PositionActuatorParams("lfootrz",     [-50,   50 ], 50 , 3 ),
    PositionActuatorParams("lhandrx",     [-20,   20 ], 20 , 1 ),
    PositionActuatorParams("lhandrz",     [-20,   20 ], 20 , 1 ),
    PositionActuatorParams("lhumerusrx",  [-120,  120], 120, 6 ),
    PositionActuatorParams("lhumerusry",  [-120,  120], 120, 6 ),
    PositionActuatorParams("lhumerusrz",  [-120,  120], 120, 6 ),
    PositionActuatorParams("lowerbackrx", [-300,  300], 300, 15),
    PositionActuatorParams("lowerbackry", [-180,  180], 180, 20),
    PositionActuatorParams("lowerbackrz", [-200,  200], 200, 20),
    PositionActuatorParams("lowerneckrx", [-120,  120 ],120, 20),
    PositionActuatorParams("lowerneckry", [-120,  120 ],120, 20),
    PositionActuatorParams("lowerneckrz", [-120,  120 ],120, 20),
    PositionActuatorParams("lradiusrx",   [-90,   90 ], 90 , 5 ),
    PositionActuatorParams("lthumbrx",    [-20,   20 ], 20 , 1 ),
    PositionActuatorParams("lthumbrz",    [-20,   20 ], 20 , 1 ),
    PositionActuatorParams("ltibiarx",    [-160,  160], 160, 8 ),
    PositionActuatorParams("ltoesrx",     [-20,   20 ], 20 , 1 ),
    PositionActuatorParams("lwristry",    [-20,   20 ], 20 , 1 ),
    PositionActuatorParams("rclaviclery", [-80,   80 ], 80 , 20),
    PositionActuatorParams("rclaviclerz", [-80,   80 ], 80 , 20),
    PositionActuatorParams("rfemurrx",    [-300,  300], 300, 15),
    PositionActuatorParams("rfemurry",    [-200,  200], 200, 10),
    PositionActuatorParams("rfemurrz",    [-200,  200], 200, 10),
    PositionActuatorParams("rfingersrx",  [-20,   20 ], 20 , 1 ),
    PositionActuatorParams("rfootrx",     [-120,  120], 120, 6 ),
    PositionActuatorParams("rfootrz",     [-50,   50 ], 50 , 3 ),
    PositionActuatorParams("rhandrx",     [-20,   20 ], 20 , 1 ),
    PositionActuatorParams("rhandrz",     [-20,   20 ], 20 , 1 ),
    PositionActuatorParams("rhumerusrx",  [-120,  120], 120, 6 ),
    PositionActuatorParams("rhumerusry",  [-120,  120], 120, 6 ),
    PositionActuatorParams("rhumerusrz",  [-120,  120], 120, 6 ),
    PositionActuatorParams("rradiusrx",   [-90,   90 ], 90 , 5 ),
    PositionActuatorParams("rthumbrx",    [-20,   20 ], 20 , 1 ),
    PositionActuatorParams("rthumbrz",    [-20,   20 ], 20 , 1 ),
    PositionActuatorParams("rtibiarx",    [-160,  160], 160, 8 ),
    PositionActuatorParams("rtoesrx",     [-20,   20 ], 20 , 1 ),
    PositionActuatorParams("rwristry",    [-20,   20 ], 20 , 1 ),
    PositionActuatorParams("thoraxrx",    [-300,  300], 300, 15),
    PositionActuatorParams("thoraxry",    [-80,   80],  80 , 8 ),
    PositionActuatorParams("thoraxrz",    [-200,  200], 200, 12),
    PositionActuatorParams("upperbackrx", [-300,  300], 300, 15),
    PositionActuatorParams("upperbackry", [-80,   80],  80 , 8 ),
    PositionActuatorParams("upperbackrz", [-200,  200], 200, 12),
    PositionActuatorParams("upperneckrx", [-60,   60 ], 60 , 10),
    PositionActuatorParams("upperneckry", [-60,   60 ], 60 , 10),
    PositionActuatorParams("upperneckrz", [-60,   60 ], 60 , 10),
]
# fmt: on


_UPRIGHT_POS = (0.0, 0.0, 0.94)
_UPRIGHT_POS_V2020 = (0.0, 0.0, 1.143)
_UPRIGHT_QUAT = (0.859, 1.0, 1.0, 0.859)

# Height of head above which the humanoid is considered standing.
_STAND_HEIGHT = 1.5

_TORQUE_THRESHOLD = 60


class CMUHumanoid(legacy_base.Walker):
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
        return CMUHumanoidObservables(self)

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
        return self._mjcf_root.find("body", "root")

    @composer.cached_property
    def head(self):
        return self._mjcf_root.find("body", "head")

    @composer.cached_property
    def ground_contact_geoms(self):
        return tuple(
            self._mjcf_root.find("body", "rfoot").find_all("geom")
            + self._mjcf_root.find("body", "lfoot").find_all("geom")
        )

    @composer.cached_property
    def standing_height(self):
        return _STAND_HEIGHT

    @composer.cached_property
    def end_effectors(self):
        return (
            self._mjcf_root.find("body", "rhand"),
            self._mjcf_root.find("body", "lhand"),
            self._mjcf_root.find("body", "rfoot"),
            self._mjcf_root.find("body", "lfoot"),
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


class CMUHumanoidPositionControlled(CMUHumanoid):
    """A position-controlled CMU humanoid with control range scaled to [-1, 1]."""

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


class CMUHumanoidObservables(legacy_base.WalkerObservables):
    """Observables for `CMUHumanoid`."""

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
