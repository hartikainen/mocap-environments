"""Tests for the simple humanoid."""


from absl.testing import absltest
from absl.testing import parameterized
from dm_control import mjcf
from dm_control.composer.observation.observable import base as observable_base
from dm_control.locomotion.walkers import simple_humanoid
import numpy as np


class SimpleHumanoidTest(parameterized.TestCase):
    @parameterized.parameters(
        [
            simple_humanoid.SimpleHumanoid,
            simple_humanoid.SimpleHumanoidPositionControlled,
        ]
    )
    def test_can_compile_and_step_simulation(self, walker_type):
        walker = walker_type()
        physics = mjcf.Physics.from_mjcf_model(walker.mjcf_model)
        for _ in range(100):
            physics.step()

    @parameterized.parameters(
        [
            simple_humanoid.SimpleHumanoid,
            simple_humanoid.SimpleHumanoidPositionControlled,
        ]
    )
    def test_actuators_sorted_alphabetically(self, walker_type):
        walker = walker_type()
        actuator_names = [
            actuator.name for actuator in walker.mjcf_model.find_all("actuator")
        ]
        np.testing.assert_array_equal(actuator_names, sorted(actuator_names))

    def test_actuator_to_mocap_joint_mapping(self):
        walker = simple_humanoid.SimpleHumanoid()

        with self.subTest("Forward mapping"):
            for actuator_num, simple_mocap_joint_num in enumerate(
                walker.actuator_order
            ):
                self.assertEqual(
                    walker.actuator_to_joint_order[simple_mocap_joint_num], actuator_num
                )

        with self.subTest("Inverse mapping"):
            for simple_mocap_joint_num, actuator_num in enumerate(
                walker.actuator_to_joint_order
            ):
                self.assertEqual(
                    walker.actuator_order[actuator_num], simple_mocap_joint_num
                )

    def test_simple_humanoid_position_controlled_has_correct_actuators(self):
        walker_torque = simple_humanoid.SimpleHumanoid()
        walker_pos = simple_humanoid.SimpleHumanoidPositionControlled()

        actuators_torque = walker_torque.mjcf_model.find_all("actuator")
        actuators_pos = walker_pos.mjcf_model.find_all("actuator")

        actuator_pos_params = {
            params.name: params for params in simple_humanoid._POSITION_ACTUATORS
        }

        self.assertEqual(len(actuators_torque), len(actuators_pos))

        for actuator_torque, actuator_pos in zip(actuators_torque, actuators_pos):
            self.assertEqual(actuator_pos.name, actuator_torque.name)
            self.assertEqual(
                actuator_pos.joint.full_identifier,
                actuator_torque.joint.full_identifier,
            )
            self.assertEqual(actuator_pos.tag, "general")
            self.assertEqual(actuator_pos.ctrllimited, "true")
            np.testing.assert_array_equal(actuator_pos.ctrlrange, (-1, 1))

            expected_params = actuator_pos_params[actuator_pos.name]
            self.assertEqual(actuator_pos.biasprm[1], -expected_params.kp)
            np.testing.assert_array_equal(
                actuator_pos.forcerange, expected_params.forcerange
            )

    @parameterized.parameters(
        [
            "body_camera",
            "egocentric_camera",
            "head",
            "left_arm_root",
            "right_arm_root",
            "root_body",
        ]
    )
    def test_get_element_property(self, name):
        attribute_value = getattr(simple_humanoid.SimpleHumanoid(), name)
        self.assertIsInstance(attribute_value, mjcf.Element)

    @parameterized.parameters(
        [
            "actuators",
            "bodies",
            "end_effectors",
            "marker_geoms",
            "mocap_joints",
            "observable_joints",
        ]
    )
    def test_get_element_tuple_property(self, name):
        attribute_value = getattr(simple_humanoid.SimpleHumanoid(), name)
        self.assertNotEmpty(attribute_value)
        for item in attribute_value:
            self.assertIsInstance(item, mjcf.Element)

    def test_set_name(self):
        name = "fred"
        walker = simple_humanoid.SimpleHumanoid(name=name)
        self.assertEqual(walker.mjcf_model.model, name)

    def test_set_marker_rgba(self):
        marker_rgba = (1.0, 0.0, 1.0, 0.5)
        walker = simple_humanoid.SimpleHumanoid(marker_rgba=marker_rgba)
        for marker_geom in walker.marker_geoms:
            np.testing.assert_array_equal(marker_geom.rgba, marker_rgba)

    @parameterized.parameters(
        "actuator_activation",
        "appendages_pos",
        "body_camera",
        "head_height",
        "sensors_torque",
    )
    def test_evaluate_observable(self, name):
        walker = simple_humanoid.SimpleHumanoid()
        observable = getattr(walker.observables, name)
        physics = mjcf.Physics.from_mjcf_model(walker.mjcf_model)
        observation = observable(physics)
        self.assertIsInstance(observation, (float, np.ndarray))

    def test_proprioception(self):
        walker = simple_humanoid.SimpleHumanoid()
        for item in walker.observables.proprioception:
            self.assertIsInstance(item, observable_base.Observable)

    def test_simple_pose_to_actuation(self):
        walker = simple_humanoid.SimpleHumanoidPositionControlled()
        random_state = np.random.RandomState(123)

        expected_actuation = random_state.uniform(-1, 1, len(walker.actuator_order))

        simple_limits = zip(*(joint.range for joint in walker.mocap_joints))
        simple_lower, simple_upper = (np.array(limit) for limit in simple_limits)
        simple_pose = (
            simple_lower
            + (simple_upper - simple_lower)
            * (1 + expected_actuation[walker.actuator_to_joint_order])
            / 2
        )

        actual_actuation = walker.simple_pose_to_actuation(simple_pose)

        np.testing.assert_allclose(actual_actuation, expected_actuation)


if __name__ == "__main__":
    absltest.main()
