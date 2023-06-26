"""Tests for the simple humanoid."""

from absl.testing import absltest
from absl.testing import parameterized
from dm_control import mjcf
from dm_control.composer.observation.observable import base as observable_base
import numpy as np

from . import simple_humanoid


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

    def test_simple_humanoid_position_controlled_has_correct_actuators(self):
        walker_torque = simple_humanoid.SimpleHumanoid()
        walker_pos = simple_humanoid.SimpleHumanoidPositionControlled()

        actuators_torque = walker_torque.mjcf_model.find_all("actuator")
        actuators_pos = walker_pos.mjcf_model.find_all("actuator")

        actuator_pos_params = {
            params.name: params
            for params in simple_humanoid._POSITION_ACTUATORS  # pylint: disable=protected-access
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
            "egocentric_camera",
            "head",
            "root_body",
        ]
    )
    def test_get_element_property(self, name):
        attribute_value = getattr(simple_humanoid.SimpleHumanoid(), name)
        self.assertIsInstance(attribute_value, mjcf.Element)

    @parameterized.parameters(
        [
            "actuators",
            "mocap_tracking_sites",
            "end_effectors",
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

    @parameterized.parameters(
        "actuator_activation",
        "appendages_pos",
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


if __name__ == "__main__":
    absltest.main()
