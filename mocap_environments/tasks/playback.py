"""Motion playback `dm_control.composer.Task`."""

from dm_control import mjcf
import numpy as np

from mocap_environments.tasks import tracking as tracking_task


class PlaybackTask(tracking_task.TrackingTask):
    """Task that simply replays the motion sequences."""

    def before_step(
        self,
        physics: mjcf.Physics,
        action: np.ndarray,
        random_state: np.random.RandomState,
    ):
        return super(PlaybackTask, self).before_step(physics, action, random_state)

    def after_step(self, physics: mjcf.Physics, random_state: np.random.RandomState):
        result = super().after_step(physics, random_state)
        self._set_walker(physics)
        return result
