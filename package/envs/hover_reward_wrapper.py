import gym
from gym.core import ActType

from package.envs.drone import MAX_ALTITUDE, MIN_ALTITUDE

HOVER_ALTITUDE = 3


class HowerRewardWrapper(gym.Wrapper):
    """Class gives reward for just hovering"""

    def __init__(self, env: gym.Env):
        super().__init__(env)

        self.hover_altitude = HOVER_ALTITUDE / MAX_ALTITUDE

    def step(self, action: ActType):
        obs, reward, terminated, info = super().step(action)
        altitude = obs[5]
        above_ground = altitude > MIN_ALTITUDE / MAX_ALTITUDE

        if above_ground:
            reward = reward + 1

        if abs(altitude - self.hover_altitude) < self.hover_altitude * 0.1:
            reward = reward + 5

        terminated = (
            True if not above_ground and info["step_number"] > 50 else terminated
        )

        return obs, reward, terminated, info
