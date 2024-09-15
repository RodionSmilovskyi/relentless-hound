import gym
from gym.core import ActType

from package.envs.drone import MAX_ALTITUDE, MIN_ALTITUDE

HOVER_ALTITUDE = 3


class HowerRewardWrapper(gym.Wrapper):
    """Class gives reward for just hovering"""

    def __init__(self, env: gym.Env):
        super().__init__(env)

        self.hover_altitude = round(HOVER_ALTITUDE / MAX_ALTITUDE, 4)
        self.entered_range = False
        self.prev_altitude_diff = self.hover_altitude

    def step(self, action: ActType):
        obs, reward, terminated, info = super().step(action)
        altitude = round(obs[5], 4)
        reward = -1
        current_altitude_diff = round(abs(altitude - self.hover_altitude), 4)
        above_ground = altitude > MIN_ALTITUDE / MAX_ALTITUDE

        if current_altitude_diff < self.hover_altitude * 0.1:
            reward = reward + 10
        else:
            if current_altitude_diff > self.prev_altitude_diff:
                reward = reward - 5
            elif current_altitude_diff < self.prev_altitude_diff:
                reward = reward + 5

        self.prev_altitude_diff = current_altitude_diff

        terminated = (
            True if not above_ground and info["step_number"] > 50 else terminated
        )

        return obs, reward, terminated, info
