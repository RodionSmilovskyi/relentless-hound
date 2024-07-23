import gym
from gym.core import ActType

APPROACH_DISTANCE = 0.5

class RewardWrapper(gym.Wrapper):
   
   
    def step(self, action: ActType):
        frames, reward, terminated, info = self.env.step(action)
        not_empty_frames = [f for f in frames if f[0] != 0]
        
        if len(not_empty_frames):
            last_frame = not_empty_frames[-1]
            # dist_rate = 1 / last_frame[4]
            area_rate = abs(last_frame[0] - last_frame[2]) * abs(last_frame[1] - last_frame[3])
            # area_rate = area_rate if area_rate != 0 else -1
            # reward = reward +  dist_rate + 2 * area_rate
            
            reward = reward + area_rate
            
            return  frames, reward, terminated, info
        
        return frames, reward, terminated, info