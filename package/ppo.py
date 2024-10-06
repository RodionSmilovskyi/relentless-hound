import torch
import os
import imageio
import numpy as np
from torch.optim import Adam
from torch import nn
from torch.distributions import MultivariateNormal
from package import settings
from package.network import FeedForwardNN
from torch.utils.tensorboard import SummaryWriter


class PPO:
    """Class responsible for PPO learning"""

    def __init__(self, env):
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        self.actor = FeedForwardNN(self.obs_dim, self.act_dim)
        self.critic = FeedForwardNN(self.obs_dim, 1)

        self._init_hyperparameters()

        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)
        # Create our variable for the matrix.
        # Note that I chose 0.5 for stdev arbitrarily.
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)

        # Create the covariance matrix
        self.cov_mat = torch.diag(self.cov_var)

        self.writer = SummaryWriter(settings.OUTPUT_PATH + "/tensorboard")

    def _init_hyperparameters(self):
        self.timesteps_per_batch = settings.TIMESTEPS_PER_BATCH  # timesteps per batch
        self.max_timesteps_per_episode = 1000  # timesteps per episode
        self.gamma = 0.95
        self.n_updates_per_iteration = 5
        self.clip = 0.2  # As recommended by the paper
        self.lr = settings.LEARN_RATE

    def get_action(self, obs):
        """Query the actor network for a mean action."""
        # Same thing as calling self.actor.forward(obs)
        mean = self.actor(obs)
        # Create our Multivariate Normal Distribution
        dist = MultivariateNormal(mean, self.cov_mat)
        # Sample an action from the distribution and get its log prob
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # Return the sampled action and the log prob of that action
        # Note that I'm calling detach() since the action and log_prob
        # are tensors with computation graphs, so I want to get rid
        # of the graph and just convert the action to numpy array.
        # log prob as tensor is fine. Our computation graph will
        # start later down the line.
        return action.detach().cpu().numpy(), log_prob.detach()

    def compute_rtgs(self, batch_rews):
        """The rewards-to-go (rtg) per episode per batch to return."""
        # The shape will be (num timesteps per episode)
        batch_rtgs = []
        # Iterate through each episode backwards to maintain same order
        # in batch_rtgs
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0  # The discounted reward so far
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        return batch_rtgs

    def rollout(self):
        """Env rollout"""
        batch_obs = []  # batch observations
        batch_acts = []  # batch actions
        batch_log_probs = []  # log probs of each action
        batch_rews = []  # batch rewards
        batch_rtgs = []  # batch rewards-to-go
        batch_lens = []  # episodic lengths in batch

        # Episodic data. Keeps track of rewards per episode, will get cleared
        # upon each new episode
        ep_rews = []

        # Number of timesteps run so far this batch
        t = 0
        while t < self.timesteps_per_batch:
            # Rewards this episode
            ep_rews = []

            obs, _ = self.env.reset()
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                # Increment timesteps ran this batch so far
                t += 1

                # Collect observation
                batch_obs.append(obs)

                action, log_prob = self.get_action(obs)
                obs, rew, terminated, truncated, _ = self.env.step(action)

                # Don't really care about the difference between terminated or truncated in this, so just combine them
                done = terminated | truncated

                # Collect reward, action, and log prob
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break

            # Collect episodic length and rewards
            batch_lens.append(ep_t + 1)  # plus 1 because timestep starts at 0
            batch_rews.append(ep_rews)

        # Reshape data as tensors in the shape specified before returning
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        # ALG STEP #4
        batch_rtgs = self.compute_rtgs(batch_rews)
        # Return the batch data
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def evaluate_policy(self, step, iteration, num_of_episodes):
        """Evaluate policy performance"""
        ep_returns = []

        for _ in range(num_of_episodes):
            ep_ret = 0
            done = False

            obs, _ = self.env.reset()

            while not done:
                action = self.actor(obs).detach().cpu().numpy()
                obs, rew, terminated, truncated, _ = self.env.step(action)
                done = terminated | truncated

                # Sum all episodic rewards as we go along
                ep_ret += rew

            ep_returns.append(ep_ret)

        avg_return = np.mean(np.array(ep_returns))

        print(f"Iteration {iteration} Step {step} Average return {avg_return}")
        self.writer.add_scalar("Average return", avg_return, step)

    def create_policy_eval_video(self, filename, num_episodes=5, fps=30):
        """Record video of policy in action"""
        filename = filename + ".mp4"

        directory = os.path.dirname(filename)

        if not os.path.exists(directory):
            os.makedirs(directory)

        with imageio.get_writer(filename, fps=fps) as video:
            for _ in range(num_episodes):
                done = False
                obs, _ = self.env.reset()
                video.append_data(self.env.render())

                while not done:
                    action = self.actor(obs).detach().cpu().numpy()
                    obs, rew, terminated, truncated, _ = self.env.step(action)
                    done = terminated | truncated

                    video.append_data(self.env.render())

                video.close()

    def evaluate(self, batch_obs, batch_acts):
        """Query critic network for a value V for each obs in batch_obs."""
        V = self.critic(batch_obs).squeeze()

        # Calculate the log probabilities of batch actions using most
        # recent actor network.
        # This segment of code is similar to that in get_action()
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)
        # Return predicted values V and log probs log_probs
        return V, log_probs

    def learn(self, total_timesteps):
        """Main method"""

        t_so_far = 0
        i_so_far = 0  # Iterations ran so far

        while t_so_far < total_timesteps:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = (
                self.rollout()
            )

            # Calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_lens)
            i_so_far += 1

            # Calculate V_{phi, k}
            V, _ = self.evaluate(batch_obs, batch_acts)
            # ALG STEP 5
            # Calculate advantage
            A_k = batch_rtgs - V.detach()

            # Normalize advantages
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            if i_so_far % 10 == 0:
                self.evaluate_policy(t_so_far, i_so_far, 10)
                self.create_policy_eval_video(
                    f"{settings.OUTPUT_PATH}/data/videos/{i_so_far}", 1
                )
                torch.save(
                    self.actor.state_dict(), f"{settings.CHECKPOINT_PATH}/ppo_actor.pth"
                )
                torch.save(
                    self.critic.state_dict(),
                    f"{settings.CHECKPOINT_PATH}/ppo_critic.pth",
                )

            for _ in range(self.n_updates_per_iteration):
                frac = (t_so_far - 1.0) / total_timesteps
                new_lr = self.lr * (1.0 - frac)

                # Make sure learning rate doesn't go below 0
                new_lr = max(new_lr, 0.0)
                self.actor_optim.param_groups[0]["lr"] = new_lr
                self.critic_optim.param_groups[0]["lr"] = new_lr

                self.writer.add_scalar("Learning rate", new_lr, t_so_far)

                # Calculate pi_theta(a_t | s_t)
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                # Calculate ratios
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # Calculate surrogate losses
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                # Calculate gradients and perform backward propagation for actor
                # network
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()

                # Calculate gradients and perform backward propagation for critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

        state, _ = self.env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        onnx_program = torch.onnx.export(self.actor, (state,), dynamo=True)
        onnx_program.save(settings.MODEL_PATH + "/model.onnx")
        torch.save(self.actor.state_dict(), settings.MODEL_PATH + "/model.pth")

        self.writer.flush()
        self.writer.close()
