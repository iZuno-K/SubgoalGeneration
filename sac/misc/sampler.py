import numpy as np
import time

from rllab.misc import logger
# my
import misc.mylogger as mylogger
from misc.mylogger import MyJsonLogger
import copy

def rollout(env, policy, path_length, render=False, speedup=None):
    Da = env.action_space.flat_dim
    Do = env.observation_space.flat_dim

    observation = env.reset()
    policy.reset()

    observations = np.zeros((path_length + 1, Do))
    actions = np.zeros((path_length, Da))
    terminals = np.zeros((path_length, ))
    rewards = np.zeros((path_length, ))
    agent_infos = []
    env_infos = []

    t = 0
    for t in range(path_length):

        action, agent_info = policy.get_action(observation)
        next_obs, reward, terminal, env_info = env.step(action)

        agent_infos.append(agent_info)
        env_infos.append(env_info)

        actions[t] = action
        terminals[t] = terminal
        rewards[t] = reward
        observations[t] = observation

        observation = next_obs

        if render:
            env.render()
            time_step = 0.05
            time.sleep(time_step / speedup)

        if terminal:
            break

    observations[t + 1] = observation

    path = {
        'observations': observations[:t + 1],
        'actions': actions[:t + 1],
        'rewards': rewards[:t + 1],
        'terminals': terminals[:t + 1],
        'next_observations': observations[1:t + 2],
        'agent_infos': agent_infos,
        'env_infos': env_infos
    }

    return path


def rollouts(env, policy, path_length, n_paths):
    paths = [
        rollout(env, policy, path_length)
        for i in range(n_paths)
    ]

    return paths


class Sampler(object):
    def __init__(self, max_path_length, min_pool_size, batch_size):
        self._max_path_length = max_path_length
        self._min_pool_size = min_pool_size
        self._batch_size = batch_size

        self.env = None
        self.policy = None
        self.pool = None

    def initialize(self, env, policy, pool):
        self.env = env
        self.policy = policy
        self.pool = pool

    def sample(self):
        raise NotImplementedError

    def batch_ready(self):
        enough_samples = self.pool.size >= self._min_pool_size
        return enough_samples

    def random_batch(self):
        return self.pool.random_batch(self._batch_size)

    def terminate(self):
        self.env.terminate()

    def log_diagnostics(self):
        logger.record_tabular('pool-size', self.pool.size)


class SimpleSampler(Sampler):
    def __init__(self, **kwargs):
        super(SimpleSampler, self).__init__(**kwargs)
        self._path_length = 0
        self._path_return = 0
        self._last_path_return = 0
        self._max_path_return = -np.inf
        self._n_episodes = 0
        self._current_observation = None
        self._total_samples = 0

    def sample(self):
        if self._current_observation is None:
            self._current_observation = self.env.reset()

        action, _ = self.policy.get_action(self._current_observation)
        next_observation, reward, terminal, info = self.env.step(action)
        self._path_length += 1
        self._path_return += reward
        self._total_samples += 1

        self.pool.add_sample(
            observation=self._current_observation,
            action=action,
            reward=reward,
            terminal=terminal,
            next_observation=next_observation)

        info = copy.deepcopy(info)  # to avoid changing info of this scope by reset function by self.env.reset
        if terminal or self._path_length >= self._max_path_length:
            # my
            mylogger.data_append(key='mean_return', val=self._path_return)
            mylogger.data_update(key='total_step', val=self._total_samples)
            mylogger.data_update(key='total_episode', val=self._n_episodes)
            # my end

            self.policy.reset()
            self._current_observation = self.env.reset()
            self._path_length = 0
            self._max_path_return = max(self._max_path_return,
                                        self._path_return)
            self._last_path_return = self._path_return

            self._path_return = 0
            self._n_episodes += 1

        else:
            self._current_observation = next_observation

        # my
        return terminal, self._n_episodes, next_observation, info["reached_goal"]

    def log_diagnostics(self):
        super(SimpleSampler, self).log_diagnostics()
        logger.record_tabular('max-path-return', self._max_path_return)
        logger.record_tabular('last-path-return', self._last_path_return)
        logger.record_tabular('episodes', self._n_episodes)
        logger.record_tabular('total-samples', self._total_samples)

# my
class NormalizeSampler(Sampler):
    def __init__(self, **kwargs):
        super(NormalizeSampler, self).__init__(**kwargs)
        self._path_length = 0
        self._path_return = 0
        self._last_path_return = 0
        self._max_path_return = -np.inf
        self._n_episodes = 0
        self._current_observation = None
        self._total_samples = 0
        self.obs_mean = 0.
        self.obs_var = 1.
        self._obs_alpha = 0.001

    def _update_obs_estimate(self, flat_obs):
        self.obs_mean = (1 - self._obs_alpha) * self.obs_mean + self._obs_alpha * flat_obs
        self.obs_var = (1 - self._obs_alpha) * self.obs_var + self._obs_alpha * np.square(flat_obs - self.obs_mean)

    def apply_normalize_obs(self, obs):
        return (obs - self.obs_mean) / (np.sqrt(self.obs_var) + 1e-8)

    def sample(self):
        if self._current_observation is None:
            self._current_observation = self.env.reset()

        self._update_obs_estimate(self._current_observation)
        action, _ = self.policy.get_action(self.apply_normalize_obs(self._current_observation))
        next_observation, reward, terminal, info = self.env.step(action)
        self._path_length += 1
        self._path_return += reward
        self._total_samples += 1

        self.pool.add_sample(
            observation=self._current_observation,
            action=action,
            reward=reward,
            terminal=terminal,
            next_observation=next_observation)

        info = copy.deepcopy(info)  # to avoid changing info of this scope by reset function by self.env.reset
        if terminal or self._path_length >= self._max_path_length:
            # my
            mylogger.data_append(key='mean_return', val=self._path_return)
            mylogger.data_update(key='total_step', val=self._total_samples)
            mylogger.data_update(key='total_episode', val=self._n_episodes)
            mylogger.data_update(key='obs_mean', val=self.obs_mean.tolist())
            mylogger.data_update(key='obs_var', val=self.obs_var.tolist())
            # my end

            self.policy.reset()
            self._current_observation = self.env.reset()
            self._path_length = 0
            self._max_path_return = max(self._max_path_return,
                                        self._path_return)
            self._last_path_return = self._path_return

            self._path_return = 0
            self._n_episodes += 1

        else:
            self._current_observation = next_observation

        # my
        return terminal, self._n_episodes, next_observation, info["reached_goal"]

    def random_batch(self):
        batch = self.pool.random_batch(self._batch_size)
        batch['observations'] = self.apply_normalize_obs(batch['observations'])
        batch['next_observations'] = self.apply_normalize_obs(batch['next_observations'])
        return batch

    def log_diagnostics(self):
        super(NormalizeSampler, self).log_diagnostics()
        logger.record_tabular('max-path-return', self._max_path_return)
        logger.record_tabular('last-path-return', self._last_path_return)
        logger.record_tabular('episodes', self._n_episodes)
        logger.record_tabular('total-samples', self._total_samples)
        logger.record_tabular('obs-mean', self.obs_mean)
        logger.record_tabular('obs-var', self.obs_var)


class DummySampler(Sampler):
    def __init__(self, batch_size, max_path_length):
        super(DummySampler, self).__init__(
            max_path_length=max_path_length,
            min_pool_size=0,
            batch_size=batch_size)

    def sample(self):
        pass
