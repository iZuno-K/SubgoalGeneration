import gym
from gym import wrappers
import os
from glob import glob

from gym.wrappers.monitoring import Monitor
from gym.utils import atomic_write, closer
import json, logging
from gym.utils.json_utils import json_encode_np
from gym.monitoring import stats_recorder, video_recorder
from gym import error
import tempfile

logger = logging.getLogger(__name__)


class MyStatsRecorder(stats_recorder.StatsRecorder):
    def flush(self):
        return


def touch(path):
    open(path, 'a').close()


class MyVideoRecorder(video_recorder.VideoRecorder):

    def __init__(self, env, path=None, metadata=None, enabled=True, base_path=None):
        modes = env.metadata.get('render.modes', [])
        self._async = env.metadata.get('semantics.async')
        self.enabled = enabled

        # Don't bother setting anything else if not enabled
        if not self.enabled:
            return

        self.ansi_mode = False
        if 'rgb_array' not in modes:
            if 'ansi' in modes:
                self.ansi_mode = True
            else:
                logger.info(
                    'Disabling video recorder because {} neither supports video mode "rgb_array" nor "ansi".'.format(
                        env))
                # Whoops, turns out we shouldn't be enabled after all
                self.enabled = False
                return

        if path is not None and base_path is not None:
            raise error.Error("You can pass at most one of `path` or `base_path`.")

        self.last_frame = None
        self.env = env

        required_ext = '.json' if self.ansi_mode else '.mp4'
        if path is None:
            if base_path is not None:
                # Base path given, append ext
                path = base_path + required_ext
            else:
                # Otherwise, just generate a unique filename
                with tempfile.NamedTemporaryFile(suffix=required_ext, delete=False) as f:
                    path = f.name
        self.path = path

        path_base, actual_ext = os.path.splitext(self.path)

        if actual_ext != required_ext:
            hint = " HINT: The environment is text-only, therefore we're recording its text output in a structured JSON format." if self.ansi_mode else ''
            raise error.Error(
                "Invalid path given: {} -- must have file extension {}.{}".format(self.path, required_ext, hint))
        # Touch the file in any case, so we know it's present. (This
        # corrects for platform platform differences. Using ffmpeg on
        # OS X, the file is precreated, but not on Linux.
        touch(path)

        self.frames_per_sec = env.metadata.get('video.frames_per_second', 30)
        self.encoder = None  # lazily start the process
        self.broken = False

        # Dump metadata
        self.metadata = metadata or {}
        self.metadata['content_type'] = 'video/vnd.openai.ansivid' if self.ansi_mode else 'video/mp4'
        self.metadata_path = '{}.meta.json'.format(path_base)
        # self.write_metadata()

        logger.info('Starting new video recorder writing to %s', self.path)
        self.empty = True

    def write_metadata(self):
        return


class MyMonitor(Monitor):

    def _flush(self, force=False):
        """Flush all relevant monitor information to disk."""
        if not self.write_upon_reset and not force:
            return

    def additional_init(self):
        self.stats_recorder = MyStatsRecorder(self.directory,
                                              '{}.episode_batch.{}'.format(self.file_prefix, self.file_infix),
                                              autoreset=self.env_semantics_autoreset, env_id=self.env.spec.id)

    def set_video_path(self, path=None, video_name=None):
        if self.video_recorder:
            self._close_video_recorder()

        obs = self.reset()

        # Start recording the next video.
        #
        # TODO: calculate a more correct 'episode_id' upon merge
        # base_path = os.path.join(self.directory,
        #                          '{}.video.{}.video{:06}'.format(self.file_prefix, self.file_infix, self.episode_id)),
        if path is not None:

            directory = os.path.join(path, video_name)
        else:
            directory = os.path.join(self.directory, 'myvideo{:03}'.format(self.episode_id))

        self.video_recorder = MyVideoRecorder(
            env=self.env,
            base_path=directory,
            metadata={'episode_id': self.episode_id},
            enabled=self._video_enabled(),
        )
        self.video_recorder.path = directory + '.mp4'
        self.video_recorder.capture_frame()

        return obs

    def _after_step(self, observation, reward, done, info):
        if not self.enabled: return done

        if done and self.env_semantics_autoreset:
            # For envs with BlockingReset wrapping VNCEnv, this observation will be the first one of the new episode
            # self._reset_video_recorder()
            # self.episode_id += 1
            self._flush()

        if info.get('true_reward',
                    None):  # Semisupervised envs modify the rewards, but we want the original when scoring
            reward = info['true_reward']

        # Record stats
        self.stats_recorder.after_step(observation, reward, done, info)
        # Record video
        self.video_recorder.capture_frame()

        return done

    def _after_reset(self, observation):
        if not self.enabled: return

        # Reset the stat count
        self.stats_recorder.after_reset(observation)

        self._reset_video_recorder()

        # Bump *after* all reset activity has finished
        # self.episode_id += 1

        self._flush()

    def _reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return observation

    def _step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.video_recorder.capture_frame()
        return observation, reward, done, info

    def _reset_video_recorder(self):
        # Close any existing video recorder
        if self.video_recorder:
            self._close_video_recorder()

        # Start recording the next video.
        #
        # TODO: calculate a more correct 'episode_id' upon merge
        # base_path = os.path.join(self.directory,
        #                          '{}.video.{}.video{:06}'.format(self.file_prefix, self.file_infix, self.episode_id)),
        self.video_recorder = MyVideoRecorder(
            env=self.env,
            base_path=os.path.join(self.directory, 'myvideo{:03}'.format(self.episode_id)),
            metadata={'episode_id': self.episode_id},
            enabled=self._video_enabled(),
        )
        self.video_recorder.capture_frame()


if __name__ == '__main__':
    env = gym.make('HalfCheetah-v1')
    path = os.path.expanduser('~/baselines/data/log-files/test')
    env = MyMonitor(env, directory=path, force=True)

    obs = env.reset()
    for i in range(100):
        print(i)
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)

    env.close()
    rem = glob(path + '/*.json')
    print(rem)
    for f in rem:
        os.remove(f)