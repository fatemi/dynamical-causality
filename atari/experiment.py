import os
import time
import numpy as np
from utils import write_to_csv, plot


class DQNExperiment(object):
    def __init__(self, env, ai, episode_max_len, annealing=False, history_len=1, max_start_nullops=1, test_epsilon=0.0,
                 replay_min_size=0, score_window_size=100, folder_location='/experiments/', folder_name='expt',
                 saving_period=10, rng=None, make_folder=True):
        self.rng = rng
        self.fps = 0
        self.episode_num = 0
        self.last_episode_steps = 0
        self.total_training_steps = 0
        self.score_computer = 0
        self.score_agent = 0
        self.eval_scores = []
        self.eval_steps = []
        self.env = env
        self.ai = ai
        self.history_len = history_len
        self.annealing = annealing
        self.test_epsilon = test_epsilon
        self.max_start_nullops = max_start_nullops
        self.saving_period = saving_period  # after each `saving_period` epochs, the results so far will be saved.
        self.episode_max_len = episode_max_len
        self.score_agent_window = np.zeros(score_window_size)
        self.steps_agent_window = np.zeros(score_window_size)
        self.replay_min_size = max(self.ai.minibatch_size, replay_min_size)
        self.last_state = np.empty(tuple([self.history_len] + self.env.state_shape), dtype=np.uint8)
        if make_folder:
            self.folder_name = self._create_folder(folder_location, folder_name)
        self.curr_epoch = 0
        self.all_rewards = []

    def do_epochs(self, number=1, steps_per_epoch=10000, is_learning=True, is_testing=True, steps_per_test=10000):
        for epoch in range(self.curr_epoch, number):
            print('=' * 30)
            print('>>>>> Epoch  ' + str(epoch + 1) + '/' + str(number) + '  >>>>>')
            steps = 0
            while steps < steps_per_epoch:
                self.do_episodes(number=1, is_learning=is_learning)
                steps += self.last_episode_steps
            if is_testing:
                eval_steps = 0
                eval_episodes = 0
                eval_scores = 0
                print('Evaluation ...')
                while eval_steps < steps_per_test:
                    if eval_episodes % 10 == 0:
                        print('Evaluate episode: ' + str(eval_episodes))
                    eval_scores += self.evaluate(number=1)
                    eval_steps += self.last_episode_steps
                    eval_episodes += 1
                self.eval_scores.append(eval_scores / eval_episodes)
                self.eval_steps.append(eval_steps / eval_episodes)
                self._plot_and_write(plot_dict={'scores': self.eval_scores}, loc=self.folder_name + "/scores",
                                    x_label="Epochs", y_label="Mean Score", title="", kind='line', legend=True,
                                    moving_average=True)
                self._plot_and_write(plot_dict={'steps': self.eval_steps}, loc=self.folder_name + "/steps",
                                    x_label="Epochs", y_label="Mean Steps", title="", kind='line', legend=True)
                if epoch % self.saving_period == 0:
                    self.ai.dump_network(weights_file_path=self.folder_name + '/ai/q_network_weights_' + str(epoch + 1) + '.pt')
                self.all_rewards.append(eval_scores / eval_episodes)

    def do_episodes(self, number=1, is_learning=True):
        all_rewards = []
        for _ in range(number):
            reward = self._do_episode(is_learning=is_learning)
            all_rewards.append(reward)
            self.score_agent_window = self._update_window(self.score_agent_window, self.score_agent)
            self.steps_agent_window = self._update_window(self.steps_agent_window, self.last_episode_steps)
            if self.episode_num % 10 == 0:
                print_string = ("\nSteps: {0} | Fps: {1} | Eps: {2} | Score: {3} | Agent Moving Avg: {4} | "
                                "Agent Moving Steps: {5} | Total Steps: {6} ")
                print('=' * 30)
                print('::Episode::  ' + str(self.episode_num))
                print(print_string.format(self.last_episode_steps, self.fps, round(self.ai.epsilon, 2),
                                        round(self.score_agent, 2), round(np.mean(self.score_agent_window), 2),
                                        np.mean(self.steps_agent_window), self.total_training_steps))
            self.episode_num += 1
        return all_rewards
    
    def evaluate(self, number=10):
        for _ in range(number):
            self._do_episode(is_learning=False, evaluate=True)
        return self.score_agent

    def _do_episode(self, is_learning=True, evaluate=False):
        rewards = []
        self.env.reset()
        self._episode_reset()
        term = False
        self.fps = 0
        start_time = time.time()
        while not term:
            reward, term = self._step(evaluate=evaluate)
            rewards.append(reward)
            if self.ai.transitions.size >= self.replay_min_size and is_learning and \
               self.last_episode_steps % self.ai.learning_frequency == 0:
                self.ai.learn()
            self.score_agent += reward
            if not term and self.last_episode_steps >= self.episode_max_len:
                print('Reaching maximum number of steps in the current episode.')
                term = True
        self.fps = int(self.last_episode_steps * self.env.frame_skip / max(time.time() - start_time, 0.01))
        return rewards

    def _step(self, evaluate=False):
        self.last_episode_steps += 1
        prev_lives = self.env.get_lives()
        action = self.ai.get_action(self.last_state, evaluate)
        new_obs, reward, game_over, _ = self.env.step(action)
        if new_obs.ndim == 1 and len(self.env.state_shape) == 2:
            new_obs = new_obs.reshape(self.env.state_shape)
        if not evaluate:
            self.ai.transitions.add(s=self.last_state[-1].astype('float32'), a=action, r=reward, t=game_over)
            if self.annealing:
                if self.total_training_steps >= self.replay_min_size:
                    self.ai.anneal_eps(self.total_training_steps - self.replay_min_size)
            self.total_training_steps += 1
        self._update_state(new_obs)
        return reward, game_over

    def _episode_reset(self):
        self.last_episode_steps = 0
        self.score_agent = 0
        self.score_computer = 0
        assert self.max_start_nullops >= self.history_len or self.max_start_nullops == 0
        if self.max_start_nullops != 0:
            num_nullops = self.rng.randint(self.history_len, self.max_start_nullops)
            for i in range(num_nullops - self.history_len):
                self.env.step(0)
        for i in range(self.history_len):
            if i > 0:
                self.env.step(0)
            obs = self.env.get_state()
            if obs.ndim == 1 and len(self.env.state_shape) == 2:
                obs = obs.reshape(self.env.state_shape)
            self.last_state[i] = obs

    def _update_state(self, new_obs):
        temp_buffer = np.empty(self.last_state.shape, dtype=np.uint8)
        temp_buffer[:-1] = self.last_state[-self.history_len + 1:]
        temp_buffer[-1] = new_obs
        self.last_state = temp_buffer

    @staticmethod
    def _plot_and_write(plot_dict, loc, x_label="", y_label="", title="", kind='line', legend=True,
                        moving_average=False):
        for key in plot_dict:
            plot(data={key: plot_dict[key]}, loc=loc + ".pdf", x_label=x_label, y_label=y_label, title=title,
                 kind=kind, legend=legend, index_col=None, moving_average=moving_average)
            write_to_csv(data={key: plot_dict[key]}, loc=loc + ".csv")

    @staticmethod
    def _create_folder(folder_location, folder_name):
        i = 0
        while os.path.exists(folder_location + folder_name + str(i)):
            i += 1
        folder_name = folder_location + folder_name + str(i)
        os.makedirs(folder_name)
        os.mkdir(folder_name + '/ai')
        return folder_name

    @staticmethod
    def _update_window(window, new_value):
        window[:-1] = window[1:]
        window[-1] = new_value
        return window
