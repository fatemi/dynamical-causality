import numpy as np
from zmq import device
from utils import ExperienceReplay
from model import NatureNetwork
import torch
import torch.nn.functional as F
import torch.optim as optim


class RL(object):
    def __init__(self, state_shape, nb_actions, action_dim, reward_dim, history_len, gamma,
                 learning_rate, epsilon, final_epsilon, test_epsilon, annealing_steps,
                 minibatch_size, replay_max_size, update_freq, learning_frequency, ddqn,
                 network_size, normalize, event, sided_Q, rng, device):
        """
        `event` : negative / positive
        `sided_Q` : grit / reachability
        """
        self.rng = rng
        self.history_len = history_len
        self.state_shape = state_shape
        self.nb_actions = nb_actions
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.start_epsilon = epsilon
        self.test_epsilon = test_epsilon
        self.final_epsilon = final_epsilon
        self.decay_steps = annealing_steps
        self.minibatch_size = minibatch_size
        self.network_size = network_size
        self.update_freq = update_freq
        self.update_counter = 0
        self.normalize = normalize
        self.learning_frequency = learning_frequency
        self.replay_max_size = replay_max_size
        self.transitions = ExperienceReplay(max_size=self.replay_max_size, history_len=history_len,
                                            state_shape=state_shape, action_dim=action_dim, reward_dim=reward_dim)
        self.ddqn = ddqn
        self.device = device
        self.network = self._build_network()
        self.target_network = self._build_network()
        self.weight_transfer(from_model=self.network, to_model=self.target_network)
        self.network.to(self.device)
        self.target_network.to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate, amsgrad=True)
        assert event in ['negative', 'positive'], "Incorrect event."
        assert sided_Q in ['grit', 'reachability'], "Incorrect sided_Q."
        self.event = event
        self.sided_Q = sided_Q

    def _build_network(self):
        if self.network_size == 'small':
            return Network()
        elif self.network_size == 'large':
            return LargeNetwork(state_shape=self.state_shape, nb_channels=4, nb_actions=self.nb_actions)
        elif self.network_size == 'nature':
            return NatureNetwork(state_shape=self.state_shape, nb_channels=4, nb_actions=self.nb_actions)
        else:
            raise ValueError('Invalid network_size.')

    def train_on_batch(self, s, a, r, s2, t):
        s = torch.FloatTensor(np.float32(s)).to(self.device)
        s2 = torch.FloatTensor(np.float32(s2)).to(self.device)
        a = torch.LongTensor(np.int64(a)).to(self.device)
        r = torch.FloatTensor(np.float32(r)).to(self.device)
        t = torch.FloatTensor(np.float32(t)).to(self.device)

        # r.clamp_(min=-1, max=1)  # not needed

        q = self.network(s / self.normalize)
        q2 = self.target_network(s2 / self.normalize).detach()
        q_pred = q.gather(1, a.unsqueeze(1)).squeeze(1) 
        if self.ddqn:
            q2_net = self.network(s2 / self.normalize).detach()
            q2_max = q2.gather(1, torch.max(q2_net, 1)[1].unsqueeze(1)).squeeze(1)
        else:
            q2_max = torch.max(q2, 1)[0]
        
        if self.sided_Q == 'grit':
            if self.event == 'positive':
                r = torch.clamp(r, min=0, max=1)  # get rid of negative rewards
                r = -r  # switch +1 to -1 for grit
            elif self.event == 'negative':
                r = torch.clamp(r, min=-1, max=0) # get rid of positive rewards
            bellman_target = r + self.gamma * torch.clamp(q2_max.detach(), min=-1, max=0) * (1 - t)  # clip the target at [-1,0]
        elif self.sided_Q == 'reachability':
            if self.event == 'positive':
                r = torch.clamp(r, min=0, max=1)
            elif self.event == 'negative':
                r = torch.clamp(r, min=-1, max=0)
                r = -r  # switch -1 to +1 for reachability
            bellman_target = r + self.gamma * torch.clamp(q2_max.detach(), min=0, max=1) * (1 - t)  # clip the target at [0,1]
        
        loss = F.smooth_l1_loss(q_pred, bellman_target)
            
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_q(self, s):
        # For external use
        if type(s) == np.ndarray:
            s = torch.FloatTensor(s).to(self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.network(s / self.normalize).detach().cpu()
            if self.sided_Q == 'grit':
                torch.clamp_(q, min=-1, max=0)
            elif self.sided_Q == 'reachability':
                torch.clamp_(q, min=0, max=1)
            return q.numpy()

    def get_max_action(self, s):
        s = torch.FloatTensor(np.float32(s)).to(self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.network(s / self.normalize).detach()
            return q.max(1)[1].cpu().numpy()

    def get_action(self, states, evaluate):
        # get action WITH e-greedy exploration
        eps = self.epsilon if not evaluate else self.test_epsilon
        if self.rng.binomial(1, eps):
            return self.rng.randint(self.nb_actions)
        else:
            return self.get_max_action(states)[0]

    def learn(self):
        """ Learning from one minibatch """
        assert self.minibatch_size <= self.transitions.size, 'not enough data in the pool'
        s, a, r, s2, term = self.transitions.sample(self.minibatch_size)
        self.train_on_batch(s, a, r, s2, term)
        if self.update_counter == self.update_freq:
            self.weight_transfer(from_model=self.network, to_model=self.target_network)
            self.update_counter = 0
        else:
            self.update_counter += 1

    def anneal_eps(self, step):
        if self.epsilon > self.final_epsilon:
            decay = (self.start_epsilon - self.final_epsilon) * step / self.decay_steps
            self.epsilon = self.start_epsilon - decay
        if step >= self.decay_steps:
            self.epsilon = self.final_epsilon
    
    def get_grad(self, s):
        grads = []
        s  = torch.FloatTensor(np.float32(s)).to(self.device).unsqueeze(0)
        s.requires_grad = True
        s = s / self.normalize
        q = self.network(s)
        
        if self.sided_Q == 'grit':
            torch.clamp_(q, min=-1, max=0)
        elif self.sided_Q == 'reachability':
            torch.clamp_(q, min=0, max=1)
        else:
            raise NotImplementedError("Unknown sided_Q.")
                
        for a_idx in range(self.nb_actions):
            a = torch.LongTensor([a_idx]).to(self.device)
            q_a = q.gather(1, a.unsqueeze(1)).squeeze(1)
            grads.append( torch.autograd.grad(inputs=s, outputs=q_a, retain_graph=True)[0].to('cpu').numpy().tolist() )
            # TODO: ^ autograd makes an extra redundant inner dimension; should be removed. Test what happens if input `s` is multi-dim.
        grads = torch.Tensor(grads)   # rows == actions, cols == state_variables (extera dim between rows and cols)
        return grads

    def dump_network(self, weights_file_path):
        torch.save(self.network.state_dict(), weights_file_path)

    def load_weights(self, weights_file_path, target=False):
        self.network.load_state_dict(torch.load(weights_file_path, map_location=torch.device(self.device)))
        if target:
            self.weight_transfer(from_model=self.network, to_model=self.target_network)

    @staticmethod
    def weight_transfer(from_model, to_model):
        to_model.load_state_dict(from_model.state_dict())

    def __getstate__(self):
        _dict = {k: v for k, v in self.__dict__.items()}
        del _dict['device']  # is not picklable
        del _dict['transitions']  # huge object (if you need the replay buffer, save it with np.save)
        return _dict
