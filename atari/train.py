import os
import sys
import click
import yaml
import numpy as np
from rl import RL
from experiment import DQNExperiment
import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = ROOT_DIR
sys.path.append(ROOT_DIR)

np.set_printoptions(suppress=True, linewidth=200, precision=2)


@click.command()
@click.option('--domain', '-d', default='catch', help="'catch' or 'atari'")
@click.option('--options', '-o', multiple=True, nargs=2, type=click.Tuple([str, str]))
def run(domain, options):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    cfg_file = os.path.join(dir_path, 'config_' + domain + '.yaml')
    params = yaml.safe_load(open(cfg_file, 'r'))

    # replacing params with command line options
    for opt in options:
        assert opt[0] in params
        dtype = type(params[opt[0]])
        if dtype == bool:
            new_opt = False if opt[1] != 'True' else True
        else:
            new_opt = dtype(opt[1])
        params[opt[0]] = new_opt

    print('\n')
    print('Parameters ')
    for key in params:
        print(key, params[key])
    print('\n')

    np.random.seed(params['random_seed'])
    torch.manual_seed(params['random_seed'])
    random_state = np.random.RandomState(params['random_seed'])
    device = torch.device(params["device"])

    if domain == 'atari':
        from atari import AtariEnv
        env = AtariEnv(game_name=params['game_name'], rendering=params['test'], sticky_actions=params['sticky_actions'], 
                       frame_skip=params['frame_skip'], terminal_on_life_loss=params['terminal_on_life_loss'], screen_size=params['screen_size'])
    else:
        raise ValueError('domain must be either catch or atari')
    
    for ex in range(params['num_experiments']):
        print('\n')
        print('>>>>> Experiment ', ex, ' >>>>>')
        print('\n')
        network_size = 'small' if env.state_shape[0] < 80 else params['network_size']  # always use small for catch
        ai = RL(state_shape=env.state_shape, nb_actions=env.nb_actions, action_dim=params['action_dim'],
                reward_dim=params['reward_dim'], history_len=params['history_len'], gamma=params['gamma'], 
                learning_rate=params['learning_rate'], epsilon=params['epsilon'], final_epsilon=params['final_epsilon'],
                test_epsilon=params['test_epsilon'], annealing_steps=params['annealing_steps'], minibatch_size=params['minibatch_size'],
                replay_max_size=params['replay_max_size'], update_freq=params['update_freq'], 
                learning_frequency=params['learning_frequency'], ddqn=params['ddqn'], network_size=network_size, 
                normalize=params['normalize'], event=params['event'], sided_Q=params['sided_Q'], rng=random_state, device=device)
        if params['test']:  # note to pass correct folder name
            network_weights_file = os.path.join(ROOT_DIR, 'results', params['folder_name'],
                                                'q_network_weights.pt')
            ai.load_weights(weights_file_path=network_weights_file)

        expt = DQNExperiment(env=env, ai=ai, episode_max_len=params['episode_max_len'], annealing=params['annealing'],
                             history_len=params['history_len'], max_start_nullops=params['max_start_nullops'],
                             replay_min_size=params['replay_min_size'], test_epsilon=params['test_epsilon'],
                             folder_location=os.path.join(OUTPUT_DIR, params['folder_location']),
                             folder_name=params['folder_name'], score_window_size=100, rng=random_state)
        env.reset()
        if not params['test']:
            with open(expt.folder_name + '/config.yaml', 'w') as y:
                yaml.safe_dump(params, y)  # saving params for reference
            expt.do_epochs(number=params['num_epochs'], is_learning=params['is_learning'],
                           steps_per_epoch=params['steps_per_epoch'], is_testing=params['is_testing'],
                           steps_per_test=params['steps_per_test'])
        else:
            if params['human']:
                expt.do_human_episode()
            else:
                expt.evaluate(number=1)


if __name__ == '__main__':
    run()
