import gym
import numpy as np
import time
import preprocessing
import click


ALL_GAMES = ['AirRaid', 'Alien', 'Amidar', 'Assault', 'Asterix', 'Asteroids',
             'Atlantis', 'BankHeist', 'BattleZone', 'BeamRider', 'Berzerk',
             'Bowling', 'Boxing', 'Breakout', 'Carnival', 'Centipede',
             'ChopperCommand', 'CrazyClimber', 'DemonAttack', 'DoubleDunk',
             'ElevatorAction', 'Enduro', 'FishingDerby', 'Freeway', 'Frostbite',
             'Gopher', 'Gravitar', 'Hero', 'IceHockey', 'Jamesbond',
             'JourneyEscape', 'Kangaroo', 'Krull', 'KungFuMaster',
             'MontezumaRevenge', 'MsPacman', 'NameThisGame', 'Phoenix',
             'Pitfall', 'Pong', 'Pooyan', 'PrivateEye', 'Qbert', 'Riverraid',
             'RoadRunner', 'Robotank', 'Seaquest', 'Skiing', 'Solaris',
             'SpaceInvaders', 'StarGunner', 'Tennis', 'TimePilot', 'Tutankham',
             'UpNDown', 'Venture', 'VideoPinball', 'WizardOfWor', 'YarsRevenge',
             'Zaxxon']



class AtariEnv(object):
    def __init__(self, game_name='pong', rendering=True, sticky_actions=True, frame_skip=4,
                 terminal_on_life_loss=False, screen_size=84):
        self.game_name = None
        for g in ALL_GAMES:
            if game_name.lower() in g.lower():
                self.game_name = g
                break
        if self.game_name is None:
            raise ValueError('Invalid game_name.')
        game_version = 'v0' if sticky_actions else 'v4'
        full_game_name = '{}NoFrameskip-{}'.format(self.game_name, game_version)
        self.frame_skip = frame_skip
        self.rendering = rendering
        if rendering:
            env = gym.make(full_game_name, render_mode='human')
        else:
            env = gym.make(full_game_name, render_mode='rgb_array')
        env = env.env
        self.env = preprocessing.AtariPreprocessing(env, frame_skip=frame_skip, terminal_on_life_loss=terminal_on_life_loss, screen_size=screen_size)
        self.nb_actions = self.env.action_space.n
        self.state_shape = [screen_size, screen_size]

    def render(self):
        if self.rendering == True:
            self.env.render()

    def step(self, action):
        s, r, term, info = self.env.step(action)
        self.render()
        return s.reshape(self.state_shape), r, term, info

    def get_lives(self):
        return self.env.lives
    
    def reset(self):
        return self.env.reset().reshape(self.state_shape)
    
    def get_state(self):
        return self.env._pool_and_resize().reshape(self.state_shape)
    
    def save_img(self, path):
        self.env.environment.ale.saveScreenPNG(path.encode())



@click.command()
@click.option('--human/--no-human', default=False, help='Activates the flat agent.')
@click.option('--game', default='pong', help='Game to play.')
@click.option('--pause', default=0., type=float, help='Pause in seconds.')
def play(human, game, pause):
    env = AtariEnv(game_name=game, rendering=True)
    term = False
    while not term:
        if not human:
            a = env.env.action_space.sample()
        else:
            print('-'*30)
            print('Num actions: ', env.nb_actions)
            a = input('Action >> ')
            if a == 'q':
                return
            elif not a.isdigit():
                continue
            a = int(a)
            if a > env.nb_actions:
                print('Invalid action.')
                continue
        obs, r, term, _ = env.step(a)
        print('action>>  {0:2d}  |  reward>>  {1:3.0f}  |  lives>>  {2}'.format(
            a, r, env.get_lives()))
        if pause != 0.:
            time.sleep(pause)


if __name__ == "__main__":
    play()
