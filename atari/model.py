import torch
import torch.nn as nn

floatX = 'float32'


def init_weights(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        torch.nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.)


class NatureNetwork(nn.Module):
    def __init__(self, state_shape=[84, 84], nb_channels=4, nb_actions=None):
        super(NatureNetwork, self).__init__()

        self.state_shape = state_shape
        self.nb_channels = nb_channels
        self.nb_actions = nb_actions
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=self.nb_channels, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.features.apply(init_weights)
        
        self.fc = nn.Sequential(
            nn.Linear(self._feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.nb_actions)
        )
        self.fc.apply(init_weights)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def _feature_size(self):
        return self.features(torch.zeros(1, 4, 84, 84)).view(-1).size(0)
