import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, in_shape, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels  = in_shape[0],
                out_channels = 32,
                kernel_size  = 8,
                stride       = 4
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels  = 32,
                out_channels = 64,
                kernel_size  = 4,
                stride       = 2
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels  = 64,
                out_channels = 64,
                kernel_size  = 3,
                stride       = 1
            ),
            nn.ReLU(),
            nn.Flatten()
        )
        conv_out_size = self.conv(
                torch.zeros(1, *in_shape)
        ).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(
                in_features  = conv_out_size,
                out_features = 512
            ),
            nn.ReLU(),
            nn.Linear(
               in_features   = 512,
               out_features  = n_actions 
            )
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    model = DQN(
        in_shape  = (4, 84, 84),
        n_actions = 6
    )
    print(model)