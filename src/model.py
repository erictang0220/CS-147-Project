import torch
import torch.nn as nn

class RNNClassifier(nn.Module):
    def __init__(self, input_dim=22, hidden_dim=20, output_dim=4):
        super(RNNClassifier, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, h_0=None):
        # x has shape (batch_size, input_dim, timesteps)
        # h_0 has shape (D*num_layers, hidden_dim)

        z = x.permute(0, 2, 1) # (batch_size, input_dim, timesteps) -> (batch_size, timesteps, input_dim)
        if h_0 is None:
            z, hn = self.rnn(z) # (batch_size, timesteps, input_dim) -> (batch_size, timesteps, hidden_dim)
        else:
            # We detach h_0 (not required, just recommended) so that the computational graph does not extend too far.
            z, hn = self.rnn(z, h_0.detach()) # (batch_size, timesteps, input_dim) -> (batch_size, timesteps, hidden_dim)
        # hn is the stored hidden state after computation
        out = self.fc(z[:, -1, :])
        return out