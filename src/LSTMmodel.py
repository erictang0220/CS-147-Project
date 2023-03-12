import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=22, hidden_dim=20, output_dim=4):
        super(LSTMClassifier, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, h_0=None, c_0=None):
        # x has shape (batch_size, input_dim, timesteps)
        # h_0 has shape (num_layers * num_directions, batch_size, hidden_dim)
        # c_0 has shape (num_layers * num_directions, batch_size, hidden_dim)

        z = x.permute(0, 2, 1) # (batch_size, input_dim, timesteps) -> (batch_size, timesteps, input_dim)
        if h_0 is None and c_0 is None:
            z, (hn, cn) = self.lstm(z) # (batch_size, timesteps, input_dim) -> (batch_size, timesteps, hidden_dim)
        else:
            # We detach h_0 and c_0 (not required, just recommended) so that the computational graph does not extend too far.
            z, (hn, cn) = self.lstm(z, (h_0.detach(), c_0.detach())) # (batch_size, timesteps, input_dim) -> (batch_size, timesteps, hidden_dim)
        # hn and cn are the stored hidden states and cell states after computation
        
        out = self.fc(z[:, -1, :])
        # add ReLU activation function
        out = F.relu(out)
        return out
