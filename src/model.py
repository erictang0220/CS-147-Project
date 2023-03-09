import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        
        # Define the size of the RNN's input, hidden, and output layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Create the RNN's internal state variables
        self.Wxh = nn.Linear(input_size, hidden_size)
        
        self.Whh = nn.Linear(hidden_size, hidden_size)
        self.Why = nn.Linear(hidden_size, output_size)
        self.relu = nn.reLU()
        self.ELU = nn.ELU()
        self.softmax = nn.Softmax(dim=1)
    
    # TODO: no need for loss fuction

    def forward(self, x, h):
        # Apply the RNN's weights to the input and hidden state
        h = self.tanh(self.Wxh(x) + self.Whh(h))
        
        # Compute the output of the RNN
        y = self.softmax(self.Why(h))
        
        return y, h
    
    def init_hidden(self, batch_size):
        # Initialize the hidden state with zeros
        return torch.zeros(batch_size, self.hidden_size)
