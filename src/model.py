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
        
        # Define activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x, h):
        """
        Performs forward pass through the RNN.
        Parameters:
        x (torch.Tensor): The input tensor with shape (batch_size, input_size).
        h (torch.Tensor): The hidden state tensor with shape (batch_size, hidden_size).
        Returns:
        y (torch.Tensor): The output tensor with shape (batch_size, output_size).
        h_new (torch.Tensor): The new hidden state tensor with shape (batch_size, hidden_size).
        """
        # Apply the RNN's weights to the input and hidden state
        h_new = self.relu(self.Wxh(x) + self.Whh(h))
        
        # Compute the output of the RNN
        y = self.softmax(self.Why(h_new))
        
        return y, h_new
    
    def init_hidden(self, batch_size):
        """
        Initializes the hidden state with zeros.
        Parameters:
        batch_size (int): The size of the batch.
        Returns:
        torch.Tensor: The initial hidden state tensor with shape (batch_size, hidden_size).
        """
        return torch.zeros(batch_size, self.hidden_size)