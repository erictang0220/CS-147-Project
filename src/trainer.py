# class for the trainer
#   def train
#   def validation
#   def kfold_cross_validation?

import torch.optim as optim
import torch.nn as nn

class RNNTrainer:
    def __init__(self, rnn, learning_rate):
        self.rnn = rnn
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.rnn.parameters(), lr=self.learning_rate)

    def train(self, train_loader, num_epochs):
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                self.optimizer.zero_grad()

                outputs = self.rnn(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            print('Epoch: %d | Loss: %.4f' % (epoch+1, running_loss/len(train_loader)))

