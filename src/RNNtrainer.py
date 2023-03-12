# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.model_selection import KFold

# class RNNTrainer:
#     """
#     A class for training an RNN using PyTorch.
#     """

#     def __init__(self, rnn, learning_rate):
#         """
#         Initialize the trainer with an RNN model and a learning rate.
#         Parameters:
#         - rnn (nn.Module): The RNN model to train.
#         - learning_rate (float): The learning rate for optimization.
#         """
#         self.rnn = rnn
#         self.learning_rate = learning_rate
#         self.criterion = nn.CrossEntropyLoss()
#         self.optimizer = optim.Adam(self.rnn.parameters(), lr=self.learning_rate)

#     def train(self, train_loader, num_epochs):
#         """
#         Train the RNN model using a DataLoader for a fixed number of epochs.
#         Parameters:
#         - train_loader (DataLoader): The DataLoader containing the training dataset.
#         - num_epochs (int): The number of epochs to train for.
#         """
#         for epoch in range(num_epochs):
#             running_loss = 0.0
#             self.rnn.train() # Set the model to train mode
#             for i, data in enumerate(train_loader, 0):
#                 inputs, labels = data
#                 self.optimizer.zero_grad()

#                 outputs = self.rnn(inputs)
#                 loss = self.criterion(outputs, labels)
#                 loss.backward()
#                 self.optimizer.step()

#                 running_loss += loss.item()

#             print('Epoch: %d | Train Loss: %.4f' % (epoch+1, running_loss/len(train_loader)))

#     def validate(self, valid_loader):
#         """
#         Evaluate the RNN model on a validation DataLoader and return the accuracy.
#         Parameters:
#         - valid_loader (DataLoader): The DataLoader containing the validation dataset.
#         Returns:
#         - accuracy (float): The accuracy of the model on the validation dataset.
#         """
#         self.rnn.eval() # Set the model to evaluation mode
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             for data in valid_loader:
#                 inputs, labels = data
#                 outputs = self.rnn(inputs)
#                 _, predicted = torch.max(outputs.data, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()

#         accuracy = 100 * correct / total
#         return accuracy

#     def k_fold_cross_validation(self, train_dataset, k=5):
#         """
#         Perform k-fold cross-validation on the RNN model using a training dataset.

#         Parameters:
#         - train_dataset (Dataset): The training dataset to use for k-fold cross-validation.
#         - k (int, optional): The number of folds for cross-validation. Default is 5.

#         Returns:
#         - accuracies (List[float]): A list of accuracies for each fold of cross-validation.
#         """
#         kf = KFold(n_splits=k)
#         accuracies = []
#         for train_index, valid_index in kf.split(train_dataset):
#             train_subset = torch.utils.data.Subset(train_dataset, train_index)
#             valid_subset = torch.utils.data.Subset(train_dataset, valid_index)

#             train_loader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)
#             valid_loader = torch.utils.data.DataLoader(valid_subset, batch_size=64, shuffle=False)

#             self.train(train_loader, num_epochs=5)
#             accuracy = self.validate(valid_loader)
#             accuracies.append(accuracy)
        
#         return accuracies
