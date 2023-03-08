# the main():

from src.model import # our model
from src.dataloader import EEGData
from src.trainer import Trainer

if __name__ == "__main__":
    # run the trainer 
    trainer = Trainer() #???
    trainer.train_and_test()