import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime as dt

from abc import ABC, abstractmethod


class Model(ABC):
    """Base Class for Machine Learning methods."""

    def __init__(self, path, name):
        self.path = path
        self.name = name
        pass


    @abstractmethod
    def create_model(self):
        pass


    @abstractmethod
    def train(self):
        pass


    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def predict(self):
        pass


    @abstractmethod
    def save(self):
        pass


    @abstractmethod
    def load(self):
        pass


    @classmethod
    def _plot_history(cls, history, path, name):
        pd.DataFrame(history.history).plot(figsize=(8, 5))
        plt.grid(True)
        plt.gca().set_ylim(0, 1)
        plt.savefig(
            '{}/{}/{}.png'.format(path, name, cls.get_now()))


    @staticmethod
    def get_now():
        return dt.now().strftime("%Y-%m-%dT%H:%M:%S.%f")


    @staticmethod
    def create_history_folder(path, name):
        path = '{}/{}'.format(path, name)
        if not os.path.isdir(path):
            os.makedirs(path)
