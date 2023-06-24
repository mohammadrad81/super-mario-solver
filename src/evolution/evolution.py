from typing import Dict, List, Set, Tuple
from abc import ABC


class Evolution(ABC):

    def initial_population(self, *args, **kwargs):
        pass

    def evaluate(self, *args, **kwargs):
        pass

    def selection(self, *args, **kwargs):
        pass

    def crossover(self, *args, **kwargs):
        pass

    def mutate(self, *args, **kwargs):
        pass
