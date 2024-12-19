from abc import ABC, abstractmethod
from typing import List

from src.vias.path import Path
from vias.scenario import Scenario


class Simulator(ABC):
    """This is an abstract class (~interface) for any Simulator that is implemented in the framework. Every simulator
        has to override the abstract methods of this interface in order to work properly in the framework."""

    def __init__(self, **kwargs):
        self.inputs = []
        self.processing = {}
        self.scenario = Scenario()
        self.parameters = kwargs["parameters"]
        self.objective_str = kwargs["identifier_str"]

    @abstractmethod
    def simulate(self, path: Path) -> float:
        """Starts the simulation. The inputs for the simulation can be found in self.input. After the simulation
        its outputs should be written into self.quality_criteria. This is then returned by the method."""
        raise NotImplementedError('Users must define simulate_traffic to use this base class.')

    def load_inputs_and_init(self, inputs_str: List[str], input_data_folder: str):
        self.load_inputs(inputs_str, input_data_folder)
        self.init()

    @abstractmethod
    def load_inputs(self, inputs_str: List[str], input_data_folder: str):
        raise NotImplementedError('Users must define load_inputs_and_init to be implemented in the base class.')

    @abstractmethod
    def init(self):
        raise NotImplementedError('Users must define init to be implemented in the base class.')




