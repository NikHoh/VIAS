from abc import ABC, abstractmethod

from vias.path import Path
from vias.scenario import Scenario


class Constraint:
    def __init__(self, name: str, value: float, critical: bool, satisfied: bool):
        self.name = name
        self.value = value
        self.critical = critical
        self.satisfied = satisfied


class ConstraintChecker(ABC):
    """This is an abstract class (~interface) for any ConstraintChecker (Constraint
    Simulator) that is implemented in the framework. Every constraint checker
    has to override the abstract methods of this interface in order to work properly
    in the framework."""

    def __init__(self, **kwargs):
        self.inputs = []
        self.processing = {}
        self.scenario = Scenario()
        self.parameters = kwargs["parameters"]
        self.constraint_str = kwargs["identifier_str"]

    def load_inputs_and_init(self, inputs_str: list[str], input_data_folder: str):
        self.load_inputs(inputs_str, input_data_folder)
        self.init()

    @abstractmethod
    def load_inputs(self, inputs_str: list[str], input_data_folder: str):
        raise NotImplementedError(
            "Users must define load_inputs() to use this base class."
        )

    @abstractmethod
    def init(self):
        raise NotImplementedError("Users must define init() to use this base class")

    @abstractmethod
    def check_constraint(self, path: Path) -> tuple[float, Constraint]:
        """Starts the conulation. The inputs for the conulation can be found in
        self.input. After the conulation
        its outputs should be written into self.quality_criteria. This is then
        returned by the method."""
        raise NotImplementedError(
            "Users must define check_constraint to use this base class."
        )
