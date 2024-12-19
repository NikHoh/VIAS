from typing import Concatenate

from vias.constraint_checkers.constraint_checker import Constraint, ConstraintChecker
from vias.path import Path
from vias.simulators.function_category import FunctionCategory
from vias.simulators.simulator import Simulator


class Evaluator:
    """This class is used by an optimizer to calculate the quality criteria for a
    solution the optimizer has found. In
    general multiple quality criteria can be calculated by multiple simulators. All
    simulators are stored in a list.
    Besides the set of all simulators inputs are saved as dict (
    self.simulator_inputs) as well as the simulators outputs
    (self.criteria). Caution: the output of one simulator can also be the input of
    another simulator. If this is the
    case the simulator that produces input has always to be declared before the
    others."""

    def __init__(self):
        self.simulators: list[Concatenate[Simulator, FunctionCategory]] = []
        self.constraint_checkers: list[
            Concatenate[ConstraintChecker, FunctionCategory]
        ] = []
        self.criteria = {}
        self.simulator_inputs = {}
        self.constraint_checker_inputs = {}
        self.num_simulators = 0
        self.num_constraint_checkers = 0

    def add_constraint_checker(self, constraint_checker: ConstraintChecker):
        """Adds a new simulator to the evaluator object."""
        self.constraint_checkers.append(constraint_checker)
        self.num_constraint_checkers += 1

    def add_simulator(self, simulator: Simulator):
        """Adds a new simulator to the evaluator object."""
        self.simulators.append(simulator)
        self.num_simulators += 1

    def evaluate(
        self, path: Path, norm_to_bee_line=True
    ) -> tuple[dict[str, float], list[Constraint]]:
        """Initializes and starts all simulators that have been added to the
        Evaluator class object."""

        if norm_to_bee_line:
            bee_line_distance = path.bee_line_distance
        else:
            bee_line_distance = 1.0

        # // TODO parallelize this with concurrent.futures ThreadPoolExecutor
        for simulator in self.simulators:
            cost = simulator.simulate(path)
            # the simulation results are added to the quality criteria dict
            self.criteria[simulator.objective_str] = cost

        list_of_constraints = []
        sum_of_punishments = 0
        for constraint_checker in self.constraint_checkers:
            punishment, constraint = constraint_checker.check_constraint(path)
            sum_of_punishments += punishment
            list_of_constraints.append(constraint)

        # // TODO maybe think about using some numpy array instead of iterating over
        #  lists here
        # add all punishments from constraints to all criteria
        for key in self.criteria:
            self.criteria[key] += sum_of_punishments
            self.criteria[key] /= (
                bee_line_distance  # normalization to bee_line_distance
            )

        return self.criteria, list_of_constraints
