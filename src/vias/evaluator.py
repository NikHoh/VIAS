from typing import List, Dict, Tuple

from src.vias.constraint_checkers.constraint_checker import Constraint
from src.vias.path import Path
from src.vias.simulators.simulator import Simulator


# from vias.constraint_checkers.constraint_checker import ConstraintChecker


class Evaluator(object):
    """This class is used by an optimizer to calculate the quality criteria for a solution the optimizer has found. In
    general multiple quality criteria can be calculated by multiple simulators. All simulators are stored in a list.
    Besides the set of all simulators inputs are saved as dict (self.simulator_inputs) as well as the simulators outputs
    (self.criteria). Caution: the output of one simulator can also be the input of another simulator. If this is the
    case the simulator that produces input has always to be declared before the others."""

    def __init__(self):
        self.simulators: List[Simulator] = []
        self.constraint_checkers: List[Simulator] = []
        self.criteria = {}
        self.simulator_inputs = {}
        self.constraint_checker_inputs = {}
        self.num_simulators = 0
        self.num_constraint_checkers = 0

    def add_constraint_checker(self, conulator: Simulator):
        """Adds a new simulator to the evaluator object."""
        self.constraint_checkers.append(conulator)
        self.num_constraint_checkers += 1

    def add_simulator(self, simulator: Simulator):
        """Adds a new simulator to the evaluator object."""
        self.simulators.append(simulator)
        self.num_simulators += 1

    # def add_simulator_input(self, input_name: str, input_data):
    #     """Adds input for simulators to the evaluator object."""
    #     if input_data is None:
    #         return
    #     self.simulator_inputs[input_name] = input_data

    # def add_constraint_checker_input(self, input_name: str, input_data):
    #     """Adds input for conulators to the evaluator object."""
    #     if input_data is None:
    #         return
    #     self.constraint_checker_inputs[input_name] = input_data

    def evaluate(self, path: Path, norm_to_bee_line=True, normalization=None) -> Tuple[Dict[str, float], List[Constraint]]:
        """Initializes and starts all simulators that have been added to the Evaluator class object."""

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

        # // TODO maybe think about using some numpy array instead of iterating over lists here
        # add all punishments from constraints to all criteria
        for idx, key in enumerate(self.criteria.keys()):
            self.criteria[key] += sum_of_punishments
            self.criteria[key] /= bee_line_distance  # normalization to bee_line_distance

        return self.criteria, list_of_constraints


# // TODO delete these functions

#
#
# def init_simulation(self, simulator: Simulator):
#     """Before a simulator begins its work the different data for its input is gathered either from
#     self.simulator_inputs or from self.criteria as the input of one simulator could also be derived from the
#     output of another one."""
#     for input_name in simulator.inputs.keys():
#         if input_name in self.criteria.keys():
#             simulator[input_name] = self.criteria[input_name]
#         elif input_name in self.simulator_inputs.keys():
#             simulator.inputs[input_name] = self.simulator_inputs[input_name]
#         else:
#             assert 1 == 0, "Simulator input {} was not found".format(input_name)
#
# def init_constraint_checking(self, constraint_checker: Simulator):
#     """Before a conulator begins its work the different data for its input is gathered either from
#     self.constraint_checker_inputs."""
#     for input_name in constraint_checker.inputs.keys():
#         constraint_checker.inputs[input_name] = self.constraint_checker_inputs[input_name]
