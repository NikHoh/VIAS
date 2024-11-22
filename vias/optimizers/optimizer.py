from abc import ABC, abstractmethod
from typing import Optional

try:
    # try importing the C version
    from deap.tools._hypervolume import hv as hv
except ImportError:
    # fallback on python version
    from deap.tools._hypervolume import pyhv as hv
from vias.scenario import Scenario

from vias.evaluator import Evaluator

try:
    import pickle5 as pickle
except ModuleNotFoundError:
    import pickle
from vias.config import get_config
import queue
from multiprocessing import Queue, Process
import numpy as np
from vias.pareto_handler import ParetoHandler

from vias.coder import Coder
from vias.data_manager import DataManager
from vias.console_manager import console


class OptimizationEndsInNiches(Exception):
    """Raised when the optimization terminates while niches have not resolved yet."""

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message


class Optimizer(ABC):
    """This is an abstract class (~interface) for any evolutionary Optimizer that is implemented in the framework. Every optimizer
    has to override the abstract methods of this interface in order to work properly in the framework."""

    def __init__(self, *args, **kwargs):
        config = get_config()
        self.path = DataManager().data_output_path
        self.name = kwargs["optimizer_identifier"]
        params = kwargs["optimizer_parameters"]
        self.evaluator: Optional[Evaluator] = kwargs["evaluator"]
        self.initial_paths = kwargs["initial_paths"]
        self.num_initial_paths = len(self.initial_paths)
        self.remote = config.remote
        self.seed = config.seed
        self.objectives = config.objectives
        self.num_objectives = len(self.objectives)

        self.scenario = Scenario()
        self.coder = Coder()
        self.num_iterations = params.num_iterations
        self.pareto_handler = ParetoHandler()

        self.generation_count: int = 0

        self.eval_count = 0
        self.raw_calc_time = 0
        self.eval_calc_time = 0

    def init_optimizer(self):
        pass

    @abstractmethod
    def optimize(self):
        """This method starts the optimizer. It should run for self.iterations iterations. The input of the optimizer
            can be found in self.coder.encoding_result. Objectives are given in self.objectives. During the execution the optimizer has to call
            self.evaluate() in order to get criteria values for the current result. Statistics can be saved by the help
            of self.add_statistic_line() and self.save_generation_statistic_plot(). At the end of this method the
            attributes self.final_solution and self.final_fitness should be written."""
        raise NotImplementedError('Users must define optimize to use this base class.')

    @abstractmethod
    def finish_after_optimization(self):
        """This method cleans stuff after the optimization"""
        raise NotImplementedError('Users must define clean_after_optimization to use this base class.')

    @abstractmethod
    def close_optimization(self):
        raise NotImplementedError('Users must define close_optimization to use this base class.')

    def evaluate(self, individual: np.ndarray):
        """This method will be called in the optimize() method. After the optimizer has provided an intermediate result/
            solution one or more quality criteria are calculated here, dependent on how many simulators had been added
            to the evaluator by the user. The provided code is ready to use but can be extended by quality/constraint
            checks for example."""
        path = self.coder.decode(individual)
        self.eval_count += 1
        criteria, constraints = self.evaluator.evaluate(path)
        val = []
        for objective in self.objectives:
            val.append(criteria[objective])
        return tuple(val), constraints

    def evaluate_parallel(self, pop):
        """Does the same like the evaluate() function but supposed to work in a parallel manner through multiple
        processes."""
        config = get_config()
        individuals_to_process = Queue()
        individuals_processed = Queue()
        ind_fit_con_tuple = []
        processes = []
        for i in range(len(pop)):
            individuals_to_process.put((i, pop[i]))
        before_qsize = individuals_to_process.qsize()
        # creating processes
        num_processes = config.cores
        for w in range(num_processes):
            p = Process(target=self.evaluate_single_ind, args=(individuals_to_process, individuals_processed, w))
            processes.append(p)
            p.start()

        while True:
            ind_fit_con_tuple.append(individuals_processed.get())
            if len(ind_fit_con_tuple) == before_qsize:
                break

        # completing processes IMPORTANT: queue must have been emptied before the join()
        for p in processes:
            p.join()

        self.eval_count += before_qsize
        assert len(ind_fit_con_tuple) == before_qsize, "Not all individuals had been processed, there is probabily" \
                                                       "something wrong with the multiprocessing."
        ind_fit_con_tuple.sort()  # sorts indices of individuals so that the order that may be disturbed due to multiprocessing is like before
        fitnesses = [fit[1] for fit in ind_fit_con_tuple]
        constraints = [fit[2] for fit in ind_fit_con_tuple]
        return fitnesses, constraints

    @abstractmethod
    def evaluate_single_ind(self, individuals_to_process, individuals_processed, idx):
        raise NotImplementedError('Users must define evaluate_single_ind to use this base class.')

    def get_num_unsatisfied_niches(self, pop):
        # console.log("Niching is enabled. Check if all constraints are met.")
        # all_constraints_satisfied = True
        num_unsatisfied_niches = len(pop.get_niches())
        for niche_id, niche in enumerate(pop.get_niches()):
            if niche_id == 0:
                num_unsatisfied_niches -= 1
                continue  # this is the niche, where all the straight line paths live, we won't expect them to satisfy all constraints, therefore we skip them
            niche_satisfied = False
            for individual_idx, ind in enumerate(niche):
                assert len(ind.constraints) > 0, "Constraint satisfaction stop criterion was chosen, but there are no constraints"
                all_constraints_satisfied = True
                for constraint in ind.constraints:
                    if not constraint.satisfied:
                        # console.log(
                        #     f"Constraint {constraint.name} with value {constraint.value} for individual {individual_idx} in niche {niche_id} is not satisfied.")
                        all_constraints_satisfied = False
                        break
                if all_constraints_satisfied:
                    niche_satisfied = True
                    break
            if niche_satisfied:
                num_unsatisfied_niches -= 1
        if num_unsatisfied_niches == 0:
            console.log("All niches are satisfied. Go on with complete population.")
        else:
            console.log(f"{num_unsatisfied_niches} are still unsatisfied. Go on with niches.")
        return num_unsatisfied_niches


def dump_queue(queue_to_be_dumped):
    """
    Empties all pending items in a queue and returns them in a list.
    """
    result = []
    while True:
        try:
            result.append(queue_to_be_dumped.get_nowait())
        except queue.Empty:
            break
    return result

