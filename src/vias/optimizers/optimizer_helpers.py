from collections import UserList
from typing import List, Optional, Set

from deap.base import Fitness as deap_fitness

from src.vias.constraint_checkers.constraint_checker import Constraint


class Fitness(deap_fitness):
    def __init__(self, weights):
        self.weights = weights
        super().__init__()

    def __deepcopy__(self, memo):
        """Replace the basic deepcopy function with a faster one.

        It assumes that the elements in the :attr:`values` tuple are
        immutable and the fitness does not contain any other object
        than :attr:`values` and :attr:`weights`.
        """
        copy_ = self.__class__(self.weights)
        copy_.wvalues = self.wvalues
        return copy_


class Individual(UserList):
    def __init__(self, solution, weights):
        super().__init__(solution)
        self.fitness = Fitness(weights)
        self.constraints: Set[Constraint] = set()

    def __hash__(self):
        return hash(tuple(self))

    def satisfies_constraints(self):
        for constraint in self.constraints:
            if not constraint.satisfied:
                return False
        return True


class Population(UserList):
    def __init__(self, individuals: List[Individual], niche_numbers: Optional[List[int]] = None,
                 enable_niching: bool = False):
        super().__init__([])
        self._niching_enabled = enable_niching
        if niche_numbers is None:
            assert not self._niching_enabled, "If niching is enabled you have to provide niche numbers."
            niche_numbers = []
        else:
            assert len(individuals) == len(niche_numbers), "Niche numbers and individuals need the same size"
        self.niches = dict(zip([i for i in niche_numbers], [[] for _ in niche_numbers]))

        self.extend(individuals, niche_numbers)

    def append(self, item, niche_number: Optional[int] = None):
        assert isinstance(item, Individual), "Population can only constist of elements of type individual"
        super().append(item)
        if self._niching_enabled:
            assert 0 <= niche_number < len(self.niches), "Niche number does not fit initialized niches"
            assert niche_number is not None, "You have to provide a niche number"
            self.niches[niche_number].append(item)

    def extend(self, individuals, niche_numbers: Optional[List[int]] = None):
        if self._niching_enabled:
            assert niche_numbers is not None, "You have to provide a list of niche numbers"
            assert len(individuals) == len(niche_numbers), "Niche numbers and individuals need the same size"
        for ind_idx, individual in enumerate(individuals):
            if self._niching_enabled:
                self.append(individual, niche_number=niche_numbers[ind_idx])
            else:
                self.append(individual)

    def set_niches(self, new_niches: List[List[Individual]]):
        assert len(new_niches) == len(self.niches), "Number of new niches does not fit to number of current niches"
        self.clear_population()
        for niche_idx, niche in enumerate(new_niches):
            for individual in niche:
                self.append(individual, niche_idx)

    def clear_population(self):
        self.clear()
        for key in self.niches.keys():
            self.niches[key] = []

    def get_niches(self) -> List[List[Individual]]:
        if self._niching_enabled:
            return list(self.niches.values())
        else:
            return [self.data]

    def disable_niching(self):
        self._niching_enabled = False
        self.niches = {0: []}

# class cFitness(object):
#     def __init__(self):
#         self.valid = True
#         self.values = ()
#         self.weights = ()
#         self.wvalues = ()


# def convert_ind(ind):
#     c_fitness = cFitness()
#     c_fitness.valid = ind.fitness.valid
#     c_fitness.values = ind.fitness.values
#     c_fitness.weights = ind.fitness.weights
#     c_fitness.wvalues = ind.fitness.wvalues
#
#     c_ind = Individual(list(ind), ind.fitness.weights)
#     c_ind.fitness = c_fitness
#     c_ind.constraints = ind.constraints
#     try:
#         c_ind.strategy = list(ind.strategy)
#     except AttributeError:
#         pass
#     return c_ind
