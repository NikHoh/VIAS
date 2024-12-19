from collections import UserList

from deap.base import Fitness as deap_fitness

from vias.constraint_checkers.constraint_checker import Constraint


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
        self.constraints: set[Constraint] = set()

    def __hash__(self):
        return hash(tuple(self))

    def satisfies_constraints(self):
        return all(constraint.satisfied for constraint in self.constraints)


class Population(UserList):
    def __init__(
        self,
        individuals: list[Individual],
        niche_numbers: list[int] | None = None,
        enable_niching: bool = False,
    ):
        super().__init__([])
        self._niching_enabled = enable_niching
        if niche_numbers is None:
            assert not self._niching_enabled, (
                "If niching is enabled you have to " "provide niche numbers."
            )
            niche_numbers = []
        else:
            assert len(individuals) == len(
                niche_numbers
            ), "Niche numbers and individuals need the same size"
        self.niches: dict[int, list[Individual]] = dict(
            zip(niche_numbers, [[] for _ in niche_numbers], strict=False)
        )

        self.extend(individuals, niche_numbers)

    def append(self, item, niche_number: int | None = None):
        assert isinstance(item, Individual), (
            "Population can only constist of elements of " "type individual"
        )
        super().append(item)
        if self._niching_enabled:
            assert niche_number is not None, "You have to provide a niche number"
            assert (
                0 <= niche_number < len(self.niches)
            ), "Niche number does not fit initialized niches"
            self.niches[niche_number].append(item)

    def extend(self, individuals, niche_numbers: list[int] | None = None):
        if self._niching_enabled:
            assert niche_numbers is not None, (
                "You have to provide a list of niche " "numbers"
            )
            assert len(individuals) == len(
                niche_numbers
            ), "Niche numbers and individuals need the same size"
            for ind_idx, individual in enumerate(individuals):
                self.append(individual, niche_number=niche_numbers[ind_idx])
        else:
            for individual in individuals:
                self.append(individual)

    def set_niches(self, new_niches: list[list[Individual]]):
        assert len(new_niches) == len(
            self.niches
        ), "Number of new niches does not fit to number of current niches"
        self.clear_population()
        for niche_idx, niche in enumerate(new_niches):
            for individual in niche:
                self.append(individual, niche_idx)

    def clear_population(self):
        self.clear()
        for key in self.niches:
            self.niches[key] = []

    def get_niches(self) -> list[list[Individual]]:
        if self._niching_enabled:
            return list(self.niches.values())
        else:
            return [self.data]

    def disable_niching(self):
        self._niching_enabled = False
        self.niches = {0: []}
