from abc import ABC, abstractmethod

from vias.grid_graph import GridGraph


class FunctionCategory(ABC):
    @abstractmethod
    def derive_grid_graph(self) -> GridGraph | None:
        raise NotImplementedError(
            "Users must implement derive_grid_graph() to use this base class."
        )
