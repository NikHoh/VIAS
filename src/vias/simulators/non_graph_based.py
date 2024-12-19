from vias.grid_graph import GridGraph
from vias.simulators.function_category import FunctionCategory


class NonGraphBased(FunctionCategory):
    def derive_grid_graph(self) -> GridGraph | None:
        return None
