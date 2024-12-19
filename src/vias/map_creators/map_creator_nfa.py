from vias.console_manager import console
from vias.grid_map import GridMap
from vias.map_creators.map_creator import MapCreator


class MapCreatorNfa(MapCreator):
    def create_map(self, map_name) -> GridMap:
        obstacle_map: GridMap = self.inputs["obstacle_map"]
        nfa_map = self._get_map_blueprint(map_name, 3)
        minimum_flight_height_map: GridMap = self.inputs["minimum_flight_height_map"]

        console.log(f"Map {nfa_map.name} is being rendered. This may take a while ...")

        # NFA grid_map consists of 1) the obstacles grid_map
        nfa_map.set_from_array(obstacle_map.grid_tensor)

        # ... the min flight height --> set all cells below min flight height to one
        nfa_map.add_from_array(
            minimum_flight_height_map.grid_tensor, operation_type="max"
        )

        return nfa_map
