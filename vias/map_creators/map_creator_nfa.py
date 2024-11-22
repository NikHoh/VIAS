from vias.grid_map import GridMap
from vias.map_creators.map_creator import MapCreator
from vias.console_manager import console

class MapCreatorNfa(MapCreator):
    def create_map(self, map_name) -> GridMap:
        obstacle_map: GridMap = self.inputs["obstacle_map"]
        min_flight_height = self.scenario_info.min_flight_height
        nfa_map = self._get_map_blueprint(map_name, 3)
        minimum_flight_height_map: GridMap = self.inputs["minimum_flight_height_map"]

        # # create a min flight array indicating all cells below the min flight height with one
        # min_flight_array = np.zeros(nfa_map.grid_tensor.shape)
        # lay_idx = np.floor(min_flight_height / self.scenario_info.z_res).astype(int)
        # min_flight_array[:, :, 0:lay_idx] = 1.0  # set all elements below lay idx to 1.0  # TODO delete 4 commented lines

        console.log("Map {} is being rendered. This may take a while ...".format(nfa_map.name))

        # NFA grid_map consists of 1) the obstacles grid_map
        nfa_map.set_from_array(obstacle_map.grid_tensor)

        # ... the min flight height --> set all cells below min flight height to one
        nfa_map.add_from_array(minimum_flight_height_map.grid_tensor, operation_type='max')

        # and the legality grid_map
        # // TODO

        return nfa_map