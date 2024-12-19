import os
from typing import Optional


class DataManager:
    _instance: Optional["DataManager"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, **kwargs):
        # Only initialize if it hasn't been initialized yet
        if not hasattr(self, "_initialized"):
            self._initialized = True  # Prevent further initialization
            self.data_input_path = kwargs["input_path"]
            self.data_processing_path = kwargs["processing_path"]
            self.data_output_path = kwargs["output_path"]
            self.dijkstra_paths_path = os.path.join(
                self.data_processing_path, "dijkstra_paths"
            )
            if not os.path.exists(self.dijkstra_paths_path):
                os.makedirs(self.dijkstra_paths_path)
            self.merged_grid_graphs_path = os.path.join(
                self.data_processing_path, "grid_graphs"
            )
            if not os.path.exists(self.merged_grid_graphs_path):
                os.makedirs(self.merged_grid_graphs_path)
            self.optimized_paths_path = os.path.join(
                self.data_output_path, "optimized_paths"
            )
            if not os.path.exists(self.optimized_paths_path):
                os.makedirs(self.optimized_paths_path)

    @classmethod
    def reset_instance(cls):
        cls._instance = None
