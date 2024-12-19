import os

from src.vias.data_manager import DataManager
from pymoo.indicators.hv import Hypervolume
import math
import os
import pickle
import sys
from datetime import datetime as dt
from typing import List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.igd import IGD
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.normalization import normalize

from vias.config import get_config
from src.vias.data_manager import DataManager
from vias.optimizers.optimizer_helpers import Population, Individual
from vias.utils.helpers import is_stdout_redirected
from vias.utils.math_functions import EPS

try:
    # try importing the C version
    from deap.tools._hypervolume import hv as hv
except ImportError:
    # fallback on python version
    from deap.tools._hypervolume import pyhv as hv

from vias.console_manager import console, live
from rich.table import Table


def convergence(first_front, optimal_front):
    """This is mainly a copy from the deap framework working for lists of fitnesses:
    Given a Pareto front `first_front` and the optimal Pareto front,
    this function returns a metric of convergence
    of the front as explained in the original NSGA-II article by K. Deb.
    The smaller the value is, the closer the front is to the optimal one.
    """
    distances = []

    for solution in first_front:
        distances.append(float("inf"))
        for opt_solution in optimal_front:
            dist = 0.
            for i in range(len(opt_solution)):
                dist += (solution[i] - opt_solution[i]) ** 2
            if dist < distances[-1]:
                distances[-1] = dist
        distances[-1] = math.sqrt(distances[-1])

    return sum(distances) / len(distances)


def diversity(first_front, first, last):
    """his is mainly a copy from the deap framework working for lists of fitnesses
    Given a Pareto front `first_front` and the two extreme points of the
    optimal Pareto front, this function returns a metric of the diversity
    of the front as explained in the original NSGA-II article by K. Deb.
    The smaller the value is, the better the front is.
    """
    df = math.hypot(first_front[0][0] - first[0],
                    first_front[0][1] - first[1])
    dl = math.hypot(first_front[-1][0] - last[0],
                    first_front[-1][1] - last[1])
    dt = [math.hypot(first[0] - second[0],
                     first[1] - second[1])
          for first, second in zip(first_front[:-1], first_front[1:])]

    if len(first_front) == 1:
        return df + dl

    dm = sum(dt) / len(dt)
    di = sum(abs(d_i - dm) for d_i in dt)
    delta = (df + dl + di) / (df + dl + len(dt) * dm)
    return delta


class ParetoHandler:
    """This is an  class for managing data of multi-objective based optimizers. Saving individual's fitness per
    iteration, plotting the pareto front and so on."""

    def __init__(self, path_to_init=""):
        self.non_dom_sort = NonDominatedSorting()
        self._hyp_vol = None
        self.data = {}

        self.expected_columns = ["time", "n_gen", "n_nds", "n_eval", "F", "non_dom_idx", "HV", "nadir", "ideal",
                                 "delta_f",
                                 "delta_nadir", "delta_ideal", "delta_hv", "X", "n_cv"]
        self.init(path_to_init)
        self._first_update = True
        self._first_statistics_line = True

    def is_empty(self):
        for key in self.expected_columns:
            if "delta" in key:
                if len(self.data[key]) != 1:
                    return False
            else:
                if len(self.data[key]) != 0:
                    return False
        return True

    def insert_F_X_dict(self, F_X_dict: dict):
        assert self.is_empty(), "ParetoHandler needs to be empty in order to load F_X_data"

        objective_function_values = list(F_X_dict.keys())
        individuals = list(F_X_dict.values())
        objective_function_values = np.array([np.array(el) for el in objective_function_values])

        self.data["F"].append(objective_function_values)
        self.data["X"].append(individuals)

        objective_values = np.array(self.data["F"][-1])
        non_dominated_indices = self.non_dom_sort.do(objective_values, only_non_dominated_front=True)

        self.data["non_dom_idx"].append(non_dominated_indices)
        self.data["n_nds"].append(len(non_dominated_indices))



    @property
    def hyp_vol(self):
        if self._hyp_vol is None:
            assert len(self.data["F"]) > 0, "hypervolume calculation was triggered before first individuals are known"
            reference_point = np.max(self.data["F"][-1], axis=0) * 1.2
            self._hyp_vol = Hypervolume(ref_point=reference_point, pf=self.data["F"][0], nds=False)
        return self._hyp_vol

    def init(self, path_to_init):

        if path_to_init == "":
            for key in self.expected_columns:
                self.data[key] = []
            self.data["delta_f"].append(np.nan)
            self.data["delta_nadir"].append(np.nan)
            self.data["delta_ideal"].append(np.nan)
            self.data["delta_hv"].append(np.nan)
        else:
            with open(path_to_init, 'rb') as f:
                self.data = pickle.load(f)
            assert isinstance(self.data, dict), "Input object is expected to be a dictionary"
            assert len(self.data.keys()) == len(
                self.expected_columns), f"Input dictionary is expected to have {len(self.expected_columns)} keys."
            assert set(self.data.keys()) == set(
                self.expected_columns), f"Input dictionary is expected to have the keys {self.expected_columns}."

    def update_statistic(self, generation_count: int, pop: Population, n_eval: int):
        """Calculates important metrics of a whole population that are minimum achieved fitness, maximum achieved
        fitness, mean and standard deviation of all fitnesses of one population."""
        # take only individuals that satisfy the constraints
        self.data["time"].append(dt.now())
        self.data["n_eval"].append(n_eval)
        self.data["n_gen"].append(generation_count)
        individuals = [ind for ind in pop if ind.satisfies_constraints()]
        self.data["X"].append(individuals)
        self.data["n_cv"].append(len(pop) - len(individuals))
        objective_values = np.array([list(ind.fitness.values) for ind in individuals])
        self.data["F"].append(objective_values)
        non_dominated_indices = self.non_dom_sort.do(objective_values, only_non_dominated_front=True)
        non_dominated_objective_values = objective_values[non_dominated_indices]
        self.data["non_dom_idx"].append(non_dominated_indices)
        self.data["n_nds"].append(len(non_dominated_indices))

        # mean = np.mean(objective_values, axis=0)
        # std = np.std(objective_values, axis=0)
        curr_ideal = np.min(objective_values, axis=0)
        curr_nadir = np.max(objective_values, axis=0)
        self.data["nadir"].append(curr_nadir)
        self.data["ideal"].append(curr_ideal)

        HV = self.hyp_vol.do(non_dominated_objective_values)
        self.data["HV"].append(HV)
        if not self._first_update:
            norm = curr_nadir - curr_ideal
            norm[norm < EPS] = 1.0
            prev_ideal = self.data["ideal"][-2]
            prev_nadir = self.data["nadir"][-2]
            prev_F = self.data["F"][-2]
            prev_hv = self.data["HV"][-2]
            delta_hv = HV - prev_hv
            delta_ideal = np.max(np.abs((curr_ideal - prev_ideal) / norm))
            delta_nadir = np.max(np.abs((curr_nadir - prev_nadir) / norm))
            current_norm_F = normalize(objective_values, curr_ideal, curr_nadir)
            prev_norm_F = normalize(prev_F, curr_ideal, curr_nadir)
            # calculate IGD from one to another
            delta_f = IGD(current_norm_F).do(prev_norm_F)

            self.data["delta_f"].append(delta_f)
            self.data["delta_nadir"].append(delta_nadir)
            self.data["delta_ideal"].append(delta_ideal)
            self.data["delta_hv"].append(delta_hv)

        self._first_update = False
        self.print_statistics()

    def get_extreme_solution(self, idx_objective) -> Tuple[float, Individual]:
        """Return the extreme objective function value given a objective index and the respective individual"""
        extr_idx = self._get_extreme_point_index(idx_objective)
        non_dom_objective_function_values = self.get_pareto_front()
        non_dom_individuals = self.get_pareto_solutions()
        return non_dom_objective_function_values[extr_idx][idx_objective], non_dom_individuals[extr_idx]

    def get_extreme_point_path(self, idx_objective):
        extr_idx = self._get_extreme_point_index(idx_objective)
        return [self.data["X"][-1][el] for el in self.data["non_dom_idx"][-1]][extr_idx]

    def get_extreme_point(self, idx_objective):
        extr_idx = self._get_extreme_point_index(idx_objective)
        return self.data["F"][-1][self.data["non_dom_idx"][-1]][extr_idx]

    def _get_extreme_point_index(self, idx_objective):
        non_dom_objective_function_values = self.get_pareto_front()
        extr_idx = np.argmin(non_dom_objective_function_values[:, idx_objective])
        return extr_idx

    def get_nadir_point(self):
        return self.data["nadir"][-1]

    def get_ideal_point(self):
        return self.data["ideal"][-1]

    def get_knee_point(self):
        """Return the knee point variable and its fitness value"""
        knee_idx = self._get_knee_point_index()
        return self.data["F"][-1][knee_idx]

    def get_knee_point_path(self):
        knee_idx = self._get_knee_point_index()
        return self.data["X"][-1][knee_idx]

    def _get_knee_point_index(self):
        normalized_data = np.array(self.data["F"])
        normalized_data = (normalized_data - np.min(normalized_data, axis=1))/ (np.max(normalized_data, axis=1) - np.min(normalized_data, axis=1))
        knee_idx = np.argmin(np.linalg.norm(normalized_data[-1], axis=1))
        return knee_idx

    def get_hypervolume(self):
        return self.data["hv"][-1]

    def get_F_X_dict(self):
        return dict(zip([tuple(row) for row in self.data["F"][-1]], self.data["X"][-1]))

    def get_objective_function_values(self, obj_idx: Optional[List] = None):
        # returns the current objective function values (can also include dominated solutions)
        if obj_idx is None:
            return self.data["F"][-1]
        else:
            return self.data["F"][-1][:, obj_idx]

    def get_pareto_front(self):
        # returns current non-dominated objective function values
        return self.data["F"][-1][self.data["non_dom_idx"][-1]]

    def get_delta_f(self):
        return self.data["delta_f"][-1]

    def get_delta_nadir(self):
        return self.data["delta_nadir"][-1]

    def get_delta_ideal(self):
        return self.data["delta_ideal"][-1]

    def get_pareto_solutions(self):
        # returns the solutions (individuals) for all non-dominated objective function values
        return [self.data["X"][-1][el] for el in self.data["non_dom_idx"][-1]]

    def print_statistics(self):
        output_redirectd = is_stdout_redirected()

        headers = ["n_gen", "n_eval", "n_nds", "n_cv", "delta_f", "ideal", "delta_ideal", "nadir", "delta_nadir"]
        if not output_redirectd or self._first_statistics_line:
            table = Table(title="Optimization statistics")
            self._first_statistics_line = False
        else:
            table = Table(title=None, show_header=False)
        for header in headers:
            table.add_column(header, justify="center")

        if not output_redirectd:
            columns = [self.data[header][-10:] for header in headers]  # the latest 10 rows
        else:
            columns = [self.data[header][-1:] for header in headers]  # the last row

        rows = list(zip(*columns))
        for row in rows:
            table.add_row(*[str(x) for x in row])
        if not output_redirectd:
            live.update(table, refresh=True)
        else:
            console.print(table)
        sys.stdout.flush()  # Ensure immediate output

    def save_and_clean(self):
        self.save_generation_statistic_plot()
        with open(os.path.join(DataManager().data_output_path, "F_X_dict.pkl"), 'wb') as handle:
            pickle.dump(self.get_F_X_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(DataManager().data_output_path, "optimizer_statistics.pkl"), 'wb') as handle:
            pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # save it as csv as well (redundant)
        # df = pd.DataFrame(self.data)
        # Save the DataFrame to a CSV file
        # df.to_csv(os.path.join(DataManager().data_output_path, "optimizer_statistics.csv"), index=False)

    def save_generation_statistic_plot(self, show_figure=False):
        """This method can be used during each iteration of the optimizer to save an updated version of the result's
        figure plot."""
        config = get_config()
        data = self.data
        n_gen = data["n_gen"]

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        # plot HV
        fig.add_trace(
            go.Scatter(x=n_gen, y=data["HV"], mode="lines", name="Hypervolume"), secondary_y=False)
        fig.add_trace(go.Scatter(x=n_gen, y=data["delta_hv"], mode="lines", name="delta_hv"), secondary_y=True)
        fig.add_trace(go.Scatter(x=n_gen, y=data["delta_f"], mode="lines", name="delta_f/current IGD"),
                      secondary_y=True)
        fig.add_trace(go.Scatter(x=n_gen, y=data["delta_nadir"], mode="lines", name="delta_nadir"), secondary_y=True)
        fig.add_trace(go.Scatter(x=n_gen, y=data["delta_ideal"], mode="lines", name="delta_ideal"), secondary_y=True)
        fig.add_trace(go.Scatter(x=n_gen, y=data["n_cv"], mode="lines", name="# constraint violations"),
                      secondary_y=True)
        for i in range(0, data["nadir"][-1].size):
            fig.add_trace(go.Scatter(x=n_gen, y=[el[i] for el in data["nadir"]], mode="lines",
                                     name=f"Nadir axis {i + 1}"), secondary_y=False)
            fig.add_trace(go.Scatter(x=n_gen, y=[el[i] for el in data["ideal"]], mode="lines",
                                     name=f"Ideal axis {i + 1}"), secondary_y=True)

        fig.update_layout(title=dict(text=f"Optimization Callback Data"))
        fig.update_yaxes(title="Values")
        fig.update_xaxes(title="Generation")

        save_path = os.path.join(DataManager().data_output_path, 'optimizer_statistics')

        if save_path != '':
            fig.write_image(save_path + '.png')
            fig.write_html(save_path + '.html')
            if config.save_as_pdf:
                fig.write_image(save_path + '.pdf')  # Plotly supports SVG, but not EPS directly
        # Show the image if requested
        if not config.suppress_grid_image_plot and show_figure:
            fig.show()
