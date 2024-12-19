from collections import deque
import copy as cp
import os
import queue
import random
import time
from typing import List, Optional

from vias.config import get_config
from vias.grid_graph import load_grid_graph
from src.vias.path import Path


import numpy as np

from deap.tools import initRepeat, selNSGA2, selNSGA3WithMemory, mutGaussian
from pymoo.util.ref_dirs import get_reference_directions
from deap.tools.selection import selTournament, selBest
from deap import base as deap_base
from deap.tools.crossover import cxOnePointDeterminedLength
from deap.tools.emo import selTournamentDCD
from deap.tools.emo import selTournamentConstraint

from src.vias.optimizers.optimizer import Optimizer
from vias.utils.tools import bcolors as bc
from optimizer import OptimizationEndsInNiches
from vias.optimizers.optimizer_helpers import Population, Individual
from src.vias.data_manager import DataManager
from vias.console_manager import console


class OptimizerGA(Optimizer):
    """This is an implementation of the abstract Optimizer class. It describes an genetic algorithm (GA), whose
    individuals are floating point lists ([a,b,c,d,e,f,...]). The selection strategy is based on DEBs NSGAII selection
     for biobjective optimization problems, thus giving it the same NSGA2 (Non-dominated sorting genetic algorithm).
     The GA uses a conventional real coded mutation and crossover operators."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # optimizer specific attributes
        params = kwargs["optimizer_parameters"]
        config = get_config()
        self.pop_size = params['population_size']
        self.prob_ind_gene_mut = None  # 1/D in init_genetic_algorithm()
        self.prob_mut = params['mutation_probability']
        self.prob_cross = params['crossover_probability']

        self.low_bound, self.up_bound = None, None

        self.enable_niching = params['enable_niching']

        self.stop_criterion_fifo = deque(50 * [np.nan], maxlen=50)
        self.tracked_termination_value: Optional[float] = np.nan
        self.termination_percentage = 0.0
        self.initial_path_selection_idx = 0
        self.unsatisfied_iteration_count = 0
        self.multiprocessing_enabled = config.multiprocessing_enabled

        if self.enable_niching:
            assert config.preprocessor.advanced_individual_init, "Niching makes only sense with precomputed good solutions"

    def optimize(self):
        """In this method the genetic optimization takes place."""
        config = get_config()
        start_time = time.time()
        toolbox = self.init_genetic_algorithm()

        pop = self.init_pop(toolbox)

        console.log("Start of Multi-objective Path Planning")

        # Evaluate the entire population
        self.evaluate_individuals(pop, toolbox)

        if self.num_objectives == 2:
            # This is just to assign the crowding distance to the individuals (needed for Tournament)
            # no actual selection is done
            pop.set_niches([toolbox.select(niche, len(niche)) for niche in pop.get_niches()])

        self.raw_calc_time += time.time() - start_time
        # Variable keeping track of the number of generations


        # Add the properties of the initial population to the statistic.
        self.pareto_handler.update_statistic(self.generation_count, pop, self.eval_count)

        # Begin the evolution
        while not self.termination_criterion_valid():
            # A new generation
            start_time = time.time()

            # Vary the population
            offspring = cp.deepcopy(pop)
            offspring.set_niches(
                [self.var_individual(niche, toolbox, len(niche), self.prob_cross, self.prob_mut) for niche in
                 pop.get_niches()])

            # Evaluate invalid individuals (individuals whose paths have changed)
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            self.evaluate_individuals(invalid_ind, toolbox)

            # Select the next generation population
            pop.set_niches(
                [toolbox.select(pop_niche + offspring_niche, len(pop_niche)) for pop_niche, offspring_niche
                 in zip(pop.get_niches(), offspring.get_niches())])



            if self.enable_niching:
                self.update_niching(pop)

            self.generation_count += 1
            self.raw_calc_time += time.time() - start_time
            # Gather all the fitnesses in one list and print the stats
            self.pareto_handler.update_statistic(self.generation_count, pop, self.eval_count)
            # save network representations for all time best individuals with respect to every single objective

            if not config.suppress_iteration_based_paths_plot:
                for idx, objective in enumerate(self.objectives):
                    extreme_individual_fitness, extreme_individual = self.pareto_handler.get_extreme_solution(idx)
                    self.plot_extreme_path(extreme_individual, extreme_individual_fitness, objective)
                    # check extreme individuals for constraints
                    self.check_extreme_solution_constraints(extreme_individual, objective)



    def check_extreme_solution_constraints(self, extreme_individual, objective):
        for constraint in extreme_individual.constraints:
            if not constraint.satisfied:
                console.log(
                    bc.FAIL + f"Critical {constraint.name} constraint was not satisfied (penalty: {constraint.value}) for best solution regarding {objective} objective." + bc.ENDC)

    def plot_extreme_path(self, extreme_individual, extreme_individual_fitness, objective):
        config = get_config()
        extreme_path = self.coder.decode(extreme_individual)
        path_plot_path_per_objective = os.path.join(DataManager().optimized_paths_path, objective)
        if not os.path.exists(path_plot_path_per_objective):
            os.makedirs(path_plot_path_per_objective)
        grid = self.scenario.grid_maps[config.get(objective).map_to_plot_on]
        suffix = f'_gen_{self.generation_count}'
        title = f'Obj: {objective} Fit:{str(round(extreme_individual_fitness, 1))} Gen: {self.generation_count}'
        grid.plot_paths(extreme_path, savepath=path_plot_path_per_objective,
                        suffix=suffix, prefix='2D',
                        title=title)
        grid.plot_3D_paths(extreme_path, savepath=path_plot_path_per_objective, suffix=suffix, prefix='3D')

    def close_optimization(self):
        console.log("-- End of (successful) evolution --")
        config = get_config()
        if self.enable_niching:
            raise OptimizationEndsInNiches("optimizer_ga()",
                                           'End of iterations although population was still searching for constraint-satisfying solutions in some niches.')
        for i, objective in enumerate(self.objectives):
            grid = self.scenario.grid_maps[config.get(objective).map_to_plot_on]
            fitness, extreme_individual = self.pareto_handler.get_extreme_solution(i)
            extreme_path = self.coder.decode(extreme_individual)
            # plot paths
            title = f'{self.name}\'s best found path for {objective} with fitness {str(round(fitness, 1))}'
            suffix = f"_{objective}"
            grid.plot_paths(extreme_path, savepath=DataManager().optimized_paths_path, suffix=suffix,
                            title=title, prefix="2D")
            grid.plot_3D_paths(extreme_path, savepath=DataManager().optimized_paths_path, suffix=suffix, prefix="3D")


    def termination_criterion_valid(self):
        if 0 <= self.num_iterations <= self.generation_count:
            console.log("Finished optimization due to 'set number of iterations reached'")
            return True

        if self.num_objectives == 1:  #evaluate stop criterion considering fitness that is not improving
            fitness = self.pareto_handler.get_ideal_point().item()
            tracked_val = fitness
        else:
            delta_f = self.pareto_handler.get_delta_f()
            delta_nadir = self.pareto_handler.get_delta_nadir()
            delta_ideal = self.pareto_handler.get_delta_ideal()
            if np.all(np.isnan(np.array([delta_f, delta_nadir, delta_ideal]))):
                tracked_val = np.nan
            else:
                tracked_val = np.nanmax(np.array([delta_f, delta_nadir, delta_ideal]))
        return self.terminate_based_on_tracked_val(tracked_val)

    def terminate_based_on_tracked_val(self, tracked_val):
        tolerance = 0.0001
        n_skip = 5
        previous_val = self.tracked_termination_value
        if previous_val is np.nan:
            termination_percentage = 0.0
        elif self.generation_count > 0 and self.generation_count % (n_skip+1) != 0:
            termination_percentage = self.termination_percentage
        else:
            delta = np.abs(tracked_val - previous_val)
            if delta < tolerance:
                termination_percentage = 1.0
            else:
                v = (delta - tolerance)
                termination_percentage = 1 / (1 + v)
        self.termination_percentage = termination_percentage
        self.tracked_termination_value = tracked_val

        self.stop_criterion_fifo.append(termination_percentage)

        windowed_termination_percentage = np.nanmin(self.stop_criterion_fifo)
        if windowed_termination_percentage >= 1.0:
            console.log("Finished optimization due to no change in tracked value.")
            return True
        else:
            return False

    def update_niching(self,pop):
        config = get_config()
        num_unsat_niches = self.get_num_unsatisfied_niches(pop)
        num_niches = self.num_initial_paths
        if num_unsat_niches == 0:
            self.enable_niching = False
            pop.disable_niching()
            console.log(f"Disabling niching in iteration {self.generation_count}.")
        elif num_unsat_niches == 1:  # additional quit criterion: if only one single (nasty) unsatisfied niche is left
            self.unsatisfied_iteration_count += 1
        elif num_unsat_niches <= int(0.1*num_niches) and int(self.eval_count/num_niches) >= 5500: # additional quit criterion: if 90% of all niches are satisfied and a fair amount of function evaluations has been spent
            self.unsatisfied_iteration_count += 1
        if self.unsatisfied_iteration_count >= 10: # if additional quit criteria have been valid 10 times, force quit niching
            console.log(
                "Warning: one niche is not satisfying a constraint for 10 iterations in a row as only nicht. Quit niching.")
            self.enable_niching = False
            pop.disable_niching()
            console.log(f"Force disabling niching in iteration {self.generation_count}.")

    def evaluate_individuals(self, individuals: List[Individual], toolbox):
        eval_start_time = time.time()
        if len(individuals) > 0:
            if self.multiprocessing_enabled:
                fitnesses, constraints = self.evaluate_parallel(individuals)
            else:
                fitnesses, constraints = zip(*map(toolbox.evaluate, individuals))

            for ind, fit, constraint in zip(individuals, fitnesses, constraints):
                ind.fitness.values = fit
                ind.constraints = constraint
        self.eval_calc_time += time.time() - eval_start_time

    def init_pop(self, toolbox):
        if self.pop_size < 10 * self.num_initial_paths:
            console.log(bc.WARNING + f"There are {self.num_initial_paths} pre-processed solutions (incl. straight line) but only {self.pop_size} individuals. This is {self.pop_size / self.num_initial_paths} per niche. We recommend at least 10 individuals per niche." + bc.ENDC)
        individuals_without_noise, niche_numbers = zip(*[toolbox.first_ind() for _ in range(self.num_initial_paths)])
        pop = Population(individuals_without_noise, niche_numbers, self.enable_niching)  # first individual_germ_cells won't be applied by noise
        if len(pop) < self.pop_size:
            individuals_with_noise, niche_numbers = zip(
                *[toolbox.individual() for _ in range(self.pop_size - self.num_initial_paths)])
            pop.extend(individuals_with_noise, niche_numbers=niche_numbers)  # apply noise to the rest
        return pop

    def finish_after_optimization(self):
        self.pareto_handler.save_and_clean()

        # plot the result paths
        pareto_solutions = self.pareto_handler.get_pareto_solutions()

        output_paths = [self.coder.decode(ind) for ind in pareto_solutions]

        grid_map = load_grid_graph(DataManager().data_input_path, self.scenario.scenario_info, "buildings_map")
        grid_map.plot_3D_paths(output_paths, savepath=DataManager().data_output_path, prefix="3D_non_dominated_paths_")


    def init_genetic_algorithm(self):
        """In this method the NSGA2 is initialized, meaning the form of an individual is defined, the mutation and
        selection operators are chosen and the objectives are derived."""

        config = get_config()
        # Fitness class and representation of individual
        weights = tuple([-1 for _ in self.objectives])  # minimization is assumed for all objectives

        self.prob_ind_gene_mut = 1 / self.coder.size_individual
        self.low_bound, self.up_bound = self.coder.get_bounds()

        toolbox = deap_base.Toolbox()

        toolbox.register("individual", self.generate_individual, Individual,
                         self.initial_paths, weights, True)
        toolbox.register("first_ind", self.generate_individual, Individual,
                         self.initial_paths, weights, False)

        # define the population to be a list of individuals
        toolbox.register("population", initRepeat, list, toolbox.individual)

        # ----------
        # Operator registration
        # ----------
        # register the goal / fitness function
        toolbox.register("evaluate", self.evaluate)



        # operator for selecting individuals for next generation
        if self.num_objectives == 1:
            toolbox.register("choose", selTournament)
            toolbox.register("select", selBest)
            eta_cross = 15
            eta_mut = 20
        elif self.num_objectives == 2:
            toolbox.register("choose", selTournamentDCD)
            toolbox.register("select", selNSGA2)
            eta_cross = 15
            eta_mut = 20
        elif self.num_objectives > 2:
            toolbox.register("choose", selTournamentConstraint)
            eta_cross = 30
            eta_mut = 20
            num_ref_dirs = int(self.pop_size/3)
            console.log(f"The number of reference directions {num_ref_dirs} for the NSGA3 selection is automatically chosen to be 1/3 of the population size.")
            ref_dirs = get_reference_directions("energy", self.num_objectives, num_ref_dirs, seed=config.seed)
            toolbox.register("select", selNSGA3WithMemory(ref_dirs))

        # register the crossover operator
        # toolbox.register("mate", cxSimulatedBinaryBounded, eta=eta_cross, low=self.low_bound, up=self.up_bound)
        toolbox.register("mate", cxOnePointDeterminedLength, cut_length=self.coder.size_gene_per_cp)

        # register a mutation operator
        # toolbox.register("mutate", mutPolynomialBounded, eta=eta_mut, low=self.low_bound, up=self.up_bound,
        #                  indpb=self.prob_ind_gene_mut)
        mu = list(np.zeros(self.coder.size_individual))
        sigma_x = self.scenario.x_res
        sigma_y = self.scenario.y_res
        sigma_z = self.scenario.z_res
        sigma_w = 1.0
        if self.coder.adaptive_weights and self.coder.use_z_component:
            sigma_piece = [sigma_x, sigma_y, sigma_z, sigma_w]
        elif self.coder.adaptive_weights:
            sigma_piece = [sigma_x, sigma_y, sigma_w]
        elif self.coder.use_z_component:
            sigma_piece = [sigma_x, sigma_y, sigma_z]
        else:
            sigma_piece = [sigma_x, sigma_y]
        sigma = sigma_piece*self.coder.path_factory.num_variable_control_points
        toolbox.register("mutate", mutGaussian, mu=mu, sigma=sigma, indpb=self.prob_ind_gene_mut)


        return toolbox

    def evaluate_single_ind(self, individuals_to_process, individuals_processed, idx):
        """This function is only called by the evaluate_parallel() function of the super class and is called by every
         single evaluation (multiprocessing) process there. Individuals that still have to be processed and those that
         are already processed (means that there fitness has been calculated) are stored in queues."""
        while True:
            try:
                ind_id, individual = individuals_to_process.get_nowait()
            except queue.Empty:
                break
            else:
                path = self.coder.decode(individual)

                criteria, constraints = self.evaluator.evaluate(path)
                val = []
                for objective in self.objectives:
                    val.append(criteria[objective])
                individuals_processed.put(
                    (ind_id, tuple(val), constraints))  # caution --> the parameter constraints has not been tested

    def var_individual(self, population, toolbox, N, cxpb, mutpb):
        """This is mainly a copy of DEAP's VarAnd() algorithm."""
        # do tournament to increase selection pressure
        if self.num_objectives == 1:
            offspring = toolbox.choose(population, N, 2)
        else:
            offspring = toolbox.choose(population, N)
        offspring = [toolbox.clone(ind) for ind in offspring]
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            changed = False
            # Apply crossover
            if random.random() < cxpb:
                toolbox.mate(ind1, ind2)
                changed = True

            # Apply mutation
            if random.random() < mutpb:
                changed = True
                toolbox.mutate(ind1)
                toolbox.mutate(ind2)

            if changed:
                del ind1.fitness.values
                del ind2.fitness.values

        # through niching there is the possibility of an uneven number of individuals in a niche:
        if len(offspring) % 2 != 0:
            toolbox.mutate(offspring[-1])
            del offspring[-1].fitness.values

        return offspring

    def generate_individual(self, icls, initial_paths: List[Path], weights, apply_noise):
        """Used to initialize an individual which consists of a list of genes. Currently there are two nurbs curve
        control points per link in the list of genes, so that the list looks like
        [x_00, y_00, x_01, y_01, x_10, y_10, x_11, y_11, ...]. The initial position lies on the line between the NURBS
        curves' end points and are randomly varied a bit. Through the flags "use_z_component" and "adaptive_weights"
        more parameters can be enabled.
        """
        # germ_cell_count_out = None

        initial_path = initial_paths[self.initial_path_selection_idx]
        initial_path_selection_idx = cp.deepcopy(self.initial_path_selection_idx)
        self.initial_path_selection_idx = (self.initial_path_selection_idx + 1) % self.num_initial_paths

        if apply_noise:
            control_points_array = np.array(initial_path.nurbs_curve.ctrlpts)
            x_std = self.scenario.x_res
            y_std = self.scenario.y_res
            z_std = self.scenario.z_res
            control_points_array_with_noise = control_points_array + np.random.normal(0,
                                                                                      np.array([x_std, y_std, z_std]),
                                                                                      size=control_points_array.shape)
            control_points_with_noise = [row.tolist() for row in control_points_array_with_noise]
            initial_nurbs_curve_with_noise = cp.deepcopy(initial_path.nurbs_curve)
            initial_nurbs_curve_with_noise.ctrlpts = control_points_with_noise

            initial_path_with_noise = cp.deepcopy(initial_path)
            initial_path_with_noise.nurbs_curve = initial_nurbs_curve_with_noise
            initial_individual = self.coder.encode(initial_path_with_noise)
        else:
            initial_individual = self.coder.encode(initial_path)

        ind = icls(initial_individual.flatten().tolist(), weights)
        return ind, initial_path_selection_idx

    def select_best(self, individuals, k, objective_index):
        return sorted(individuals, key=lambda x: x.fitness.values[objective_index], reverse=True)[:k]

