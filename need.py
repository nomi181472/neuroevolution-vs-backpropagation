import os
import copy
import gymnasium as gym
import torch
from src.evotorch.algorithms import Cosyne
from src.evotorch.logging import StdOutLogger
from src.evotorch.neuroevolution import GymNE
import torch.nn as nn
from typing import Callable, Iterable, Optional, Union

from src.evotorch.core import Problem, SolutionBatch
from src.evotorch.operators import CosynePermutation, CrossOver, GaussianMutation, OnePointCrossOver, SimulatedBinaryCrossOver
from src.evotorch.algorithms.searchalgorithm import SearchAlgorithm, SinglePopulationAlgorithmMixin



from custom_logger import CSVLogger
from evotorch.decorators import pass_info
from torch import nn
class Need(SearchAlgorithm, SinglePopulationAlgorithmMixin):
    """
    Implementation of the CoSyNE algorithm.

    References:

        F.Gomez, J.Schmidhuber, R.Miikkulainen, M.Mitchell (2008).
        Accelerated Neural Evolution through Cooperatively Coevolved Synapses.
        Journal of Machine Learning Research 9 (5).
    """

    def __init__(
        self,
        problem: Problem,
        *,
        popsize: int,
        tournament_size: int,
        mutation_stdev: Optional[float],
        mutation_probability: Optional[float] = None,
        permute_all: bool = False,
        num_elites: Optional[int] = None,
        elitism_ratio: Optional[float] = None,
        eta: Optional[float] = None,
        num_children: Optional[int] = None,
    ):
        """
        `__init__(...)`: Initialize the Cosyne instance.

        Args:
            problem: The problem object to work on.
            popsize: Population size, as an integer.
            tournament_size: Tournament size, for tournament selection.
            mutation_stdev: Standard deviation of the Gaussian mutation.
                See [GaussianMutation][evotorch.operators.real.GaussianMutation] for more information.
            mutation_probability: Elementwise Gaussian mutation probability.
                Defaults to None.
                See [GaussianMutation][evotorch.operators.real.GaussianMutation] for more information.
            permute_all: If given as True, all solutions are subject to
                permutation. If given as False (which is the default),
                there will be a selection procedure for each decision
                variable.
            num_elites: Optionally expected as an integer, specifying the
                number of elites to pass to the next generation.
                Cannot be used together with the argument `elitism_ratio`.
            elitism_ratio: Optionally expected as a real number between
                0 and 1, specifying the amount of elites to pass to the
                next generation. For example, 0.1 means that the best 10%
                of the population are accepted as elites and passed onto
                the next generation.
                Cannot be used together with the argument `num_elites`.
            eta: Optionally expected as an integer, specifying the eta
                hyperparameter for the simulated binary cross-over (SBX).
                If left as None, one-point cross-over will be used instead.
            num_children: Number of children to generate at each iteration.
                If left as None, then this number is half of the population
                size.
        """

        problem.ensure_numeric()

        SearchAlgorithm.__init__(self, problem)

        if mutation_stdev is None:
            if mutation_probability is not None:
                raise ValueError(
                    f"`mutation_probability` was set to {mutation_probability}, but `mutation_stdev` is None, "
                    "which means, mutation is disabled. If you want to enable the mutation, be sure to provide "
                    "`mutation_stdev` as well."
                )
            self.mutation_op = None
        else:
            self.mutation_op = GaussianMutation(
                self._problem,
                stdev=mutation_stdev,
                mutation_probability=mutation_probability,
            )

        cross_over_kwargs = {"tournament_size": tournament_size}
        if num_children is None:
            cross_over_kwargs["cross_over_rate"] = 2.0
        else:
            cross_over_kwargs["num_children"] = num_children

        if eta is None:
            self._cross_over_op = OnePointCrossOver(self._problem, **cross_over_kwargs)
        else:
            self._cross_over_op = SimulatedBinaryCrossOver(self._problem, eta=eta, **cross_over_kwargs)

        self._permutation_op = CosynePermutation(self._problem, permute_all=permute_all)

        self._popsize = int(popsize)

        if num_elites is not None and elitism_ratio is None:
            self._num_elites = int(num_elites)
        elif num_elites is None and elitism_ratio is not None:
            self._num_elites = int(self._popsize * elitism_ratio)
        elif num_elites is None and elitism_ratio is None:
            self._num_elites = None
        else:
            raise ValueError(
                "Received both `num_elites` and `elitism_ratio`. Please provide only one of them, or none of them."
            )
        self.reward_tracker = {i: [] for i in range(popsize)}  # Initialize reward tracker
        self._population = SolutionBatch(problem, device=problem.device, popsize=self._popsize)
        self._first_generation: bool = True

        # GAStatusMixin.__init__(self)
        SinglePopulationAlgorithmMixin.__init__(self)

    @property
    def population(self) -> SolutionBatch:
        return self._population

    def _step(self):
        if self._first_generation:
            self._first_generation = False
            self._problem.evaluate(self._population)

        to_merge = []

        num_elites = self._num_elites
        num_parents = int(self._popsize / 4)
        num_relevant = max((0 if num_elites is None else num_elites), num_parents)

        sorted_relevant = self._population.take_best(num_relevant)

        if self._num_elites is not None and self._num_elites >= 1:
            to_merge.append(sorted_relevant[:num_elites].clone())

        parents = sorted_relevant[:num_parents]
        children = self._cross_over_op(parents)
        if self.mutation_op is not None:
            children = self.mutation_op(children)

        permuted = self._permutation_op(self._population)

        to_merge.extend([children, permuted])

        extended_population = SolutionBatch(merging_of=to_merge)
        self._problem.evaluate(extended_population)


        self._population = extended_population.take_best(self._popsize)
    def get_best_population(self,n:int)->SolutionBatch:
        return self._population.take_best(n)
    def get_ensembled_models(self,n:int,)->nn.Module:
        decisions=[]
        for sol in self.get_best_population(n):
            model=self.problem.parameterize_net(sol.access_values())



