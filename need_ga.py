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
from src.evotorch.algorithms.ga import ExtendedPopulationMixin


from custom_logger import CSVLogger
from evotorch.decorators import pass_info
from torch import nn
class NeedGA(SearchAlgorithm, SinglePopulationAlgorithmMixin, ExtendedPopulationMixin):
    """
    A genetic algorithm implementation.

    **Basic usage.**
    Let us consider a single-objective optimization problem where the goal is to
    minimize the L2 norm of a continuous tensor of length 10:

    ```python
    from evotorch import Problem
    from evotorch.algorithms import GeneticAlgorithm
    from evotorch.operators import OnePointCrossOver, GaussianMutation

    import torch


    def f(x: torch.Tensor) -> torch.Tensor:
        return torch.linalg.norm(x)


    problem = Problem(
        "min",
        f,
        initial_bounds=(-10.0, 10.0),
        solution_length=10,
    )
    ```

    For solving this problem, a genetic algorithm could be instantiated as
    follows:

    ```python
    ga = GeneticAlgorithm(
        problem,
        popsize=100,
        operators=[
            OnePointCrossOver(problem, tournament_size=4),
            GaussianMutation(problem, stdev=0.1),
        ],
    )
    ```

    The genetic algorithm instantiated above is configured to have a population
    size of 100, and is configured to perform the following operations on the
    population at each generation:
    (i) select solutions with a tournament of size 4, and produce children from
    the selected solutions by applying one-point cross-over;
    (ii) apply a gaussian mutation on the values of the solutions produced by
    the previous step, the amount of the mutation being sampled according to a
    standard deviation of 0.1.
    Once instantiated, this GeneticAlgorithm instance can be used with an API
    compatible with other search algorithms, as shown below:

    ```python
    from evotorch.logging import StdOutLogger

    _ = StdOutLogger(ga)  # Report the evolution's progress to standard output
    ga.run(100)  # Run the algorithm for 100 generations
    print("Solution with best fitness ever:", ga.status["best"])
    print("Current population's best:", ga.status["pop_best"])
    ```

    Please also note:

    - The operators are always executed according to the order specified within
      the `operators` argument.
    - There are more operators available in the namespace
      [evotorch.operators][evotorch.operators].
    - By default, GeneticAlgorithm is elitist. In the elitist mode, an extended
      population is formed from parent solutions and child solutions, and the
      best n solutions of this extended population are accepted as the next
      generation. If you wish to switch to a non-elitist mode (where children
      unconditionally replace the worst-performing parents), you can use the
      initialization argument `elitist=False`.
    - It is not mandatory to specify a cross-over operator. When a cross-over
      operator is missing, the GeneticAlgorithm will work like a simple
      evolution strategy implementation which produces children by mutating
      the parents, and then replaces the parents (where the criteria for
      replacing the parents depend on whether or not elitism is enabled).
    - To be able to deal with stochastic fitness functions correctly,
      GeneticAlgorithm re-evaluates previously evaluated parents as well.
      When you are sure that the fitness function is deterministic,
      you can pass the initialization argument `re_evaluate=False` to prevent
      unnecessary computations.

    **Integer decision variables.**
    GeneticAlgorithm can be used on problems with `dtype` declared as integer
    (e.g. `torch.int32`, `torch.int64`, etc.).
    Within the field of discrete optimization, it is common to encounter
    one or more of these scenarios:

    - The search space of the problem has a special structure that one will
      wish to exploit (within the cross-over and/or mutation operators) to
      be able to reach the (near-)optimum within a reasonable amount of time.
    - The problem is partially or fully combinatorial.
    - The problem is constrained in such a way that arbitrarily sampling
      discrete values for its decision variables might cause infeasibility.

    Considering all these scenarios, it is difficult to come up with general
    cross-over and mutation operators that will work across various discrete
    optimization problems, and it is common to design problem-specific
    operators. In EvoTorch, it is possible to define custom operators and
    use them with GeneticAlgorithm, which is required when using
    GeneticAlgorithm on a problem with a non-float dtype.

    As an example, let us consider the following discrete optimization
    problem:

    ```python
    def f(x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x)


    problem = Problem(
        "min",
        f,
        bounds=(-10, 10),
        solution_length=10,
        dtype=torch.int64,
    )
    ```

    Although EvoTorch does provide a very simple and generic (usable with float
    and int dtypes) cross-over named
    [OnePointCrossOver][evotorch.operators.real.OnePointCrossOver]
    (a cross-over which randomly decides a cutting point for each pair of
    parents, cuts them from those points and recombines them), it can be
    desirable and necessary to implement a custom cross-over operator.
    One can inherit from [CrossOver][evotorch.operators.base.CrossOver] to
    define a custom cross-over operator, as shown below:

    ```python
    from evotorch import SolutionBatch
    from evotorch.operators import CrossOver


    class CustomCrossOver(CrossOver):
        def _do_cross_over(
            self,
            parents1: torch.Tensor,
            parents2: torch.Tensor,
        ) -> SolutionBatch:
            # parents1 is a tensor storing the decision values of the first
            # half of the chosen parents.
            # parents2 is a tensor storing the decision values of the second
            # half of the chosen parents.

            # We expect that the lengths of parents1 and parents2 are equal.
            assert len(parents1) == len(parents2)

            # Allocate an empty SolutionBatch that will store the children
            childpop = SolutionBatch(self.problem, popsize=num_parents, empty=True)

            # Gain access to the decision values tensor of the newly allocated
            # childpop
            childpop_values = childpop.access_values()

            # Here we somehow fill `childpop_values` by recombining the parents.
            # The most common thing to do is to produce two children by
            # combining parents1[0] and parents2[0], to produce the next two
            # children parents1[1] and parents2[1], and so on.
            childpop_values[:] = ...

            # Return the child population
            return childpop
    ```

    One can define a custom mutation operator by inheriting from
    [Operator][evotorch.operators.base.Operator], as shown below:

    ```python
    class CustomMutation(Operator):
        def _do(self, solutions: SolutionBatch):
            # Get the decision values tensor of the solutions
            sln_values = solutions.access_values()

            # do in-place modifications to the decision values
            sln_values[:] = ...
    ```

    Alternatively, you could define the mutation operator as a function:

    ```python
    def my_mutation_function(original_values: torch.Tensor) -> torch.Tensor:
        # Somehow produce mutated copies of the original values
        mutated_values = ...

        # Return the mutated values
        return mutated_values
    ```

    With these defined operators, we are now ready to instantiate our
    GeneticAlgorithm:

    ```python
    ga = GeneticAlgorithm(
        problem,
        popsize=100,
        operators=[
            CustomCrossOver(problem, tournament_size=4),
            CustomMutation(problem),
            # -- or, if you chose to define the mutation as a function: --
            # my_mutation_function,
        ],
    )
    ```

    **Non-numeric or variable-length solutions.**
    GeneticAlgorithm can also work on problems whose `dtype` is declared
    as `object`, where `dtype=object` means that a solution's value(s) can be
    expressed via a tensor, a numpy array, a scalar, a tuple, a list, a
    dictionary.

    Like in the previously discussed case (where dtype is an integer type),
    one has to define custom operators when working on problems with
    `dtype=object`. A custom cross-over definition specialized for
    `dtype=object` looks like this:

    ```python
    from evotorch.tools import ObjectArray


    class CrossOverForObjectDType(CrossOver):
        def _do_cross_over(
            self,
            parents1: ObjectArray,
            parents2: ObjectArray,
        ) -> SolutionBatch:
            # Allocate an empty SolutionBatch that will store the children
            childpop = SolutionBatch(self.problem, popsize=num_parents, empty=True)

            # Gain access to the decision values ObjectArray of the newly allocated
            # childpop
            childpop_values = childpop.access_values()

            # Here we somehow fill `childpop_values` by recombining the parents.
            # The most common thing to do is to produce two children by
            # combining parents1[0] and parents2[0], to produce the next two
            # children parents1[1] and parents2[1], and so on.
            childpop_values[:] = ...

            # Return the child population
            return childpop
    ```

    A custom mutation operator specialized for `dtype=object` looks like this:

    ```python
    class MutationForObjectDType(Operator):
        def _do(self, solutions: SolutionBatch):
            # Get the decision values ObjectArray of the solutions
            sln_values = solutions.access_values()

            # do in-place modifications to the decision values
            sln_values[:] = ...
    ```

    A custom mutation function specialized for `dtype=object` looks like this:

    ```python
    def mutation_for_object_dtype(original_values: ObjectArray) -> ObjectArray:
        # Somehow produce mutated copies of the original values
        mutated_values = ...

        # Return the mutated values
        return mutated_values
    ```

    With these operators defined, one can instantiate the GeneticAlgorithm:

    ```python
    ga = GeneticAlgorithm(
        problem_with_object_dtype,
        popsize=100,
        operators=[
            CrossOverForObjectDType(problem_with_object_dtype, tournament_size=4),
            MutationForObjectDType(problem_with_object_dtype),
            # -- or, if you chose to define the mutation as a function: --
            # mutation_for_object_dtype,
        ],
    )
    ```

    **Multiple objectives.**
    GeneticAlgorithm can work on problems with multiple objectives.
    When there are multiple objectives, GeneticAlgorithm will compare the
    solutions according to their pareto-ranks and their crowding distances,
    like done by the NSGA-II algorithm (Deb, 2002).

    References:

        Sean Luke, 2013, Essentials of Metaheuristics, Lulu, second edition
        available for free at http://cs.gmu.edu/~sean/book/metaheuristics/

        Kalyanmoy Deb, Amrit Pratap, Sameer Agarwal, T. Meyarivan (2002).
        A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II.
    """

    def __init__(
        self,
        problem: Problem,
        *,
        operators: Iterable,
        popsize: int,
        elitist: bool = True,
        re_evaluate: bool = True,
        n_ensemble:int=5,
        re_evaluate_parents_first: Optional[bool] = None,
        _allow_empty_operator_list: bool = False,
    ):
        """
        `__init__(...)`: Initialize the GeneticAlgorithm.

        Args:
            problem: The problem to optimize.
            operators: Operators to be used by the genetic algorithm.
                Expected as an iterable, such as a list or a tuple.
                Each item within this iterable object is expected either
                as an instance of [Operator][evotorch.operators.base.Operator],
                or as a function which receives the decision values of
                multiple solutions in a PyTorch tensor (or in an
                [ObjectArray][evotorch.tools.objectarray.ObjectArray]
                for when dtype is `object`) and returns a modified copy.
            popsize: Population size.
            elitist: Whether or not this genetic algorithm will behave in an
                elitist manner. This argument controls how the genetic
                algorithm will form the next generation from the parents
                and the children. In elitist mode (i.e. with `elitist=True`),
                the procedure to be followed by this genetic algorithm is:
                (i) form an extended population which consists of
                both the parents and the children,
                (ii) sort the extended population from best to worst,
                (iii) select the best `n` solutions as the new generation where
                `n` is `popsize`.
                In non-elitist mode (i.e. with `elitist=False`), the worst `m`
                solutions within the parent population are replaced with
                the children, `m` being the number of produced children.
            re_evaluate: Whether or not to evaluate the solutions
                that were already evaluated in the previous generations.
                By default, this is set as True.
                The reason behind this default setting is that,
                in problems where the evaluation procedure is noisy,
                by re-evaluating the already-evaluated solutions,
                we prevent the bad solutions that were luckily evaluated
                from hanging onto the population.
                Instead, at every generation, each solution must go through
                the evaluation procedure again and prove their worth.
                For problems whose evaluation procedures are NOT noisy,
                the user might consider turning re_evaluate to False
                for saving computational cycles.
            re_evaluate_parents_first: This is to be specified only when
                `re_evaluate` is True (otherwise to be left as None).
                If this is given as True, then it will be assumed that the
                provided operators require the parents to be evaluated.
                If this is given as False, then it will be assumed that the
                provided operators work without looking at the parents'
                fitnesses (in which case both parents and children can be
                evaluated in a single vectorized computation cycle).
                If this is left as None, then whether or not the operators
                need to know the parent evaluations will be determined
                automatically as follows:
                if the operators contain at least one cross-over operator
                then `re_evaluate_parents_first` will be internally set as
                True; otherwise `re_evaluate_parents_first` will be internally
                set as False.
        """
        SearchAlgorithm.__init__(self, problem)

        self._popsize = int(popsize)
        self._elitist: bool = bool(elitist)
        self._population = problem.generate_batch(self._popsize)

        ExtendedPopulationMixin.__init__(
            self,
            re_evaluate=re_evaluate,
            re_evaluate_parents_first=re_evaluate_parents_first,
            operators=operators,
            allow_empty_operators_list=_allow_empty_operator_list,
        )
        self.n_ensemble = n_ensemble
        self.ensembled_models = []

        SinglePopulationAlgorithmMixin.__init__(self)

    @property
    def population(self) -> SolutionBatch:
        """Get the population"""
        return self._population

    def _step(self):
        # Get the population size
        popsize = self._popsize

        if self._elitist:
            # This is where we handle the elitist mode.

            # Produce and get an extended population in a single SolutionBatch
            extended_population = self._make_extended_population(split=False)

            # From the extended population, take the best n solutions, n being the popsize.
            self._population = extended_population.take_best(popsize)
        else:
            # This is where we handle the non-elitist mode.

            # Take the parent solutions (ensured to be evaluated) and the children separately.
            parents, children = self._make_extended_population(split=True)

            # Get the number of children
            num_children = len(children)

            if num_children < popsize:
                # If the number of children is less than the population size, then we keep the best m solutions from
                # the parents, m being `popsize - num_children`.
                chosen_parents = self._population.take_best(popsize - num_children)

                # Combine the children with the chosen parents, and declare them as the new population.
                self._population = SolutionBatch.cat([chosen_parents, children])
            elif num_children == popsize:
                # If the number of children is the same with the population size, then these children are declared as
                # the new population.
                self._population = children
            else:
                # If the number of children is more than the population size, then we take the best n solutions from
                # these children, n being the population size. These chosen children are then declared as the new
                # population.
                self._population = children.take_best(self._popsize)

        self.ensembled_models = self.get_ensembled_models(
                self.get_best_population(self.n_ensemble)
            )

    def get_best_population(self, n: int) -> SolutionBatch:
        return self._population.take_best(n)

    def get_ensembled_models(self, solultions: SolutionBatch) -> nn.ModuleList:
        models = []

        for sol in solultions:
            model = self.problem.parameterize_net(sol.access_values())
            models.append(model)
        return nn.ModuleList(models)