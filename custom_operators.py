import random
import torch
from src.evotorch.operators import CopyingOperator,Operator,CrossOver
from src.evotorch.core import  SolutionBatch,Optional
from src.evotorch.core import Problem



import torch
from evotorch.operators import CopyingOperator
from evotorch import SolutionBatch

import torch
from src.evotorch.operators import CrossOver
from src.evotorch import SolutionBatch

class GreedyCrossover(CrossOver):
    def __init__(self, problem,tournament_size=5, top_n=2, num_children=20, crossover_strategy=None):
        """
        Args:
            problem: The problem being solved.
            top_n: Number of top solutions to preserve (elitism).
            num_children: Number of children to generate.
            solution_length: Length of each solution (dimensionality).
            crossover_strategy: A callable for the crossover strategy. If None, defaults to single-point crossover.
        """
        super().__init__(problem,tournament_size=tournament_size)
        self.top_n = top_n
        self.num_children = num_children

        self.crossover_strategy = crossover_strategy or self.two_point_crossover

    def _do(self, batch: SolutionBatch) -> SolutionBatch:
        """
        Main method to perform the crossover operation.

        Args:
            batch: The input population as a SolutionBatch.

        Returns:
            A new SolutionBatch with the offspring replacing the worst solutions.
        """
        parents1, parents2,sorted_batch = self._make_tournament_and_selection(batch)
        if len(parents1) != len(parents2):
            raise ValueError(
                f"_make_tournament_and_selection() returned parents1 and parents2 with incompatible sizes. "
                f"len(parents1): {len(parents1)}; len(parents2): {len(parents2)}."
            )
        offspring = self._do_cross_over(parents1, parents2)
        return self._replace_worst(batch, offspring,sorted_batch)

    def _make_tournament_and_selection(self, batch: SolutionBatch):
        """
        Select parents greedily based on evaluation scores.

        Args:
            batch: The input population as a SolutionBatch.

        Returns:
            parents1, parents2: Two groups of selected parents for crossover.
        """
        # Step 1: Evaluate and sort the population
        evals = batch.utility()  # Get evaluation scores
        sorted_indices = torch.argsort(evals, descending=True)  # Higher is better
        sorted_batch = batch[sorted_indices]

        # Step 2: Select parents greedily
        parents1 = sorted_batch[:self.num_children // 2]._data
        parents2 = sorted_batch[1:self.num_children // 2 + 1]._data  # Shifted pairing

        return parents1, parents2,sorted_batch

    def _do_cross_over(self, parents1: torch.Tensor, parents2: torch.Tensor) -> torch.Tensor:
        """
        Perform crossover using the selected strategy.

        Args:
            parents1: The first half of the parents.
            parents2: The second half of the parents.

        Returns:
            A tensor containing the offspring.
        """
        return self.crossover_strategy(parents1, parents2)
    def two_point_crossover(self, parents1: torch.Tensor, parents2: torch.Tensor,eta=0.2) -> torch.Tensor:
        # Generate u_i values which determine the spread
        u = self.problem.make_uniform_shaped_like(parents1)

        # Compute beta_i values from u_i values as the actual spread per dimension
        betas = (2 * u).pow(1.0 / (eta + 1.0))  # Compute all values for u_i < 0.5 first
        betas[u > 0.5] = (1.0 / (2 * (1.0 - u[u > 0.5]))).pow(
            1.0 / (eta + 1.0)
        )  # Replace the values for u_i >= 0.5
        children1 = 0.5 * (
            (1 + betas) * parents1 + (1 - betas) * parents2
        )  # Create the first set of children from the beta values
        children2 = 0.5 * (
            (1 + betas) * parents2 + (1 - betas) * parents1
        )  # Create the second set of children as a mirror of the first set of children

        # Combine the children tensors in one big tensor
        children = torch.cat([children1, children2], dim=0)

        # Respect the lower and upper bounds defined by the problem object
        children = self._respect_bounds(children)

        return children

    def _single_point_crossover(self, parents1: torch.Tensor, parents2: torch.Tensor) -> torch.Tensor:
        """
        Default single-point crossover strategy.

        Args:
            parents1: The first half of the parents.
            parents2: The second half of the parents.

        Returns:
            A tensor containing the offspring.
        """
        num_children = parents1.size(0) * 2
        solution_length = parents1.size(1)
        device = parents1.device

        child_values = torch.empty((num_children, solution_length), device=device)

        for i in range(parents1.size(0)):
            crossover_point = torch.randint(1, solution_length, (1,)).item()

            # Generate two children per pair
            child_values[2 * i, :crossover_point] = parents1[i, :crossover_point]
            child_values[2 * i, crossover_point:] = parents2[i, crossover_point:]
            child_values[2 * i + 1, :crossover_point] = parents2[i, :crossover_point]
            child_values[2 * i + 1, crossover_point:] = parents1[i, crossover_point:]

        return child_values

    def _replace_worst(self, batch: SolutionBatch, offspring: torch.Tensor,sorted_batch:SolutionBatch) -> SolutionBatch:
        """
        Replace the worst solutions in the population with the offspring.

        Args:
            batch: The input population as a SolutionBatch.
            offspring: A tensor containing the offspring.

        Returns:
            A new SolutionBatch with the worst solutions replaced.
        """
        offspring_batch = self._make_children_batch(offspring)


        # Step 2: Extract the preserved top N solutions
        preserved_solutions = sorted_batch[:self.top_n]._data

        # Step 3: Replace the worst solutions with offspring
        # Combine preserved solutions and offspring
        combined_solutions = torch.cat([preserved_solutions, offspring_batch._data], dim=0)

        # Step 4: Fill the remaining slots with other solutions from the sorted batch
        remaining_solutions = sorted_batch[self.top_n + len(offspring_batch):]._data
        num_remaining_slots = len(batch) - len(combined_solutions)
        if num_remaining_slots > 0:
            combined_solutions = torch.cat(
                [combined_solutions, remaining_solutions[:num_remaining_slots]], dim=0
            )
        return self._make_children_batch(combined_solutions)


    def _make_children_batch(self, child_values: torch.Tensor) -> SolutionBatch:
        """
        Convert a tensor of offspring into a SolutionBatch.

        Args:
            child_values: A tensor containing the offspring.

        Returns:
            A SolutionBatch containing the offspring.
        """
        result = SolutionBatch(self.problem, device=child_values.device, empty=True, popsize=child_values.shape[0])
        result._data = child_values
        return result


# Custom TournamentSelection



class MultiPointCrossOver(CrossOver):
    """
    Representation of a multi-point cross-over operator.

    When this operator is applied on a SolutionBatch, a tournament selection
    technique is used for selecting parent solutions from the batch, and then
    those parent solutions are mated via cutting from a random position and
    recombining. The result of these recombination operations is a new
    SolutionBatch, containing the children solutions. The original
    SolutionBatch stays unmodified.

    This operator is a generalization over the standard cross-over operators
    [OnePointCrossOver][evotorch.operators.real.OnePointCrossOver]
    and [TwoPointCrossOver][evotorch.operators.real.TwoPointCrossOver].
    In more details, instead of having one or two cutting points, this operator
    is configurable in terms of how many cutting points is desired.
    This generalized cross-over implementation follows the procedure described
    in:

        Sean Luke, 2013, Essentials of Metaheuristics, Lulu, second edition
        available for free at http://cs.gmu.edu/~sean/book/metaheuristics/
    """

    def __init__(
        self,
        problem: Problem,
        *,
        tournament_size: int,
        obj_index: Optional[int] = None,
        num_points: Optional[int] = None,
        num_children: Optional[int] = None,
        cross_over_rate: Optional[float] = None,
    ):
        """
        `__init__(...)`: Initialize the MultiPointCrossOver.

        Args:
            problem: The problem object to work on.
            tournament_size: What is the size (or length) of a tournament
                when selecting a parent candidate from a population
            obj_index: Objective index according to which the selection
                will be done.
            num_points: Number of cutting points for the cross-over operator.
            num_children: Optionally a number of children to produce by the
                cross-over operation.
                Not to be used together with `cross_over_rate`.
                If `num_children` and `cross_over_rate` are both None,
                then the number of children is equal to the number
                of solutions received.
            cross_over_rate: Optionally expected as a real number between
                0.0 and 1.0. Specifies the number of cross-over operations
                to perform. 1.0 means `1.0 * len(solution_batch)` amount of
                cross overs will be performed, resulting in
                `2.0 * len(solution_batch)` amount of children.
                Not to be used together with `num_children`.
                If `num_children` and `cross_over_rate` are both None,
                then the number of children is equal to the number
                of solutions received.
        """

        super().__init__(
            problem,
            tournament_size=tournament_size,
            obj_index=obj_index,
            num_children=num_children,
            cross_over_rate=cross_over_rate,
        )

        self._num_points = int(num_points)
        if self._num_points < 1:
            raise ValueError(
                f"Invalid `num_points`: {self._num_points}."
                f" Please provide a `num_points` which is greater than or equal to 1"
            )

    @torch.no_grad()
    def _do_cross_over(self, parents1: torch.Tensor, parents2: torch.Tensor) -> SolutionBatch:
        # What we expect here is this:
        #
        #    parents1      parents2
        #    ==========    ==========
        #    parents1[0]   parents2[0]
        #    parents1[1]   parents2[1]
        #    ...           ...
        #    parents1[N]   parents2[N]
        #
        # where parents1 and parents2 are 2D tensors, each containing values of N solutions.
        # For each row i, we will apply cross-over on parents1[i] and parents2[i].
        # From each cross-over, we will obtain 2 children.
        # This means, there are N pairings, and 2N children.

        num_pairings = parents1.shape[0]
        # num_children = num_pairings * 2

        device = parents1[0].device
        solution_length = len(parents1[0])
        num_points = self._num_points

        # For each pairing, generate all gene indices (i.e. [0, 1, 2, ...] for each pairing)
        gene_indices = (
            torch.arange(0, solution_length, device=device).unsqueeze(0).expand(num_pairings, solution_length)
        )

        if num_points == 1:
            # For each pairing, generate a gene index at which the parent solutions will be cut and recombined
            crossover_point = self.problem.make_randint((num_pairings, 1), n=(solution_length - 1), device=device) + 1

            # Make a mask for crossing over
            # (False: take the value from one parent, True: take the value from the other parent).
            # For gene indices less than crossover_point of that pairing, the mask takes the value 0.
            # Otherwise, the mask takes the value 1.
            crossover_mask = gene_indices >= crossover_point
        else:
            # For each pairing, generate gene indices at which the parent solutions will be cut and recombined
            crossover_points = self.problem.make_randint(
                (num_pairings, num_points), n=(solution_length + 1), device=device
            )

            # From `crossover_points`, extract each cutting point for each solution.
            cutting_points = [crossover_points[:, i].reshape(-1, 1) for i in range(num_points)]

            # Initialize `crossover_mask` as a tensor filled with False.
            crossover_mask = torch.zeros((num_pairings, solution_length), dtype=torch.bool, device=device)

            # For each cutting point p, toggle the boolean values of `crossover_mask`
            # for indices bigger than the index pointed to by p
            for p in cutting_points:
                crossover_mask ^= gene_indices >= p

        # Using the mask, generate two children.
        children1 = torch.where(crossover_mask, parents1, parents2)
        children2 = torch.where(crossover_mask, parents2, parents1)

        # Combine the children tensors in one big tensor
        children = torch.cat([children1, children2], dim=0)

        # Write the children solutions into a new SolutionBatch, and return the new batch
        result = self._make_children_batch(children)
        return result