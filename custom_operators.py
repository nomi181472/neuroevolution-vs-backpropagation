import random
import torch
from src.evotorch.operators import CopyingOperator,Operator,CrossOver
from src.evotorch.core import  SolutionBatch,Optional
from src.evotorch.core import Problem

# Custom TournamentSelection
class CustomTournamentSelection(CopyingOperator):
    def __init__(self, problem:Problem,tournament_size=3):
        super().__init__(problem=problem)  # No specific problem is required for selection
        self.tournament_size = tournament_size

    def _do(self, population):
        selected = []
        pop_size = len(population)

        for _ in range(pop_size):  # Perform selection for the entire population
            competitors = random.sample(population, self.tournament_size)
            best = max(competitors, key=lambda ind: ind.evaluation)  # Choose the best individual
            selected.append(best)

        return selected



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