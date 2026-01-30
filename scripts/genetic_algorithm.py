from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional, Any


# ----------------------------
# Types
# ----------------------------
Design = Dict[str, float]
EvaluateFn = Callable[[Design], Tuple[float, float]]   # returns (Nu, Eu)
ConstraintFn = Callable[[Design, float, float], float] # returns cv >= 0, 0 means feasible


@dataclass
class NSGA2Config:
    pop_size: int = 120
    generations: int = 80

    # SBX + polynomial mutation params (typical defaults)
    p_crossover: float = 0.9
    eta_c: float = 15.0

    p_mutation: float = 0.2         # per-variable mutation prob
    eta_m: float = 20.0

    tournament_k: int = 2
    seed: int = 42


@dataclass
class Individual:
    x: Design
    Nu: float = float("nan")  # objective 1 (maximize)
    Eu: float = float("nan")  # objective 2 (minimize)
    cv: float = float("nan")  # constraint violation (0 feasible)
    rank: int = 0
    crowd: float = 0.0


class NSGA2:
    """
    NSGA-II for 2 objectives:
      - maximize Nu
      - minimize Eu

    Constraint handling (Deb-style):
      - feasible dominates infeasible
      - among infeasible: smaller cv dominates
      - among feasible: Pareto dominance on (Nu, Eu)
    """

    def __init__(
        self,
        bounds: Dict[str, Tuple[float, float]],
        evaluate: EvaluateFn,
        constraint: Optional[ConstraintFn] = None,
        cfg: NSGA2Config = NSGA2Config(),
    ):
        self.bounds = bounds
        self.evaluate_fn = evaluate
        self.constraint_fn = constraint
        self.cfg = cfg
        random.seed(cfg.seed)

    # ----------------------------
    # Public API
    # ----------------------------
    def run(self) -> Tuple[List[Individual], List[Individual]]:
        """
        Returns:
          pop: final population
          pareto: feasible Pareto front (rank=0 and cv=0 if available;
                  else returns rank=0 non-dominated set even if infeasible)
        """
        pop = [self._random_individual() for _ in range(self.cfg.pop_size)]
        for ind in pop:
            self._evaluate(ind)

        for gen in range(1, self.cfg.generations + 1):
            fronts = self._fast_non_dominated_sort(pop)
            for f in fronts:
                self._crowding_distance(pop, f)

            offspring: List[Individual] = []
            while len(offspring) < self.cfg.pop_size:
                p1 = self._tournament_select(pop)
                p2 = self._tournament_select(pop)
                c1, c2 = self._sbx(p1, p2)
                self._poly_mutate(c1)
                self._poly_mutate(c2)
                self._evaluate(c1)
                self._evaluate(c2)
                offspring.append(c1)
                if len(offspring) < self.cfg.pop_size:
                    offspring.append(c2)

            # Environmental selection
            combined = pop + offspring
            fronts = self._fast_non_dominated_sort(combined)

            new_pop: List[Individual] = []
            for f in fronts:
                self._crowding_distance(combined, f)
                # sort by rank asc, crowd desc
                f_inds = sorted([combined[i] for i in f], key=lambda z: (z.rank, -z.crowd))
                if len(new_pop) + len(f_inds) <= self.cfg.pop_size:
                    new_pop.extend(f_inds)
                else:
                    needed = self.cfg.pop_size - len(new_pop)
                    f_by_crowd = sorted(f_inds, key=lambda z: z.crowd, reverse=True)
                    new_pop.extend(f_by_crowd[:needed])
                    break

            pop = new_pop

        # Extract Pareto
        fronts = self._fast_non_dominated_sort(pop)
        f0 = [pop[i] for i in fronts[0]]
        feasible = [ind for ind in f0 if ind.cv == 0.0]
        pareto = feasible if len(feasible) > 0 else f0
        return pop, pareto

    # ----------------------------
    # Core: evaluation
    # ----------------------------
    def _evaluate(self, ind: Individual) -> None:
        Nu, Eu = self.evaluate_fn(ind.x)

        # guards (optional; keep minimal)
        if not (isinstance(Nu, (int, float)) and math.isfinite(Nu)):
            Nu = -1e30  # terrible
        if not (isinstance(Eu, (int, float)) and math.isfinite(Eu)):
            Eu = 1e30   # terrible

        ind.Nu = float(Nu)
        ind.Eu = float(Eu)

        if self.constraint_fn is None:
            ind.cv = 0.0
        else:
            cv = self.constraint_fn(ind.x, ind.Nu, ind.Eu)
            if not (isinstance(cv, (int, float)) and math.isfinite(cv)):
                cv = 1e30
            ind.cv = float(max(0.0, cv))

    # ----------------------------
    # Dominance + sorting
    # ----------------------------
    def _dominates(self, a: Individual, b: Individual) -> bool:
        # Feasible dominates infeasible
        if a.cv == 0.0 and b.cv > 0.0:
            return True
        if a.cv > 0.0 and b.cv == 0.0:
            return False

        # Both infeasible: smaller violation dominates
        if a.cv > 0.0 and b.cv > 0.0:
            return a.cv < b.cv

        # Both feasible: Pareto dominance on (Nu maximize, Eu minimize)
        better_or_equal = (a.Nu >= b.Nu) and (a.Eu <= b.Eu)
        strictly_better = (a.Nu > b.Nu) or (a.Eu < b.Eu)
        return better_or_equal and strictly_better

    def _fast_non_dominated_sort(self, pop: List[Individual]) -> List[List[int]]:
        S = [[] for _ in range(len(pop))]
        n = [0 for _ in range(len(pop))]
        fronts: List[List[int]] = [[]]

        for p in range(len(pop)):
            for q in range(len(pop)):
                if p == q:
                    continue
                if self._dominates(pop[p], pop[q]):
                    S[p].append(q)
                elif self._dominates(pop[q], pop[p]):
                    n[p] += 1
            if n[p] == 0:
                pop[p].rank = 0
                fronts[0].append(p)

        i = 0
        while fronts[i]:
            next_front: List[int] = []
            for p in fronts[i]:
                for q in S[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        pop[q].rank = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)

        fronts.pop()
        return fronts

    def _crowding_distance(self, pop: List[Individual], front: List[int]) -> None:
        if len(front) == 0:
            return
        for idx in front:
            pop[idx].crowd = 0.0
        if len(front) <= 2:
            for idx in front:
                pop[idx].crowd = float("inf")
            return

        # Objective 1: Nu (maximize) -> crowding uses sorted by Nu
        f_sorted = sorted(front, key=lambda i: pop[i].Nu)
        pop[f_sorted[0]].crowd = float("inf")
        pop[f_sorted[-1]].crowd = float("inf")
        Nu_min = pop[f_sorted[0]].Nu
        Nu_max = pop[f_sorted[-1]].Nu
        Nu_range = Nu_max - Nu_min if Nu_max != Nu_min else 1.0
        for k in range(1, len(f_sorted) - 1):
            pop[f_sorted[k]].crowd += (pop[f_sorted[k + 1]].Nu - pop[f_sorted[k - 1]].Nu) / Nu_range

        # Objective 2: Eu (minimize) -> still compute on Eu values
        f_sorted = sorted(front, key=lambda i: pop[i].Eu)
        pop[f_sorted[0]].crowd = float("inf")
        pop[f_sorted[-1]].crowd = float("inf")
        Eu_min = pop[f_sorted[0]].Eu
        Eu_max = pop[f_sorted[-1]].Eu
        Eu_range = Eu_max - Eu_min if Eu_max != Eu_min else 1.0
        for k in range(1, len(f_sorted) - 1):
            pop[f_sorted[k]].crowd += (pop[f_sorted[k + 1]].Eu - pop[f_sorted[k - 1]].Eu) / Eu_range

    # ----------------------------
    # Selection
    # ----------------------------
    def _tournament_select(self, pop: List[Individual]) -> Individual:
        contenders = random.sample(range(len(pop)), k=self.cfg.tournament_k)
        best = pop[contenders[0]]
        for idx in contenders[1:]:
            c = pop[idx]
            if c.rank < best.rank:
                best = c
            elif c.rank == best.rank and c.crowd > best.crowd:
                best = c
        return best

    # ----------------------------
    # Variation operators
    # ----------------------------
    def _sbx(self, p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
        # SBX crossover on each variable
        c1 = Individual(x=dict(p1.x))
        c2 = Individual(x=dict(p2.x))

        if random.random() > self.cfg.p_crossover:
            return c1, c2

        for key, (lb, ub) in self.bounds.items():
            x1, x2 = p1.x[key], p2.x[key]
            if math.isclose(x1, x2):
                continue

            # ensure x1 <= x2 for formula, then map back
            swap = False
            if x1 > x2:
                x1, x2 = x2, x1
                swap = True

            rand = random.random()
            eta_c = self.cfg.eta_c

            beta = 1.0 + (2.0 * (x1 - lb) / (x2 - x1))
            alpha = 2.0 - beta ** (-(eta_c + 1.0))
            if rand <= 1.0 / alpha:
                betaq = (rand * alpha) ** (1.0 / (eta_c + 1.0))
            else:
                betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1.0))
            child1 = 0.5 * ((x1 + x2) - betaq * (x2 - x1))

            beta = 1.0 + (2.0 * (ub - x2) / (x2 - x1))
            alpha = 2.0 - beta ** (-(eta_c + 1.0))
            if rand <= 1.0 / alpha:
                betaq = (rand * alpha) ** (1.0 / (eta_c + 1.0))
            else:
                betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1.0))
            child2 = 0.5 * ((x1 + x2) + betaq * (x2 - x1))

            child1 = min(max(child1, lb), ub)
            child2 = min(max(child2, lb), ub)

            # map to c1/c2 respecting original ordering
            if not swap:
                c1.x[key] = float(child1)
                c2.x[key] = float(child2)
            else:
                c1.x[key] = float(child2)
                c2.x[key] = float(child1)

        return c1, c2

    def _poly_mutate(self, ind: Individual) -> None:
        eta_m = self.cfg.eta_m
        for key, (lb, ub) in self.bounds.items():
            if random.random() > self.cfg.p_mutation:
                continue
            x = ind.x[key]
            if ub <= lb:
                continue
            delta1 = (x - lb) / (ub - lb)
            delta2 = (ub - x) / (ub - lb)
            rand = random.random()
            mut_pow = 1.0 / (eta_m + 1.0)

            if rand < 0.5:
                val = 2.0 * rand + (1.0 - 2.0 * rand) * ((1.0 - delta1) ** (eta_m + 1.0))
                deltaq = val ** mut_pow - 1.0
            else:
                val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * ((1.0 - delta2) ** (eta_m + 1.0))
                deltaq = 1.0 - val ** mut_pow

            x_new = x + deltaq * (ub - lb)
            ind.x[key] = float(min(max(x_new, lb), ub))

    # ----------------------------
    # Utilities
    # ----------------------------
    def _random_individual(self) -> Individual:
        x = {}
        for key, (lb, ub) in self.bounds.items():
            x[key] = float(lb + (ub - lb) * random.random())
        return Individual(x=x)
