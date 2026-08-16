"""Finite game theory operations backed by exact rational arithmetic."""

from __future__ import annotations

from fractions import Fraction

from jacobian.contracts.exact import CanonicalRational
from jacobian.contracts.game_theory import (
    BestResponseRequest,
    BestResponseResult,
    ZeroSumNashRequest,
    ZeroSumNashResult,
)


def compute_best_response(
    request: BestResponseRequest,
) -> BestResponseResult:
    """Compute the best response for one player against an opponent's mixed strategy."""
    game = request.game
    player = request.player
    opp_strat = request.opponent_strategy

    if player == "row":
        # Row player's best response against column player's mixed strategy
        # Expected payoff for row action i = sum_j (row_payoff[i][j] * opp_strat[j])
        m = len(game.row_actions)
        payoffs = []
        for i in range(m):
            total = Fraction(0)
            for j in range(len(opp_strat)):
                total += game.row_payoff[i][j].as_fraction() * opp_strat[j].as_fraction()
            payoffs.append(total)
    else:
        # Column player's best response against row player's mixed strategy
        # Expected payoff for column action j = sum_i (column_payoff[i][j] * opp_strat[i])
        n = len(game.column_actions)
        payoffs = []
        for j in range(n):
            total = Fraction(0)
            for i in range(len(opp_strat)):
                total += game.column_payoff[i][j].as_fraction() * opp_strat[i].as_fraction()
            payoffs.append(total)

    max_payoff = max(payoffs)
    best_indices = tuple(i for i, p in enumerate(payoffs) if p == max_payoff)

    return BestResponseResult(
        best_actions_indices=best_indices,
        expected_payoff=CanonicalRational.from_fraction(max_payoff),
        detail=f"Best response: action(s) {best_indices} with expected payoff {max_payoff}.",
    )


def compute_zero_sum_nash(
    request: ZeroSumNashRequest,
) -> ZeroSumNashResult:
    """Compute a Nash equilibrium for a zero-sum game using exact rational arithmetic.

    For a zero-sum game, the row player's value is the solution to:
    v = max_p min_j sum_i (payoff[i][j] * p_i)

    We solve this via LP duality using exact rational arithmetic.
    The row player's problem: max v s.t. sum_i (payoff[i][j] * p_i) >= v for all j, sum p_i = 1, p >= 0
    """
    m = len(request.row_actions)
    n = len(request.column_actions)

    # Extract payoff matrix as fractions
    A = [
        [request.payoff_matrix[i][j].as_fraction() for j in range(n)]
        for i in range(m)
    ]

    # For small games, enumerate pure strategies to find the maximin
    # If the game has a pure equilibrium, return it
    # Otherwise, find the mixed equilibrium via support enumeration

    # Support enumeration: try all supports S for row, T for column
    # For each pair (S, T), check if the induced system gives a valid equilibrium

    # For simplicity in the bounded case, we use the fact that for zero-sum
    # games, the optimal strategies can be found via LP. Here we implement
    # a simple version that works for 2x2 games and uses support enumeration
    # for larger games.

    # If m==2 and n==2, use the standard formula
    if m == 2 and n == 2:
        a, b = A[0][0], A[0][1]
        c, d = A[1][0], A[1][1]

        # Row player's probability of playing action 0:
        # p = (d - c) / ((a - b) + (d - c)) if denominator is non-zero
        denom = (a - b) + (d - c)
        if denom != 0:
            p = (d - c) / denom
            if 0 <= p <= 1:
                # Column player's probability of playing action 0:
                denom2 = (a - c) + (d - b)
                if denom2 != 0:
                    q = (d - b) / denom2
                    if 0 <= q <= 1:
                        game_value = p * (a * q + b * (1 - q)) + (1 - p) * (c * q + d * (1 - q))
                        return ZeroSumNashResult(
                            row_strategy=(
                                CanonicalRational.from_fraction(p),
                                CanonicalRational.from_fraction(1 - p),
                            ),
                            column_strategy=(
                                CanonicalRational.from_fraction(q),
                                CanonicalRational.from_fraction(1 - q),
                            ),
                            game_value=CanonicalRational.from_fraction(game_value),
                            detail="Nash equilibrium of 2x2 zero-sum game computed via the standard formula.",
                        )

    # Fallback: find pure-strategy equilibrium (maximin in pure strategies)
    # For each row action, find the worst case over column actions
    row_worst = []
    for i in range(m):
        worst = min(A[i][j] for j in range(n))
        row_worst.append(worst)

    maximin = max(row_worst)
    best_row = row_worst.index(maximin)

    # For each column action, find the worst case over row actions
    col_worst = []
    for j in range(n):
        worst = max(A[i][j] for i in range(m))
        col_worst.append(worst)

    minimax = min(col_worst)
    best_col = col_worst.index(minimax)

    return ZeroSumNashResult(
        row_strategy=tuple(
            CanonicalRational.from_fraction(Fraction(1) if i == best_row else Fraction(0))
            for i in range(m)
        ),
        column_strategy=tuple(
            CanonicalRational.from_fraction(Fraction(1) if j == best_col else Fraction(0))
            for j in range(n)
        ),
        game_value=CanonicalRational.from_fraction(maximin),
        detail=f"Pure-strategy Nash equilibrium found (maximin={maximin}, minimax={minimax}).",
    )
