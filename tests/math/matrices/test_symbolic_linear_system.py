"""Tests for symbolic linear system solving over QQ(t_1, ..., t_n)."""

from __future__ import annotations

from jacobian.math.matrices.symbolic._models import (
    SymbolicLinearSystemRequest,
    SymbolicMatrix,
)
from jacobian.math.matrices.symbolic._operations import compute_symbolic_linear_system
from jacobian.math.polynomials.values import RationalFunction


def _rf(
    variables: tuple[str, ...],
    *numerator_terms: tuple[int, tuple[int, ...]],
) -> RationalFunction:
    """Build a rational function from coefficient, exponent tuples."""
    return RationalFunction.model_validate(
        {
            "rational_function_schema_version": "1",
            "domain": "QQ",
            "variables": list(variables),
            "numerator": {
                "terms": [
                    {
                        "coefficient": {"num": str(c), "den": "1"},
                        "exponents": list(e),
                    }
                    for c, e in numerator_terms
                ]
            },
            "denominator": {
                "terms": [
                    {
                        "coefficient": {"num": "1", "den": "1"},
                        "exponents": [0] * len(variables),
                    }
                ]
            },
        }
    )


def _matrix(
    variables: tuple[str, ...],
    entries: tuple[tuple[RationalFunction, ...], ...],
) -> SymbolicMatrix:
    return SymbolicMatrix.model_validate(
        {
            "variables": list(variables),
            "entries": entries,
        }
    )


class TestSymbolicLinearSystem:
    """Tests for ``matrix.symbolic.linear_system.solve``."""

    def test_unique_solution_1x1(self):
        """Solve [[t]] * x = [1] -> x = 1/t."""
        vars_ = ("t",)
        matrix = _matrix(vars_, ((_rf(vars_, (1, (1,))),),))
        rhs = (_rf(vars_, (1, (0,))),)
        req = SymbolicLinearSystemRequest(matrix=matrix, rhs=rhs)
        result = compute_symbolic_linear_system(req)
        assert result.classification == "UNIQUE"
        assert result.solution is not None
        assert len(result.solution) == 1

    def test_unique_solution_identity_2x2(self):
        """Solve [[1, a], [b, 1]] * x = [1, 1] for a symbolic 2x2 system."""
        vars_ = ("a", "b")
        one = _rf(vars_, (1, (0, 0)))
        a_val = _rf(vars_, (1, (1, 0)))
        b_val = _rf(vars_, (1, (0, 1)))
        matrix = _matrix(
            vars_,
            (
                (one, a_val),
                (b_val, one),
            ),
        )
        rhs = (one, one)
        req = SymbolicLinearSystemRequest(matrix=matrix, rhs=rhs)
        result = compute_symbolic_linear_system(req)
        assert result.classification == "UNIQUE"
        assert result.solution is not None
        assert len(result.solution) == 2

    def test_rectangular_full_column_rank_system(self):
        """An overdetermined full-rank system [[1],[2]] x = [1,2] is UNIQUE.

        SymPy's LUsolve rejects rectangular systems, so the unique branch
        must read the solution from an exact RREF instead.
        """
        vars_: tuple[str, ...] = ()
        one = _rf(vars_, (1, ()))
        two = _rf(vars_, (2, ()))
        matrix = _matrix(vars_, ((one,), (two,)))
        rhs = (one, two)
        req = SymbolicLinearSystemRequest(matrix=matrix, rhs=rhs)
        result = compute_symbolic_linear_system(req)
        assert result.classification == "UNIQUE"
        assert result.solution is not None
        assert result.solution[0].numerator.terms[0].coefficient.num == "1"

    def test_inconsistent_system(self):
        """Solve [[1], [1]] * x = [1, 2] -> INCONSISTENT."""
        vars_: tuple[str, ...] = ()
        one = _rf(vars_, (1, ()))
        matrix = _matrix(vars_, ((one,), (one,)))
        rhs = (_rf(vars_, (1, ())), _rf(vars_, (2, ())))
        req = SymbolicLinearSystemRequest(matrix=matrix, rhs=rhs)
        result = compute_symbolic_linear_system(req)
        assert result.classification == "INCONSISTENT"
        assert result.solution is None
        assert result.particular_solution is None

    def test_orthocenter_system(self):
        """Solve a 2x2 symbolic system from the orthocenter derivation."""
        vars_ = ("a", "b", "c", "d", "e", "f")
        a = _rf(vars_, (1, (1, 0, 0, 0, 0, 0)))
        b = _rf(vars_, (1, (0, 1, 0, 0, 0, 0)))
        c = _rf(vars_, (1, (0, 0, 1, 0, 0, 0)))
        d = _rf(vars_, (1, (0, 0, 0, 1, 0, 0)))
        e = _rf(vars_, (1, (0, 0, 0, 0, 1, 0)))
        f = _rf(vars_, (1, (0, 0, 0, 0, 0, 1)))
        matrix = _matrix(
            vars_,
            (
                (a, b),
                (c, d),
            ),
        )
        rhs = (e, f)
        req = SymbolicLinearSystemRequest(matrix=matrix, rhs=rhs)
        result = compute_symbolic_linear_system(req)
        assert result.classification == "UNIQUE"
        assert result.solution is not None
        assert len(result.solution) == 2

    def test_constant_system(self):
        """Solve [[2]] * x = [6] -> x = 3 (over QQ)."""
        vars_: tuple[str, ...] = ()
        two = _rf(vars_, (2, ()))
        six = _rf(vars_, (6, ()))
        matrix = _matrix(vars_, ((two,),))
        rhs = (six,)
        req = SymbolicLinearSystemRequest(matrix=matrix, rhs=rhs)
        result = compute_symbolic_linear_system(req)
        assert result.classification == "UNIQUE"
        assert result.solution is not None
        # Solution should be 3
        sol = result.solution[0]
        assert sol.numerator.terms[0].coefficient.num == "3"

    def test_rational_function_rhs(self):
        """Solve [[1]] * x = [1/t] -> x = 1/t."""
        vars_ = ("t",)
        one = _rf(vars_, (1, (0,)))
        matrix = _matrix(vars_, ((one,),))
        # Build rhs = 1/t as a rational function
        rhs = RationalFunction.model_validate(
            {
                "rational_function_schema_version": "1",
                "domain": "QQ",
                "variables": list(vars_),
                "numerator": {
                    "terms": [
                        {
                            "coefficient": {"num": "1", "den": "1"},
                            "exponents": [0],
                        }
                    ]
                },
                "denominator": {
                    "terms": [
                        {
                            "coefficient": {"num": "1", "den": "1"},
                            "exponents": [1],
                        }
                    ]
                },
            }
        )
        req = SymbolicLinearSystemRequest(matrix=matrix, rhs=(rhs,))
        result = compute_symbolic_linear_system(req)
        assert result.classification == "UNIQUE"
        assert result.solution is not None


class TestSolutionGrowthAdmission:
    """Derived-solution growth is bounded at request admission."""

    def test_solution_exponent_overflow_rejected_at_request(self):
        """[[1/t^64]] * x = [t^64] would solve to t^128, outside the result
        type; the request is rejected before the backend runs."""
        import pytest
        from pydantic import ValidationError

        vars_ = ("t",)
        inv = RationalFunction.model_validate(
            {
                "variables": ["t"],
                "numerator": {
                    "terms": [
                        {"coefficient": {"num": "1", "den": "1"}, "exponents": [0]}
                    ]
                },
                "denominator": {
                    "terms": [
                        {"coefficient": {"num": "1", "den": "1"}, "exponents": [64]}
                    ]
                },
            }
        )
        matrix = _matrix(vars_, ((inv,),))
        rhs = (_rf(vars_, (1, (64,))),)
        with pytest.raises(ValidationError, match="exponent"):
            SymbolicLinearSystemRequest(matrix=matrix, rhs=rhs)

    def test_rank_deficient_large_coefficients_rejected(self):
        """All work-size minors can be structurally zero while a smaller
        minor drives a large particular solution; lower-rank minors are
        included in the growth bound."""
        import pytest
        from pydantic import ValidationError

        def rf(num: int, den: int) -> RationalFunction:
            terms = (
                [{"coefficient": {"num": str(num), "den": str(den)}, "exponents": [0]}]
                if num != 0
                else []
            )
            return RationalFunction.model_validate(
                {
                    "variables": ["t"],
                    "numerator": {"terms": terms},
                    "denominator": {
                        "terms": [
                            {"coefficient": {"num": "1", "den": "1"}, "exponents": [0]}
                        ]
                    },
                }
            )

        big = 10**127
        matrix = _matrix(("t",), ((rf(1, big), rf(0, 1)), (rf(0, 1), rf(0, 1))))
        rhs = (rf(big, 1), rf(0, 1))
        with pytest.raises(ValidationError, match="coefficient"):
            SymbolicLinearSystemRequest(matrix=matrix, rhs=rhs)

    def test_rank_deficient_particular_solution_is_bounded_by_small_minor(self):
        """The reviewer's rank-deficient shape with representable sizes:
        [[1/N, 0], [0, 0]] x = [N, 0] has the exact particular solution
        x_0 = N^2 driven by a size-1 minor while every work-size (size-2)
        augmented minor is structurally zero; a small N is admitted and
        solved exactly."""

        def rf(num: int, den: int) -> RationalFunction:
            terms = (
                [{"coefficient": {"num": str(num), "den": str(den)}, "exponents": [0]}]
                if num != 0
                else []
            )
            return RationalFunction.model_validate(
                {
                    "variables": ["t"],
                    "numerator": {"terms": terms},
                    "denominator": {
                        "terms": [
                            {"coefficient": {"num": "1", "den": "1"}, "exponents": [0]}
                        ]
                    },
                }
            )

        n = 3
        matrix = _matrix(("t",), ((rf(1, n), rf(0, 1)), (rf(0, 1), rf(0, 1))))
        rhs = (rf(n, 1), rf(0, 1))
        request = SymbolicLinearSystemRequest(matrix=matrix, rhs=rhs)
        result = compute_symbolic_linear_system(request)
        assert result.classification == "NON_UNIQUE"
        assert result.particular_solution is not None
        assert result.particular_solution[0] == rf(n**2, 1)
        assert result.nullspace_basis is not None


class TestAdmissionWorkBounding:
    """Growth admission derives its bounds without factorial enumeration."""

    def _identity(self, size: int):
        one = _rf((), (1, ()))
        zero = _rf(())
        entries = tuple(
            tuple(one if i == j else zero for j in range(size)) for i in range(size)
        )
        return _matrix((), entries), one, zero

    def test_identity_eight_by_eight_zero_rhs_round_trips(self):
        """The trivial 8x8 identity system with a zero right-hand side is
        admitted, solved, and revalidated from its serialized result."""
        from jacobian.math.matrices.symbolic._models import (
            SymbolicLinearSystemResult,
        )

        matrix, _, zero = self._identity(8)
        request = SymbolicLinearSystemRequest(matrix=matrix, rhs=(zero,) * 8)
        result = compute_symbolic_linear_system(request)
        assert result.classification == "UNIQUE"
        assert result.solution is not None
        assert all(
            value.numerator.terms == ()
            and value.denominator.terms[0].coefficient.num == "1"
            for value in result.solution
        )
        revalidated = SymbolicLinearSystemResult.model_validate(result.model_dump())
        assert revalidated.solution == result.solution

    def test_diagonal_system_with_unit_rhs_keeps_exact_sparse_bounds(self):
        """An 8x8 diagonal system with unit right-hand side stays admitted.

        Its exact per-size expansion is 1; the loose closed-form fallback
        would charge the right-hand-side column degree 8 and reject on the
        coefficient budget, so acceptance pins the exact enumeration path.
        """
        import time

        matrix, one, _ = self._identity(8)
        started = time.perf_counter()
        request = SymbolicLinearSystemRequest(matrix=matrix, rhs=(one,) * 8)
        assert time.perf_counter() - started < 5.0
        result = compute_symbolic_linear_system(request)
        assert result.classification == "UNIQUE"

    def test_exhausted_enumeration_falls_back_to_sound_closed_form(self, monkeypatch):
        """With the exact enumeration budget at zero every size uses the
        closed-form injection bound: still sound, coarser on sparse data.

        The star system's exact expansion is 1, but its first row and
        column support degrees force a closed-form bound of 8, which
        exceeds the coefficient budget; both verdicts stay typed."""
        import pytest
        from pydantic import ValidationError

        from jacobian.math.matrices.symbolic import _models

        one = _rf((), (1, ()))
        zero = _rf(())
        entries = tuple(
            tuple(one for _ in range(8))
            if i == 0
            else tuple(one if j == 0 else zero for j in range(8))
            for i in range(8)
        )
        matrix = _matrix((), entries)
        request = SymbolicLinearSystemRequest(matrix=matrix, rhs=(zero,) * 8)
        assert request.matrix.entries == entries
        monkeypatch.setattr(_models, "_EXPANSION_ENUMERATION_NODE_BUDGET", 0)
        with pytest.raises(ValidationError, match="coefficient"):
            SymbolicLinearSystemRequest(matrix=matrix, rhs=(zero,) * 8)

    def test_dense_eight_by_eight_system_is_rejected_quickly(self):
        """A fully dense 8x8 unit system exceeds the derived term budget."""
        import time

        import pytest
        from pydantic import ValidationError

        one = _rf((), (1, ()))
        matrix = _matrix((), ((one,) * 8,) * 8)
        started = time.perf_counter()
        with pytest.raises(ValidationError, match="term budget"):
            SymbolicLinearSystemRequest(matrix=matrix, rhs=(one,) * 8)
        assert time.perf_counter() - started < 5.0


class TestSourceBoundResult:
    """The retained result must be verifiable against its source system."""

    def _request(self):
        vars_ = ("a", "b")
        one = _rf(vars_, (1, (0, 0)))
        a_val = _rf(vars_, (1, (1, 0)))
        b_val = _rf(vars_, (1, (0, 1)))
        matrix = _matrix(vars_, ((one, a_val), (b_val, one)))
        return SymbolicLinearSystemRequest(matrix=matrix, rhs=(one, one))

    def test_result_retains_source_system(self):
        request = self._request()
        result = compute_symbolic_linear_system(request)
        assert result.system == request

    def test_serialized_result_revalidates(self):
        from jacobian.math.matrices.symbolic._models import (
            SymbolicLinearSystemResult,
        )

        result = compute_symbolic_linear_system(self._request())
        revalidated = SymbolicLinearSystemResult.model_validate(result.model_dump())
        assert revalidated.classification == "UNIQUE"
        assert revalidated.solution == result.solution

    def test_forged_solution_for_other_system_is_rejected(self):
        import pytest
        from pydantic import ValidationError

        from jacobian.math.matrices.symbolic._models import (
            SymbolicLinearSystemResult,
        )

        result = compute_symbolic_linear_system(self._request())
        payload = result.model_dump()
        vars_: tuple[str, ...] = ()
        three = _rf(vars_, (3, ()))
        payload["system"] = {
            "matrix": {"variables": [], "entries": ((three,),)},
            "rhs": (_rf(vars_, (7, ())),),
        }
        with pytest.raises(ValidationError, match="exact solve"):
            SymbolicLinearSystemResult.model_validate(payload)

    def test_wrong_solution_vector_is_rejected(self):
        import pytest
        from pydantic import ValidationError

        from jacobian.math.matrices.symbolic._models import (
            SymbolicLinearSystemResult,
        )

        result = compute_symbolic_linear_system(self._request())
        payload = result.model_dump()
        wrong = _rf(("a", "b"), (5, (1, 1)))
        payload["solution"] = (wrong, wrong)
        with pytest.raises(ValidationError, match="exact solve"):
            SymbolicLinearSystemResult.model_validate(payload)


class TestNativeSystemAdmission:
    def test_native_solve_rejects_short_rhs(self):
        import pytest

        from jacobian.math.matrices.symbolic.operations import (
            symbolic_linear_system_solve,
        )

        entries = ((_rf(("t",), (1, (0,))),),)
        with pytest.raises(ValueError, match="right-hand side length"):
            symbolic_linear_system_solve(entries, (), ())

    def test_native_solve_rejects_mismatched_variable_axis(self):
        import pytest

        from jacobian.math.matrices.symbolic.operations import (
            symbolic_linear_system_solve,
        )

        entries = ((_rf(("t",), (1, (0,))),),)
        with pytest.raises(ValueError, match="declared ordered field"):
            symbolic_linear_system_solve(entries, (_rf(("s",), (1, (0,))),), ("t",))
        with pytest.raises(ValueError, match="declared ordered field"):
            symbolic_linear_system_solve(
                (
                    (
                        _rf(("t",), (1, (0,))),
                        _rf(("t", "s"), (1, (0, 1))),
                    ),
                ),
                (_rf(("t",), (1, (0,))),),
                ("t",),
            )
