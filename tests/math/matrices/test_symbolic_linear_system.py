"""Tests for symbolic linear system solving over QQ(t_1, ..., t_n)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.matrices.symbolic._models import (
    SymbolicLinearSystemRequest,
    SymbolicMatrix,
)
from jacobian.math.matrices.symbolic._operations import (
    compute_symbolic_linear_system,
    verify_symbolic_linear_system_result,
)
from jacobian.math.polynomials.values import RationalFunction


def _rf(
    variables: tuple[str, ...],
    *numerator_terms: tuple[int, tuple[int, ...]],
) -> RationalFunction:
    """Build a rational function from coefficient, exponent tuples."""
    return RationalFunction.model_validate(
        {
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
        assert verify_symbolic_linear_system_result(result)

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
        assert verify_symbolic_linear_system_result(result)

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
        with pytest.raises(ValidationError):
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
        with pytest.raises(ValidationError):
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
        with pytest.raises(ValidationError):
            SymbolicLinearSystemRequest(matrix=matrix, rhs=(zero,) * 8)

    def test_dense_eight_by_eight_system_is_rejected_quickly(self):
        """A fully dense 8x8 unit system exceeds the derived term budget."""
        import time

        import pytest
        from pydantic import ValidationError

        one = _rf((), (1, ()))
        matrix = _matrix((), ((one,) * 8,) * 8)
        started = time.perf_counter()
        with pytest.raises(ValidationError):
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
        assert verify_symbolic_linear_system_result(revalidated)

    def test_forged_solution_for_other_system_fails_explicit_verification(self):
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
        # Same shape as the foreign 1x1 system so only explicit source-bound
        # verification, rather than transport-shape validation, rejects it.
        payload["solution"] = (three,)
        assert not verify_symbolic_linear_system_result(
            SymbolicLinearSystemResult.model_validate(payload)
        )

    def test_wrong_solution_vector_fails_explicit_verification(self):
        from jacobian.math.matrices.symbolic._models import (
            SymbolicLinearSystemResult,
        )

        result = compute_symbolic_linear_system(self._request())
        payload = result.model_dump()
        wrong = _rf(("a", "b"), (5, (1, 1)))
        payload["solution"] = (wrong, wrong)
        assert not verify_symbolic_linear_system_result(
            SymbolicLinearSystemResult.model_validate(payload)
        )


class TestNativeSystemAdmission:
    """The native solve validates the complete request before SymPy runs."""

    @staticmethod
    def _solve(entries, rhs, variables=()):
        from jacobian.math.matrices.symbolic.operations import (
            symbolic_linear_system_solve,
        )

        return symbolic_linear_system_solve(entries, rhs, variables)

    def test_short_rhs_rejected_before_sympy(self):
        vars_: tuple[str, ...] = ()
        one = _rf(vars_, (1, ()))
        with pytest.raises(ValueError):
            self._solve(((one,), (one,)), (one,), vars_)

    def test_field_mismatch_rejected(self):
        entry = _rf(("t",), (1, (0,)))
        with pytest.raises(ValueError):
            self._solve(((entry,),), (entry,), ())

    def test_growth_budget_applied_to_native_callers(self):
        """[[1/t^64]] x = [t^64] would solve to t^128: rejected natively."""
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
        rhs = (_rf(("t",), (1, (64,))),)
        with pytest.raises(ValueError):
            self._solve(((inv,),), rhs, ("t",))

    def test_oversized_native_shape_rejected_before_growth_scan(self):
        """A wide native matrix is rejected by the wire dimension limits
        before growth admission scans its columns."""
        import time

        vars_: tuple[str, ...] = ()
        one = _rf(vars_, (1, ()))
        started = time.perf_counter()
        with pytest.raises(ValueError):
            self._solve(((one,) * 20_000,), (one,), vars_)
        assert time.perf_counter() - started < 5.0


class TestNonUniqueWitnessEquivalence:
    """NON_UNIQUE witnesses validate by their defining equations."""

    @staticmethod
    def _non_unique_result():
        def rf(num: int, den: int) -> RationalFunction:
            terms = (
                [{"coefficient": {"num": str(num), "den": str(den)}, "exponents": [0]}]
                if num != 0
                else []
            )
            return RationalFunction.model_validate(
                {
                    "domain": "QQ",
                    "variables": ["t"],
                    "numerator": {"terms": terms},
                    "denominator": {
                        "terms": [
                            {
                                "coefficient": {"num": "1", "den": "1"},
                                "exponents": [0],
                            }
                        ]
                    },
                }
            )

        matrix = _matrix(("t",), ((rf(1, 3), rf(0, 1)), (rf(0, 1), rf(0, 1))))
        rhs = (rf(3, 1), rf(0, 1))
        return compute_symbolic_linear_system(
            SymbolicLinearSystemRequest(matrix=matrix, rhs=rhs)
        )

    def test_alternative_valid_witnesses_pass_explicit_verification(self):
        from jacobian.math.matrices.symbolic._models import (
            SymbolicLinearSystemResult,
        )

        def rf(num: int) -> RationalFunction:
            return _rf(("t",), (num, (0,)))

        payload = self._non_unique_result().model_dump()
        # (9, 4) solves the system exactly and (0, 5) spans the same kernel
        # line as the replayed witness; neither matches backend identity.
        payload["particular_solution"] = [rf(9).model_dump(), rf(4).model_dump()]
        basis_vector = [_rf(("t",)).model_dump(), rf(5).model_dump()]
        payload["nullspace_basis"] = [basis_vector]
        revalidated = SymbolicLinearSystemResult.model_validate(payload)
        assert revalidated.classification == "NON_UNIQUE"
        assert verify_symbolic_linear_system_result(revalidated)

    def test_wrong_particular_solution_fails_explicit_verification(self):
        from jacobian.math.matrices.symbolic._models import (
            SymbolicLinearSystemResult,
        )

        zero = _rf(("t",))
        wrong = _rf(("t",), (8, (0,)))
        payload = self._non_unique_result().model_dump()
        payload["particular_solution"] = [wrong.model_dump(), zero.model_dump()]
        assert not verify_symbolic_linear_system_result(
            SymbolicLinearSystemResult.model_validate(payload)
        )

    def test_non_kernel_basis_vector_fails_explicit_verification(self):
        from jacobian.math.matrices.symbolic._models import (
            SymbolicLinearSystemResult,
        )

        one_one = _rf(("t",), (1, (1,)), (1, (0,)))
        zero = _rf(("t",))
        payload = self._non_unique_result().model_dump()
        payload["nullspace_basis"] = [[one_one.model_dump(), zero.model_dump()]]
        assert not verify_symbolic_linear_system_result(
            SymbolicLinearSystemResult.model_validate(payload)
        )

    def test_incomplete_or_degenerate_basis_fails_explicit_verification(self):
        """Duplicated vectors cannot satisfy the verifier's basis invariant."""
        from jacobian.math.matrices.symbolic._models import (
            SymbolicLinearSystemResult,
        )

        zero = _rf(("t",))
        one = _rf(("t",), (1, (0,)))
        two = _rf(("t",), (2, (0,)))
        payload = self._non_unique_result().model_dump()
        # Both vectors lie in the kernel line but are dependent and exceed
        # the kernel dimension.
        payload["nullspace_basis"] = [
            [zero.model_dump(), one.model_dump()],
            [zero.model_dump(), two.model_dump()],
        ]
        assert not verify_symbolic_linear_system_result(
            SymbolicLinearSystemResult.model_validate(payload)
        )


class TestWitnessDeserializationHardening:
    """Relayed payloads are rejected by contract, not backend arithmetic."""

    @staticmethod
    def _inconsistent_payload():
        one = _rf((), (1, ()))
        two = _rf((), (2, ()))
        matrix = _matrix((), ((one,), (one,)))
        result = compute_symbolic_linear_system(
            SymbolicLinearSystemRequest(matrix=matrix, rhs=(one, two))
        )
        assert result.classification == "INCONSISTENT"
        return result.model_dump()

    def test_inconsistent_result_rejects_nullspace_basis(self):
        from jacobian.math.matrices.symbolic._models import (
            SymbolicLinearSystemResult,
        )

        payload = self._inconsistent_payload()
        payload["nullspace_basis"] = [[_rf((), (1, ())).model_dump()]]
        with pytest.raises(ValidationError):
            SymbolicLinearSystemResult.model_validate(payload)

    def test_witness_field_mismatch_rejected(self):
        """A zero system over QQ must not accept witnesses over QQ(z)."""
        from jacobian.math.matrices.symbolic._models import (
            SymbolicLinearSystemResult,
        )

        zero_t = _rf(("t",))
        matrix = _matrix(("t",), ((zero_t,),))
        result = compute_symbolic_linear_system(
            SymbolicLinearSystemRequest(matrix=matrix, rhs=(zero_t,))
        )
        assert result.classification == "NON_UNIQUE"
        payload = result.model_dump()
        # Every residual stays zero and the basis stays independent over the
        # foreign field, so only the declared-field check can reject it.
        foreign_zero = _rf(("z",)).model_dump()
        payload["particular_solution"] = [foreign_zero]
        payload["nullspace_basis"] = [[_rf(("z",), (1, (0,))).model_dump()]]
        with pytest.raises(ValidationError):
            SymbolicLinearSystemResult.model_validate(payload)

    def test_undersized_particular_solution_rejected_before_arithmetic(self):
        from jacobian.math.matrices.symbolic._models import (
            SymbolicLinearSystemResult,
        )

        payload = TestNonUniqueWitnessEquivalence._non_unique_result().model_dump()
        payload["particular_solution"] = [_rf(("t",), (3, (0,))).model_dump()]
        with pytest.raises(ValidationError):
            SymbolicLinearSystemResult.model_validate(payload)

    def test_undersized_nullspace_vector_rejected_before_arithmetic(self):
        from jacobian.math.matrices.symbolic._models import (
            SymbolicLinearSystemResult,
        )

        payload = TestNonUniqueWitnessEquivalence._non_unique_result().model_dump()
        payload["nullspace_basis"] = [[_rf(("t",), (1, (0,))).model_dump()]]
        with pytest.raises(ValidationError):
            SymbolicLinearSystemResult.model_validate(payload)


class TestRelayedPayloadShapeCaps:
    """Solution payloads are capped before nested rational functions parse."""

    def test_oversized_solution_tuple_rejected_pre_parsing(self):
        from jacobian.math.matrices.symbolic._models import (
            SymbolicLinearSystemResult,
        )

        payload = self._non_unique_payload()
        filler = _rf(("t",)).model_dump()
        payload["solution"] = None
        payload["particular_solution"] = [filler] * 50
        with pytest.raises(ValidationError):
            SymbolicLinearSystemResult.model_validate(payload)

    def test_oversized_nullspace_vector_rejected_pre_parsing(self):
        from jacobian.math.matrices.symbolic._models import (
            SymbolicLinearSystemResult,
        )

        payload = self._non_unique_payload()
        long_vector = [_rf(("t",)).model_dump() for _ in range(7)]
        payload["nullspace_basis"] = [long_vector]
        with pytest.raises(ValidationError):
            SymbolicLinearSystemResult.model_validate(payload)

    @staticmethod
    def _non_unique_payload():
        result = TestNonUniqueWitnessEquivalence._non_unique_result()
        return result.model_dump()
