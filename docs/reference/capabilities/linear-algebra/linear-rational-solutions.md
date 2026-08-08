# Exact rational linear-system evidence

[Documentation home](../../../index.md)

- Status: Experimental pre-stable contract
- Provider profile: Python-FLINT 0.9.0, `fmpq_mat.rref`
- Domain: finite systems `A x = b` over `QQ`
- Maximum shape: 32 equations by 32 declared variables

## Capability boundary

Solution vectors and inconsistency certificates are separate mathematical
outcomes. Their producers and verifiers are also separate operations:

| Capability | Observable outcome | Assurance |
| --- | --- | --- |
| `linear.rational_solution.compute` | One exact vector for the supplied rational system, or no vector | `COMPUTED` when the bounded provider attempt completes; never self-verified |
| `linear.rational_solution.verify` | Independent replay of every equation for one inline vector | `VERIFIED` only after the operator-authorized checker creates a durable verification record |
| `linear.rational_inconsistency.compute` | One normalized left witness `y` with proposed relations `y^T A = 0` and `y^T b = 1`, or no witness | `COMPUTED` when the bounded provider attempt completes; never self-verified |
| `linear.rational_inconsistency.verify` | Independent replay of every left-witness equation and the nonzero pairing | `VERIFIED` only after the operator-authorized checker creates a durable verification record |

The producers run only when the exact optional Python-FLINT distribution is
available. Install the pinned wheel with:

```sh
uv sync --extra flint
```

The verifiers do not import Python-FLINT, SymPy, or the producers. They use
standard-library `fractions.Fraction` arithmetic in the existing clean-process
checker boundary. Bundled checker authorization remains an explicit
`checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED` operator decision.

## Exact system input

A request declares the ordered variable list, coefficient matrix, right-hand
side, and wall-time budget:

```json
{
  "system": {
    "variables": ["x", "y"],
    "coefficients": {
      "entries": [
        [
          {"num": "2", "den": "1"},
          {"num": "1", "den": "1"}
        ],
        [
          {"num": "1", "den": "1"},
          {"num": "-1", "den": "1"}
        ]
      ]
    },
    "rhs": [
      {"num": "5", "den": "1"},
      {"num": "1", "den": "1"}
    ]
  },
  "resource_budget": {"wall_seconds": 5}
}
```

Every rational is a reduced numerator and positive denominator encoded as
canonical decimal strings. Negative zero, leading zeroes, zero denominators,
and unreduced values are rejected. Each numerator and denominator is limited
to 256 decimal digits. The coefficient row count must equal the right-hand
side length, and the column count must equal the number of unique declared
variables.

The variable order is semantic. The inline result preserves the declared order,
and the v2 verification binding covers the exact input and candidate so a vector
cannot be silently rebound to permuted columns.

## Solution producer behavior

The producer starts an isolated Python process with a fixed locale, timezone,
and hash seed. Input, output, and wall time are bounded. The worker:

1. constructs the augmented rational matrix `[A | b]`;
2. calls Python-FLINT 0.9.0 `fmpq_mat.rref`;
3. rejects a reduced row whose coefficient entries are all zero and whose
   right-hand side is nonzero;
4. otherwise sets every free variable to zero and reads one value for each
   pivot variable; and
5. returns canonical numerator and denominator strings.

The produced vector is returned inline as a typed result. No system, vector, or
candidate artifact is created for this ordinary bounded value. Its relationship
to the supplied system is still only proposed until the independent verifier
accepts the exact input/candidate binding.

`NO_SOLUTION_PRODUCED` has `UNKNOWN` conclusion and unknown completeness. It
does not assert that the system is inconsistent.

## Solution verification

The authorized `linear.rational_solution.verify` checker accepts only when all
of these hold:

- the v2 checker request, system, solution, semantics, and binding have their
  exact closed shapes;
- the solution has one canonical rational for every declared variable; and
- standard-library exact arithmetic confirms every full equation
  `sum(A[i][j] * x[j]) == b[i]`.

Wrong values, partial vectors, reordered variables, changed equations,
noncanonical rationals, provider substitutions, extra fields, and candidates
bound to another input are rejected. Rejection, timeout, cancellation, runtime
replacement, or malformed checker output remains `UNKNOWN` and cannot carry a
verification record. The accepted verification record is the durable evidence;
the ordinary mathematical values remain inline.

## Inconsistency certificate

For an inconsistent system, the certificate producer solves the dual exact
rational system

```text
A^T y = 0
b^T y = 1
```

with the same bounded isolated RREF worker. The second equality normalizes the
witness and makes its nonzero pairing explicit. For example,

```text
A = [[1, 1], [2, 2]]
b = [1, 3]
y = [-2, 1]
```

has `y^T A = [0, 0]` and `y^T b = 1`. The producer returns this ordered row
witness inline. The v2 verification request binds it to the exact supplied
system and semantics without creating a producer-side artifact.

The producer returns `CERTIFICATE_PRODUCED` with `UNKNOWN` conclusion and
`UNVERIFIED` assurance. `NO_CERTIFICATE_PRODUCED`, timeout, cancellation,
runtime replacement, or malformed output also remains `UNKNOWN`; none proves
that the system is consistent.

The authorized `linear.rational_inconsistency.verify` checker imports neither
FLINT nor the producer. In a clean process it uses standard-library exact
rational arithmetic to check every column sum
`sum(y[i] * A[i][j]) == 0`, recompute `sum(y[i] * b[i])`, and require the
stored and recomputed pairing to equal one. It also rechecks closed v2 shapes,
canonical rationals, exact input and candidate bindings, and semantics. Only
acceptance creates a verification record and the `VERIFIED_INCONSISTENT`
result.

## Runtime identity and measurements

The supported provider identity is:

- distribution: `python-flint==0.9.0`;
- digest kind: `PYTHON_DISTRIBUTION_RECORD`;
- license expression: `MIT AND LGPL-3.0-or-later`;
- install tier: `T1`;
- required API: `flint.fmpq` and `flint.fmpq_mat`;
- operation profile: exact `QQ` reduced row-echelon form, either with free
  variables fixed to zero for a solution vector or with the dual pairing
  normalized to one for an inconsistency witness.

On the 2026-07-26 Linux x86-64 development host, the repository measurement
protocol recorded:

| Measurement | Result |
| --- | ---: |
| Fresh-cache, no-dependency wheel install | 4.606 s |
| Installed distribution size | 25,950,757 bytes |
| Cold import probe | 0.038 s, 42,332,160-byte peak RSS |
| 2-by-2 exact RREF reproduction | 0.040 s, 42,332,160-byte peak RSS |

These measurements characterize one host and distribution digest; they are
not performance guarantees.

The paired agent result is recorded in the
[the committed Harbor task boundary](../../evaluations/benchmark-contracts.md#task-and-verifier-validation).

## Trust limits

Python-FLINT is a maintained exact-arithmetic provider, not an independent
verifier of its own output. Provider success, RREF rank, deterministic output,
and durable storage remain computed evidence. Only the separately registered
checker may authorize a `VERIFIED` result, and that promotion is bound to the
exact system, candidate, semantics, witness, checker digest, and verification
request.

Primary provider references are the
[Python-FLINT repository](https://github.com/flintlib/python-flint),
[Python-FLINT rational matrix API](https://python-flint.readthedocs.io/en/latest/fmpq_mat.html),
and the
[Python-FLINT package release](https://pypi.org/project/python-flint/0.9.0/).
