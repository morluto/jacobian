"""Preflight bounds for the pinned SymPy 1.14 Smith transformation kernel.

HNF preprocessing and unit elimination are computed once and retained by the
caller. These bounds inspect the residual; they do not simulate or replay SNF.
"""

from dataclasses import dataclass
from itertools import pairwise

from jacobian.catalog.models import OperationResourceAdmissionError

# Separate arithmetic work, integer height, and live mathematical storage.
# These conservative ceilings admit the full 50-vertex cycle/complete families
# and large cyclic/two-generator residuals without claiming arbitrary Smith
# transformations have the complexity of FLINT's diagonal-only algorithm.
MAX_SMITH_WORK = 250_000_000
MAX_SMITH_BITS = 4_000_000
MAX_SMITH_STORAGE_BITS = 1 << 31


@dataclass(frozen=True)
class SmithEnvelope:
    work: int
    bits: int


def _exceeded() -> SmithEnvelope:
    return SmithEnvelope(MAX_SMITH_WORK + 1, MAX_SMITH_BITS + 1)


def _two_by_two_bound(bits: int, determinant_bits: int) -> SmithEnvelope:
    """Exploit nonsingularity to bound the last Euclidean block sharply.

    Clearing the first column of [[a,b],[c,d]] gives [[g,t],[0,delta/g]],
    where g=gcd(a,c) divides delta=det(A), |t|<=2*B**2, and the first
    Bezout transform has entries <=B. Clearing that row either diagonalizes
    it or gives [[g',0],[y*delta/g,-delta/g']], where |y|<=g/g'. Thus after
    this first column/row pair every matrix entry is <=|delta|. Subsequent
    column-clearing intermediates are <=2*delta**2 and each further pair
    strictly reduces the pivot, allowing at most bit_length(delta) pairs.
    Multiplying their transforms (and the final divisibility repair) gives
    the deliberately loose 8*b + 16*d**2 bit bound below. Euclidean quotient
    work is linear in operand bits, including the initial B-sized column.
    """
    return SmithEnvelope(
        128 * (bits + 1 + (determinant_bits + 1) ** 2),
        8 * (bits + 1) + 16 * (determinant_bits + 1) ** 2,
    )


def _recursive_bound(
    n: int, bits: int, pivot_bits: int, determinant_bits: int
) -> SmithEnvelope:
    """Bound Euclidean clearing, recursive products, and divisibility repairs.

    In SymPy 1.14 each elementary update has coefficients at most the
    current maximum magnitude B (division or Bezout), and maps B to at most
    2*B**2. Each pass has <=2(n-1) updates. A further pass strictly reduces
    the positive pivot to a proper divisor, so <=pivot_bits passes suffice.
    The recursive trailing matrix has the resulting entry bound. Embedding
    and multiplying its transforms costs <=4*n**3 scalar operations; summing
    products adds ceil(log2(n)) bits. At this level <=n-1 divisibility repairs
    use three row and two column updates, with coefficients bounded by the
    determinant: every partial diagonal factor divides its product.
    """
    if n <= 1:
        return SmithEnvelope(1, max(1, bits))
    if n == 2:
        return _two_by_two_bound(bits, determinant_bits)
    updates = 2 * (n - 1) * max(1, pivot_bits)
    if updates >= MAX_SMITH_BITS.bit_length():
        return _exceeded()
    cleared = (bits + 1) * (1 << updates)
    if cleared > MAX_SMITH_BITS:
        return _exceeded()
    child = _recursive_bound(n - 1, cleared, cleared, determinant_bits)
    height = (
        child.bits + cleared + n.bit_length() + 3 * (n - 1) * (determinant_bits + 2)
    )
    # Extended Euclid on b-bit operands needs O(b) quotient steps, each
    # using a bounded number of integer-ring operations. Include those
    # steps as well as matrix updates; this is not a bit-operation count.
    work = child.work + 12 * updates * (n * n + cleared + 1) + 4 * n**3 + 30 * n * n
    return SmithEnvelope(work, height)


def admit_smith_residual(matrix: list[list[int]], determinant: int) -> SmithEnvelope:
    """Admit a nonsingular lower column-HNF residual before SymPy expansion."""
    n = len(matrix)
    bits = max((abs(v).bit_length() for row in matrix for v in row), default=1)
    det_bits = determinant.bit_length()
    # For this visible triangular class, every column clears by exact
    # division and the trailing matrix is unchanged. The only possible gcd
    # updates are final diagonal divisibility repairs. There are <=n(n-1)/2
    # repairs in the whole recursion, each with coefficients <=determinant.
    # Unit triangular elimination has coefficient bound (1+n*B)**n.
    divisible = all(
        matrix[i][j] % matrix[j][j] == 0 for j in range(n) for i in range(j + 1, n)
    )
    if divisible:
        diagonal = [matrix[i][i] for i in range(n)]
        ordered = all(b % a == 0 for a, b in pairwise(diagonal))
        repairs = 0 if ordered else n * (n - 1) // 2
        # A diagonal matrix needs no clearing transforms at all; in
        # particular a Smith diagonal keeps both transformations equal to I.
        clearing_bits = (
            (n + 1) * (bits + n.bit_length() + 2)
            if any(matrix[i][j] for j in range(n) for i in range(j + 1, n))
            else bits
        )
        envelope = SmithEnvelope(
            20 * max(1, n) ** 4 + 12 * repairs * (det_bits + 1),
            clearing_bits + 3 * repairs * (det_bits + 2),
        )
    else:
        envelope = _recursive_bound(n, bits, matrix[0][0].bit_length(), det_bits)
    # The worker also multiplies the left transform by a vector reduced
    # modulo determinant. Reserve its products and accumulation explicitly.
    envelope = SmithEnvelope(
        envelope.work + 2 * n * n, envelope.bits + det_bits + n.bit_length() + 1
    )
    # Recursive sources, both transforms, embedded products, and temporaries
    # occupy O(n^3) integer slots; 12*n^3 is a conservative retained-slot
    # bound for this implementation. Include multiplication temporaries by
    # doubling the entry bound. This is distinct from worker address space.
    storage = 24 * max(1, n) ** 3 * envelope.bits
    if (
        envelope.bits > MAX_SMITH_BITS
        or envelope.work > MAX_SMITH_WORK
        or storage > MAX_SMITH_STORAGE_BITS
    ):
        raise OperationResourceAdmissionError(
            location=("graph",),
            code="chip_firing.smith_transform_bound",
            message=(
                "column-HNF residual exceeds the conservative Smith transformation bound "
                f"(dimension={n}; work={envelope.work}/{MAX_SMITH_WORK}; "
                f"intermediate_bits={envelope.bits}/{MAX_SMITH_BITS}; "
                f"storage_bits={storage}/{MAX_SMITH_STORAGE_BITS}; "
                "estimates saturate when excessive)"
            ),
        )
    return envelope
