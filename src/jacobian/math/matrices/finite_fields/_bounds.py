"""Representation and execution bounds for canonical prime-field matrices."""

# A dense 1024-square matrix carries 1,048,576 canonical residues. RREF returns
# the same number of residues; a complete right-nullspace basis carries at most
# columns**2 entries. FLINT's measured dense 1024-square rank and RREF stay well
# inside the owning test lane, while this explicit cubic ledger conservatively
# accounts for elimination work independently of wall time.
MAX_PRIME_FIELD_MATRIX_AXIS = 1_024
MAX_PRIME_FIELD_MATRIX_CELLS = 1_048_576
MAX_PRIME_FIELD_ELIMINATION_WORK = 1_073_741_824

# Catalog requests use a cross-platform word-safe modulus. Native canonical
# values retain their broader exact-prime domain through the SymPy fallback.
MAX_PRIME_FIELD_FLINT_PRIME = 2_147_483_647
MAX_PRIME_FIELD_FALLBACK_AXIS = 256

__all__ = [
    "MAX_PRIME_FIELD_ELIMINATION_WORK",
    "MAX_PRIME_FIELD_FALLBACK_AXIS",
    "MAX_PRIME_FIELD_FLINT_PRIME",
    "MAX_PRIME_FIELD_MATRIX_AXIS",
    "MAX_PRIME_FIELD_MATRIX_CELLS",
]
