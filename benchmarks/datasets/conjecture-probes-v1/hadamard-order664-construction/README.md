# Hadamard order-664 construction

This public calibration probe asks for a complete Hadamard matrix of order 664. The Oracle uses a deterministic Paley order-332 matrix over F331 tensored with the order-2 Hadamard matrix; the verifier knows none of that construction and checks only the represented matrix.

Ramos, Hulak, and de Queiroz, arXiv:2607.20765v1, identify order 668 as the smallest order unresolved at the 2026-08-14 cutoff. Because no order-668 witness is known, the repository's full-reward Oracle contract rules it out as an executable task. Order 664 is the adjacent lower multiple of four and provides the same large-matrix construction and exact-orthogonality surface without pretending to resolve 668.
