# cddlib exact H/V optional-provider spike

[Documentation home](../index.md)

- Status: provider evidence; production deferred
- Frozen report contract: `jacobian.cddlib-hv-spike/v1`
- Registered capability IDs: none

## Decision

The spike accepts cddlib `0.94n` with pycddlib `3.0.2` as a viable exact
rational H/V conversion producer, but does not add it to Jacobian's locked
environment or catalog. It must remain an operator-installed T1 provider.
cddlib and pycddlib are GPL-2.0-or-later; their source build and distribution
obligations are therefore an explicit deployment boundary, not a silent core
dependency.

The pins use the official
[cddlib 0.94n release](https://github.com/cddlib/cddlib/releases/tag/0.94n)
and the official
[pycddlib 3.0.2 source distribution](https://pypi.org/project/pycddlib/3.0.2/).
The source archives, tag commits, license files, GMP rational interface, and
pycddlib `cdd.gmp` binding sources are digest-bound in
`benchmarks/cddlib_hv_pin.json`.

pycddlib 3.x intentionally separates the floating `cdd` module from the exact
`cdd.gmp` module. This spike imports only `cdd.gmp`, supplies
`fractions.Fraction`, and confirms a large rational value round-trips exactly.
The official
[pycddlib module reference](https://pycddlib.readthedocs.io/en/stable/cdd.html)
defines the homogeneous H and V representations and their linearity sets.

## Reproduce

Linux has no official pycddlib wheel, so the recorded CPython 3.12 reproduction
used an isolated source build. One equivalent setup is:

```bash
tar -xzf /tmp/cddlib-0.94n.tar.gz -C /tmp/cddlib-source
cd /tmp/cddlib-source/cddlib-0.94n
./configure --prefix=/tmp/cddlib-prefix
make -j2
make check
make install

uv venv --python 3.12 /tmp/pycddlib-venv
env \
  CFLAGS='-I/tmp/cddlib-prefix/include' \
  LDFLAGS='-L/tmp/cddlib-prefix/lib -Wl,-rpath,/tmp/cddlib-prefix/lib' \
  uv pip install \
    --python /tmp/pycddlib-venv/bin/python \
    /tmp/pycddlib-3.0.2.tar.gz
```

Run the bounded adapter from the locked Jacobian environment while selecting
the isolated provider interpreter:

```bash
uv run python benchmarks/cddlib_hv_spike.py \
  --python-executable /tmp/pycddlib-venv/bin/python \
  --cddlib-source-archive /tmp/cddlib-0.94n.tar.gz \
  --pycddlib-source-archive /tmp/pycddlib-3.0.2.tar.gz \
  --output /tmp/jcb-cddlib-hv-spike.json
```

The frozen two-dimensional cases exercise both conversion directions:

- an affine ray with one equality, one inequality, one vertex, and one ray;
- a strip with two inequalities, two vertices, and one lineality direction;
- the reverse V-to-H conversion of each representation; and
- explicit affine dimension and homogeneous row normalization.

The observed provider output digest is
`sha256:ed554a7be4fedba9f5d6904a984030f96f9b9f8b513004d2252e0bb58df77e84`.
All four same-provider round trips match the frozen normalized inputs. This is
reproduction evidence, not independent completeness evidence.

## Assurance boundary

The controller independently replays every returned
constraint/generator incidence with stdlib `Fraction` arithmetic and recomputes
affine dimension from exact ranks. The four cases perform 4, 6, 4, and 6 exact
checks. This establishes the demonstrated soundness direction:

- for H-to-V, every returned vertex, ray, and lineality direction respects the
  input constraints; and
- for V-to-H, every returned equality and inequality contains the input
  vertices, rays, and lineality directions.

It does not establish the reverse containment. In particular, an omitted
facet, vertex, or ray can pass one-direction soundness. Converting the result
back with the same cddlib implementation is useful provider evidence but is not
an independent checker. The spike therefore remains `COMPUTED`-eligible only
and reports `REVISE` for checker feasibility.

## Production contract gate

Do not register `polytope.rational.h_to_v.compute` or
`polytope.rational.v_to_h.compute` until a later change supplies:

- separate typed H and V artifacts with ambient and affine dimensions;
- explicit equalities, inequalities, vertices, rays, and lineality;
- a versioned `[b,a]` / `[t,x]` homogeneous convention and normalization;
- full request validation before provider execution or artifact writes;
- bounded subprocess failure semantics for timeout, cancellation, crash,
  malformed output, and version mismatch;
- installed cddlib shared-library identity measurement, because pycddlib does
  not expose a cddlib runtime version API;
- independent bidirectional containment, Farkas/extremality/completeness
  evidence, or an equivalently strong exact checker; and
- adversarial omitted-row, omitted-generator, sign, scope, and artifact-binding
  tests with zero false certification.

Only an operator-authorized checker package independent of cddlib, pycddlib,
proposal, search, and evaluation may return `VERIFIED`. Provider availability,
GPL deployment approval, compatibility, and checker authority remain separate.

## Handoff

- Baseline: the git tree containing the spike; record the exact tree in the PR.
- Provider state: cddlib and pycddlib absent from the locked Jacobian
  environment; present only in the isolated reproduction prefix and venv.
- Model/prompt settings: not applicable; this is a deterministic provider
  reproduction.
- Public/held-out role: all four cases are answer-visible regression evidence.
- Raw report: `/tmp/jcb-cddlib-hv-spike.json` on the reproducing host.
- Decision: retain the spike, register no capabilities, and require the
  production contract and independent completeness gates above.
