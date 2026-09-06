# Backend requirements

Jacobian's maintained Python backends, including Z3, are normal package
dependencies. `sat.solve` and `smt.solve` call those bindings in process.

Installing Jacobian installs its declared Python dependencies automatically.
Importing the library and using operations backed by those dependencies does not
require Docker, Singular, or QEPCAD. `uv sync` and `make setup` install Python
dependencies only; they do not provision system executables. The complete local
operation library also needs the system executables Singular and QEPCAD.
On Debian or Ubuntu, install them explicitly:

```sh
sudo apt-get update
sudo apt-get install -y --no-install-recommends singular qepcad
```

Check the versions below; distribution package revisions can differ. The
service image pins both packages and verifies their versions at build time.

## Singular

General multivariate ideal radical and quotient computations use the fixed
Singular 4.4 backend. The maintained service image installs the pinned Debian
package and checks Singular's numeric capability version while building.

Jacobian accepts the maintained Singular 4.4 release line. Confirm the local
numeric capability version with:

```sh
Singular -q --execute 'system("version");quit;'
```

Jacobian invokes Singular through the shared bounded process runner. An operation
may require several backend phases within its request deadline. The
commutative-algebra domain owns the strict polynomial and ideal codec; callers never submit Singular source or receive Singular values.
An unavailable backend, timeout, process-limit failure, or invalid backend
output is an execution outcome and does not establish a mathematical ideal.

## QEPCAD

`real_algebraic.plane_semialgebraic.component_profile.compute` uses QEPCAD B
1.74 for exact plane connectivity and point classification. It is a separate
project from Singular, but Debian and Ubuntu package it with a Singular
dependency, and its default configuration uses Singular for some algebra
computations. Installing QEPCAD therefore may also install Singular.

```sh
qepcad -v
make test-qepcad
```

The version output must contain `Version B 1.74,`. The service image and CI pin
Debian `qepcad=1.74+ds-5`; Ubuntu's `1.74+ds-5build1` has the same upstream
version. The adapter discovers a standard package installation automatically.
For a custom installation, put `qepcad` on PATH and, if necessary, set
`QEPCAD_ROOT` to its support directory containing `default.qepcadrc` and `bin/`.

Missing or unsupported QEPCAD causes an execution error on nondegenerate
requests; it never produces an approximate component count. Other operations
do not require this executable. See the
[backend contract and replacement assessment](../reference/mathematical-backends.md#qepcad)
for the exact dependency scope and alternatives.

The server does not install operations, create state, migrate databases, or
manage external tool lifecycles.

## Discover requirements before execution

Optional system runtimes currently serve these operations:

| Runtime | Operations |
| --- | --- |
| Singular 4.4.x | `polynomial.ideal.minimal_primes.compute`, `polynomial.ideal.radical.compute`, `polynomial.ideal.quotient.compute`, `polynomial.ideal.saturation.compute`, `polynomial.map.generic_degree.compute`, `algebraic_geometry.projective_plane_curve.singularity_profile.compute` |
| QEPCAD B 1.74 | `real_algebraic.plane_semialgebraic.component_profile.compute` |

Declarations expose `runtime_requirements`. These are stable requirements, not
live availability claims. An operation can handle a degenerate request without
starting its declared backend. Missing runtimes do not remove operations from the
catalog or prevent unrelated operations from running.

The projective plane-curve singularity profile requires Singular for degree-two
and degree-three inputs. Degree-one inputs use a native smoothness shortcut.
Its runtime declaration records the possible requirement for the operation.

Python callers can explicitly check the execution environment:

```python
from jacobian.backends import check_backend

availability = check_backend("singular")
print(availability.status)
print(availability.installation)
```

The status is `AVAILABLE`, `MISSING`, `UNSUPPORTED`, or `CHECK_FAILED`.
When the executable exists, the check runs a version probe with a five-second
limit and bounded output; results are not cached. `AVAILABLE` means the diagnostic detected a
supported runtime, not that every mathematical request will succeed. Importing
Jacobian and constructing its catalog never run these probes.

For MCP, match and browse results expose `runtime_requirements`. `math.find`
inspection also returns `backend_availability` for the selected operation in the
server's environment. If execution needs an absent or unsupported runtime,
Python raises `jacobian.backends.BackendUnavailableError`; MCP returns a tool
execution error with `code: BACKEND_UNAVAILABLE`, the backend, required version,
and an installation hint. This is an environment failure, not a mathematical
answer or malformed input.

## Where installation belongs

Install runtimes where computation executes. A local Python application or local
MCP server needs them locally; a remote MCP server needs them on the server, not
on each agent's machine. Users do not need to write a Dockerfile to import
Jacobian or connect an agent to a running remote MCP server.

Jacobian currently does not bundle Singular or QEPCAD into Python wheels, install
them automatically, or provide a pip extra that provisions them. Native runtime
installation varies by operating system. For the complete runtime on Windows,
a Linux environment such as WSL or the maintained service image is an option;
this is not a claim of native Windows support for these adapters. The maintained
service image provides the pinned deployment environment. Default Python
installation still depends on dependency support for the chosen Python/platform.
