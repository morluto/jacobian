# Backend requirements

Jacobian's maintained Python backends, including Z3, are normal package
dependencies. `sat.solve` and `smt.solve` call those bindings in process.

`uv sync` and `make setup` install Python dependencies only. The complete local
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

Jacobian invokes Singular once per accepted request through the shared bounded
process runner. The commutative-algebra domain owns the strict polynomial and
ideal codec; callers never submit Singular source or receive Singular values.
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
