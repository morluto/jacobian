# Backend requirements

Jacobian's maintained Python backends, including Z3, are normal package
dependencies. `sat.solve` and `smt.solve` call those bindings in process.

General multivariate ideal radical and quotient computations use the fixed
Singular 4.4 backend. The maintained service image installs the pinned Debian
package and checks Singular's numeric capability version while building. A
local installation can be checked with:

```sh
sudo apt-get update
sudo apt-get install -y --no-install-recommends singular
```

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

`lean.check` and `lean.declarations.search` are the retained formal-runtime
operations. The service image must contain the fixed Lean toolchain and the
repository-pinned Mathlib workspace and compiled cache before it starts. A
source checkout prepares both with:

```sh
make setup-lean
```

Other deployments set `JACOBIAN_MATHLIB_ROOT` to the prepared `lean/`
workspace. Jacobian validates the complete manifest, toolchain, Mathlib
revision, and umbrella module before checking or searching. It does not
download, select, or upgrade either environment while serving. Confirm the Lean
executable with:

```sh
lean --version
```

The complete Mathlib compiled cache is a multi-gigabyte runtime dependency.
Declaration search and a source check that imports the umbrella `Mathlib`
module run with a 10,240 MB Lean memory threshold; provision the service host
accordingly.

The server does not install operations, create state, migrate databases, or
manage external tool lifecycles.
