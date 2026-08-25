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

The server does not install operations, create state, migrate databases, or
manage external tool lifecycles.
