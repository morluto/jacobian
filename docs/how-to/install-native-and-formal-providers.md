# Backend requirements

Jacobian's maintained Python backends, including Z3, are normal package
dependencies. `sat.solve` and `smt.solve` call those bindings in process.

`lean.check` is the only retained formal-runtime operation. The service image
must contain the fixed Lean toolchain before it starts; Jacobian does not
download, select, or upgrade toolchains while serving. Confirm the image with:

```sh
lean --version
```

The server does not install operations, create state, migrate databases, or
manage external tool lifecycles.
