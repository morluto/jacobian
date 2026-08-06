# Troubleshoot Z3 installation on macOS

[Documentation home](../index.md)

The locked environment uses `z3-solver` 5.0.0.0. Its upstream macOS wheels
target macOS 13 or newer on Apple silicon and Intel. On an older release, `uv`
falls back to a source build that requires CMake, `make`, and a C++20 compiler.

Install the Xcode Command Line Tools and CMake before retrying `uv sync --dev`.
These commands report the relevant environment without changing it:

```sh
sw_vers -productVersion
uname -m
xcode-select -p
clang++ --version
cmake --version
make --version
```

See the
[`z3-solver` 5.0.0.0 files on PyPI](https://pypi.org/project/z3-solver/5.0.0.0/#files)
for the upstream wheel tags.
