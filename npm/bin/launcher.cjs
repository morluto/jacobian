"use strict";

const { spawnSync } = require("node:child_process");
const { existsSync, mkdirSync } = require("node:fs");
const { homedir } = require("node:os");
const { join } = require("node:path");

/**
 * Jacobian npm launcher.
 *
 * Detects a usable Python runtime (preferring uv), ensures the
 * `jacobian` package is installed in a shared virtual
 * environment, and spawns the requested Jacobian entry point.
 *
 * The launcher never runs npm lifecycle scripts.  It is a thin Node wrapper
 * that delegates all heavy work to the Python package on PyPI or git.
 */

const PACKAGE_NAME = "jacobian";
const npmPackageVersion = require("../package.json").version;

/**
 * Convert the npm release spelling to the Python spelling used by
 * importlib.metadata and pip.
 *
 * @param {string} version
 * @returns {string}
 */
function pythonVersionFromNpmVersion(version) {
  const match = version.match(
    /^(\d+\.\d+\.\d+)(?:-(alpha|beta|rc)\.(\d+))?$/,
  );
  if (!match) throw new Error(`unsupported Jacobian npm version: ${version}`);
  const prerelease = { alpha: "a", beta: "b", rc: "rc" }[match[2]];
  return prerelease ? `${match[1]}${prerelease}${match[3]}` : match[1];
}

const PYTHON_PACKAGE_VERSION = pythonVersionFromNpmVersion(npmPackageVersion);
const USING_DEFAULT_PACKAGE = !process.env.JACOBIAN_PACKAGE;
const PACKAGE_SPEC =
  process.env.JACOBIAN_PACKAGE || `${PACKAGE_NAME}==${PYTHON_PACKAGE_VERSION}`;
const VENV_NAME = "jacobian-venv";
const STATE_DIR_ENV = "JACOBIAN_STATE_DIR";

/**
 * Resolve the shared virtual-environment root.
 *
 * Uses XDG data directory when available, falling back to
 * `~/.local/share/jacobian`.
 *
 * @returns {string}
 */
function venvRoot() {
  const xdgData = process.env.XDG_DATA_HOME;
  if (xdgData) return join(xdgData, "jacobian");
  return join(homedir(), ".local", "share", "jacobian");
}

/**
 * Return the path to the virtual-environment Python executable.
 *
 * @returns {string}
 */
function venvPython() {
  const root = venvRoot();
  if (process.platform === "win32") return join(root, VENV_NAME, "Scripts", "python.exe");
  return join(root, VENV_NAME, "bin", "python");
}

/**
 * Run a command synchronously and return its result.
 *
 * @param {string} program
 * @param {string[]} args
 * @param {object} [options]
 * @returns {import("node:child_process").SpawnSyncReturns<string>}
 */
function run(program, args, options = {}) {
  return spawnSync(program, args, {
    encoding: "utf8",
    stdio: options.silent ? "pipe" : "inherit",
    ...options,
  });
}

/**
 * Detect whether a program is available on PATH.
 *
 * @param {string} program
 * @returns {boolean}
 */
function hasProgram(program) {
  const result = run(program, ["--version"], { silent: true, shell: process.platform === "win32" });
  return result.error === undefined && result.status === 0;
}

/**
 * Detect uv, preferring it over raw Python.
 *
 * @returns {{ kind: "uv" | "python", path: string } | null}
 */
function detectRuntime() {
  if (hasProgram("uv")) return { kind: "uv", path: "uv" };
  for (const program of ["python3", "python"]) {
    const version = run(
      program,
      ["-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
      { silent: true, shell: process.platform === "win32" },
    );
    if (
      version.error === undefined &&
      version.status === 0 &&
      ["3.12", "3.13"].includes(version.stdout.trim())
    ) {
      return { kind: "python", path: program };
    }
  }
  return null;
}

/**
 * Return whether the cached environment needs the configured package.
 *
 * @param {string | null} installedVersion
 * @returns {boolean}
 */
function packageNeedsRefresh(installedVersion) {
  return (
    installedVersion === null ||
    (USING_DEFAULT_PACKAGE && installedVersion !== PYTHON_PACKAGE_VERSION)
  );
}

/**
 * Install or upgrade the configured Python package in the shared environment.
 *
 * @param {{ kind: "uv" | "python", path: string }} runtime
 * @param {string} python
 */
function installPackage(runtime, python) {
  const args =
    runtime.kind === "uv"
      ? ["pip", "install", "--upgrade", "--python", python, PACKAGE_SPEC]
      : ["-m", "pip", "install", "--upgrade", PACKAGE_SPEC];
  const result = run(runtime.path, args);
  if (result.error || result.status !== 0) {
    process.exitCode = 1;
    throw new Error(
      "Jacobian could not install its Python package. Check network and package " +
        "index access, then retry `npx jacobian doctor`.",
    );
  }
}

/**
 * Ensure the shared virtual environment exists and the Jacobian package is
 * installed.  Returns the path to the venv Python executable.
 *
 * @param {{ kind: "uv" | "python", path: string }} runtime
 * @param {{ forceUpgrade?: boolean }} [options]
 * @returns {string}
 */
function ensureEnvironment(runtime, options = {}) {
  const root = venvRoot();
  const python = venvPython();

  if (!existsSync(python)) {
    mkdirSync(root, { recursive: true });
    if (runtime.kind === "uv") {
      const result = run(runtime.path, ["venv", join(root, VENV_NAME), "--python", "3.12"]);
      if (result.error || result.status !== 0) {
        process.exitCode = 1;
        throw new Error(
          `Jacobian could not create its local Python environment in ${root}. ` +
            "Check write access and that Python 3.12 is available, then retry.",
        );
      }
    } else {
      const result = run(runtime.path, ["-m", "venv", join(root, VENV_NAME)]);
      if (result.error || result.status !== 0) {
        process.exitCode = 1;
        throw new Error(
          `Jacobian could not create its local Python environment in ${root}. ` +
            "Check write access and install Python's venv support, then retry.",
        );
      }
    }
  }

  // Check if the package is already installed.
  const check = run(python, ["-c", `import jacobian; print(jacobian.__version__)`], { silent: true });
  const installedVersion = check.status === 0 ? check.stdout.trim() : null;
  if (options.forceUpgrade || packageNeedsRefresh(installedVersion)) {
    installPackage(runtime, python);
  }

  return python;
}

/**
 * Refresh the configured Python package in the launcher-managed environment.
 *
 * @returns {{ package: string, python: string }}
 */
function upgrade() {
  const runtime = requireRuntime();
  const python = ensureEnvironment(runtime, { forceUpgrade: true });
  return { package: PACKAGE_SPEC, python };
}

/**
 * Resolve a usable Python runtime or raise the standard launcher error.
 *
 * @returns {{ kind: "uv" | "python", path: string }}
 */
function requireRuntime() {
  const runtime = detectRuntime();
  if (!runtime) {
    process.exitCode = 1;
    throw new Error(
      "Jacobian requires Python 3.12/3.13 or uv on PATH. Install one, then retry " +
        "`npx jacobian doctor`.",
    );
  }
  return runtime;
}

/**
 * Resolve the Python executable to use for launching Jacobian.
 *
 * Ensures the virtual environment and package are installed.
 *
 * @returns {string}
 */
function resolvePython() {
  return ensureEnvironment(requireRuntime());
}

/**
 * Spawn a Jacobian entry point and forward signals.
 *
 * @param {string} module The Python module to run (e.g. "jacobian.cli" or "jacobian.adapters.mcp.server")
 * @param {string[]} extraArgs Arguments to pass through
 * @param {object} [options]
 * @param {string[]} [options.extraEnv] Additional env vars to set
 */
function launch(module, extraArgs, options = {}) {
  const python = resolvePython();
  const args = ["-m", module, ...extraArgs];
  const env = { ...process.env };
  if (!env[STATE_DIR_ENV]) {
    env[STATE_DIR_ENV] = join(process.cwd(), ".jacobian");
  }

  const { spawn } = require("node:child_process");
  const child = spawn(python, args, {
    stdio: "inherit",
    env,
    windowsHide: true,
  });

  const signals =
    process.platform === "win32"
      ? ["SIGINT", "SIGTERM"]
      : ["SIGHUP", "SIGINT", "SIGTERM"];
  const handlers = new Map(
    signals.map((signal) => [
      signal,
      () => {
        if (!child.killed) child.kill(signal);
      },
    ]),
  );
  for (const [signal, handler] of handlers) process.on(signal, handler);

  child.once("error", (error) => {
    console.error(
      `Jacobian could not start: ${error.message}. Run \`npx jacobian doctor\` ` +
        "to check the local installation.",
    );
    process.exitCode = 1;
  });

  child.once("close", (code, signal) => {
    for (const [forwarded, handler] of handlers)
      process.removeListener(forwarded, handler);
    if (signal && process.platform !== "win32") {
      process.kill(process.pid, signal);
      return;
    }
    process.exitCode = code ?? 1;
  });
}

module.exports = {
  PACKAGE_NAME,
  PYTHON_PACKAGE_VERSION,
  PACKAGE_SPEC,
  VENV_NAME,
  packageNeedsRefresh,
  upgrade,
  pythonVersionFromNpmVersion,
  venvRoot,
  venvPython,
  detectRuntime,
  ensureEnvironment,
  resolvePython,
  launch,
};
