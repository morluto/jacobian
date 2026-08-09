#!/usr/bin/env node

"use strict";

const { stderr } = require("node:process");
const { spawn } = require("node:child_process");

const NPM_UPGRADE_HANDOFF = "JACOBIAN_NPM_UPGRADE_HANDOFF";

/**
 * Jacobian CLI entry point.
 *
 * Subcommands handled in Node:
 *   jacobian setup    — MCP client configuration wizard
 *   jacobian upgrade  — Refresh the launcher-managed Python package
 *   jacobian doctor   — MCP handshake and tool catalog verification
 *   jacobian remove   — Remove Jacobian MCP from client configs
 *   jacobian mcp      — Run the MCP server over stdio
 *
 * Everything else is forwarded to the Python `jacobian` CLI.
 */

const HELP = `Jacobian — composable mathematical capabilities for AI agents

Usage:
  jacobian setup [--client <id>...] [--all] [--yes] [--dry-run] [--json] [--plain]
                 [--source <checkout> --state-dir <path> --profile <name>]
    Configure MCP clients to use Jacobian.
  jacobian upgrade
    Refresh the launcher-managed Python package.
  jacobian doctor [--client <id>...] [--all] [--json]
    Verify configured client launchers, the MCP handshake, and the tool catalog.
  jacobian remove [--client <id>...] [--all] [--yes] [--dry-run] [--json] [--plain]
    Remove Jacobian from MCP client configs.
  jacobian mcp
    Run the Jacobian MCP server over stdio.
  jacobian <command> [args...]
    Forward to the Python Jacobian CLI.

Clients:
  claude, cursor, opencode, codex, gemini

Environment:
  JACOBIAN_STATE_DIR    State directory (default: ./.jacobian)
  JACOBIAN_PACKAGE      Python package spec (default: jacobian)
`;

function requiredOptionValue(value, option) {
  if (!value || value.startsWith("-")) {
    throw new Error(`${option} requires a value.`);
  }
  return value;
}

function reportSetupFailure(error) {
  stderr.write(
    `Jacobian setup did not finish: ${error.message}\n` +
      "No additional setup changes will be made. Correct the reported problem, " +
      "then retry `npx jacobian setup`.\n",
  );
  process.exitCode = 1;
}

/**
 * Resolve the latest npm bootstrap before upgrading the managed Python package.
 *
 * An unqualified npx invocation can execute an older cached launcher. The
 * handoff guard lets the latest package enter the pinned-runtime path once.
 *
 * @param {string[]} args
 */
function resolveLatestUpgrade(args) {
  const executable =
    process.env.JACOBIAN_NPX_EXECUTABLE ||
    (process.platform === "win32" ? "npx.cmd" : "npx");
  const child = spawn(
    executable,
    ["--yes", "--prefer-online", "jacobian@latest", "upgrade", ...args.slice(1)],
    {
      env: { ...process.env, [NPM_UPGRADE_HANDOFF]: "1" },
      shell: process.platform === "win32",
      stdio: "inherit",
    },
  );
  const signals = ["SIGINT", "SIGTERM", "SIGHUP"];
  const handlers = new Map(
    signals.map((signal) => [
      signal,
      () => {
        if (!child.killed) child.kill(signal);
      },
    ]),
  );
  for (const [signal, handler] of handlers) process.on(signal, handler);

  const removeSignalHandlers = () => {
    for (const [signal, handler] of handlers) {
      process.removeListener(signal, handler);
    }
  };

  child.once("error", (error) => {
    removeSignalHandlers();
    stderr.write(
      `Jacobian could not resolve its latest npm bootstrap: ${error.message}\n` +
        "Run `npx jacobian@latest upgrade` after checking that npx is available.\n",
    );
    process.exitCode = 1;
  });
  child.once("exit", (code, signal) => {
    removeSignalHandlers();
    if (signal) {
      process.kill(process.pid, signal);
      return;
    }
    process.exitCode = code ?? 1;
  });
}

function main() {
  const args = process.argv.slice(2);
  const command = args[0];

  if (!command || command === "--help" || command === "-h") {
    stderr.write(HELP);
    return;
  }

  if (command === "--version" || command === "-v") {
    const pkg = require("../package.json");
    console.log(`jacobian ${pkg.version}`);
    return;
  }

  if (command === "setup") {
    const setup = require("./setup.cjs");
    const rest = args.slice(1);
    const options = { operation: "setup" };
    try {
      for (let i = 0; i < rest.length; i++) {
        const arg = rest[i];
        if (arg === "--all") {
          options.all = true;
        } else if (arg === "--yes" || arg === "-y") {
          options.yes = true;
        } else if (arg === "--dry-run") {
          options.dryRun = true;
        } else if (arg === "--json") {
          options.json = true;
        } else if (arg === "--plain") {
          options.plain = true;
        } else if (arg === "--source") {
          options.source = requiredOptionValue(rest[++i], arg);
        } else if (arg.startsWith("--source=")) {
          options.source = requiredOptionValue(arg.slice(9), "--source");
        } else if (arg === "--state-dir") {
          options.stateDir = requiredOptionValue(rest[++i], arg);
        } else if (arg.startsWith("--state-dir=")) {
          options.stateDir = requiredOptionValue(arg.slice(12), "--state-dir");
        } else if (arg === "--uv-bin") {
          options.uvBin = requiredOptionValue(rest[++i], arg);
        } else if (arg.startsWith("--uv-bin=")) {
          options.uvBin = requiredOptionValue(arg.slice(9), "--uv-bin");
        } else if (arg === "--profile") {
          options.profile = requiredOptionValue(rest[++i], arg);
        } else if (arg.startsWith("--profile=")) {
          options.profile = requiredOptionValue(arg.slice(10), "--profile");
        } else if (arg === "--provider-path") {
          options.providerPath = requiredOptionValue(rest[++i], arg);
        } else if (arg.startsWith("--provider-path=")) {
          options.providerPath = requiredOptionValue(arg.slice(16), "--provider-path");
        } else if (arg === "--project-environment") {
          options.projectEnvironment = requiredOptionValue(rest[++i], arg);
        } else if (arg.startsWith("--project-environment=")) {
          options.projectEnvironment = requiredOptionValue(
            arg.slice(22),
            "--project-environment",
          );
        } else if (arg === "--elan-home") {
          options.elanHome = requiredOptionValue(rest[++i], arg);
        } else if (arg.startsWith("--elan-home=")) {
          options.elanHome = requiredOptionValue(arg.slice(12), "--elan-home");
        } else if (arg === "--lean-runtime") {
          options.leanRuntime = requiredOptionValue(rest[++i], arg);
        } else if (arg.startsWith("--lean-runtime=")) {
          options.leanRuntime = requiredOptionValue(arg.slice(15), "--lean-runtime");
        } else if (arg === "--client" || arg === "-c") {
          const value = requiredOptionValue(rest[++i], arg);
          options.clients = options.clients || [];
          options.clients.push(...value.split(",").map((s) => s.trim()));
        } else if (arg.startsWith("--client=")) {
          const value = requiredOptionValue(arg.slice(9), "--client");
          options.clients = options.clients || [];
          options.clients.push(...value.split(",").map((s) => s.trim()));
        } else {
          throw new Error(`Unknown setup option: ${arg}.`);
        }
      }
    } catch (error) {
      reportSetupFailure(error);
      return;
    }
    setup
      .run(options)
      .then((result) => {
        if (result.cancelled) {
          stderr.write("Jacobian setup cancelled; no client configuration was applied.\n");
          process.exitCode = 1;
        }
      })
      .catch((error) => {
        reportSetupFailure(error);
      });
    return;
  }

  if (command === "upgrade") {
    if (process.env[NPM_UPGRADE_HANDOFF] !== "1") {
      stderr.write(
        "Resolving the latest Jacobian bootstrap before upgrading the managed Python runtime.\n",
      );
      resolveLatestUpgrade(args);
      return;
    }
    const { PACKAGE_SPEC, upgrade } = require("./launcher.cjs");
    try {
      upgrade();
      console.log(`Jacobian Python package upgraded to ${PACKAGE_SPEC}.`);
    } catch (error) {
      stderr.write(
        `Jacobian upgrade did not finish: ${error.message}\n` +
          "Check the local Python runtime and package index, then retry `npx jacobian upgrade`.\n",
      );
      process.exitCode = 1;
    }
    return;
  }

  if (command === "remove") {
    const setup = require("./setup.cjs");
    const rest = args.slice(1);
    const options = { operation: "remove" };
    try {
      for (let i = 0; i < rest.length; i++) {
        const arg = rest[i];
        if (arg === "--all") {
          options.all = true;
        } else if (arg === "--yes" || arg === "-y") {
          options.yes = true;
        } else if (arg === "--json") {
          options.json = true;
        } else if (arg === "--dry-run") {
          options.dryRun = true;
        } else if (arg === "--plain") {
          options.plain = true;
        } else if (arg === "--client" || arg === "-c") {
          const value = requiredOptionValue(rest[++i], arg);
          options.clients = options.clients || [];
          options.clients.push(...value.split(",").map((s) => s.trim()));
        } else if (arg.startsWith("--client=")) {
          const value = requiredOptionValue(arg.slice(9), "--client");
          options.clients = options.clients || [];
          options.clients.push(...value.split(",").map((s) => s.trim()));
        } else {
          throw new Error(`Unknown remove option: ${arg}.`);
        }
      }
    } catch (error) {
      stderr.write(
        `Jacobian removal did not start: ${error.message}\n` +
          "Correct the option and retry `npx jacobian remove`.\n",
      );
      process.exitCode = 1;
      return;
    }
    setup
      .run(options)
      .then((result) => {
        if (result.cancelled) {
          stderr.write("Jacobian removal cancelled; no client configuration was changed.\n");
        }
      })
      .catch((error) => {
        stderr.write(
          `Jacobian removal did not finish: ${error.message}\n` +
            "Inspect the named client configuration, then retry `npx jacobian remove`.\n",
        );
        process.exitCode = 1;
      });
    return;
  }

  if (command === "doctor") {
    const doctor = require("./doctor.cjs");
    const rest = args.slice(1);
    const options = {};
    try {
      for (let i = 0; i < rest.length; i++) {
        const arg = rest[i];
        if (arg === "--json" || arg === "-j") {
          options.json = true;
        } else if (arg === "--all") {
          options.all = true;
        } else if (arg === "--client" || arg === "-c") {
          const value = requiredOptionValue(rest[++i], arg);
          options.clients = options.clients || [];
          options.clients.push(...value.split(",").map((item) => item.trim()));
        } else if (arg.startsWith("--client=")) {
          const value = requiredOptionValue(arg.slice(9), "--client");
          options.clients = options.clients || [];
          options.clients.push(...value.split(",").map((item) => item.trim()));
        } else {
          throw new Error(`Unknown doctor option: ${arg}.`);
        }
      }
    } catch (error) {
      stderr.write(`Jacobian diagnostics did not start: ${error.message}\n`);
      process.exitCode = 1;
      return;
    }
    doctor.run(options).catch((error) => {
      stderr.write(
        `Jacobian diagnostics did not finish: ${error.message}\n` +
          "Run `npx jacobian setup`, then retry `npx jacobian doctor`.\n",
      );
      process.exitCode = 1;
    });
    return;
  }

  if (command === "mcp") {
    const { launch } = require("./launcher.cjs");
    try {
      launch("jacobian.adapters.mcp.server", args.slice(1));
    } catch (error) {
      stderr.write(`Jacobian MCP could not start: ${error.message}\n`);
      process.exitCode = 1;
    }
    return;
  }

  // Forward everything else to the Python CLI.
  const { launch } = require("./launcher.cjs");
  try {
    launch("jacobian.cli", args);
  } catch (error) {
    stderr.write(`Jacobian could not start: ${error.message}\n`);
    process.exitCode = 1;
  }
}

main();
