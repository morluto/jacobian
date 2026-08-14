#!/usr/bin/env node

"use strict";

const { spawn } = require("node:child_process");
const { stderr } = require("node:process");

/**
 * Jacobian npm carrier.
 *
 * This package does not implement the kernel, install Python, manage
 * environments, edit MCP client configuration, or forward the full Python
 * CLI. It is a thin deterministic carrier that invokes the one canonical
 * Jacobian MCP command:
 *
 *   uvx --from jacobian==<exact-version> jacobian-mcp [args...]
 *
 * The npm package version is the single release manifest; the spelling below
 * maps it to the Python package spec. Install, upgrade, and remove this
 * carrier with npm. Configure MCP clients with the published command snippet.
 */

const HELP = `Jacobian — MCP server carrier for AI agents.

Usage:
  jacobian mcp [args...]
    Run the canonical Jacobian MCP server over stdio.

The carrier invokes the exact Python MCP command:
  uvx --from jacobian==<version> jacobian-mcp [args...]

Requires uv on PATH (or set JACOBIAN_UV_BIN). Override the resolved Python
package spec with JACOBIAN_PACKAGE. Install, upgrade, and remove this carrier
with npm; point MCP clients at the published command snippet.
`;

/**
 * Map the npm release spelling to the Python spelling used by pip/uvx.
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

/**
 * Resolve the exact Python package spec the carrier pins for `uvx`.
 *
 * @returns {string}
 */
function packageSpec() {
  const override = process.env.JACOBIAN_PACKAGE;
  if (override) return override;
  const npmVersion = require("../package.json").version;
  return `jacobian==${pythonVersionFromNpmVersion(npmVersion)}`;
}

/**
 * Spawn the canonical Jacobian MCP server and forward signals.
 *
 * @param {string[]} extraArgs
 */
function launchMcp(extraArgs) {
  const uv = process.env.JACOBIAN_UV_BIN || "uvx";
  const child = spawn(
    uv,
    ["--from", packageSpec(), "jacobian-mcp", ...extraArgs],
    {
      stdio: "inherit",
      env: { ...process.env },
      windowsHide: true,
    },
  );

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
    for (const [signal, handler] of handlers)
      process.removeListener(signal, handler);
    stderr.write(`Jacobian MCP could not start: ${error.message}\n`);
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

function main() {
  const args = process.argv.slice(2);
  const command = args[0];

  if (!command || command === "--help" || command === "-h") {
    stderr.write(HELP);
    return;
  }

  if (command === "--version" || command === "-v") {
    console.log(`jacobian ${require("../package.json").version}`);
    return;
  }

  if (command === "mcp") {
    launchMcp(args.slice(1));
    return;
  }

  stderr.write(`Unknown command: ${command}\n\n${HELP}`);
  process.exitCode = 1;
}

module.exports = { pythonVersionFromNpmVersion, packageSpec };

if (require.main === module) {
  main();
}
