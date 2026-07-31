import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import test from "node:test";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const npmRoot = dirname(fileURLToPath(import.meta.url));
const packageMetadata = require("./package.json");

const pythonPrereleaseNames = {
  a: "alpha",
  alpha: "alpha",
  b: "beta",
  beta: "beta",
  c: "rc",
  pre: "rc",
  preview: "rc",
  rc: "rc",
};

/**
 * Convert the Python release spellings used by the project to npm semver.
 *
 * Python's version normalizer accepts aliases and separators such as
 * `0.3.0-alpha.0`, while npm uses `0.3.0-alpha.0` for the same release.
 *
 * @param {string} pythonVersion
 * @returns {string}
 */
function npmVersionFromPythonVersion(pythonVersion) {
  const match = pythonVersion.match(
    /^(\d+\.\d+\.\d+)(?:(?:[-_.]?)(alpha|a|beta|b|rc|c|pre|preview)(?:[-_.]?)(\d+))?$/,
  );
  assert.ok(match, `unsupported Python release version: ${pythonVersion}`);

  const prerelease = match[2];
  return prerelease
    ? `${match[1]}-${pythonPrereleaseNames[prerelease]}.${match[3]}`
    : match[1];
}

import {
  clientDefinitions,
  isClientDetected,
  buildLauncher,
  resolveClientEdit,
  SERVER_NAME,
} from "./bin/setup.cjs";
import {
  PACKAGE_SPEC,
  PYTHON_PACKAGE_VERSION,
  packageNeedsRefresh,
  pythonVersionFromNpmVersion,
  uvInstallArgs,
} from "./bin/launcher.cjs";
import {
  EXPECTED_TOOLS,
  handshakeFailure,
  timeoutMessage,
} from "./bin/doctor.cjs";

/**
 * Create a fake home directory with selected client markers.
 *
 * @param {string} base
 * @param {string[]} clients
 * @returns {Promise<string>}
 */
async function fakeHome(base, clients) {
  const home = await mkdtemp(join(base, "home-"));
  for (const client of clients) {
    switch (client) {
      case "claude":
        await mkdir(join(home, ".claude"), { recursive: true });
        break;
      case "cursor":
        await mkdir(join(home, ".cursor"), { recursive: true });
        break;
      case "opencode":
        await mkdir(join(home, ".config", "opencode"), { recursive: true });
        break;
      case "codex":
        await mkdir(join(home, ".codex"), { recursive: true });
        break;
      case "gemini":
        await mkdir(join(home, ".gemini"), { recursive: true });
        break;
    }
  }
  return home;
}

test("clientDefinitions returns all five supported clients", () => {
  const home = "/tmp/fake";
  const defs = clientDefinitions(home);
  assert.equal(defs.length, 5);
  assert.equal(defs[0].id, "claude");
  assert.equal(defs[1].id, "cursor");
  assert.equal(defs[2].id, "opencode");
  assert.equal(defs[3].id, "codex");
  assert.equal(defs[4].id, "gemini");
});

test("isClientDetected recognizes installed client markers", async () => {
  const base = await mkdtemp(join(tmpdir(), "jacobian-detect-"));
  try {
    const home = await fakeHome(base, ["claude", "cursor", "codex"]);
    assert.equal(isClientDetected(home, "claude"), true);
    assert.equal(isClientDetected(home, "cursor"), true);
    assert.equal(isClientDetected(home, "codex"), true);
    assert.equal(isClientDetected(home, "opencode"), false);
    assert.equal(isClientDetected(home, "gemini"), false);
  } finally {
    await rm(base, { recursive: true, force: true });
  }
});

test("buildLauncher returns a version-matching launcher with mcp subcommand", () => {
  const launcher = buildLauncher();
  assert.equal(launcher.version, packageMetadata.version);
  assert.ok(launcher.command.length > 0);
  assert.ok(launcher.args.length > 0);
  // The last arg should be "mcp" (the subcommand).
  assert.equal(launcher.args[launcher.args.length - 1], "mcp");
});

test("normalizes release-please Python prerelease versions for npm", () => {
  assert.equal(npmVersionFromPythonVersion("0.2.0a0"), "0.2.0-alpha.0");
  assert.equal(
    npmVersionFromPythonVersion("0.3.0-alpha.0"),
    "0.3.0-alpha.0",
  );
});

test("launcher pins and refreshes stale default Python packages", () => {
  const expectedPythonVersion = pythonVersionFromNpmVersion(packageMetadata.version);
  assert.equal(PYTHON_PACKAGE_VERSION, expectedPythonVersion);
  assert.equal(PACKAGE_SPEC, `${packageMetadata.name}==${expectedPythonVersion}`);
  assert.equal(packageNeedsRefresh("0.4.0"), true);
  assert.equal(packageNeedsRefresh(null), true);
  assert.equal(packageNeedsRefresh(PYTHON_PACKAGE_VERSION), false);
});

test("launcher allows pre-release resolution when installing through uv", () => {
  const args = uvInstallArgs("/tmp/venv/bin/python");

  assert.ok(args.includes("--prerelease=if-necessary"));
  assert.deepEqual(args, [
    "pip",
    "install",
    "--upgrade",
    "--prerelease=if-necessary",
    "--python",
    "/tmp/venv/bin/python",
    PACKAGE_SPEC,
  ]);
});

test("npm and Python packages publish the same release version", async () => {
  const pyproject = await readFile(join(npmRoot, "..", "pyproject.toml"), "utf8");
  const match = pyproject.match(/^version = "([^"]+)"$/m);
  assert.ok(match, "pyproject.toml must declare a project version");

  const pythonVersion = match[1];
  const expectedNpmVersion = npmVersionFromPythonVersion(pythonVersion);

  assert.equal(packageMetadata.version, expectedNpmVersion);
});

test("doctor errors hide request ids and provide a recovery command", () => {
  const timeout = timeoutMessage(10000);
  assert.match(timeout, /Jacobian did not answer within 10000 ms/);
  assert.match(timeout, /npx jacobian setup/);
  assert.doesNotMatch(timeout, /response id=/);

  const handshake = handshakeFailure("tool-catalog request");
  assert.match(handshake, /MCP tool-catalog request/);
  assert.match(handshake, /npx jacobian doctor/);
});

test("doctor validates the default capability-first MCP profile", () => {
  assert.deepEqual(EXPECTED_TOOLS, [
    "capability.describe",
    "capability.invoke",
  ]);
});

test("launcher explains how to recover when no runtime is on PATH", () => {
  const script = [
    "try {",
    "  require('./bin/launcher.cjs').resolvePython();",
    "} catch (error) {",
    "  process.stdout.write(error.message);",
    "}",
  ].join("\n");
  const result = spawnSync(process.execPath, ["-e", script], {
    cwd: npmRoot,
    encoding: "utf8",
    env: { ...process.env, PATH: "" },
  });
  assert.equal(result.status, 1);
  assert.match(result.stdout, /requires Python 3\.12 or uv on PATH/);
  assert.match(result.stdout, /npx jacobian doctor/);
  assert.doesNotMatch(result.stdout, /no Python runtime found/);
});

test("setup writes a JSON config for Claude Code", async () => {
  const base = await mkdtemp(join(tmpdir(), "jacobian-json-"));
  try {
    const home = await fakeHome(base, ["claude"]);
    const defs = clientDefinitions(home);
    const claude = defs.find((d) => d.id === "claude");
    const launcher = { command: "/usr/bin/node", args: ["/path/to/jacobian", "mcp"], version: packageMetadata.version, package: null };

    const edit = resolveClientEdit("setup", claude, launcher);
    assert.equal(edit.action, "create");
    assert.equal(edit.original, null);
    assert.ok(edit.updated !== null);

    // Apply the edit.
    mkdirSync(join(home, ".claude"), { recursive: true });
    writeFileSync(claude.configPath, edit.updated);

    // Re-read and verify.
    const config = JSON.parse(await readFile(claude.configPath, "utf8"));
    assert.ok(config.mcpServers);
    assert.ok(config.mcpServers[SERVER_NAME]);
    assert.equal(config.mcpServers[SERVER_NAME].command, "/usr/bin/node");
    assert.deepEqual(config.mcpServers[SERVER_NAME].args, ["/path/to/jacobian", "mcp"]);

    // Re-resolving should report already_current.
    const edit2 = resolveClientEdit("setup", claude, launcher);
    assert.equal(edit2.action, "already_current");
  } finally {
    await rm(base, { recursive: true, force: true });
  }
});

test("setup writes a TOML config for Codex", async () => {
  const base = await mkdtemp(join(tmpdir(), "jacobian-toml-"));
  try {
    const home = await fakeHome(base, ["codex"]);
    const defs = clientDefinitions(home);
    const codex = defs.find((d) => d.id === "codex");
    const launcher = { command: "/usr/bin/node", args: ["/path/to/jacobian", "mcp"], version: packageMetadata.version, package: null };

    const edit = resolveClientEdit("setup", codex, launcher);
    assert.equal(edit.action, "create");
    assert.ok(edit.updated !== null);
    assert.ok(edit.updated.includes("[mcp_servers]"));
    assert.ok(edit.updated.includes("jacobian"));

    // Apply and re-read.
    mkdirSync(join(home, ".codex"), { recursive: true });
    writeFileSync(codex.configPath, edit.updated);

    // Re-resolving should report already_current.
    const edit2 = resolveClientEdit("setup", codex, launcher);
    assert.equal(edit2.action, "already_current");

    // Remove should work.
    const edit3 = resolveClientEdit("remove", codex, launcher);
    assert.equal(edit3.action, "remove");
  } finally {
    await rm(base, { recursive: true, force: true });
  }
});

test("remove on a non-configured client reports not_configured", async () => {
  const base = await mkdtemp(join(tmpdir(), "jacobian-remove-"));
  try {
    const home = await fakeHome(base, ["claude"]);
    const defs = clientDefinitions(home);
    const claude = defs.find((d) => d.id === "claude");
    const launcher = { command: "/usr/bin/node", args: ["/path/to/jacobian", "mcp"], version: packageMetadata.version, package: null };

    const edit = resolveClientEdit("remove", claude, launcher);
    assert.equal(edit.action, "not_configured");
  } finally {
    await rm(base, { recursive: true, force: true });
  }
});

test("setup updates an existing JSON config without losing other servers", async () => {
  const base = await mkdtemp(join(tmpdir(), "jacobian-update-"));
  try {
    const home = await fakeHome(base, ["claude"]);
    const defs = clientDefinitions(home);
    const claude = defs.find((d) => d.id === "claude");
    const configPath = claude.configPath;

    // Pre-populate with an existing server.
    mkdirSync(join(home, ".claude"), { recursive: true });
    const existing = {
      mcpServers: {
        "other-server": { command: "other", args: ["--foo"] },
      },
      someOtherSetting: true,
    };
    writeFileSync(configPath, JSON.stringify(existing, null, 2) + "\n");

    const launcher = { command: "/usr/bin/node", args: ["/path/to/jacobian", "mcp"], version: packageMetadata.version, package: null };
    const edit = resolveClientEdit("setup", claude, launcher);
    assert.equal(edit.action, "update");

    // Apply.
    writeFileSync(configPath, edit.updated);
    const config = JSON.parse(await readFile(configPath, "utf8"));
    assert.ok(config.mcpServers["other-server"]);
    assert.ok(config.mcpServers[SERVER_NAME]);
    assert.equal(config.someOtherSetting, true);
  } finally {
    await rm(base, { recursive: true, force: true });
  }
});

test("invalid JSON explains that setup made no changes", async () => {
  const base = await mkdtemp(join(tmpdir(), "jacobian-invalid-json-"));
  try {
    const home = await fakeHome(base, ["claude"]);
    const claude = clientDefinitions(home).find((d) => d.id === "claude");
    const launcher = {
      command: "/usr/bin/node",
      args: ["/path/to/jacobian", "mcp"],
      version: packageMetadata.version,
      package: null,
    };
    await writeFile(claude.configPath, "{ invalid", "utf8");

    assert.throws(
      () => resolveClientEdit("setup", claude, launcher),
      /No changes were written\. Repair this file, then retry/,
    );
    assert.equal(await readFile(claude.configPath, "utf8"), "{ invalid");
  } finally {
    await rm(base, { recursive: true, force: true });
  }
});

test("top-level setup failure gives a retry action without a stack trace", async () => {
  const home = await mkdtemp(join(tmpdir(), "jacobian-cli-invalid-json-"));
  try {
    const configPath = join(home, ".claude.json");
    await writeFile(configPath, "{ invalid", "utf8");
    const result = spawnSync(
      process.execPath,
      [
        join(npmRoot, "bin", "jacobian.cjs"),
        "setup",
        "--client",
        "claude",
        "--yes",
      ],
      {
        encoding: "utf8",
        env: { ...process.env, HOME: home },
      },
    );

    assert.equal(result.status, 1);
    assert.match(result.stderr, /Jacobian setup did not finish/);
    assert.match(result.stderr, /retry `npx jacobian setup`/);
    assert.doesNotMatch(result.stderr, /\n\s+at /);
    assert.equal(await readFile(configPath, "utf8"), "{ invalid");
  } finally {
    await rm(home, { recursive: true, force: true });
  }
});
