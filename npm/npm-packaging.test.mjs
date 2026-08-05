import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import {
  chmod,
  mkdtemp,
  mkdir,
  lstat,
  readFile,
  readdir,
  rm,
  stat,
  symlink,
  writeFile,
} from "node:fs/promises";
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
  buildSourceLauncher,
  applyEdits,
  resolveClientEdit,
  SERVER_NAME,
} from "./bin/setup.cjs";
import {
  PACKAGE_SPEC,
  PYTHON_PACKAGE_VERSION,
  packageNeedsRefresh,
  pythonVersionFromNpmVersion,
} from "./bin/launcher.cjs";
import {
  classifyStartupFailure,
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

test("buildLauncher does not pass npx-cli.js an extra exec subcommand", () => {
  const original = {
    npm_lifecycle_event: process.env.npm_lifecycle_event,
    npm_node_execpath: process.env.npm_node_execpath,
    npm_execpath: process.env.npm_execpath,
  };
  process.env.npm_lifecycle_event = "npx";
  process.env.npm_node_execpath = "/usr/bin/node";
  process.env.npm_execpath = "/usr/share/nodejs/npm/bin/npx-cli.js";

  try {
    const launcher = buildLauncher();
    assert.deepEqual(launcher.args, [
      "/usr/share/nodejs/npm/bin/npx-cli.js",
      "--yes",
      `--package=jacobian@${packageMetadata.version}`,
      "--",
      "jacobian",
      "mcp",
    ]);
  } finally {
    for (const [key, value] of Object.entries(original)) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
  }
});

test("buildLauncher uses npm exec when npx exposes npm-cli.js", () => {
  const original = {
    npm_lifecycle_event: process.env.npm_lifecycle_event,
    npm_node_execpath: process.env.npm_node_execpath,
    npm_execpath: process.env.npm_execpath,
  };
  process.env.npm_lifecycle_event = "npx";
  process.env.npm_node_execpath = "/usr/bin/node";
  process.env.npm_execpath = "/usr/share/nodejs/npm/bin/npm-cli.js";

  try {
    const launcher = buildLauncher();
    assert.deepEqual(launcher.args, [
      "/usr/share/nodejs/npm/bin/npm-cli.js",
      "exec",
      "--yes",
      `--package=jacobian@${packageMetadata.version}`,
      "--",
      "jacobian",
      "mcp",
    ]);
  } finally {
    for (const [key, value] of Object.entries(original)) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
  }
});

test("buildSourceLauncher binds the exact checkout and state directory", () => {
  const source = join(npmRoot, "..");
  const state = join(source, ".jacobian-source-test");
  const launcher = buildSourceLauncher(
    source,
    state,
    "/opt/uv",
    "full-python",
    "/providers/bin:/usr/bin",
    "/environments/jacobian",
  );
  assert.equal(launcher.command, "/opt/uv");
  assert.deepEqual(launcher.args, [
    "run",
    "--project",
    source,
    "--locked",
    "--no-sync",
    "jacobian-mcp",
    "--state-dir",
    state,
  ]);
  assert.equal(launcher.profile, "full-python");
  assert.equal(launcher.package, null);
  assert.deepEqual(launcher.env, {
    PATH: "/providers/bin:/usr/bin",
    UV_PROJECT_ENVIRONMENT: "/environments/jacobian",
  });
});

test("buildSourceLauncher preserves a nondefault Lean toolchain home", () => {
  const source = join(npmRoot, "..");
  const launcher = buildSourceLauncher(
    source,
    join(source, ".jacobian-source-test"),
    "/opt/uv",
    "lean",
    "/providers/bin:/usr/bin",
    "",
    "/toolchains/elan",
    "/runtimes/lean",
  );
  assert.deepEqual(launcher.env, {
    PATH: "/providers/bin:/usr/bin",
    ELAN_HOME: "/toolchains/elan",
    JACOBIAN_LEAN_RUNTIME: "/runtimes/lean",
  });
});

test("buildSourceLauncher rejects an unrelated uv project", async () => {
  const base = await mkdtemp(join(tmpdir(), "jacobian-unrelated-source-"));
  try {
    await writeFile(join(base, "pyproject.toml"), '[project]\nname = "unrelated"\n');
    await writeFile(join(base, "uv.lock"), "version = 1\n");
    await mkdir(join(base, "src", "jacobian"), { recursive: true });
    await writeFile(join(base, "src", "jacobian", "__init__.py"), "");
    assert.throws(
      () => buildSourceLauncher(base, join(base, ".state")),
      /not a Jacobian source checkout/,
    );
  } finally {
    await rm(base, { recursive: true, force: true });
  }
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

test("doctor classifies incompatible persisted state without exposing tracebacks", () => {
  const diagnostic = classifyStartupFailure(
    "StateDatabaseError: state migration 3 identity or checksum changed",
    "Connection closed",
  );
  assert.equal(diagnostic.code, "STATE_MIGRATION_INCOMPATIBLE");
  assert.match(diagnostic.message, /migration 3/);
  assert.match(diagnostic.recovery, /fresh state directory/);
  assert.doesNotMatch(diagnostic.message, /StateDatabaseError/);
});

test("doctor classifies a missing managed Jacobian runtime", () => {
  const diagnostic = classifyStartupFailure(
    "ModuleNotFoundError: No module named 'jacobian'",
    "Connection closed",
  );
  assert.equal(diagnostic.code, "JACOBIAN_RUNTIME_UNAVAILABLE");
  assert.match(diagnostic.recovery, /source-agent doctor/);
});

test("doctor validates the canonical math MCP surface", () => {
  assert.deepEqual(EXPECTED_TOOLS, [
    "math.find",
    "math.run",
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
    const launcher = { command: "/usr/bin/node", args: ["/path/to/jacobian", "mcp"], version: packageMetadata.version, package: null, env: { PATH: "/providers/bin" } };

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
    assert.deepEqual(config.mcpServers[SERVER_NAME].env, { PATH: "/providers/bin" });

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
    const launcher = { command: "/usr/bin/node", args: ["/path/to/jacobian", "mcp"], version: packageMetadata.version, package: null, env: { PATH: "/providers/bin" } };

    const edit = resolveClientEdit("setup", codex, launcher);
    assert.equal(edit.action, "create");
    assert.ok(edit.updated !== null);
    assert.ok(edit.updated.includes("[mcp_servers]"));
    assert.ok(edit.updated.includes("jacobian"));
    assert.ok(edit.updated.includes('env = { PATH = "/providers/bin" }'));

    // Apply and re-read.
    mkdirSync(join(home, ".codex"), { recursive: true });
    writeFileSync(codex.configPath, edit.updated);

    // Re-resolving should report already_current.
    const edit2 = resolveClientEdit("setup", codex, launcher);
    assert.equal(edit2.action, "already_current");

    // Remove should work.
    await writeFile(
      codex.configPath,
      edit2.original === null
        ? `${edit.updated}jacobian_theme = "preserve"\n`
        : `${edit2.original}jacobian_theme = "preserve"\n`,
    );
    const edit3 = resolveClientEdit("remove", codex, launcher);
    assert.equal(edit3.action, "remove");
    assert.match(edit3.updated, /jacobian_theme = "preserve"/);
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

test("remove preserves unrelated JSON settings and MCP servers", async () => {
  const base = await mkdtemp(join(tmpdir(), "jacobian-remove-preserve-"));
  try {
    const home = await fakeHome(base, ["claude"]);
    const claude = clientDefinitions(home).find((d) => d.id === "claude");
    const original = {
      mcpServers: {
        jacobian: { command: "old", args: [] },
        other: { command: "other", args: ["--safe"] },
      },
      theme: "dark",
    };
    await writeFile(claude.configPath, JSON.stringify(original, null, 2) + "\n");
    const launcher = buildLauncher();
    const edit = resolveClientEdit("remove", claude, launcher);
    applyEdits([edit]);
    const updated = JSON.parse(await readFile(claude.configPath, "utf8"));
    assert.equal(updated.mcpServers.jacobian, undefined);
    assert.deepEqual(updated.mcpServers.other, original.mcpServers.other);
    assert.equal(updated.theme, "dark");
  } finally {
    await rm(base, { recursive: true, force: true });
  }
});

test("multi-client config writes roll back as one transaction", async () => {
  const base = await mkdtemp(join(tmpdir(), "jacobian-rollback-"));
  try {
    const firstPath = join(base, "first.json");
    const blockedParent = join(base, "blocked");
    await writeFile(firstPath, "before\n");
    await writeFile(blockedParent, "not a directory\n");
    assert.throws(
      () =>
        applyEdits([
          {
            path: firstPath,
            original: "before\n",
            updated: "after\n",
            action: "update",
          },
          {
            path: join(blockedParent, "config.json"),
            original: null,
            updated: "{}\n",
            action: "create",
          },
        ]),
      /Earlier config writes were rolled back/,
    );
    assert.equal(await readFile(firstPath, "utf8"), "before\n");
  } finally {
    await rm(base, { recursive: true, force: true });
  }
});

test("rollback excludes no-op edits and preserves concurrent config creation", async () => {
  const base = await mkdtemp(join(tmpdir(), "jacobian-noop-rollback-"));
  try {
    const concurrentPath = join(base, "concurrent.json");
    const blockedParent = join(base, "blocked");
    await writeFile(concurrentPath, "created concurrently\n");
    await writeFile(blockedParent, "not a directory\n");
    assert.throws(
      () =>
        applyEdits([
          {
            path: concurrentPath,
            original: null,
            updated: null,
            action: "not_configured",
          },
          {
            path: join(blockedParent, "config.json"),
            original: null,
            updated: "{}\n",
            action: "create",
          },
        ]),
      /Earlier config writes were rolled back/,
    );
    assert.equal(await readFile(concurrentPath, "utf8"), "created concurrently\n");
  } finally {
    await rm(base, { recursive: true, force: true });
  }
});

test("rollback does not overwrite a config changed after Jacobian wrote it", async () => {
  const base = await mkdtemp(join(tmpdir(), "jacobian-concurrent-rollback-"));
  try {
    const firstPath = join(base, "first.json");
    const blockedParent = join(base, "blocked");
    await writeFile(firstPath, "before\n");
    await writeFile(blockedParent, "not a directory\n");
    const failingEdit = {
      original: null,
      updated: "{}\n",
      action: "create",
      get path() {
        writeFileSync(firstPath, "concurrent update\n");
        return join(blockedParent, "config.json");
      },
    };
    assert.throws(
      () =>
        applyEdits([
          {
            path: firstPath,
            original: "before\n",
            updated: "after\n",
            action: "update",
          },
          failingEdit,
        ]),
      /concurrent value was left untouched/,
    );
    assert.equal(await readFile(firstPath, "utf8"), "concurrent update\n");
  } finally {
    await rm(base, { recursive: true, force: true });
  }
});

test("config creation uses a private atomic replacement", async () => {
  const base = await mkdtemp(join(tmpdir(), "jacobian-atomic-config-"));
  try {
    const configPath = join(base, "nested", "config.json");
    applyEdits([
      {
        path: configPath,
        original: null,
        updated: "{}\n",
        action: "create",
      },
    ]);
    assert.equal(await readFile(configPath, "utf8"), "{}\n");
    assert.equal((await stat(configPath)).mode & 0o777, 0o600);
    assert.deepEqual(await readdir(join(base, "nested")), ["config.json"]);
  } finally {
    await rm(base, { recursive: true, force: true });
  }
});

test("atomic config replacement preserves an existing mode under umask", async () => {
  const base = await mkdtemp(join(tmpdir(), "jacobian-config-mode-"));
  try {
    const configPath = join(base, "config.json");
    await writeFile(configPath, "before\n");
    await chmod(configPath, 0o664);
    const previousUmask = process.umask(0o077);
    try {
      applyEdits([
        {
          path: configPath,
          original: "before\n",
          updated: "after\n",
          action: "update",
        },
      ]);
    } finally {
      process.umask(previousUmask);
    }
    assert.equal(await readFile(configPath, "utf8"), "after\n");
    assert.equal((await stat(configPath)).mode & 0o777, 0o664);
  } finally {
    await rm(base, { recursive: true, force: true });
  }
});

test("config writes reject symbolic links without detaching them", async () => {
  const base = await mkdtemp(join(tmpdir(), "jacobian-symlink-config-"));
  try {
    const targetPath = join(base, "dotfiles-config.json");
    const configPath = join(base, "client-config.json");
    await writeFile(targetPath, "before\n");
    await symlink(targetPath, configPath);
    assert.throws(
      () =>
        applyEdits([
          {
            path: configPath,
            original: "before\n",
            updated: "after\n",
            action: "update",
          },
        ]),
      /symbolic link/,
    );
    assert.equal((await lstat(configPath)).isSymbolicLink(), true);
    assert.equal(await readFile(targetPath, "utf8"), "before\n");
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

test("declining setup exits nonzero without claiming completion", async () => {
  const home = await mkdtemp(join(tmpdir(), "jacobian-cli-cancel-"));
  try {
    const result = spawnSync(
      process.execPath,
      [join(npmRoot, "bin", "jacobian.cjs"), "setup", "--client", "codex"],
      {
        encoding: "utf8",
        env: { ...process.env, HOME: home },
        input: "n\n",
      },
    );
    assert.equal(result.status, 1);
    assert.match(result.stderr, /setup cancelled/);
    assert.doesNotMatch(result.stderr, /Setup complete/);
    assert.equal(existsSync(join(home, ".codex", "config.toml")), false);
  } finally {
    await rm(home, { recursive: true, force: true });
  }
});

test("source setup rejects an unsupported client before writing config", async () => {
  const home = await mkdtemp(join(tmpdir(), "jacobian-cli-unknown-client-"));
  try {
    const result = spawnSync(
      process.execPath,
      [
        join(npmRoot, "bin", "jacobian.cjs"),
        "setup",
        "--source",
        join(npmRoot, ".."),
        "--profile",
        "full-python",
        "--client",
        "codxe",
        "--yes",
      ],
      { encoding: "utf8", env: { ...process.env, HOME: home } },
    );
    assert.equal(result.status, 1);
    assert.match(result.stderr, /Unknown MCP client: codxe/);
    assert.equal(existsSync(join(home, ".codex", "config.toml")), false);
  } finally {
    await rm(home, { recursive: true, force: true });
  }
});

test("source setup rejects every missing option value", async () => {
  const home = await mkdtemp(join(tmpdir(), "jacobian-cli-missing-value-"));
  try {
    const flags = [
      "--source",
      "--state-dir",
      "--uv-bin",
      "--profile",
      "--provider-path",
      "--project-environment",
      "--elan-home",
      "--lean-runtime",
      "--client",
    ];
    for (const flag of flags) {
      const result = spawnSync(
        process.execPath,
        [
          join(npmRoot, "bin", "jacobian.cjs"),
          "setup",
          "--client",
          "codex",
          "--yes",
          flag,
        ],
        { encoding: "utf8", env: { ...process.env, HOME: home } },
      );
      assert.equal(result.status, 1, flag);
      assert.ok(result.stderr.includes(`${flag} requires a value`), result.stderr);
    }
    assert.equal(existsSync(join(home, ".codex", "config.toml")), false);
  } finally {
    await rm(home, { recursive: true, force: true });
  }
});
