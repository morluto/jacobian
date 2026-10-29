import assert from "node:assert/strict";
import { chmod, lstat, mkdir, mkdtemp, readFile, readlink, rm, stat, symlink, writeFile } from "node:fs/promises";
import { spawnSync } from "node:child_process";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import test from "node:test";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const npmRoot = dirname(fileURLToPath(import.meta.url));
const packageMetadata = require("./package.json");
const TOML = require("@iarna/toml");
const { pythonVersionFromNpmVersion, packageSpec } = require("./bin/jacobian.cjs");

async function setupEnvironment(base) {
  const bin = join(base, "bin");
  const home = join(base, "home");
  await mkdir(bin, { recursive: true });
  await writeFile(join(bin, "uvx"), "#!/bin/sh\nexit 0\n", "utf8");
  await chmod(join(bin, "uvx"), 0o755);
  return {
    ...process.env,
    HOME: home,
    PATH: `${bin}:${process.env.PATH}`,
  };
}

function runCarrier(args, env) {
  return spawnSync(process.execPath, [join(npmRoot, "bin", "jacobian.cjs"), ...args], {
    encoding: "utf8",
    env,
  });
}

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

test("pythonVersionFromNpmVersion maps release and prerelease spellings", () => {
  assert.equal(pythonVersionFromNpmVersion("0.12.0"), "0.12.0");
  assert.equal(pythonVersionFromNpmVersion("0.12.0-alpha.1"), "0.12.0a1");
  assert.equal(pythonVersionFromNpmVersion("0.12.0-beta.2"), "0.12.0b2");
  assert.equal(pythonVersionFromNpmVersion("0.12.0-rc.3"), "0.12.0rc3");
  assert.throws(() => pythonVersionFromNpmVersion("0.12"), /unsupported/);
});

test("packageSpec pins the exact matching Python package by default", () => {
  const saved = process.env.JACOBIAN_PACKAGE;
  delete process.env.JACOBIAN_PACKAGE;
  try {
    assert.equal(
      packageSpec(),
      `jacobian==${pythonVersionFromNpmVersion(packageMetadata.version)}`,
    );
  } finally {
    if (saved !== undefined) process.env.JACOBIAN_PACKAGE = saved;
  }
});

test("packageSpec honors the JACOBIAN_PACKAGE override", () => {
  const saved = process.env.JACOBIAN_PACKAGE;
  process.env.JACOBIAN_PACKAGE = "git+https://example/jacobian.git@deadbeef";
  try {
    assert.equal(packageSpec(), process.env.JACOBIAN_PACKAGE);
  } finally {
    if (saved === undefined) delete process.env.JACOBIAN_PACKAGE;
    else process.env.JACOBIAN_PACKAGE = saved;
  }
});

test(
  "jacobian mcp execs the exact canonical uvx command with forwarded args",
  { skip: process.platform === "win32" },
  async () => {
    const base = await mkdtemp(join(tmpdir(), "jacobian-carrier-mcp-"));
    try {
      const log = join(base, "argv.json");
      const fakeUvx = join(base, "uvx");
      await writeFile(
        fakeUvx,
        `#!/usr/bin/env node
require("node:fs").writeFileSync(
  process.env.JACOBIAN_CARRIER_LOG,
  JSON.stringify(process.argv.slice(2)),
);
`,
        "utf8",
      );
      await chmod(fakeUvx, 0o755);

      const result = spawnSync(
        process.execPath,
        [
          join(npmRoot, "bin", "jacobian.cjs"),
          "mcp",
          "--managed-by-setup",
          "--state-dir",
          join(base, "state"),
        ],
        {
          encoding: "utf8",
          env: {
            ...process.env,
            JACOBIAN_UV_BIN: fakeUvx,
            JACOBIAN_CARRIER_LOG: log,
          },
        },
      );
      assert.equal(result.status, 0, result.stderr);
      assert.deepEqual(JSON.parse(await readFile(log, "utf8")), [
        "--python",
        "3.12",
        "--from",
        `jacobian==${pythonVersionFromNpmVersion(packageMetadata.version)}`,
        "jacobian-mcp",
        "--state-dir",
        join(base, "state"),
      ]);
    } finally {
      await rm(base, { recursive: true, force: true });
    }
  },
);

test("jacobian --version prints the carrier package version", () => {
  const result = spawnSync(
    process.execPath,
    [join(npmRoot, "bin", "jacobian.cjs"), "--version"],
    { encoding: "utf8" },
  );
  assert.equal(result.status, 0, result.stderr);
  assert.equal(result.stdout.trim(), `jacobian ${packageMetadata.version}`);
});

test("jacobian with no command prints help to stderr and exits zero", () => {
  const result = spawnSync(
    process.execPath,
    [join(npmRoot, "bin", "jacobian.cjs")],
    { encoding: "utf8" },
  );
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stderr, /jacobian mcp \[args\.\.\.\]/);
  assert.match(result.stderr, /uvx --python 3\.12 --from jacobian==<version> jacobian-mcp/);
});

test("jacobian rejects an unknown command without forwarding to the Python CLI", async () => {
  const base = await mkdtemp(join(tmpdir(), "jacobian-carrier-unknown-"));
  try {
    const marker = join(base, "forwarded");
    const fakeUvx = join(base, "uvx");
    await writeFile(
      fakeUvx,
      `#!/usr/bin/env node
require("node:fs").writeFileSync(${JSON.stringify(marker)}, "forwarded");
`,
      "utf8",
    );
    await chmod(fakeUvx, 0o755);

    const result = spawnSync(
      process.execPath,
      [join(npmRoot, "bin", "jacobian.cjs"), "run", "matrix.determinant.compute"],
      {
        encoding: "utf8",
        env: { ...process.env, JACOBIAN_UV_BIN: fakeUvx },
      },
    );
    assert.equal(result.status, 1, result.stderr);
    assert.match(result.stderr, /Unknown command: run/);
    assert.match(result.stderr, /jacobian mcp \[args\.\.\.\]/);
    await assert.rejects(readFile(marker), { code: "ENOENT" });
  } finally {
    await rm(base, { recursive: true, force: true });
  }
});

test("setup dry-run emits a pinned, non-mutating Codex plan", async () => {
  const base = await mkdtemp(join(tmpdir(), "jacobian-carrier-setup-plan-"));
  try {
    const env = await setupEnvironment(base);
    const result = runCarrier(["setup", "--codex", "--dry-run", "--json"], env);

    assert.equal(result.status, 0, result.stderr);
    const report = JSON.parse(result.stdout);
    assert.equal(report.status, "planned");
    assert.equal(report.dry_run, true);
    assert.deepEqual(report.clients.map((client) => client.client), ["codex"]);
    assert.deepEqual(report.launcher, {
      command: "npx",
      args: ["--yes", `jacobian@${packageMetadata.version}`, "mcp", "--managed-by-setup"],
    });
    await assert.rejects(readFile(join(env.HOME, ".codex", "config.toml")), {
      code: "ENOENT",
    });
    await assert.rejects(readFile(join(env.HOME, ".agents", "skills", "jacobian-math", "SKILL.md")), {
      code: "ENOENT",
    });
  } finally {
    await rm(base, { recursive: true, force: true });
  }
});

test("setup configures every supported client and preserves unrelated configuration", async () => {
  const base = await mkdtemp(join(tmpdir(), "jacobian-carrier-setup-apply-"));
  try {
    const env = await setupEnvironment(base);
    const claudePath = join(env.HOME, ".claude.json");
    await mkdir(dirname(claudePath), { recursive: true });
    await writeFile(
      claudePath,
      JSON.stringify({ mcpServers: { other: { command: "other" } }, theme: "dark" }),
      "utf8",
    );
    const result = runCarrier(["setup", "--all", "--yes", "--json"], env);

    assert.equal(result.status, 0, result.stderr);
    const report = JSON.parse(result.stdout);
    assert.equal(report.status, "configured");
    assert.deepEqual(report.clients.map((client) => client.client), [
      "claude",
      "opencode",
      "codex",
      "cursor",
      "gemini",
      "antigravity",
    ]);

    const claude = JSON.parse(await readFile(claudePath, "utf8"));
    assert.equal(claude.theme, "dark");
    assert.deepEqual(claude.mcpServers.other, { command: "other" });
    assert.deepEqual(claude.mcpServers.jacobian, {
      command: "npx",
      args: ["--yes", `jacobian@${packageMetadata.version}`, "mcp", "--managed-by-setup"],
    });

    const codex = await readFile(join(env.HOME, ".codex", "config.toml"), "utf8");
    assert.match(codex, /# Managed by Jacobian setup\./);
    assert.match(codex, /\[mcp_servers\.jacobian\]/);
    assert.match(codex, /startup_timeout_sec = 120/);
    const opencode = JSON.parse(
      await readFile(join(env.HOME, ".config", "opencode", "opencode.json"), "utf8"),
    );
    assert.deepEqual(opencode.mcp.jacobian, {
      type: "local",
      command: ["npx", "--yes", `jacobian@${packageMetadata.version}`, "mcp", "--managed-by-setup"],
      cwd: ".",
      enabled: true,
    });
    const universalSkill = await readFile(
      join(env.HOME, ".agents", "skills", "jacobian-math", "SKILL.md"),
      "utf8",
    );
    assert.match(universalSkill, /^---\nname: jacobian-math\n/);
    assert.match(universalSkill, /<!-- Managed by Jacobian setup\. -->/);
    for (const relative of [
      ".claude/skills/jacobian-math/SKILL.md",
      ".cursor/skills/jacobian-math/SKILL.md",
      ".gemini/skills/jacobian-math/SKILL.md",
      ".agent/skills/jacobian-math/SKILL.md",
    ]) {
      assert.equal(await readFile(join(env.HOME, relative), "utf8"), universalSkill);
    }

    const repeat = runCarrier(["setup", "--codex", "--dry-run", "--json"], env);
    assert.equal(repeat.status, 0, repeat.stderr);
    assert.equal(JSON.parse(repeat.stdout).clients[0].action, "already current");
    assert.equal(JSON.parse(repeat.stdout).clients[0].skill_action, "already current");
  } finally {
    await rm(base, { recursive: true, force: true });
  }
});

test("setup refreshes a selected unmanaged Jacobian registration", async () => {
  const base = await mkdtemp(join(tmpdir(), "jacobian-carrier-setup-conflict-"));
  try {
    const env = await setupEnvironment(base);
    const configPath = join(env.HOME, ".claude.json");
    const original = JSON.stringify({
      mcpServers: { jacobian: { command: "python", args: ["-m", "jacobian.mcp"] } },
    });
    await mkdir(dirname(configPath), { recursive: true });
    await writeFile(configPath, original, "utf8");

    const result = runCarrier(["setup", "--claude", "--yes", "--json"], env);
    assert.equal(result.status, 0, result.stderr);
    assert.deepEqual(JSON.parse(await readFile(configPath, "utf8")).mcpServers.jacobian, {
      command: "npx",
      args: ["--yes", `jacobian@${packageMetadata.version}`, "mcp", "--managed-by-setup"],
    });
  } finally {
    await rm(base, { recursive: true, force: true });
  }
});

test("setup preserves existing configuration and managed skill permissions", { skip: process.platform === "win32" }, async () => {
  const base = await mkdtemp(join(tmpdir(), "jacobian-carrier-setup-mode-"));
  try {
    const env = await setupEnvironment(base);
    const configPath = join(env.HOME, ".claude.json");
    const skillPath = join(env.HOME, ".claude", "skills", "jacobian-math", "SKILL.md");
    await mkdir(dirname(skillPath), { recursive: true });
    await writeFile(configPath, '{"theme":"dark","mcpServers":{"jacobian":{"command":"old"}}}', "utf8");
    await writeFile(skillPath, "<!-- Managed by Jacobian setup. -->\nold skill\n", "utf8");
    await chmod(configPath, 0o600);
    await chmod(skillPath, 0o640);

    const result = runCarrier(["setup", "--claude", "--yes", "--json"], env);

    assert.equal(result.status, 0, result.stderr);
    assert.equal(JSON.parse(await readFile(configPath, "utf8")).mcpServers.jacobian.command, "npx");
    assert.equal((await stat(configPath)).mode & 0o777, 0o600);
    assert.equal((await stat(skillPath)).mode & 0o777, 0o640);
    assert.match(await readFile(skillPath, "utf8"), /^---\nname: jacobian-math\n/);
  } finally {
    await rm(base, { recursive: true, force: true });
  }
});

test("setup leaves an existing temporary-name collision untouched", async () => {
  const base = await mkdtemp(join(tmpdir(), "jacobian-carrier-setup-temp-collision-"));
  try {
    const env = await setupEnvironment(base);
    const configPath = join(env.HOME, ".claude.json");
    const collisionLog = join(base, "collision-path");
    const preload = join(base, "fixed-temp-name.cjs");
    const original = '{"theme":"dark"}';
    await mkdir(dirname(configPath), { recursive: true });
    await writeFile(configPath, original, "utf8");
    await writeFile(preload, `
const fs = require("node:fs");
const crypto = require("node:crypto");
crypto.randomBytes = (size) => Buffer.alloc(size);
const collision = process.env.JACOBIAN_TEST_CONFIG + ".jacobian-" + process.pid + "-000000000000.tmp";
fs.writeFileSync(collision, "unrelated temporary file", { flag: "wx" });
fs.writeFileSync(process.env.JACOBIAN_TEST_COLLISION_LOG, collision);
`, "utf8");

    const result = runCarrier(["setup", "--claude", "--yes", "--json"], {
      ...env,
      NODE_OPTIONS: `${env.NODE_OPTIONS || ""} --require=${preload}`,
      JACOBIAN_TEST_CONFIG: configPath,
      JACOBIAN_TEST_COLLISION_LOG: collisionLog,
    });

    assert.equal(result.status, 1);
    const collision = await readFile(collisionLog, "utf8");
    assert.equal(await readFile(collision, "utf8"), "unrelated temporary file");
    assert.equal(await readFile(configPath, "utf8"), original);
    await assert.rejects(readFile(join(env.HOME, ".claude", "skills", "jacobian-math", "SKILL.md")), { code: "ENOENT" });
  } finally {
    await rm(base, { recursive: true, force: true });
  }
});

test("setup rejects ambiguous duplicate JSONC keys before any write", async () => {
  const base = await mkdtemp(join(tmpdir(), "jacobian-carrier-setup-duplicate-"));
  try {
    const env = await setupEnvironment(base);
    const configPath = join(env.HOME, ".claude.json");
    await mkdir(dirname(configPath), { recursive: true });
    for (const source of [
      '{"mcpServers":{"jacobian":{"command":"first"},"jacobian":{"command":"effective-old"}}}',
      '{"mcpServers":{},"mcpServers":{"jacobian":{"command":"effective-old"}}}',
      '{"mcpServers":{"jacobian":{"command":"first"},"jaco\\u0062ian":{"command":"effective-old"}}}',
      '{"mcpServers":{},"other":{"setting":1,"setting":2}}',
    ]) {
      await writeFile(configPath, source, "utf8");

      const result = runCarrier(["upgrade", "--claude", "--yes", "--json"], env);

      assert.equal(result.status, 1);
      assert.match(result.stderr, /duplicate JSON object key in configuration/);
      assert.equal(await readFile(configPath, "utf8"), source);
      await assert.rejects(readFile(join(env.HOME, ".claude", "skills", "jacobian-math", "SKILL.md")), { code: "ENOENT" });
    }
    await writeFile(configPath, '{\n// valid JSONC\n"mcpServers":{},"other":{"setting":1},"another":{"setting":2},\n}', "utf8");
    const accepted = runCarrier(["setup", "--claude", "--yes", "--json"], env);
    assert.equal(accepted.status, 0, accepted.stderr);
    const updated = await readFile(configPath, "utf8");
    assert.match(updated, /\/\/ valid JSONC/);
    const { parse } = require("jsonc-parser");
    assert.equal(parse(updated).mcpServers.jacobian.command, "npx");
  } finally {
    await rm(base, { recursive: true, force: true });
  }
});

test("upgrade accepts quoted, spaced, and commented Codex registration headers", async () => {
  const base = await mkdtemp(join(tmpdir(), "jacobian-carrier-setup-toml-header-"));
  try {
    const env = await setupEnvironment(base);
    const configPath = join(env.HOME, ".codex", "config.toml");
    await mkdir(dirname(configPath), { recursive: true });
    for (const registration of [
      '[mcp_servers."jacobian"]\ncommand="old"\n',
      "[ 'mcp_servers' . 'jacobian' ] # registration\ncommand='old'\n",
      '[mcp_servers.jacobian] # registration\r\ncommand="old"\r\n',
      '["mcp_servers"] # parent\n  "jacobian" = {command="old"}\n',
    ]) {
      const neighbor = '\n  [mcp_servers.other] # keep this comment\ncommand = "other"\n';
      await writeFile(configPath, registration + neighbor, "utf8");

      const result = runCarrier(["upgrade", "--codex", "--yes", "--json"], env);

      assert.equal(result.status, 0, result.stderr);
      const updated = await readFile(configPath, "utf8");
      const parsed = TOML.parse(updated);
      assert.equal(parsed.mcp_servers.jacobian.command, "npx");
      assert.equal(parsed.mcp_servers.jacobian.args[1], `jacobian@${packageMetadata.version}`);
      assert.deepEqual(parsed.mcp_servers.other, { command: "other" });
      assert.ok(updated.includes(neighbor.trimStart()));
      const repeated = runCarrier(["setup", "--codex", "--yes", "--json"], env);
      assert.equal(repeated.status, 0, repeated.stderr);
      assert.equal(JSON.parse(repeated.stdout).clients[0].action, "already current");
      assert.equal(await readFile(configPath, "utf8"), updated);
    }
  } finally {
    await rm(base, { recursive: true, force: true });
  }
});

test("setup refuses TOML edits that mistake multiline-string text for a table", async () => {
  const base = await mkdtemp(join(tmpdir(), "jacobian-carrier-setup-toml-string-"));
  try {
    const env = await setupEnvironment(base);
    const configPath = join(env.HOME, ".codex", "config.toml");
    await mkdir(dirname(configPath), { recursive: true });
    for (const note of [
      'note = """\n[mcp_servers.jacobian]\ncommand="text only"\n"""\n',
      "note = '''\n[mcp_servers.jacobian]\ncommand='text only'\n[elsewhere]\n'''\n",
    ]) {
      const source = note + '[mcp_servers."jacobian"] # actual\ncommand="old"\n[mcp_servers.other]\ncommand="other"\n';
      assert.equal(TOML.parse(source).mcp_servers.jacobian.command, "old");
      await writeFile(configPath, source, "utf8");

      const result = runCarrier(["setup", "--codex", "--yes", "--json"], env);

      assert.equal(result.status, 1);
      assert.match(result.stderr, /Jacobian entry cannot be safely updated/);
      assert.equal(await readFile(configPath, "utf8"), source);
      await assert.rejects(readFile(join(env.HOME, ".agents", "skills", "jacobian-math", "SKILL.md")), { code: "ENOENT" });
    }
  } finally {
    await rm(base, { recursive: true, force: true });
  }
});

test("setup reads only authored JSONC object properties", async () => {
  const base = await mkdtemp(join(tmpdir(), "jacobian-carrier-setup-own-properties-"));
  try {
    const env = await setupEnvironment(base);
    const configPath = join(env.HOME, ".claude.json");
    const expected = {
      command: "npx",
      args: ["--yes", `jacobian@${packageMetadata.version}`, "mcp", "--managed-by-setup"],
    };
    await mkdir(dirname(configPath), { recursive: true });
    for (const original of [
      { ["__proto__"]: { mcpServers: { jacobian: expected } } },
      { mcpServers: { ["__proto__"]: { jacobian: expected } } },
      { ["__proto__"]: { mcpServers: "not an authored section" } },
    ]) {
      const source = JSON.stringify(original);
      await writeFile(configPath, source, "utf8");

      const result = runCarrier(["setup", "--claude", "--yes", "--json"], env);

      assert.equal(result.status, 0, result.stderr);
      assert.equal(JSON.parse(result.stdout).clients[0].action, "create");
      const updated = JSON.parse(await readFile(configPath, "utf8"));
      assert.ok(Object.hasOwn(updated, "mcpServers"));
      assert.ok(Object.hasOwn(updated.mcpServers, "jacobian"));
      assert.deepEqual(updated.mcpServers.jacobian, expected);
      if (Object.hasOwn(original, "__proto__")) {
        assert.deepEqual(updated.__proto__, original.__proto__);
      } else {
        assert.deepEqual(updated.mcpServers.__proto__, original.mcpServers.__proto__);
      }
      const repeated = runCarrier(["setup", "--claude", "--yes", "--json"], env);
      assert.equal(repeated.status, 0, repeated.stderr);
      assert.equal(JSON.parse(repeated.stdout).clients[0].action, "already current");
    }
  } finally {
    await rm(base, { recursive: true, force: true });
  }
});

test("setup updates symbolic-link targets without replacing the links", { skip: process.platform === "win32" }, async () => {
  const base = await mkdtemp(join(tmpdir(), "jacobian-carrier-setup-symlink-"));
  try {
    const env = await setupEnvironment(base);
    const configPath = join(env.HOME, ".claude.json");
    const target = join(base, "dotfiles", "claude.json");
    const intermediate = join(env.HOME, "claude-link");
    const skillPath = join(env.HOME, ".claude", "skills", "jacobian-math", "SKILL.md");
    const skillTarget = join(base, "dotfiles", "jacobian-skill.md");
    await mkdir(dirname(target), { recursive: true });
    await mkdir(dirname(skillPath), { recursive: true });
    await writeFile(target, '{"theme":"dark"}', "utf8");
    await writeFile(skillTarget, "<!-- Managed by Jacobian setup. -->\nold skill\n", "utf8");
    await symlink(target, intermediate);
    await symlink("claude-link", configPath);
    await symlink(skillTarget, skillPath);

    const result = runCarrier(["setup", "--claude", "--yes", "--json"], env);

    assert.equal(result.status, 0, result.stderr);
    assert.ok((await lstat(configPath)).isSymbolicLink());
    assert.ok((await lstat(intermediate)).isSymbolicLink());
    assert.ok((await lstat(skillPath)).isSymbolicLink());
    assert.equal(await readlink(configPath), "claude-link");
    assert.equal(await readlink(intermediate), target);
    const updated = JSON.parse(await readFile(target, "utf8"));
    assert.equal(updated.theme, "dark");
    assert.equal(updated.mcpServers.jacobian.command, "npx");
    assert.equal(await readFile(configPath, "utf8"), await readFile(target, "utf8"));
    assert.match(await readFile(skillTarget, "utf8"), /^---\nname: jacobian-math\n/);
  } finally {
    await rm(base, { recursive: true, force: true });
  }
});

test("setup refuses dangling configuration or skill links before any write", { skip: process.platform === "win32" }, async () => {
  const base = await mkdtemp(join(tmpdir(), "jacobian-carrier-setup-dangling-link-"));
  try {
    for (const kind of ["configuration", "skill"]) {
      const env = await setupEnvironment(join(base, kind));
      const configPath = join(env.HOME, ".claude.json");
      const skillPath = join(env.HOME, ".claude", "skills", "jacobian-math", "SKILL.md");
      const link = kind === "configuration" ? configPath : skillPath;
      await mkdir(dirname(link), { recursive: true });
      await symlink("missing-target", link);

      const result = runCarrier(["setup", "--claude", "--yes", "--json"], env);

      assert.equal(result.status, 1);
      assert.match(result.stderr, /symbolic link has a missing target/);
      assert.ok((await lstat(link)).isSymbolicLink());
      assert.equal(await readlink(link), "missing-target");
      const untouched = kind === "configuration" ? skillPath : configPath;
      await assert.rejects(lstat(untouched), { code: "ENOENT" });
      await assert.rejects(lstat(join(dirname(link), "missing-target")), { code: "ENOENT" });
    }
  } finally {
    await rm(base, { recursive: true, force: true });
  }
});

test("setup coalesces identical updates through multiple configuration links", { skip: process.platform === "win32" }, async () => {
  const base = await mkdtemp(join(tmpdir(), "jacobian-carrier-setup-shared-target-"));
  try {
    const env = await setupEnvironment(base);
    const target = join(env.HOME, "shared.json");
    const claude = join(env.HOME, ".claude.json");
    const cursor = join(env.HOME, ".cursor", "mcp.json");
    await mkdir(dirname(cursor), { recursive: true });
    await writeFile(target, '{"theme":"dark"}', "utf8");
    await symlink("shared.json", claude);
    await symlink("../shared.json", cursor);

    const result = runCarrier(["setup", "--claude", "--cursor", "--yes", "--json"], env);

    assert.equal(result.status, 0, result.stderr);
    assert.ok((await lstat(claude)).isSymbolicLink());
    assert.ok((await lstat(cursor)).isSymbolicLink());
    assert.equal(JSON.parse(await readFile(target, "utf8")).mcpServers.jacobian.command, "npx");
    for (const client of JSON.parse(result.stdout).clients) {
      assert.match(await readFile(client.skill_path, "utf8"), /^---\nname: jacobian-math\n/);
    }
  } finally {
    await rm(base, { recursive: true, force: true });
  }
});

test("setup coalesces a missing shared skill through a linked parent directory", { skip: process.platform === "win32" }, async () => {
  const base = await mkdtemp(join(tmpdir(), "jacobian-carrier-setup-shared-skill-"));
  try {
    const env = await setupEnvironment(base);
    const sharedSkills = join(env.HOME, ".agents", "skills");
    const claudeSkills = join(env.HOME, ".claude", "skills");
    await mkdir(sharedSkills, { recursive: true });
    await mkdir(dirname(claudeSkills), { recursive: true });
    await symlink("../.agents/skills", claudeSkills);

    const result = runCarrier(["setup", "--claude", "--codex", "--yes", "--json"], env);

    assert.equal(result.status, 0, result.stderr);
    assert.ok((await lstat(claudeSkills)).isSymbolicLink());
    assert.equal(await readlink(claudeSkills), "../.agents/skills");
    const report = JSON.parse(result.stdout);
    assert.deepEqual(report.clients.map((client) => client.client), ["claude", "codex"]);
    const skill = await readFile(join(sharedSkills, "jacobian-math", "SKILL.md"), "utf8");
    assert.match(skill, /^---\nname: jacobian-math\n/);
    for (const client of report.clients) {
      assert.equal(await readFile(client.skill_path, "utf8"), skill);
    }
    const claude = JSON.parse(await readFile(join(env.HOME, ".claude.json"), "utf8"));
    assert.equal(claude.mcpServers.jacobian.command, "npx");
    const codex = TOML.parse(await readFile(join(env.HOME, ".codex", "config.toml"), "utf8"));
    assert.equal(codex.mcp_servers.jacobian.command, "npx");
  } finally {
    await rm(base, { recursive: true, force: true });
  }
});

test("setup refuses conflicting updates to one linked target before any write", { skip: process.platform === "win32" }, async () => {
  const base = await mkdtemp(join(tmpdir(), "jacobian-carrier-setup-conflicting-target-"));
  try {
    const env = await setupEnvironment(base);
    const target = join(env.HOME, "shared.json");
    const claude = join(env.HOME, ".claude.json");
    const opencode = join(env.HOME, ".config", "opencode", "opencode.json");
    const source = '{"mcpServers":{},"mcp":{}}';
    await mkdir(dirname(opencode), { recursive: true });
    await writeFile(target, source, "utf8");
    await symlink("shared.json", claude);
    await symlink("../../shared.json", opencode);

    const result = runCarrier(["setup", "--claude", "--opencode", "--yes", "--json"], env);

    assert.equal(result.status, 1);
    assert.match(result.stderr, /selected setup entries propose conflicting changes/);
    assert.equal(await readFile(target, "utf8"), source);
    assert.ok((await lstat(claude)).isSymbolicLink());
    assert.ok((await lstat(opencode)).isSymbolicLink());
    await assert.rejects(lstat(join(env.HOME, ".claude", "skills", "jacobian-math", "SKILL.md")), { code: "ENOENT" });
    await assert.rejects(lstat(join(env.HOME, ".agents", "skills", "jacobian-math", "SKILL.md")), { code: "ENOENT" });
  } finally {
    await rm(base, { recursive: true, force: true });
  }
});

test("setup migrates a Codex inline registration and preserves neighboring tables", async () => {
  const base = await mkdtemp(join(tmpdir(), "jacobian-carrier-setup-codex-inline-"));
  try {
    const env = await setupEnvironment(base);
    const configPath = join(env.HOME, ".codex", "config.toml");
    const original = `[mcp_servers]\njacobian = { command = "uv", args = ["run", "jacobian-mcp"] }\nother_inline = { command = "other-inline" }\n\n[mcp_servers.other]\ncommand = "other"\n`;
    await mkdir(dirname(configPath), { recursive: true });
    await writeFile(configPath, original, "utf8");

    const result = runCarrier(["setup", "--codex", "--yes", "--json"], env);
    assert.equal(result.status, 0, result.stderr);
    const updated = await readFile(configPath, "utf8");
    assert.match(updated, /# Managed by Jacobian setup\.\n\[mcp_servers\.jacobian\]/);
    assert.match(updated, /\[mcp_servers\.other\]\ncommand = "other"/);
    assert.match(updated, /^other_inline = \{ command = "other-inline" \}$/m);
    assert.deepEqual(TOML.parse(updated).mcp_servers.other_inline, { command: "other-inline" });
    assert.doesNotMatch(updated, /^jacobian =/m);
  } finally {
    await rm(base, { recursive: true, force: true });
  }
});

test("upgrade runs the pinned setup journey", async () => {
  const base = await mkdtemp(join(tmpdir(), "jacobian-carrier-upgrade-"));
  try {
    const env = await setupEnvironment(base);
    const result = runCarrier(["upgrade", "--codex", "--dry-run", "--json"], env);
    assert.equal(result.status, 0, result.stderr);
    const report = JSON.parse(result.stdout);
    assert.equal(report.status, "planned");
    assert.equal(report.launcher.args[1], `jacobian@${packageMetadata.version}`);
    await assert.rejects(readFile(join(env.HOME, ".codex", "config.toml")), { code: "ENOENT" });
  } finally {
    await rm(base, { recursive: true, force: true });
  }
});

test("upgrade migrates a CRLF Codex registration", async () => {
  const base = await mkdtemp(join(tmpdir(), "jacobian-carrier-upgrade-crlf-"));
  try {
    const env = await setupEnvironment(base);
    const configPath = join(env.HOME, ".codex", "config.toml");
    await mkdir(dirname(configPath), { recursive: true });
    await writeFile(
      configPath,
      `[mcp_servers]\r\njacobian = { command = "uv", args = ["run", "jacobian-mcp"] }\r\nother = { command = "other" }\r\n`,
      "utf8",
    );

    const result = runCarrier(["upgrade", "--codex", "--yes", "--json"], env);
    assert.equal(result.status, 0, result.stderr);
    const parsed = TOML.parse(await readFile(configPath, "utf8"));
    assert.equal(parsed.mcp_servers.jacobian.args[1], `jacobian@${packageMetadata.version}`);
    assert.deepEqual(parsed.mcp_servers.other, { command: "other" });
  } finally {
    await rm(base, { recursive: true, force: true });
  }
});

test("setup protects an unmanaged Jacobian skill", async () => {
  const base = await mkdtemp(join(tmpdir(), "jacobian-carrier-setup-skill-conflict-"));
  try {
    const env = await setupEnvironment(base);
    const skillPath = join(env.HOME, ".agents", "skills", "jacobian-math", "SKILL.md");
    await mkdir(dirname(skillPath), { recursive: true });
    await writeFile(skillPath, "user-owned skill\n", "utf8");

    const result = runCarrier(["setup", "--codex", "--dry-run", "--json"], env);
    assert.equal(result.status, 1);
    assert.match(result.stderr, /refusing to replace an unmanaged Jacobian skill/);
    assert.equal(await readFile(skillPath, "utf8"), "user-owned skill\n");
  } finally {
    await rm(base, { recursive: true, force: true });
  }
});

test("setup requires explicit clients when non-interactive", async () => {
  const base = await mkdtemp(join(tmpdir(), "jacobian-carrier-setup-nontty-"));
  try {
    const env = await setupEnvironment(base);
    const result = runCarrier(["setup", "--dry-run"], env);
    assert.equal(result.status, 1);
    assert.match(result.stderr, /requires one or more client flags or --all/);
  } finally {
    await rm(base, { recursive: true, force: true });
  }
});

test("setup blocks before any write when uvx is unavailable", async () => {
  const base = await mkdtemp(join(tmpdir(), "jacobian-carrier-setup-prerequisite-"));
  try {
    const env = { ...process.env, HOME: join(base, "home"), PATH: "" };
    const result = runCarrier(["setup", "--codex", "--dry-run", "--json"], env);
    assert.equal(result.status, 1);
    assert.match(result.stderr, /requires uvx on PATH/);
    await assert.rejects(readFile(join(env.HOME, ".codex", "config.toml")), {
      code: "ENOENT",
    });
  } finally {
    await rm(base, { recursive: true, force: true });
  }
});

test("npm and Python packages publish the same release version", async () => {
  const pyproject = await readFile(join(npmRoot, "..", "pyproject.toml"), "utf8");
  const match = pyproject.match(/^version = "([^"]+)"$/m);
  assert.ok(match, "pyproject.toml must declare a project version");
  assert.equal(packageMetadata.version, npmVersionFromPythonVersion(match[1]));
});

test("package metadata carries setup dependencies and packs the setup adapter", async () => {
  assert.equal(packageMetadata.engines.node, ">=23.5.0 || ^22.13.0 || ^20.17.0");
  assert.deepEqual(Object.keys(packageMetadata.dependencies).sort(), [
    "@iarna/toml",
    "@inquirer/prompts",
    "jsonc-parser",
  ]);
  assert.ok(
    Object.values(packageMetadata.dependencies).every(
      (range) => typeof range === "string" && range.length > 0,
    ),
    "runtime dependency ranges must be non-empty manifest strings",
  );
  assert.equal(packageMetadata.bundleDependencies, undefined);
  assert.deepEqual(packageMetadata.files, ["bin", "lib", "skill", "README.md"]);
  assert.deepEqual(Object.keys(packageMetadata.bin), ["jacobian"]);

  const base = await mkdtemp(join(tmpdir(), "jacobian-carrier-pack-"));
  try {
    const pack = spawnSync(
      "npm",
      ["pack", "--json", "--pack-destination", base],
      { cwd: npmRoot, encoding: "utf8" },
    );
    assert.equal(pack.status, 0, pack.stderr);
    const metadata = JSON.parse(pack.stdout);
    assert.equal(metadata.length, 1);
    const tarball = join(base, metadata[0].filename);

    const list = spawnSync("npm", ["pack", "--dry-run", "--json"], {
      cwd: npmRoot,
      encoding: "utf8",
    });
    assert.equal(list.status, 0, list.stderr);
    const entries = JSON.parse(list.stdout)[0].files.map((file) => file.path);
    assert.ok(entries.includes("bin/jacobian.cjs"));
    assert.ok(entries.includes("lib/setup.cjs"));
    assert.ok(entries.includes("skill/jacobian-math/SKILL.md"));
    assert.ok(entries.includes("README.md"));
    assert.ok(!entries.some((path) => path.startsWith("install.sh")));
    assert.ok(tarball.length > 0);
  } finally {
    await rm(base, { recursive: true, force: true });
  }
});

test("package-lock.json agrees with package.json dependencies", async () => {
  const lock = JSON.parse(await readFile(join(npmRoot, "package-lock.json"), "utf8"));
  assert.equal(lock.version, packageMetadata.version);
  assert.equal(lock.packages[""].version, packageMetadata.version);
  assert.deepEqual(lock.packages[""].dependencies, packageMetadata.dependencies);
  assert.equal(lock.packages[""].bundleDependencies, undefined);
});
