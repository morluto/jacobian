import assert from "node:assert/strict";
import { chmod, mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { spawnSync } from "node:child_process";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import test from "node:test";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const npmRoot = dirname(fileURLToPath(import.meta.url));
const packageMetadata = require("./package.json");
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
  assert.match(result.stderr, /uvx --from jacobian==<version> jacobian-mcp/);
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
    assert.match(codex, /startup_timeout_sec = 30/);
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

test("setup protects an unmanaged Jacobian registration", async () => {
  const base = await mkdtemp(join(tmpdir(), "jacobian-carrier-setup-conflict-"));
  try {
    const env = await setupEnvironment(base);
    const configPath = join(env.HOME, ".claude.json");
    const original = JSON.stringify({
      mcpServers: { jacobian: { command: "python", args: ["-m", "jacobian.mcp"] } },
    });
    await mkdir(dirname(configPath), { recursive: true });
    await writeFile(configPath, original, "utf8");

    const result = runCarrier(["setup", "--claude", "--dry-run", "--json"], env);
    assert.equal(result.status, 1);
    assert.match(result.stderr, /refusing to replace an unmanaged Jacobian entry/);
    assert.equal(await readFile(configPath, "utf8"), original);
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
  assert.deepEqual(packageMetadata.dependencies, {
    "@iarna/toml": "^2.2.5",
    "@inquirer/prompts": "^8.6.0",
    "jsonc-parser": "^3.3.1",
  });
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
