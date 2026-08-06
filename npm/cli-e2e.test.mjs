import assert from "node:assert/strict";
import { chmod, mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { spawn, spawnSync } from "node:child_process";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import test from "node:test";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const npmRoot = dirname(fileURLToPath(import.meta.url));
const packageMetadata = require("./package.json");

function runNpx(tarball, args, cwd, env) {
  return new Promise((resolve, reject) => {
    const child = spawn(
      "npx",
      ["--yes", "--offline", "--package", tarball, "jacobian", ...args],
      {
        cwd,
        env: {
          ...process.env,
          HOME: join(cwd, "home"),
          npm_config_cache: join(cwd, "npm-cache"),
          npm_config_prefix: join(cwd, "global"),
          ...env,
        },
        stdio: ["ignore", "pipe", "pipe"],
      },
    );
    let stdout = "";
    let stderr = "";
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk) => {
      stdout += chunk;
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk;
    });
    const timer = setTimeout(() => child.kill("SIGTERM"), 120_000);
    child.once("error", (error) => {
      clearTimeout(timer);
      reject(error);
    });
    child.once("close", (status, signal) => {
      clearTimeout(timer);
      resolve({ status, signal, stdout, stderr });
    });
  });
}

async function packNpmPackage(destination) {
  const result = spawnSync(
    "npm",
    ["pack", "--json", "--pack-destination", destination],
    { cwd: npmRoot, encoding: "utf8" },
  );
  assert.equal(result.status, 0, result.stderr);
  const metadata = JSON.parse(result.stdout);
  assert.equal(metadata.length, 1);
  return join(destination, metadata[0].filename);
}

async function writeExecutable(path, source) {
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, source, "utf8");
  await chmod(path, 0o755);
}

test("npx jacobian setup writes and reapplies a client configuration", async () => {
  const base = await mkdtemp(join(tmpdir(), "jacobian-npx-setup-"));
  try {
    await mkdir(join(base, "home"), { recursive: true });
    const tarball = await packNpmPackage(base);
    const env = { XDG_CONFIG_HOME: join(base, "config") };

    const first = await runNpx(
      tarball,
      ["setup", "--client", "claude", "--yes", "--json"],
      base,
      env,
    );
    assert.equal(first.status, 0, first.stderr);
    const firstReport = JSON.parse(first.stdout);
    assert.equal(firstReport.operation, "setup");
    assert.equal(firstReport.results[0].status, "create");

    const configPath = join(base, "home", ".claude.json");
    const config = JSON.parse(await readFile(configPath, "utf8"));
    const server = config.mcpServers.jacobian;
    assert.equal(server.command, process.execPath);
    const npmLauncher = server.args[0];
    assert.ok(
      npmLauncher.endsWith("npx-cli.js") || npmLauncher.endsWith("npm-cli.js"),
    );
    if (npmLauncher.endsWith("npm-cli.js")) {
      assert.equal(server.args[1], "exec");
    }
    assert.ok(server.args.includes(`--package=jacobian@${packageMetadata.version}`));
    assert.ok(server.args.includes("--"));
    assert.ok(server.args.includes("jacobian"));
    assert.equal(server.args.at(-1), "mcp");
    assert.equal(server.args.some((arg) => arg.includes("npm-cache")), false);

    const second = await runNpx(
      tarball,
      ["setup", "--client", "claude", "--yes", "--json"],
      base,
      env,
    );
    assert.equal(second.status, 0, second.stderr);
    const secondReport = JSON.parse(second.stdout);
    assert.equal(secondReport.results[0].status, "already_current");
  } finally {
    await rm(base, { recursive: true, force: true });
  }
});

test("npx jacobian setup manages the Codex visibility skill safely", async () => {
  const base = await mkdtemp(join(tmpdir(), "jacobian-npx-codex-skill-"));
  try {
    await mkdir(join(base, "home", ".codex"), { recursive: true });
    const tarball = await packNpmPackage(base);
    const env = { XDG_CONFIG_HOME: join(base, "config") };
    const skillPath = join(
      base,
      "home",
      ".codex",
      "skills",
      "jacobian-math",
      "SKILL.md",
    );

    const setup = await runNpx(
      tarball,
      ["setup", "--client", "codex", "--yes", "--json"],
      base,
      env,
    );
    assert.equal(setup.status, 0, setup.stderr);
    const setupReport = JSON.parse(setup.stdout);
    assert.equal(setupReport.results[0].visibilitySkill.status, "create");
    const managedSkill = await readFile(skillPath, "utf8");
    assert.match(managedSkill, /name: jacobian-math/);
    assert.match(managedSkill, /Trigger even when the user does not name Jacobian/);

    const remove = await runNpx(
      tarball,
      ["remove", "--client", "codex", "--yes", "--json"],
      base,
      env,
    );
    assert.equal(remove.status, 0, remove.stderr);
    assert.equal(existsSync(skillPath), false);

    const reinstall = await runNpx(
      tarball,
      ["setup", "--client", "codex", "--yes", "--json"],
      base,
      env,
    );
    assert.equal(reinstall.status, 0, reinstall.stderr);
    const modifiedSkill = `${await readFile(skillPath, "utf8")}\nUser note.\n`;
    await writeFile(skillPath, modifiedSkill, "utf8");

    const refuseOverwrite = await runNpx(
      tarball,
      ["setup", "--client", "codex", "--yes", "--json"],
      base,
      env,
    );
    assert.equal(refuseOverwrite.status, 1);
    assert.match(refuseOverwrite.stderr, /differs from the managed Jacobian skill/);
    assert.equal(await readFile(skillPath, "utf8"), modifiedSkill);

    const preserve = await runNpx(
      tarball,
      ["remove", "--client", "codex", "--yes", "--json"],
      base,
      env,
    );
    assert.equal(preserve.status, 0, preserve.stderr);
    const preserveReport = JSON.parse(preserve.stdout);
    assert.equal(
      preserveReport.results[0].visibilitySkill.status,
      "preserve_modified",
    );
    assert.equal(await readFile(skillPath, "utf8"), modifiedSkill);
  } finally {
    await rm(base, { recursive: true, force: true });
  }
});

test(
  "npx jacobian upgrade refreshes the launcher-managed Python package",
  { skip: process.platform === "win32" },
  async () => {
    const base = await mkdtemp(join(tmpdir(), "jacobian-npx-upgrade-"));
    try {
      const tarball = await packNpmPackage(base);
      const xdgDataHome = join(base, "xdg-data");
      const python = join(
        xdgDataHome,
        "jacobian",
        "jacobian-venv",
        "bin",
        "python",
      );
      const uv = join(base, "bin", "uv");
      const uvLog = join(base, "uv.log");

      await writeExecutable(
        python,
        `#!/usr/bin/env node
if (process.argv[2] === "-c") {
  console.log(${JSON.stringify(packageMetadata.version)});
}
`,
      );
      await writeExecutable(
        uv,
        `#!/usr/bin/env node
const fs = require("node:fs");
if (process.argv[2] === "--version") {
  console.log("uv 0.8.0");
} else {
  fs.appendFileSync(process.env.FAKE_UV_LOG, JSON.stringify(process.argv.slice(2)) + "\\n");
}
`,
      );

      const result = await runNpx(tarball, ["upgrade"], base, {
        FAKE_UV_LOG: uvLog,
        JACOBIAN_PACKAGE: `jacobian==${packageMetadata.version}`,
        PATH: `${dirname(uv)}${process.platform === "win32" ? ";" : ":"}${process.env.PATH ?? ""}`,
        XDG_DATA_HOME: xdgDataHome,
      });
      assert.equal(result.status, 0, result.stderr);
      assert.match(result.stdout, /Python package upgraded/);

      const calls = (await readFile(uvLog, "utf8"))
        .trim()
        .split("\n")
        .map((line) => JSON.parse(line));
      assert.deepEqual(calls, [
        [
          "pip",
          "install",
          "--upgrade",
          "--python",
          python,
          `jacobian==${packageMetadata.version}`,
        ],
      ]);
      assert.equal(existsSync(join(base, "global")), false);
    } finally {
      await rm(base, { recursive: true, force: true });
    }
  },
);
