import assert from "node:assert/strict";
import { chmod, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
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

test("npm and Python packages publish the same release version", async () => {
  const pyproject = await readFile(join(npmRoot, "..", "pyproject.toml"), "utf8");
  const match = pyproject.match(/^version = "([^"]+)"$/m);
  assert.ok(match, "pyproject.toml must declare a project version");
  assert.equal(packageMetadata.version, npmVersionFromPythonVersion(match[1]));
});

test("package metadata carries no dependencies and packs only the carrier", async () => {
  assert.equal(packageMetadata.dependencies, undefined);
  assert.equal(packageMetadata.bundleDependencies, undefined);
  assert.deepEqual(packageMetadata.files, ["bin", "README.md"]);
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
    assert.ok(entries.includes("README.md"));
    assert.ok(!entries.some((path) => path.startsWith("install.sh")));
    assert.ok(
      !entries.some((path) =>
        ["setup.cjs", "doctor.cjs", "launcher.cjs", "toml-config.mjs"].some(
          (removed) => path === `bin/${removed}`,
        ),
      ),
    );
    assert.ok(tarball.length > 0);
  } finally {
    await rm(base, { recursive: true, force: true });
  }
});

test("package-lock.json agrees with package.json and declares no dependencies", async () => {
  const lock = JSON.parse(await readFile(join(npmRoot, "package-lock.json"), "utf8"));
  assert.equal(lock.version, packageMetadata.version);
  assert.equal(lock.packages[""].version, packageMetadata.version);
  assert.equal(lock.packages[""].dependencies, undefined);
  assert.equal(lock.packages[""].bundleDependencies, undefined);
});
