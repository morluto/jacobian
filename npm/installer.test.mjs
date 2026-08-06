import assert from "node:assert/strict";
import { chmod, lstat, mkdir, mkdtemp, readFile, readlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { spawnSync } from "node:child_process";
import test from "node:test";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const repositoryRoot = dirname(here);
const installer = join(here, "install.sh");

async function writeExecutable(path, source) {
  await writeFile(path, source, "utf8");
  await chmod(path, 0o755);
}

async function fakeEnvironment(base, version = "0.8.0") {
  const fakeBin = join(base, "fake-bin");
  const home = join(base, "home");
  const dataDir = join(base, "data");
  const binDir = join(base, "bin");
  const log = join(base, "commands.log");
  await mkdir(fakeBin, { recursive: true });
  await mkdir(home, { recursive: true });

  const npm = join(fakeBin, "npm");
  await writeExecutable(
    npm,
    `#!/bin/sh
set -eu
printf 'npm:%s\\n' "$*" >> "$JACOBIAN_FAKE_LOG"
case "\${1:-}" in
  --version)
    printf '10.0.0\\n'
    ;;
  view)
    printf '"%s"\\n' "$JACOBIAN_FAKE_VERSION"
    ;;
  install)
    shift
    prefix=
    while [ "$#" -gt 0 ]; do
      if [ "$1" = --prefix ]; then
        prefix="$2"
        shift 2
      else
        shift
      fi
    done
    mkdir -p "$prefix/bin"
    cat > "$prefix/bin/jacobian" <<'SCRIPT'
#!/bin/sh
set -eu
printf 'jacobian:%s\\n' "$*" >> "$JACOBIAN_FAKE_LOG"
case "\${1:-}" in
  --version) printf 'jacobian %s\\n' "$JACOBIAN_FAKE_VERSION" ;;
  setup) [ "\${JACOBIAN_FAKE_SETUP_FAILURE:-0}" -eq 0 ] ;;
  doctor)
    printf 'doctor: ok\\n'
    [ "\${JACOBIAN_FAKE_DOCTOR_FAILURE:-0}" -eq 0 ]
    ;;
  *) exit 2 ;;
esac
SCRIPT
    chmod +x "$prefix/bin/jacobian"
    ;;
  *) exit 2 ;;
esac
`,
  );

  const uv = join(fakeBin, "uv");
  await writeExecutable(uv, "#!/bin/sh\nprintf 'uv 0.11.28\\n'\n");

  return {
    env: {
      ...process.env,
      HOME: home,
      JACOBIAN_BIN_DIR: binDir,
      JACOBIAN_DATA_DIR: dataDir,
      JACOBIAN_FAKE_LOG: log,
      JACOBIAN_FAKE_VERSION: version,
      JACOBIAN_NODE_BIN: process.execPath,
      JACOBIAN_NPM_BIN: npm,
      JACOBIAN_UV_BIN: uv,
      NO_COLOR: "1",
    },
    binDir,
    dataDir,
    fakeBin,
    home,
    log,
  };
}

function runInstaller(args, env) {
  return spawnSync("sh", [installer, ...args], {
    cwd: repositoryRoot,
    encoding: "utf8",
    env,
  });
}

test("installer dry-run is mutation-free and describes the runtime cost", async () => {
  const base = await mkdtemp(join(tmpdir(), "jacobian-installer-dry-"));
  const home = join(base, "home");
  const dataDir = join(base, "data");
  const binDir = join(base, "bin");
  await mkdir(home);

  const result = runInstaller(
    ["--client", "codex", "--yes", "--dry-run", "--plain"],
    { ...process.env, HOME: home, JACOBIAN_DATA_DIR: dataDir, JACOBIAN_BIN_DIR: binDir },
  );

  assert.equal(result.status, 0, result.stderr);
  assert.match(
    result.stdout,
    /math runtime:\s+installed and verified.*~160 MB.*~110 MB.*Python 3\.12/,
  );
  assert.match(result.stdout, /changes:\s+none \(dry-run\)/);
  await assert.rejects(lstat(dataDir), { code: "ENOENT" });
  await assert.rejects(lstat(binDir), { code: "ENOENT" });
});

test("installer pins npm resolution, activates the launcher, configures Codex, and verifies", async () => {
  const base = await mkdtemp(join(tmpdir(), "jacobian-installer-full-"));
  const fixture = await fakeEnvironment(base);

  const result = runInstaller(["--client", "codex", "--yes", "--plain"], fixture.env);

  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /Jacobian 0\.8\.0 is ready/);
  assert.equal(
    await readlink(join(fixture.binDir, "jacobian")),
    join(fixture.dataDir, "npm-releases", "0.8.0", "bin", "jacobian"),
  );
  const log = await readFile(fixture.log, "utf8");
  assert.match(log, /npm:view jacobian@latest version --json/);
  assert.match(
    log,
    /npm:install --global --prefix .* --ignore-scripts --omit=dev --no-audit --no-fund jacobian@0\.8\.0/,
  );
  assert.match(log, /jacobian:setup --client codex --yes/);
  assert.match(log, /jacobian:doctor/);
});

test("installer can defer the heavy runtime and reuse a validated launcher release", async () => {
  const base = await mkdtemp(join(tmpdir(), "jacobian-installer-defer-"));
  const fixture = await fakeEnvironment(base);
  const args = ["--client", "codex", "--yes", "--defer-runtime", "--plain"];

  const first = runInstaller(args, fixture.env);
  const second = runInstaller(args, fixture.env);

  assert.equal(first.status, 0, first.stderr);
  assert.equal(second.status, 0, second.stderr);
  assert.match(second.stdout, /reusing .*npm-releases\/0\.8\.0/);
  const log = await readFile(fixture.log, "utf8");
  assert.equal((log.match(/npm:install /g) || []).length, 1);
  assert.equal((log.match(/jacobian:doctor/g) || []).length, 0);
  assert.match(first.stdout, /first use installs ~160 MB.*~110 MB.*Python 3\.12/);
});

test("installer rolls stable launcher activation back when setup fails", async () => {
  const base = await mkdtemp(join(tmpdir(), "jacobian-installer-rollback-"));
  const oldFixture = await fakeEnvironment(base, "0.7.0");
  const args = ["--client", "codex", "--yes", "--defer-runtime", "--plain"];
  const installed = runInstaller(args, oldFixture.env);
  assert.equal(installed.status, 0, installed.stderr);
  const command = join(oldFixture.binDir, "jacobian");
  const oldTarget = await readlink(command);

  const failed = runInstaller(args, {
    ...oldFixture.env,
    JACOBIAN_FAKE_VERSION: "0.8.0",
    JACOBIAN_FAKE_SETUP_FAILURE: "1",
  });

  assert.notEqual(failed.status, 0);
  assert.match(failed.stderr, /activation was rolled back/);
  assert.equal(await readlink(command), oldTarget);
});

test("installer keeps the activated launcher when post-setup doctor fails", async () => {
  const base = await mkdtemp(join(tmpdir(), "jacobian-installer-doctor-failure-"));
  const fixture = await fakeEnvironment(base);
  const command = join(fixture.binDir, "jacobian");

  const failed = runInstaller(["--client", "codex", "--yes", "--plain"], {
    ...fixture.env,
    JACOBIAN_FAKE_DOCTOR_FAILURE: "1",
  });

  assert.notEqual(failed.status, 0);
  assert.doesNotMatch(failed.stderr, /activation was rolled back/);
  assert.equal(
    await readlink(command),
    join(fixture.dataDir, "npm-releases", "0.8.0", "bin", "jacobian"),
  );
});

test("installer never executes an uv installer with the wrong checksum", async () => {
  const base = await mkdtemp(join(tmpdir(), "jacobian-installer-uv-checksum-"));
  const fixture = await fakeEnvironment(base);
  const marker = join(base, "uv-installer-executed");
  const curlBin = join(base, "curl-bin");
  await mkdir(curlBin);
  await writeExecutable(
    join(curlBin, "curl"),
    `#!/bin/sh
set -eu
output=
while [ "$#" -gt 0 ]; do
  if [ "$1" = -o ]; then output="$2"; shift 2; else shift; fi
done
cat > "$output" <<'SCRIPT'
#!/bin/sh
printf executed > "$JACOBIAN_FAKE_UV_MARKER"
SCRIPT
`,
  );

  const failed = runInstaller(["--client", "codex", "--yes", "--install-uv", "--plain"], {
    ...fixture.env,
    JACOBIAN_FAKE_UV_MARKER: marker,
    JACOBIAN_UV_BIN: "",
    PATH: `${curlBin}:/usr/bin:/bin`,
  });

  assert.notEqual(failed.status, 0);
  assert.match(failed.stderr, /FAILED|checksum/i);
  await assert.rejects(lstat(marker), { code: "ENOENT" });
});

test("installer refuses noninteractive implicit consent and unmanaged commands", async () => {
  const base = await mkdtemp(join(tmpdir(), "jacobian-installer-safety-"));
  const fixture = await fakeEnvironment(base);
  const implicit = runInstaller(["--plain"], fixture.env);
  assert.notEqual(implicit.status, 0);
  assert.match(implicit.stderr, /requires --client or --all/);

  await mkdir(fixture.binDir, { recursive: true });
  const command = join(fixture.binDir, "jacobian");
  await writeFile(command, "unmanaged\n", "utf8");
  const unmanaged = runInstaller(
    ["--client", "codex", "--yes", "--defer-runtime", "--plain"],
    fixture.env,
  );
  assert.notEqual(unmanaged.status, 0);
  assert.match(unmanaged.stderr, /not a managed symlink/);
  assert.equal(await readFile(command, "utf8"), "unmanaged\n");
});

test("installer uv pin matches the repository and the script is valid POSIX shell", async () => {
  const source = await readFile(installer, "utf8");
  const pinnedUv = (await readFile(join(repositoryRoot, ".uv-version"), "utf8")).trim();
  assert.match(source, new RegExp(`uv_version="${pinnedUv.replaceAll(".", "\\.")}"`));
  assert.match(source, /uv_installer_sha256="[0-9a-f]{64}"/);

  const syntax = spawnSync("sh", ["-n", installer], { encoding: "utf8" });
  assert.equal(syntax.status, 0, syntax.stderr);
});
