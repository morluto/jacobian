"use strict";

const {
  chmodSync,
  existsSync,
  lstatSync,
  mkdirSync,
  readFileSync,
  renameSync,
  writeFileSync,
  rmSync,
  statSync,
} = require("node:fs");
const { randomUUID } = require("node:crypto");
const { homedir } = require("node:os");
const { basename, dirname, join, resolve } = require("node:path");
const readline = require("node:readline/promises");
const { emitKeypressEvents } = require("node:readline");
const { stdin, stdout, stderr } = require("node:process");
const { isDeepStrictEqual } = require("node:util");

/**
 * Jacobian MCP setup wizard.
 *
 * Detects installed MCP clients (Claude Code, Cursor, Codex, Gemini CLI,
 * OpenCode), shows the user which were found, and writes MCP server
 * configurations that launch `npx jacobian mcp` as a stdio server.
 *
 * The wizard never auto-selects detected clients.  Detection is context,
 * not consent.
 */

const SERVER_NAME = "jacobian";
let tomlConfigModule;

async function loadTomlConfig() {
  tomlConfigModule ??= import("./toml-config.mjs");
  return tomlConfigModule;
}

/**
 * @typedef {"claude" | "cursor" | "opencode" | "codex" | "gemini"} ClientId
 */

/**
 * @typedef {object} ClientDef
 * @property {ClientId} id
 * @property {string} displayName
 * @property {"json" | "toml"} format
 * @property {string} configPath
 * @property {string} jsonSection  Section key for JSON configs.
 * @property {"command_args" | "opencode"} jsonShape
 */

/**
 * @param {string} home
 * @returns {ClientDef[]}
 */
function clientDefinitions(home) {
  return [
    {
      id: "claude",
      displayName: "Claude Code",
      format: "json",
      configPath: join(home, ".claude.json"),
      jsonSection: "mcpServers",
      jsonShape: "command_args",
    },
    {
      id: "cursor",
      displayName: "Cursor",
      format: "json",
      configPath: join(home, ".cursor", "mcp.json"),
      jsonSection: "mcpServers",
      jsonShape: "command_args",
    },
    {
      id: "opencode",
      displayName: "OpenCode",
      format: "json",
      configPath: join(home, ".config", "opencode", "opencode.json"),
      jsonSection: "mcp",
      jsonShape: "opencode",
    },
    {
      id: "codex",
      displayName: "Codex",
      format: "toml",
      configPath: join(home, ".codex", "config.toml"),
      jsonSection: "",
      jsonShape: "command_args",
    },
    {
      id: "gemini",
      displayName: "Gemini CLI",
      format: "json",
      configPath: join(home, ".gemini", "settings.json"),
      jsonSection: "mcpServers",
      jsonShape: "command_args",
    },
  ];
}

/**
 * @param {string} home
 * @param {ClientId} id
 * @returns {boolean}
 */
function isClientDetected(home, id) {
  switch (id) {
    case "claude":
      return existsSync(join(home, ".claude")) || existsSync(join(home, ".claude.json"));
    case "cursor":
      return existsSync(join(home, ".cursor"));
    case "opencode":
      return existsSync(join(home, ".config", "opencode"));
    case "codex":
      return existsSync(join(home, ".codex"));
    case "gemini":
      return existsSync(join(home, ".gemini"));
    default:
      return false;
  }
}

/**
 * Build the MCP launcher command + args that will be written to client configs.
 *
 * When running under npx, pins the exact version.  When running from a
 * persistent install, uses the installed binary directly.
 *
 * @returns {{ command: string, args: string[], version: string, package: string | null }}
 */
function buildLauncher() {
  const version = require("../package.json").version;
  const managedRuntime = require("./launcher.cjs");
  const runtime = managedRuntime.detectRuntime();
  const runtimePlan = {
    status: runtime === null ? "blocked" : "deferred_until_first_use",
    prerequisite: runtime?.kind ?? null,
    environment: managedRuntime.venvRoot(),
    pythonPackage: managedRuntime.PACKAGE_SPEC,
    approximateInstallSizeMb: 160,
    approximateManagedPythonSizeMb: runtime?.kind === "uv" ? 110 : 0,
    network: "The Python package index may be contacted on first use.",
    recovery: runtime === null
      ? "Install Python 3.12/3.13 or uv, then retry setup."
      : null,
  };

  if (process.env.npm_lifecycle_event === "npx") {
    const node = process.env.npm_node_execpath;
    const npm = process.env.npm_execpath;
    if (node && npm) {
      const pkg = `jacobian@${version}`;
      const npmArgs = basename(npm).toLowerCase() === "npx-cli.js"
        ? ["--yes", `--package=${pkg}`, "--", "jacobian", "mcp"]
        : ["exec", "--yes", `--package=${pkg}`, "--", "jacobian", "mcp"];
      return {
        command: node,
        args: [npm, ...npmArgs],
        version,
        package: pkg,
        runtimePlan,
      };
    }
  }

  // Persistent install: use the bin directly with the mcp subcommand.
  const binPath = process.argv[1] || "jacobian";
  return {
    command: process.execPath,
    args: [binPath, "mcp"],
    version,
    package: null,
    runtimePlan,
  };
}

/**
 * Build a launcher bound to one source checkout and state directory.
 *
 * The configured client never syncs dependencies on startup.  The source
 * bootstrap owns the locked sync, while every agent launch reuses that exact
 * checkout and environment.
 *
 * @param {string} source
 * @param {string} stateDir
 * @param {string} [uvBin]
 * @param {string} [profile]
 * @param {string} [providerPath]
 * @param {string} [projectEnvironment]
 * @param {string} [elanHome]
 * @param {string} [leanRuntime]
 * @returns {{ command: string, args: string[], version: string, package: null, source: string, stateDir: string, profile: string, env: object }}
 */
function buildSourceLauncher(
  source,
  stateDir,
  uvBin = "uv",
  profile = "core",
  providerPath = process.env.PATH || "",
  projectEnvironment = process.env.UV_PROJECT_ENVIRONMENT || "",
  elanHome = process.env.ELAN_HOME || "",
  leanRuntime = process.env.JACOBIAN_LEAN_RUNTIME || "",
) {
  const profiles = new Set(["core", "full-python", "lean", "external-proof"]);
  if (!profiles.has(profile)) {
    throw new Error(`Unknown Jacobian source profile: ${profile}.`);
  }
  const sourcePath = resolve(source);
  const statePath = resolve(stateDir);
  const pyprojectPath = join(sourcePath, "pyproject.toml");
  const projectLines = existsSync(pyprojectPath)
    ? readFileSync(pyprojectPath, "utf8").split(/\r?\n/)
    : [];
  let inProjectSection = false;
  let hasJacobianName = false;
  for (const line of projectLines) {
    const trimmed = line.trim();
    if (trimmed === "[project]") {
      inProjectSection = true;
      continue;
    }
    if (inProjectSection && trimmed.startsWith("[")) break;
    if (
      inProjectSection &&
      /^name\s*=\s*["']jacobian["']\s*(?:#.*)?$/.test(trimmed)
    ) {
      hasJacobianName = true;
      break;
    }
  }
  if (
    !hasJacobianName ||
    !existsSync(join(sourcePath, "uv.lock")) ||
    !existsSync(join(sourcePath, "src", "jacobian", "__init__.py"))
  ) {
    throw new Error(
      `${sourcePath} is not a Jacobian source checkout: project.name must be ` +
        "jacobian and src/jacobian/__init__.py plus uv.lock are required.",
    );
  }
  const environment = {};
  if (providerPath) environment.PATH = providerPath;
  if (projectEnvironment) {
    environment.UV_PROJECT_ENVIRONMENT = projectEnvironment;
  }
  if (elanHome) environment.ELAN_HOME = elanHome;
  if (leanRuntime) {
    environment.JACOBIAN_LEAN_RUNTIME = leanRuntime;
  }
  return {
    command: uvBin,
    args: [
      "run",
      "--project",
      sourcePath,
      "--locked",
      "--no-sync",
      "jacobian-mcp",
      "--state-dir",
      statePath,
    ],
    version: "source",
    package: null,
    source: sourcePath,
    stateDir: statePath,
    profile,
    env: environment,
    runtimePlan: {
      status: "source_managed",
      prerequisite: "uv",
      environment: projectEnvironment || join(sourcePath, ".venv"),
      pythonPackage: "source checkout",
      approximateInstallSizeMb: null,
      approximateManagedPythonSizeMb: 0,
      network: "No dependency sync occurs when the configured client starts.",
      recovery: null,
    },
  };
}

/**
 * Build the MCP server entry value for a JSON config.
 *
 * @param {ClientDef} def
 * @param {{ command: string, args: string[] }} launcher
 * @returns {object}
 */
function jsonEntry(def, launcher) {
  if (def.jsonShape === "opencode") {
    return {
      type: "local",
      command: [launcher.command, ...launcher.args].join(" "),
      cwd: ".",
      enabled: true,
      ...(Object.keys(launcher.env || {}).length > 0
        ? { environment: launcher.env }
        : {}),
    };
  }
  return {
    command: launcher.command,
    args: launcher.args,
    ...(Object.keys(launcher.env || {}).length > 0 ? { env: launcher.env } : {}),
  };
}

function publicLauncher(launcher) {
  return {
    command: launcher.command,
    args: launcher.args,
    version: launcher.version,
    package: launcher.package,
    ...(launcher.source ? { source: launcher.source, stateDir: launcher.stateDir } : {}),
    runtime: launcher.runtimePlan,
  };
}

function publicPlan(plan) {
  return {
    operation: plan.operation,
    launcher: publicLauncher(plan.launcher),
    effects: plan.edits.map((edit) => ({
      client: edit.client.id,
      displayName: edit.client.displayName,
      kind: edit.kind ?? "mcp_configuration",
      action: edit.action,
      detected: edit.detected,
      path: edit.path,
    })),
  };
}

function plainValue(value) {
  if (Array.isArray(value)) return value.map(plainValue);
  if (typeof value !== "object" || value === null) return value;
  return Object.fromEntries(
    Object.entries(value).map(([key, item]) => [key, plainValue(item)]),
  );
}

function configuredLauncher(def, configured) {
  if (def.jsonShape !== "opencode") {
    return {
      command: configured.command,
      args: configured.args,
      env: configured.env || {},
    };
  }
  if (typeof configured.command !== "string") return null;
  const source = configured.command.match(
    /^(.*?) run --project (.*?) --locked --no-sync jacobian-mcp --state-dir (.*)$/,
  );
  if (!source) return null;
  return {
    command: source[1],
    args: [
      "run",
      "--project",
      source[2],
      "--locked",
      "--no-sync",
      "jacobian-mcp",
      "--state-dir",
      source[3],
    ],
    env: configured.environment || {},
  };
}

function sourceLauncher(configured) {
  if (
    configured === null ||
    typeof configured.command !== "string" ||
    !Array.isArray(configured.args) ||
    configured.args.length !== 8 ||
    configured.args[0] !== "run" ||
    configured.args[1] !== "--project" ||
    configured.args[3] !== "--locked" ||
    configured.args[4] !== "--no-sync" ||
    configured.args[5] !== "jacobian-mcp" ||
    configured.args[6] !== "--state-dir"
  ) {
    return null;
  }
  try {
    const launcher = buildSourceLauncher(
      configured.args[2],
      configured.args[7],
      configured.command,
      "core",
      configured.env?.PATH || "",
      configured.env?.UV_PROJECT_ENVIRONMENT || "",
      configured.env?.ELAN_HOME || "",
      configured.env?.JACOBIAN_LEAN_RUNTIME || "",
    );
    return isDeepStrictEqual(plainValue(configured.env || {}), launcher.env)
      ? launcher
      : null;
  } catch {
    return null;
  }
}

function managedLauncher(configured, version) {
  if (
    configured === null ||
    typeof configured.command !== "string" ||
    !Array.isArray(configured.args)
  ) {
    return null;
  }
  const nodeExecutable = ["node", "node.exe"].includes(
    basename(configured.command).toLowerCase(),
  );
  const direct =
    nodeExecutable &&
    configured.args.length === 2 &&
    configured.args[1] === "mcp" &&
    basename(configured.args[0]).toLowerCase() === "jacobian.cjs" &&
    basename(dirname(configured.args[0])).toLowerCase() === "bin";
  const npmPackage = `--package=jacobian@${version}`;
  const npmExec =
    nodeExecutable &&
    configured.args.includes(npmPackage) &&
    configured.args.at(-3) === "--" &&
    configured.args.at(-2) === "jacobian" &&
    configured.args.at(-1) === "mcp" &&
    ["npm-cli.js", "npx-cli.js"].includes(
      basename(configured.args[0]).toLowerCase(),
    );
  return direct || npmExec ? configured : null;
}

async function inspectClientConfiguration(def, launcher) {
  const original = readOptional(def.configPath);
  let configured;
  if (def.format === "toml") {
    configured = (await loadTomlConfig()).readTomlLauncher(
      def.configPath,
      original,
    );
  } else if (original === null) {
    configured = null;
  } else {
    let root;
    try {
      root = JSON.parse(original);
    } catch (error) {
      throw new Error(`Invalid JSON in ${def.configPath}: ${error.message}.`);
    }
    configured = root?.[def.jsonSection]?.[SERVER_NAME] ?? null;
  }
  if (configured === null) {
    return { client: def.id, path: def.configPath, status: "not_configured" };
  }
  const expected = def.format === "toml"
    ? {
        command: launcher.command,
        args: launcher.args,
        startup_timeout_sec: 30,
        ...(Object.keys(launcher.env || {}).length > 0 ? { env: launcher.env } : {}),
      }
    : jsonEntry(def, launcher);
  if (isDeepStrictEqual(plainValue(configured), expected)) {
    return {
      client: def.id,
      path: def.configPath,
      status: "configured",
      launcher,
    };
  }
  const configuredProcess = configuredLauncher(def, configured);
  const source = sourceLauncher(configuredProcess);
  const managed = source === null
    ? managedLauncher(configuredProcess, launcher.version)
    : null;
  return {
    client: def.id,
    path: def.configPath,
    status: source !== null
      ? "configured_source"
      : managed !== null
        ? "configured_managed"
        : "stale",
    ...(source !== null
      ? { launcher: source }
      : managed !== null
        ? { launcher: managed }
        : {}),
  };
}

/**
 * Read a file if it exists, returning null otherwise.
 *
 * @param {string} path
 * @returns {string | null}
 */
function readOptional(path) {
  try {
    return readFileSync(path, "utf8");
  } catch (error) {
    if (error.code === "ENOENT") return null;
    throw error;
  }
}

/** Refuse to replace a symlink instead of the config file it names. */
function rejectSymlink(path) {
  try {
    if (lstatSync(path).isSymbolicLink()) {
      throw new Error(
        `${path} is a symbolic link. Jacobian will not replace linked client ` +
          "configs; update the link target directly or use a regular config file",
      );
    }
  } catch (error) {
    if (error.code === "ENOENT") return;
    throw error;
  }
}

/**
 * Write a file only if its content would change.
 *
 * @param {string} path
 * @param {string} content
 * @returns {boolean} Whether the target was replaced.
 */
function writeIfChanged(path, content) {
  const existing = readOptional(path);
  if (existing === content) return false;
  rejectSymlink(path);
  mkdirSync(dirname(path), { recursive: true });
  const mode = existsSync(path) ? statSync(path).mode & 0o777 : 0o600;
  const temporary = join(
    dirname(path),
    `.${basename(path)}.jacobian-${randomUUID()}.tmp`,
  );
  try {
    writeFileSync(temporary, content, {
      encoding: "utf8",
      flag: "wx",
      mode,
    });
    chmodSync(temporary, mode);
    renameSync(temporary, path);
    return true;
  } finally {
    if (existsSync(temporary)) {
      rmSync(temporary, { force: true });
    }
  }
}

/**
 * Resolve a JSON config edit for setup or removal.
 *
 * @param {"setup" | "remove"} operation
 * @param {ClientDef} def
 * @param {{ command: string, args: string[] }} launcher
 * @returns {{ action: string, original: string | null, updated: string | null }}
 */
function resolveJsonEdit(operation, def, launcher) {
  const original = readOptional(def.configPath);
  const source = original ?? "{}\n";
  let root;
  try {
    root = JSON.parse(source);
  } catch (error) {
    throw new Error(
      `Invalid JSON in ${def.configPath}: ${error.message}. No changes were written. ` +
        "Repair this file, then retry.",
    );
  }
  if (typeof root !== "object" || root === null || Array.isArray(root)) {
    throw new Error(
      `Top-level value in ${def.configPath} must be an object. No changes were ` +
        "written. Repair this file, then retry.",
    );
  }
  const section = root[def.jsonSection] ?? {};
  if (typeof section !== "object" || section === null || Array.isArray(section)) {
    throw new Error(
      `${def.jsonSection} in ${def.configPath} must be an object. No changes were ` +
        "written. Repair this file, then retry.",
    );
  }

  if (operation === "setup") {
    const expected = jsonEntry(def, launcher);
    if (JSON.stringify(section[SERVER_NAME]) === JSON.stringify(expected)) {
      return { action: "already_current", original, updated: null };
    }
    section[SERVER_NAME] = expected;
    root[def.jsonSection] = section;
    const updated = JSON.stringify(root, null, 2) + "\n";
    return {
      action: original === null ? "create" : "update",
      original,
      updated,
    };
  }

  // Remove
  if (!(SERVER_NAME in section)) {
    return { action: "not_configured", original, updated: null };
  }
  delete section[SERVER_NAME];
  if (Object.keys(section).length === 0) {
    delete root[def.jsonSection];
  }
  const updated = JSON.stringify(root, null, 2) + "\n";
  return { action: "remove", original, updated };
}

/**
 * Resolve one client edit.
 *
 * @param {"setup" | "remove"} operation
 * @param {ClientDef} def
 * @param {{ command: string, args: string[] }} launcher
 * @returns {Promise<{ client: ClientDef, action: string, detected: boolean, path: string, original: string | null, updated: string | null }>}
 */
async function resolveClientEdit(operation, def, launcher) {
  const original = readOptional(def.configPath);
  const result =
    def.format === "json"
      ? resolveJsonEdit(operation, def, launcher)
      : (await loadTomlConfig()).resolveTomlEdit(
          operation,
          def.configPath,
          original,
          launcher,
        );
  return {
    client: def,
    action: result.action,
    detected: isClientDetected(homedir(), def.id),
    path: def.configPath,
    original: result.original,
    updated: result.updated,
  };
}

/**
 * Apply one client edit to disk.
 *
 * @param {{ path: string, original: string | null, updated: string | null, action: string }} edit
 * @returns {boolean} Whether this edit changed disk.
 */
function applyEdit(edit) {
  if (edit.updated === null && !edit.deleteTarget) return false;
  if (readOptional(edit.path) !== edit.original) {
    throw new Error(`${edit.path} changed after setup preflight; no write was made`);
  }
  if (edit.deleteTarget) {
    rmSync(edit.path, { force: true });
    return true;
  }
  return writeIfChanged(edit.path, edit.updated);
}

/** Restore a config file to the content observed during preflight. */
function restoreEdit(edit) {
  const expectedCurrent = edit.deleteTarget ? null : edit.updated;
  if (readOptional(edit.path) !== expectedCurrent) {
    throw new Error(
      "the config changed after Jacobian wrote it; the concurrent value was left untouched",
    );
  }
  if (edit.original === null) {
    rmSync(edit.path, { force: true });
    return;
  }
  writeIfChanged(edit.path, edit.original);
}

/**
 * Apply a setup plan transactionally.  If any client write fails, every
 * earlier write in the plan is restored before the error reaches the caller.
 *
 * @param {object[]} edits
 */
function applyEdits(edits) {
  const applied = [];
  try {
    for (const edit of edits) {
      if (applyEdit(edit)) applied.push(edit);
    }
  } catch (error) {
    const rollbackErrors = [];
    for (const edit of applied.reverse()) {
      try {
        restoreEdit(edit);
      } catch (rollbackError) {
        rollbackErrors.push(`${edit.path}: ${rollbackError.message}`);
      }
    }
    const suffix = rollbackErrors.length
      ? ` Rollback also failed for ${rollbackErrors.join("; ")}.`
      : " Earlier config writes were rolled back.";
    throw new Error(`${error.message}.${suffix}`);
  }
}

/**
 * Print the preflight plan for user confirmation.
 *
 * @param {{ operation: string, launcher: object, edits: object[] }} plan
 */
function printPlan(plan) {
  const title = plan.operation === "remove" ? "MCP Removal" : "MCP Setup";
  stderr.write(`\n◆ Jacobian // ${title}\n`);
  stderr.write(`  Launcher: ${plan.launcher.command} ${plan.launcher.args.join(" ")}\n`);
  stderr.write(`  Version:  ${plan.launcher.version}\n`);
  if (plan.launcher.package) {
    stderr.write(`  Package:  ${plan.launcher.package} (may contact npm on client restart)\n`);
  }
  stderr.write(`\n  Planned changes:\n`);
  for (const edit of plan.edits) {
    const det = edit.detected ? " [detected]" : "";
    stderr.write(
      `    ${edit.client.displayName}${det}: ${edit.action} → ${edit.path}\n`,
    );
  }
  const runtime = plan.launcher.runtimePlan;
  if (plan.operation === "setup" && runtime?.status === "deferred_until_first_use") {
    stderr.write(`\n  Runtime on first use:\n`);
    stderr.write(`    Environment: ${runtime.environment}\n`);
    stderr.write(`    Package: ${runtime.pythonPackage} (~${runtime.approximateInstallSizeMb} MB)\n`);
    if (runtime.approximateManagedPythonSizeMb > 0) {
      stderr.write(
        `    Managed Python: up to ~${runtime.approximateManagedPythonSizeMb} MB\n`,
      );
    }
    stderr.write(`    Network: ${runtime.network}\n`);
  }
  stderr.write("\n");
}

function printBlocked(plan) {
  const runtime = plan.launcher.runtimePlan;
  stderr.write("\n◆ Jacobian // Setup blocked\n");
  stderr.write("  No changes were made.\n\n");
  stderr.write(`  ${runtime.recovery}\n`);
  stderr.write("  Then run `npx jacobian setup` again.\n\n");
}

/**
 * Interactive multi-select prompt using raw terminal input.
 *
 * @param {ClientDef[]} clients
 * @param {ClientId[]} detected
 * @param {NodeJS.ReadStream} [input]
 * @param {NodeJS.WriteStream} [output]
 * @returns {Promise<ClientDef[]>}
 */
async function interactiveSelect(
  clients,
  detected,
  input = stdin,
  output = stderr,
  options = {},
) {
  if (
    options.plain ||
    !input.isTTY ||
    !output.isTTY ||
    typeof input.setRawMode !== "function"
  ) {
    return typedSelect(clients, detected, input, output);
  }

  emitKeypressEvents(input);
  const selected = new Set();
  let focused = 0;
  let renderedLines = 0;

  const render = () => {
    const columns = output.columns || 80;
    const title = columns >= 36
      ? "  Select coding agents to configure"
      : "  Select agents to configure";
    const help = columns >= 76
      ? ["  ↑/↓ or Tab to move · Space to toggle · Enter to confirm · Ctrl-C to cancel"]
      : columns >= 36
        ? ["  ↑/↓ or Tab moves · Space toggles", "  Enter confirms · Ctrl-C cancels"]
        : ["  Move: ↑/↓ or Tab", "  Toggle: Space", "  Confirm: Enter", "  Cancel: Ctrl-C"];
    const lines = [
      title,
      ...help,
      "",
      ...clients.map((def, index) => {
        const cursor = index === focused ? ">" : " ";
        const mark = selected.has(index) ? "●" : "○";
        const det = detected.includes(def.id) ? " · detected" : "";
        return `  ${cursor} ${mark} ${def.displayName}${det}`;
      }),
    ];
    if (renderedLines > 0) {
      output.write(`\u001b[${renderedLines}A\r\u001b[J`);
    }
    output.write(lines.join("\n") + "\n");
    renderedLines = lines.length;
  };

  const settle = (summary) => {
    output.write(`\u001b[${renderedLines}A\r\u001b[J  ${summary}\n`);
  };

  output.write("\n");
  render();
  const wasRaw = Boolean(input.isRaw);
  input.setRawMode(true);
  input.resume();

  return new Promise((resolve) => {
    const finish = (result, summary) => {
      input.removeListener("keypress", onKeypress);
      input.setRawMode(wasRaw);
      input.pause();
      settle(summary);
      resolve(result);
    };
    const onKeypress = (_sequence, key = {}) => {
      if (key.ctrl && key.name === "c") {
        finish([], "Agents: none");
        return;
      }
      if (key.name === "up" || (key.name === "tab" && key.shift)) {
        focused = (focused - 1 + clients.length) % clients.length;
        render();
        return;
      }
      if (key.name === "down" || key.name === "tab") {
        focused = (focused + 1) % clients.length;
        render();
        return;
      }
      if (key.name === "space") {
        if (selected.has(focused)) selected.delete(focused);
        else selected.add(focused);
        render();
        return;
      }
      if (key.name === "return" || key.name === "enter") {
        const result = clients.filter((_client, index) => selected.has(index));
        const summary = result.length > 0
          ? `Agents: ${result.map((client) => client.displayName).join(", ")}`
          : "Agents: none";
        finish(result, summary);
      }
    };
    input.on("keypress", onKeypress);
  });
}

/** Typed fallback for piped input and terminals without raw-mode support. */
async function typedSelect(clients, detected, input = stdin, output = stderr) {
  output.write("\n  Select coding agents to configure:\n\n");
  for (let i = 0; i < clients.length; i++) {
    const def = clients[i];
    const det = detected.includes(def.id) ? " — detected" : "";
    output.write(`  [ ] ${i + 1}. ${def.displayName}${det}\n`);
  }
  output.write("\n  Enter comma-separated numbers (e.g. 1,3) or 'all': ");

  const rl = readline.createInterface({ input, output });
  try {
    const answer = (await rl.question("")).trim();
    rl.close();
    if (answer.toLowerCase() === "all") {
      return clients;
    }
    if (!answer) return [];
    const indices = answer
      .split(/[,\s]+/)
      .map((n) => parseInt(n, 10) - 1)
      .filter((n) => n >= 0 && n < clients.length);
    return indices.map((i) => clients[i]);
  } finally {
    rl.close();
  }
}

/**
 * Interactive yes/no confirmation.
 *
 * @returns {Promise<boolean>}
 */
async function interactiveConfirm() {
  const rl = readline.createInterface({ input: stdin, output: stderr });
  try {
    const answer = (await rl.question("Apply these changes? [y/N] ")).trim().toLowerCase();
    return answer === "y" || answer === "yes";
  } finally {
    rl.close();
  }
}

/**
 * Run the setup or removal wizard.
 *
 * @param {object} options
 * @param {"setup" | "remove"} options.operation
 * @param {string[]} [options.clients] Explicit client IDs.
 * @param {boolean} [options.all] Select all clients.
 * @param {boolean} [options.yes] Skip confirmation.
 * @param {boolean} [options.dryRun] Print plan without writing.
 * @param {boolean} [options.json] Output as JSON.
 * @param {boolean} [options.plain] Use a non-animated numbered prompt.
 * @param {string} [options.source] Absolute or relative source checkout.
 * @param {string} [options.stateDir] Fixed state directory for source launches.
 * @param {string} [options.uvBin] uv executable used by source launches.
 * @param {string} [options.profile] Source dependency profile metadata.
 * @param {string} [options.providerPath] PATH audited by the source doctor.
 * @param {string} [options.projectEnvironment] uv environment used for sync and launch.
 * @param {string} [options.elanHome] ELAN_HOME audited by the Lean source profile.
 * @param {string} [options.leanRuntime] Mathlib runtime audited by the Lean source profile.
 * @returns {Promise<object>} Setup report.
 */
async function run(options) {
  const operation = options.operation;
  const home = homedir();
  const defs = clientDefinitions(home);
  const supportedIds = new Set(defs.map((definition) => definition.id));
  const invalidClients = (options.clients || []).filter(
    (client) => !supportedIds.has(client),
  );
  if (invalidClients.length > 0) {
    throw new Error(`Unknown MCP client: ${invalidClients.join(", ")}.`);
  }
  if (options.all && options.clients && options.clients.length > 0) {
    throw new Error("--all cannot be combined with --client.");
  }
  const launcher = options.source
    ? buildSourceLauncher(
        options.source,
        options.stateDir || join(resolve(options.source), ".jacobian"),
        options.uvBin || "uv",
        options.profile || "core",
        options.providerPath || process.env.PATH || "",
        options.projectEnvironment || process.env.UV_PROJECT_ENVIRONMENT || "",
        options.elanHome || process.env.ELAN_HOME || "",
        options.leanRuntime || process.env.JACOBIAN_LEAN_RUNTIME || "",
      )
    : buildLauncher();
  const detectedIds = defs.filter((d) => isClientDetected(home, d.id)).map((d) => d.id);
  const interactive = Boolean(stdin.isTTY && stderr.isTTY);

  let selected;
  if (options.all) {
    selected = defs;
  } else if (options.clients && options.clients.length > 0) {
    const wanted = new Set(options.clients);
    selected = defs.filter((d) => wanted.has(d.id));
  } else if (!interactive) {
    throw new Error(
      "Non-interactive setup requires --client <id> or --all; detection is not consent.",
    );
  } else {
    selected = await interactiveSelect(
      defs,
      detectedIds,
      stdin,
      stderr,
      { plain: options.plain },
    );
  }

  if (selected.length === 0) {
    const report = { operation, cancelled: true, dryRun: false, results: [] };
    if (options.json) stdout.write(JSON.stringify(report, null, 2) + "\n");
    return report;
  }
  if (!interactive && !options.yes && !options.dryRun) {
    throw new Error("Non-interactive setup requires --yes after explicit client selection.");
  }

  if (operation === "setup" && launcher.runtimePlan?.status === "blocked") {
    const blockedPlan = { operation, launcher, edits: [] };
    const report = {
      operation,
      cancelled: false,
      blocked: true,
      dryRun: Boolean(options.dryRun),
      plan: publicPlan(blockedPlan),
      results: [],
    };
    if (options.json) stdout.write(JSON.stringify(report, null, 2) + "\n");
    else printBlocked(blockedPlan);
    process.exitCode = 1;
    return report;
  }

  const configEdits = await Promise.all(
    selected.map((def) => resolveClientEdit(operation, def, launcher)),
  );
  const edits = configEdits;
  const plan = { operation, launcher, edits };
  const reportPlan = publicPlan(plan);

  if (options.dryRun) {
    const report = {
      operation,
      cancelled: false,
      dryRun: true,
      plan: reportPlan,
      results: [],
    };
    if (options.json) {
      stdout.write(JSON.stringify(report, null, 2) + "\n");
    } else {
      printPlan(plan);
      stderr.write("  (dry-run: no changes written)\n\n");
    }
    return report;
  }

  if (!options.yes) {
    printPlan(plan);
    const confirmed = await interactiveConfirm();
    if (!confirmed) {
      const report = {
        operation,
        cancelled: true,
        dryRun: false,
        plan: reportPlan,
        results: [],
      };
      if (options.json) stdout.write(JSON.stringify(report, null, 2) + "\n");
      return report;
    }
  } else if (!options.json) {
    printPlan(plan);
  }

  // Apply all client edits as one transaction.
  applyEdits(edits);
  const results = configEdits.map((edit) => ({
    client: edit.client.id,
    path: edit.path,
    status: edit.action,
    error: null,
  }));

  if (options.json) {
    stdout.write(JSON.stringify({ operation, cancelled: false, dryRun: false, results }, null, 2) + "\n");
  } else {
    const noun = operation === "remove" ? "Removal" : "Configuration";
    stderr.write(`\n  ${noun} complete.\n`);
    for (const result of results) {
      const status = result.error ? `FAILED: ${result.error}` : result.status;
      stderr.write(`    ${result.client}: ${status}\n`);
    }
    if (operation === "remove") {
      stderr.write("\n  Restart or reload the affected clients.\n\n");
    } else if (launcher.source) {
      stderr.write(
        "\n  Restart or reload configured clients. Re-run the checkout's " +
          "`scripts/setup-agent` command after source updates.\n\n",
      );
    } else {
      const clientArgs = selected.map((client) => `--client ${client.id}`).join(" ");
      stderr.write(
        "\n  Runtime installation is deferred until verification or first use. " +
          "Restart or reload configured clients, then run " +
          `\`npx jacobian doctor ${clientArgs}\` to verify the configured launcher.\n\n`,
      );
    }
  }

  return { operation, cancelled: false, dryRun: false, results };
}

module.exports = {
  SERVER_NAME,
  clientDefinitions,
  isClientDetected,
  buildLauncher,
  buildSourceLauncher,
  applyEdits,
  publicPlan,
  inspectClientConfiguration,
  resolveClientEdit,
  interactiveSelect,
  run,
};
