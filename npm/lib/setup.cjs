"use strict";

const { spawnSync } = require("node:child_process");
const { randomBytes } = require("node:crypto");
const fs = require("node:fs/promises");
const os = require("node:os");
const path = require("node:path");

const TOML = require("@iarna/toml");
const { applyEdits, modify, parse, printParseErrorCode } = require("jsonc-parser");

const SERVER_NAME = "jacobian";
const MANAGED_SETUP_ARGUMENT = "--managed-by-setup";
const TOML_MARKER = "# Managed by Jacobian setup.";
const SKILL_MARKER = "<!-- Managed by Jacobian setup. -->";
const SKILL_NAME = "jacobian-math";
const BUNDLED_SKILL_PATH = path.join(__dirname, "..", "skill", SKILL_NAME, "SKILL.md");
const MAX_CONFIG_BYTES = 8 * 1024 * 1024;

const CLIENTS = [
  {
    id: "claude",
    displayName: "Claude Code",
    kind: "json",
    section: "mcpServers",
    paths: [".claude.json"],
    skillPath: ".claude/skills/jacobian-math/SKILL.md",
    detected: (home) => [".claude", ".claude.json"].some((entry) => exists(path.join(home, entry))),
  },
  {
    id: "opencode",
    displayName: "OpenCode",
    kind: "opencode",
    section: "mcp",
    paths: [
      ".config/opencode/opencode.json",
      ".config/opencode/opencode.jsonc",
      ".config/opencode/.opencode.json",
      ".config/opencode/.opencode.jsonc",
    ],
    skillPath: ".agents/skills/jacobian-math/SKILL.md",
    detected: (home) => exists(path.join(home, ".config/opencode")),
  },
  {
    id: "codex",
    displayName: "Codex",
    kind: "toml",
    paths: [".codex/config.toml"],
    skillPath: ".agents/skills/jacobian-math/SKILL.md",
    detected: (home) => exists(path.join(home, ".codex")),
  },
  {
    id: "cursor",
    displayName: "Cursor",
    kind: "json",
    section: "mcpServers",
    paths: [".cursor/mcp.json"],
    skillPath: ".cursor/skills/jacobian-math/SKILL.md",
    detected: (home) => exists(path.join(home, ".cursor")),
  },
  {
    id: "gemini",
    displayName: "Gemini CLI",
    kind: "json",
    section: "mcpServers",
    paths: [".gemini/settings.json"],
    skillPath: ".gemini/skills/jacobian-math/SKILL.md",
    detected: (home) => exists(path.join(home, ".gemini")),
  },
  {
    id: "antigravity",
    displayName: "Antigravity",
    kind: "json",
    section: "mcpServers",
    paths: [".gemini/config/mcp_config.json"],
    skillPath: ".agent/skills/jacobian-math/SKILL.md",
    detected: (home) =>
      [".gemini/antigravity", ".agent"].some((entry) => exists(path.join(home, entry))),
  },
];

class SetupError extends Error {}

function exists(candidate) {
  try {
    return require("node:fs").existsSync(candidate);
  } catch {
    return false;
  }
}

function setupHelp() {
  return `Jacobian setup — configure MCP clients with an exact Jacobian release.

Usage:
  jacobian setup [options]

Options:
  --claude, --opencode, --codex, --cursor, --gemini, --antigravity
                         Select one or more clients.
  --all                  Select every supported client.
  --dry-run              Print the resolved plan without changing files.
  --yes                  Apply an explicit selection without prompting.
  --force                Replace an existing unmanaged jacobian entry.
  --json                 Emit the resolved report as JSON (requires explicit selection).
  -h, --help             Show this help.

Setup never installs Node.js, uv, Python, or an MCP client. It writes the
selected client configuration entries and the Jacobian skill, both from the
exact npm version that performed setup.\n`;
}

function parseArgs(args) {
  const result = {
    clients: [],
    all: false,
    dryRun: false,
    yes: false,
    force: false,
    json: false,
    help: false,
  };
  const clientIds = new Set(CLIENTS.map((client) => client.id));
  for (const argument of args) {
    if (argument === "--all") result.all = true;
    else if (argument === "--dry-run") result.dryRun = true;
    else if (argument === "--yes" || argument === "-y") result.yes = true;
    else if (argument === "--force") result.force = true;
    else if (argument === "--json") result.json = true;
    else if (argument === "--help" || argument === "-h") result.help = true;
    else if (argument.startsWith("--") && clientIds.has(argument.slice(2))) {
      result.clients.push(argument.slice(2));
    } else {
      throw new SetupError(`unknown setup option: ${argument}`);
    }
  }
  if (result.all && result.clients.length > 0) {
    throw new SetupError("--all cannot be combined with individual client flags");
  }
  result.clients = [...new Set(result.clients)];
  return result;
}

function homeDirectory() {
  const home = process.env.HOME || process.env.USERPROFILE || os.homedir();
  if (!path.isAbsolute(home)) throw new SetupError("could not determine an absolute home directory");
  return path.resolve(home);
}

function isInteractive() {
  return Boolean(process.stdin.isTTY && process.stdout.isTTY && process.stderr.isTTY);
}

function detectClients(home) {
  return new Set(CLIENTS.filter((client) => client.detected(home)).map((client) => client.id));
}

function choosePath(client, home) {
  const candidates = client.paths.map((relative) => path.join(home, relative));
  return candidates.find((candidate) => exists(candidate)) || candidates[0];
}

function launcher(version) {
  return {
    command: "npx",
    args: ["--yes", `jacobian@${version}`, "mcp", MANAGED_SETUP_ARGUMENT],
  };
}

function jsonEntry(client, runtime) {
  if (client.kind === "opencode") {
    return {
      type: "local",
      command: [runtime.command, ...runtime.args],
      cwd: ".",
      enabled: true,
    };
  }
  return { command: runtime.command, args: runtime.args };
}

function isManagedJsonEntry(client, entry) {
  if (!entry || typeof entry !== "object" || Array.isArray(entry)) return false;
  const command =
    client.kind === "opencode"
      ? entry.command
      : typeof entry.command === "string" && Array.isArray(entry.args)
        ? [entry.command, ...entry.args]
        : [];
  return (
    Array.isArray(command) &&
    command.includes(MANAGED_SETUP_ARGUMENT) &&
    command.some((argument) => typeof argument === "string" && argument.startsWith("jacobian@"))
  );
}

async function readOptional(filePath) {
  try {
    const stat = await fs.stat(filePath);
    if (!stat.isFile()) throw new SetupError(`configuration path is not a regular file: ${filePath}`);
    if (stat.size > MAX_CONFIG_BYTES) {
      throw new SetupError(`refusing to read configuration above ${MAX_CONFIG_BYTES} bytes: ${filePath}`);
    }
    return await fs.readFile(filePath, "utf8");
  } catch (error) {
    if (error && error.code === "ENOENT") return null;
    throw error;
  }
}

function parseJson(source, filePath) {
  const errors = [];
  const value = parse(source, errors, { allowTrailingComma: true, disallowComments: false });
  if (errors.length > 0) {
    throw new SetupError(
      `invalid JSON configuration at ${filePath}: ${printParseErrorCode(errors[0].error)}`,
    );
  }
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new SetupError(`top-level JSON configuration must be an object: ${filePath}`);
  }
  return value;
}

function sameJson(left, right) {
  return JSON.stringify(left) === JSON.stringify(right);
}

function planJsonEdit(client, filePath, original, runtime, force) {
  const source = original === null ? "{}\n" : original;
  const root = parseJson(source, filePath);
  const section = root[client.section];
  if (section !== undefined && (!section || typeof section !== "object" || Array.isArray(section))) {
    throw new SetupError(`${client.section} must be an object: ${filePath}`);
  }
  const current = section && section[SERVER_NAME];
  if (current !== undefined && !isManagedJsonEntry(client, current) && !force) {
    throw new SetupError(
      `refusing to replace an unmanaged Jacobian entry in ${filePath}; review it, then retry with --force`,
    );
  }
  const expected = jsonEntry(client, runtime);
  if (sameJson(current, expected)) return { action: "already current", updated: null };
  const edits = modify(source, [client.section, SERVER_NAME], expected, {
    formattingOptions: { insertSpaces: true, tabSize: 2, eol: "\n" },
  });
  const updated = applyEdits(source, edits);
  parseJson(updated, filePath);
  return { action: current === undefined ? "create" : "update", updated };
}

function tomlBlock(runtime) {
  const args = runtime.args.map((argument) => JSON.stringify(argument)).join(", ");
  return `${TOML_MARKER}\n[mcp_servers.${SERVER_NAME}]\ncommand = ${JSON.stringify(runtime.command)}\nargs = [${args}]\nstartup_timeout_sec = 30\n`;
}

function isManagedTomlEntry(entry) {
  return (
    entry &&
    typeof entry === "object" &&
    Array.isArray(entry.args) &&
    entry.args.includes(MANAGED_SETUP_ARGUMENT) &&
    entry.args.some((argument) => typeof argument === "string" && argument.startsWith("jacobian@"))
  );
}

function managedTomlBlock(source) {
  const escapedMarker = TOML_MARKER.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return new RegExp(`(^|\\n)${escapedMarker}\\n\\[mcp_servers\\.${SERVER_NAME}\\]\\n[\\s\\S]*?(?=\\n\\[|$)`);
}

function planTomlEdit(filePath, original, runtime, force) {
  const source = original || "";
  let root;
  try {
    root = source.trim() ? TOML.parse(source) : {};
  } catch (error) {
    throw new SetupError(`invalid TOML configuration at ${filePath}: ${error.message}`);
  }
  const current = root.mcp_servers && root.mcp_servers[SERVER_NAME];
  if (current !== undefined && !isManagedTomlEntry(current) && !force) {
    throw new SetupError(
      `refusing to replace an unmanaged Jacobian entry in ${filePath}; review it, then retry with --force`,
    );
  }
  const block = tomlBlock(runtime);
  const expression = managedTomlBlock(source);
  if (current !== undefined) {
    const managedMatch = source.match(expression);
    if (!managedMatch && !force) {
      throw new SetupError(`managed Jacobian entry cannot be safely updated: ${filePath}`);
    }
    const table = new RegExp(`(^|\\n)\\[mcp_servers\\.${SERVER_NAME}\\]\\n[\\s\\S]*?(?=\\n\\[|$)`);
    const replacement = managedMatch || source.match(table);
    if (!replacement) {
      throw new SetupError(`Jacobian entry cannot be safely updated in ${filePath}`);
    }
    const updated =
      source.replace(managedMatch ? expression : table, `${replacement[1]}${block.trimEnd()}`) +
      (source.endsWith("\n") ? "\n" : "");
    if (updated === source) return { action: "already current", updated: null };
    return { action: "update", updated };
  }
  if (root.mcp_servers !== undefined && /^\s*mcp_servers\s*=/m.test(source)) {
    throw new SetupError(`mcp_servers must be a table, not an inline value: ${filePath}`);
  }
  const separator = source.length === 0 || source.endsWith("\n") ? "\n" : "\n\n";
  return { action: current === undefined ? "create" : "update", updated: `${source}${separator}${block}` };
}

function assertUvAvailable() {
  const result = spawnSync("uvx", ["--version"], { stdio: "ignore" });
  if (result.error || result.status !== 0) {
    throw new SetupError(
      "Jacobian requires uvx on PATH before MCP clients can start. Install uv, then rerun setup; no configuration was changed.",
    );
  }
}

async function buildPlan(clientIds, home, version, force) {
  const detected = detectClients(home);
  const runtime = launcher(version);
  const plan = [];
  const skillSource = await fs.readFile(BUNDLED_SKILL_PATH, "utf8");
  const skillPlans = new Map();
  for (const clientId of clientIds) {
    const client = CLIENTS.find((candidate) => candidate.id === clientId);
    const filePath = choosePath(client, home);
    const original = await readOptional(filePath);
    const change =
      client.kind === "toml"
        ? planTomlEdit(filePath, original, runtime, force)
        : planJsonEdit(client, filePath, original, runtime, force);
    const skillPath = path.join(home, client.skillPath);
    let skill = skillPlans.get(skillPath);
    if (!skill) {
      const skillOriginal = await readOptional(skillPath);
      if (skillOriginal !== null && !skillOriginal.includes(SKILL_MARKER) && !force) {
        throw new SetupError(
          `refusing to replace an unmanaged Jacobian skill at ${skillPath}; review it, then retry with --force`,
        );
      }
      skill = {
        path: skillPath,
        original: skillOriginal,
        updated: skillOriginal === skillSource ? null : skillSource,
        action: skillOriginal === skillSource ? "already current" : skillOriginal === null ? "create" : "update",
      };
      skillPlans.set(skillPath, skill);
    }
    plan.push({
      client,
      path: filePath,
      original,
      ...change,
      detected: detected.has(client.id),
      skill,
    });
  }
  return { plan, runtime, detected };
}

async function writeAtomic(filePath, content) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  const temporary = `${filePath}.jacobian-${process.pid}-${randomBytes(6).toString("hex")}.tmp`;
  await fs.writeFile(temporary, content, "utf8");
  await fs.rename(temporary, filePath);
}

async function restore(filePath, original) {
  if (original === null) {
    await fs.unlink(filePath).catch((error) => {
      if (error.code !== "ENOENT") throw error;
    });
  } else {
    await writeAtomic(filePath, original);
  }
}

async function applyPlan(plan) {
  const applied = [];
  const changes = [];
  const seenPaths = new Set();
  for (const entry of plan) {
    for (const change of [entry, entry.skill]) {
      if (seenPaths.has(change.path)) continue;
      seenPaths.add(change.path);
      changes.push(change);
    }
  }
  try {
    for (const entry of changes) {
      if (entry.updated === null) continue;
      const current = await readOptional(entry.path);
      if (current !== entry.original) {
        throw new SetupError(`configuration changed after preflight: ${entry.path}`);
      }
      await writeAtomic(entry.path, entry.updated);
      applied.push(entry);
    }
  } catch (error) {
    const rollbackErrors = [];
    for (const entry of applied.reverse()) {
      try {
        await restore(entry.path, entry.original);
      } catch (rollbackError) {
        rollbackErrors.push(`${entry.path}: ${rollbackError.message}`);
      }
    }
    if (rollbackErrors.length > 0) {
      throw new SetupError(`${error.message}; rollback requires recovery: ${rollbackErrors.join("; ")}`);
    }
    throw error;
  }
  for (const entry of changes) {
    if (entry.updated !== null && (await readOptional(entry.path)) !== entry.updated) {
      throw new SetupError(`could not verify configuration after write: ${entry.path}`);
    }
  }
}

function renderPreflight(plan, runtime, force) {
  const lines = ["", "◆ Jacobian MCP setup plan"];
  for (const entry of plan) {
    lines.push(`  ◇ ${entry.client.displayName} — ${entry.action}`);
    lines.push(`    ${entry.path}`);
  }
  lines.push("", "  Launcher");
  lines.push(`    ${runtime.command} ${runtime.args.join(" ")}`);
  lines.push("    Exact package version is pinned; first client start may use npm and uv caches.");
  lines.push("", "  Skill");
  for (const entry of plan) lines.push(`    ${entry.client.displayName}: ${entry.skill.path} — ${entry.skill.action}`);
  lines.push("", "  Setup writes the selected client configuration entries and Jacobian skill.");
  lines.push("  It does not install or update uv, Python, Node.js, or any agent.");
  if (force) lines.push("  Explicit override: an unmanaged Jacobian entry may be replaced.");
  return lines.join("\n");
}

function report({ plan, runtime, dryRun, cancelled, force }) {
  return {
    status: cancelled ? "cancelled" : dryRun ? "planned" : "configured",
    dry_run: dryRun,
    cancelled,
    force,
    launcher: runtime,
    clients: plan.map((entry) => ({
      client: entry.client.id,
      display_name: entry.client.displayName,
      detected: entry.detected,
      path: entry.path,
      action: entry.action,
      skill_path: entry.skill.path,
      skill_action: entry.skill.action,
    })),
  };
}

function printReport(value, json) {
  if (json) {
    process.stdout.write(`${JSON.stringify(value)}\n`);
    return;
  }
  if (value.cancelled) {
    console.log("Jacobian setup cancelled. No changes were made.");
    return;
  }
  if (value.dry_run) {
    console.log("Jacobian setup dry-run complete. No changes were made.");
    return;
  }
  const names = value.clients.map((client) => client.display_name).join(", ");
  console.log(`◆ Jacobian MCP and skill are configured for ${names}.`);
  console.log("  Restart or reload those clients, then use math.find to discover an operation.");
}

async function promptForClients(detected) {
  const { checkbox } = await import("@inquirer/prompts");
  try {
    return await checkbox(
      {
        message: "Which agents should use Jacobian?",
        choices: CLIENTS.map((client) => ({
          name: `${client.displayName}${detected.has(client.id) ? " — detected" : ""}`,
          value: client.id,
          checked: detected.has(client.id),
        })),
        loop: false,
        required: false,
      },
      { input: process.stdin, output: process.stderr },
    );
  } catch (error) {
    if (error && error.name === "ExitPromptError") return null;
    throw error;
  }
}

async function confirmPlan() {
  const { confirm } = await import("@inquirer/prompts");
  try {
    return await confirm(
      { message: "Apply these changes?", default: false },
      { input: process.stdin, output: process.stderr },
    );
  } catch (error) {
    if (error && error.name === "ExitPromptError") return false;
    throw error;
  }
}

async function runSetup(args, version) {
  const options = parseArgs(args);
  if (options.help) {
    process.stdout.write(setupHelp());
    return;
  }
  const explicit = options.all ? CLIENTS.map((client) => client.id) : options.clients;
  const interactive = isInteractive() && !options.json;
  if ((options.json || !interactive) && explicit.length === 0) {
    throw new SetupError("non-interactive setup requires one or more client flags or --all");
  }
  if (!options.dryRun && !interactive && !options.yes) {
    throw new SetupError("non-interactive setup requires --yes or --dry-run; no changes were made");
  }
  if (!options.dryRun && options.yes && explicit.length === 0) {
    throw new SetupError("--yes requires one or more client flags or --all");
  }
  const home = homeDirectory();
  const detected = detectClients(home);
  const clientIds = explicit.length > 0 ? explicit : await promptForClients(detected);
  if (clientIds === null) {
    printReport(report({ plan: [], runtime: launcher(version), dryRun: false, cancelled: true, force: options.force }), options.json);
    return;
  }
  if (clientIds.length === 0) {
    printReport(report({ plan: [], runtime: launcher(version), dryRun: options.dryRun, cancelled: true, force: options.force }), options.json);
    return;
  }
  assertUvAvailable();
  const { plan, runtime } = await buildPlan(clientIds, home, version, options.force);
  if (options.dryRun) {
    if (!options.json) process.stderr.write(`${renderPreflight(plan, runtime, options.force)}\n`);
    printReport(report({ plan, runtime, dryRun: true, cancelled: false, force: options.force }), options.json);
    return;
  }
  if (!options.yes) {
    process.stderr.write(`${renderPreflight(plan, runtime, options.force)}\n\n`);
    if (!(await confirmPlan())) {
      printReport(report({ plan, runtime, dryRun: false, cancelled: true, force: options.force }), options.json);
      return;
    }
  }
  await applyPlan(plan);
  printReport(report({ plan, runtime, dryRun: false, cancelled: false, force: options.force }), options.json);
}

module.exports = {
  CLIENTS,
  MANAGED_SETUP_ARGUMENT,
  SetupError,
  buildPlan,
  launcher,
  parseArgs,
  runSetup,
  setupHelp,
};
