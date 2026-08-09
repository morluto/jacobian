import { isDeepStrictEqual } from "node:util";

import { parse, stringify } from "@decimalturn/toml-patch";

const SERVER_NAME = "jacobian";

function parseConfig(source, path) {
  try {
    const root = parse(source);
    if (typeof root !== "object" || root === null || Array.isArray(root)) {
      throw new Error("top-level value must be a table");
    }
    if (
      root.mcp_servers !== undefined &&
      (typeof root.mcp_servers !== "object" ||
        root.mcp_servers === null ||
        Array.isArray(root.mcp_servers))
    ) {
      throw new Error("mcp_servers must be a table");
    }
    return root;
  } catch (error) {
    throw new Error(
      `Invalid TOML in ${path}: ${error.message}. No changes were written. ` +
        "Repair this file, then retry.",
    );
  }
}

function expectedEntry(launcher) {
  return {
    command: launcher.command,
    args: launcher.args,
    startup_timeout_sec: 30,
    ...(Object.keys(launcher.env || {}).length > 0 ? { env: launcher.env } : {}),
  };
}

function plainValue(value) {
  if (Array.isArray(value)) return value.map(plainValue);
  if (typeof value !== "object" || value === null) return value;
  return Object.fromEntries(
    Object.entries(value).map(([key, item]) => [key, plainValue(item)]),
  );
}

function removeJacobianEntry(source, path) {
  const newline = source.includes("\r\n") ? "\r\n" : "\n";
  const kept = [];
  let currentTable = "";
  let skippingJacobianTable = false;
  let removed = false;
  for (const line of source.split(/\r?\n/)) {
    const header = line.match(/^\s*\[([^\]]+)]\s*(?:#.*)?$/);
    if (header) {
      currentTable = header[1].replaceAll(/\s/g, "");
      skippingJacobianTable =
        currentTable === "mcp_servers.jacobian" ||
        currentTable.startsWith("mcp_servers.jacobian.");
      if (skippingJacobianTable) {
        removed = true;
        continue;
      }
    }
    if (skippingJacobianTable) continue;
    if (currentTable === "mcp_servers" && /^\s*jacobian\s*=/.test(line)) {
      removed = true;
      continue;
    }
    kept.push(line);
  }
  if (!removed) {
    throw new Error(
      `Jacobian found an unsupported MCP entry representation in ${path}. ` +
        "No changes were written. Remove that entry with Codex, then retry.",
    );
  }
  return { source: kept.join(newline).trimEnd(), newline };
}

function validateGenerated(source, path) {
  try {
    parse(source);
  } catch (error) {
    throw new Error(
      `Jacobian could not safely edit ${path}: ${error.message}. ` +
        "No changes were written.",
    );
  }
}

/** Resolve a comment-preserving Codex TOML edit. */
export function resolveTomlEdit(operation, path, original, launcher) {
  const source = original ?? "";
  const root = parseConfig(source, path);
  const servers = root.mcp_servers ?? {};
  const existing = servers[SERVER_NAME];

  if (operation === "remove") {
    if (existing === undefined) {
      return { action: "not_configured", original, updated: null };
    }
    const cleaned = removeJacobianEntry(source, path);
    const updated = cleaned.source === "" ? "" : `${cleaned.source}${cleaned.newline}`;
    validateGenerated(updated, path);
    return {
      action: "remove",
      original,
      updated: updated.endsWith("\n") ? updated : `${updated}\n`,
    };
  }

  const expected = expectedEntry(launcher);
  if (isDeepStrictEqual(plainValue(existing), expected)) {
    return { action: "already_current", original, updated: null };
  }
  let base = source.trimEnd();
  let newline = source.includes("\r\n") ? "\r\n" : "\n";
  if (existing !== undefined) {
    const cleaned = removeJacobianEntry(source, path);
    base = cleaned.source;
    newline = cleaned.newline;
  }
  const fragment = stringify({ [SERVER_NAME]: expected })
    .trimEnd()
    .replace(`[${SERVER_NAME}]`, `[mcp_servers.${SERVER_NAME}]`)
    .replaceAll("\n", newline);
  const updated = `${base}${base === "" ? "" : `${newline}${newline}`}${fragment}${newline}`;
  validateGenerated(updated, path);
  return {
    action: original === null ? "create" : "update",
    original,
    updated: updated.endsWith("\n") ? updated : `${updated}\n`,
  };
}

/** Read the configured Jacobian launcher without changing the document. */
export function readTomlLauncher(path, original) {
  if (original === null) return null;
  const root = parseConfig(original, path);
  return root.mcp_servers?.[SERVER_NAME] ?? null;
}
