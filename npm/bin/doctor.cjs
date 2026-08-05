"use strict";

const { spawn } = require("node:child_process");
const { join } = require("node:path");
const { stderr, stdout } = require("node:process");

/**
 * Jacobian MCP doctor.
 *
 * Launches the Jacobian MCP server as a subprocess, performs the MCP
 * initialize handshake, lists tools, and verifies that the expected tool
 * catalog is present.  Reports structured status to stderr (human-readable)
 * or stdout (JSON with --json).
 */

const PROTOCOL_VERSION = "2025-11-25";
const HANDSHAKE_TIMEOUT_MS = 60_000;
const RESPONSE_TIMEOUT_MS = 10_000;

const EXPECTED_TOOLS = [
  "math.find",
  "math.run",
];

/**
 * Convert a server-side startup failure into a safe, actionable diagnosis.
 * The server may have emitted a Python traceback, so only known bounded
 * failure classes are surfaced instead of forwarding arbitrary stderr.
 *
 * @param {string} stderrText
 * @param {string} fallbackMessage
 * @returns {{code: string, message: string, recovery: string}}
 */
function classifyStartupFailure(stderrText, fallbackMessage) {
  const migration = stderrText.match(
    /state migration (\d+) identity or checksum changed/,
  );
  if (migration) {
    return {
      code: "STATE_MIGRATION_INCOMPATIBLE",
      message:
        `Jacobian state migration ${migration[1]} does not match this ` +
        "Jacobian version.",
      recovery:
        "Preserve the existing state directory and export it with a compatible " +
        "checkout, or start Jacobian with a fresh state directory. Do not edit " +
        "metadata.sqlite3.",
    };
  }
  if (/state database was created by a newer unsupported revision/.test(stderrText)) {
    return {
      code: "STATE_REVISION_UNSUPPORTED",
      message: "The Jacobian state was created by a newer unsupported revision.",
      recovery:
        "Use the newer Jacobian version, or preserve the state and export it " +
        "before starting a fresh state directory.",
    };
  }
  if (/No module named ['"]?jacobian/.test(stderrText)) {
    return {
      code: "JACOBIAN_RUNTIME_UNAVAILABLE",
      message: "The doctor runtime cannot import the Jacobian package.",
      recovery:
        "Run `npx jacobian setup` for the managed installation, or use the " +
        "source-agent doctor for a source-bound checkout.",
    };
  }
  return {
    code: "MCP_STARTUP_FAILED",
    message: fallbackMessage,
    recovery:
      "Run `npx jacobian setup`, preserve the state directory, and retry the " +
      "doctor command.",
  };
}

function timeoutMessage(timeoutMs) {
  return (
    `Jacobian did not answer within ${timeoutMs} ms. Retry this command once; ` +
    "if it happens again, run `npx jacobian setup` and retry."
  );
}

function handshakeFailure(stage) {
  return (
    `Jacobian could not complete the MCP ${stage}. Run \`npx jacobian setup\`, ` +
    "restart the configured client, and retry `npx jacobian doctor`."
  );
}

/**
 * @typedef {object} DoctorReport
 * @property {string} status  "ok" | "error"
 * @property {string} serverName
 * @property {string} serverVersion
 * @property {boolean} instructionsLoaded
 * @property {string[]} tools
 * @property {object} integration
 * @property {string} integration.launcherStatus
 * @property {string} integration.handshakeStatus
 * @property {string} integration.catalogStatus
 * @property {string} integration.repairCommand
 * @property {object} firstCall
 * @property {string} firstCall.status
 */

/**
 * Send a JSON-RPC message to the server's stdin.
 *
 * @param {import("node:child_process").ChildProcessWithoutNullStreams} child
 * @param {object} message
 */
function sendMessage(child, message) {
  const data = JSON.stringify(message) + "\n";
  child.stdin.write(data);
}

/**
 * Wait for a JSON-RPC response with a specific id.
 *
 * @param {import("node:child_process").ChildProcessWithoutNullStreams} child
 * @param {number} id
 * @param {number} timeoutMs
 * @returns {Promise<object>}
 */
function waitForResponse(child, id, timeoutMs) {
  return new Promise((resolve, reject) => {
    let buffer = "";
    const timer = setTimeout(() => {
      reject(new Error(timeoutMessage(timeoutMs)));
    }, timeoutMs);

    child.stdout.on("data", (chunk) => {
      buffer += chunk.toString();
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        if (!line.trim()) continue;
        try {
          const message = JSON.parse(line);
          if (message.id === id) {
            clearTimeout(timer);
            resolve(message);
            return;
          }
        } catch {
          // Ignore non-JSON lines (server stderr noise).
        }
      }
    });

    child.once("error", (error) => {
      clearTimeout(timer);
      reject(error);
    });

    child.once("close", (code) => {
      clearTimeout(timer);
      reject(
        new Error(
          "Jacobian stopped before responding. Run `npx jacobian setup`, then retry " +
            "`npx jacobian doctor`.",
        ),
      );
    });
  });
}

/**
 * Run the doctor diagnostic.
 *
 * @param {object} [options]
 * @param {boolean} [options.json]
 * @returns {Promise<DoctorReport>}
 */
async function run(options = {}) {
  const json = options.json ?? false;
  const launcher = require("./launcher.cjs");

  if (!json) {
    stderr.write("◇ Jacobian is checking the MCP handshake and tool catalog...\n");
  }

  // Resolve the Python executable and spawn the MCP server.
  let python;
  try {
    python = launcher.resolvePython();
  } catch (error) {
    const report = {
      status: "error",
      serverName: "",
      serverVersion: "",
      instructionsLoaded: false,
      tools: [],
      integration: {
        launcherStatus: "failed",
        handshakeStatus: "not_attempted",
        catalogStatus: "not_attempted",
        repairCommand: "npx jacobian setup",
      },
      firstCall: { status: "not_attempted" },
      error: error.message,
    };
    if (json) {
      stdout.write(JSON.stringify(report, null, 2) + "\n");
    } else {
      stderr.write(`  ✗ Launcher failed: ${error.message}\n`);
      stderr.write(`  Run \`npx jacobian setup\` to configure MCP clients.\n`);
    }
    process.exitCode = 1;
    return report;
  }

  const stateDir = process.env.JACOBIAN_STATE_DIR || join(process.cwd(), ".jacobian");
  const child = spawn(python, ["-m", "jacobian.adapters.mcp.server"], {
    stdio: ["pipe", "pipe", "pipe"],
    env: { ...process.env, JACOBIAN_STATE_DIR: stateDir },
    windowsHide: true,
  });
  let stderrText = "";
  child.stderr.on("data", (chunk) => {
    stderrText = (stderrText + chunk.toString()).slice(-16_384);
  });

  const version = require("../package.json").version;

  try {
    // 1. Initialize handshake.
    sendMessage(child, {
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: {
        protocolVersion: PROTOCOL_VERSION,
        capabilities: {},
        clientInfo: {
          name: "jacobian-doctor",
          version,
        },
      },
    });

    const initResponse = await waitForResponse(child, 1, HANDSHAKE_TIMEOUT_MS);
    if (initResponse.error) {
      throw new Error(handshakeFailure("initialization handshake"));
    }
    const result = initResponse.result;
    const serverName = result?.serverInfo?.name ?? "";
    const serverVersion = result?.serverInfo?.version ?? "";
    const instructionsLoaded = typeof result?.instructions === "string";

    if (serverName !== "jacobian") {
      throw new Error(
        "The configured MCP server is not Jacobian. Run `npx jacobian setup`, " +
          "restart the configured client, and retry `npx jacobian doctor`.",
      );
    }

    // 2. Send initialized notification.
    sendMessage(child, {
      jsonrpc: "2.0",
      method: "notifications/initialized",
      params: {},
    });

    // 3. List tools.
    sendMessage(child, {
      jsonrpc: "2.0",
      id: 2,
      method: "tools/list",
      params: {},
    });

    const toolsResponse = await waitForResponse(child, 2, RESPONSE_TIMEOUT_MS);
    if (toolsResponse.error) {
      throw new Error(handshakeFailure("tool-catalog request"));
    }
    const tools = (toolsResponse.result?.tools ?? []).map((t) => t.name);
    const missingTools = EXPECTED_TOOLS.filter((t) => !tools.includes(t));
    const catalogStatus = missingTools.length === 0 ? "complete" : "partial";

    const report = {
      status: missingTools.length === 0 ? "ok" : "error",
      serverName,
      serverVersion,
      instructionsLoaded,
      tools,
      integration: {
        launcherStatus: "ok",
        handshakeStatus: "ok",
        catalogStatus,
        repairCommand: "npx jacobian setup",
      },
      firstCall: { status: "not_attempted" },
      missingTools,
    };

    child.kill("SIGTERM");

    if (json) {
      stdout.write(JSON.stringify(report, null, 2) + "\n");
    } else {
      stderr.write(`\n  ✓ Launcher: ok\n`);
      stderr.write(`  ✓ Handshake: ok (server: ${serverName} ${serverVersion})\n`);
      stderr.write(`  ${catalogStatus === "complete" ? "✓" : "✗"} Tool catalog: ${catalogStatus} (${tools.length} tools)\n`);
      if (missingTools.length > 0) {
        stderr.write(`    Missing: ${missingTools.join(", ")}\n`);
      }
      stderr.write(`\n  ${report.status === "ok" ? "✓ Jacobian MCP is ready." : "✗ Jacobian MCP has issues."}\n`);
      stderr.write(`  Run \`npx jacobian setup\` to configure or repair MCP clients.\n\n`);
    }

    if (report.status !== "ok") {
      process.exitCode = 1;
    }
    return report;
  } catch (error) {
    child.kill("SIGTERM");
    const diagnostic = classifyStartupFailure(stderrText, error.message);
    const report = {
      status: "error",
      serverName: "",
      serverVersion: "",
      instructionsLoaded: false,
      tools: [],
      integration: {
        launcherStatus: "ok",
        handshakeStatus: "failed",
        catalogStatus: "not_attempted",
        repairCommand: diagnostic.recovery,
      },
      firstCall: { status: "not_attempted" },
      error: diagnostic.message,
      diagnostic,
    };
    if (json) {
      stdout.write(JSON.stringify(report, null, 2) + "\n");
    } else {
      stderr.write(`\n  ✓ Launcher: ok\n`);
      stderr.write(`  ✗ Handshake failed: ${diagnostic.message}\n`);
      stderr.write(`  Recovery: ${diagnostic.recovery}\n\n`);
    }
    process.exitCode = 1;
    return report;
  }
}

module.exports = {
  classifyStartupFailure,
  run,
  EXPECTED_TOOLS,
  timeoutMessage,
  handshakeFailure,
};
