import { spawn } from "node:child_process";
import crypto from "node:crypto";
import { mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { accessTokenState } from "./credentials.mjs";
import { workerPython } from "./dsh-http-auth.mjs";
import { accountSlotTag } from "./pool-connection.mjs";
import { sanitizeStatsMessage } from "./pool-stats.mjs";
import { createPythonDaemonPool } from "./python-daemon-pool.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
export const WEB_TEXT_WORKER = path.resolve(__dirname, "..", "python", "web_text_worker.py");

const textDaemonPool = createPythonDaemonPool({ maxWorkers: 3, maxLineBytes: 16 * 1024 * 1024 });
export function closeTextWorkerPool() {
  textDaemonPool.closeAll();
}

export const DEFAULT_WEB_TEXT_MODEL = "gpt-5.6-luna";

// Verified working on Free/Platform ChatGPT Web accounts (/backend-api/f/conversation).
// Excludes watermarked variants (e.g. *-wm) that return 403 on Web endpoints.
export const WEB_TEXT_MODEL_SPECS = Object.freeze([
  { id: "gpt-6-luna", name: "GPT-6 Luna", reasoning: false },
  { id: "gpt-5.6-luna", name: "GPT-5.6 Luna", reasoning: false },
  { id: "gpt-5.6-terra", name: "GPT-5.6 Terra", reasoning: true },
  { id: "gpt-6", name: "GPT-6", reasoning: false },
  { id: "gpt-5-6", name: "GPT-5.6", reasoning: false },
  { id: "gpt-6-mini", name: "GPT-6 Mini", reasoning: false },
  { id: "gpt-5-6-mini", name: "GPT-5.6 Mini", reasoning: false },
  { id: "gpt-6-t-mini", name: "GPT-6 Thinking Mini", reasoning: true },
  { id: "gpt-5-6-t-mini", name: "GPT-5.6 Thinking Mini", reasoning: true },
  { id: "auto", name: "ChatGPT Auto", reasoning: false },
]);

const MAX_STDERR_BYTES = 16 * 1024;
const MAX_SLOW_FAILOVERS = 3;
const MAX_TOTAL_PROBE_MS = 120_000;
const DEFAULT_COOLDOWN_SECS = 60;
const MAX_COOLDOWN_ENTRIES = 1024;
export const DEFAULT_TEXT_WINDOW_MS = 3 * 60 * 60 * 1000;
export const DEFAULT_TEXT_WINDOW_MAX_CALLS = 10;

// Short-term rate-limit memory AND per-window call counters for text calls so:
// 1. 429'd accounts rotate cleanly;
// 2. Cache-first sticky routing keeps hitting the SAME warm account until 80%
//    of its rolling window budget is used, then switches cleanly to the next.
const textCooldownUntil = new Map();
const textWindowUsage = new Map();
let stickyTextSlot = "";
let cooldownFile = null;
let cooldownsLoaded = false;

export function configureTextCooldownStore({ file = null } = {}) {
  cooldownFile = file || null;
  cooldownsLoaded = false;
}

function loadCooldowns() {
  if (cooldownsLoaded || !cooldownFile) return;
  cooldownsLoaded = true;
  try {
    const parsed = JSON.parse(readFileSync(cooldownFile, "utf8"));
    const now = Date.now();
    for (const [tag, until] of Object.entries(parsed?.until || {})) {
      if (/^s[0-9a-f]{10}$/.test(tag) && Number.isFinite(until) && until > now &&
          textCooldownUntil.size < MAX_COOLDOWN_ENTRIES) {
        textCooldownUntil.set(tag, until);
      }
    }
    for (const [tag, win] of Object.entries(parsed?.windows || {})) {
      if (/^s[0-9a-f]{10}$/.test(tag) && win && typeof win === "object" &&
          Number.isFinite(win.start) && now - win.start < DEFAULT_TEXT_WINDOW_MS &&
          Number.isFinite(win.count) && win.count > 0 &&
          textWindowUsage.size < MAX_COOLDOWN_ENTRIES) {
        textWindowUsage.set(tag, { start: win.start, count: Math.trunc(win.count) });
      }
    }
    if (typeof parsed?.stickySlot === "string" && /^s[0-9a-f]{10}$/.test(parsed.stickySlot)) {
      stickyTextSlot = parsed.stickySlot;
    }
  } catch {
    // Missing or unreadable file: start with no cooldowns.
  }
}

function saveCooldowns() {
  if (!cooldownFile) return;
  const now = Date.now();
  const until = {};
  for (const [tag, at] of textCooldownUntil) if (at > now) until[tag] = at;
  const windows = {};
  for (const [tag, win] of textWindowUsage) {
    if (win && now - win.start < DEFAULT_TEXT_WINDOW_MS && win.count > 0) {
      windows[tag] = { start: win.start, count: win.count };
    }
  }
  try {
    mkdirSync(path.dirname(cooldownFile), { recursive: true, mode: 0o700 });
    const tmp = `${cooldownFile}.${process.pid}.tmp`;
    writeFileSync(tmp, JSON.stringify({
      until,
      windows,
      ...(stickyTextSlot ? { stickySlot: stickyTextSlot } : {}),
    }) + "\n", { mode: 0o600 });
    renameSync(tmp, cooldownFile);
  } catch {
    // Persistence is best effort; the in-memory state still applies.
  }
}

// CJK text is roughly one token per character, everything else roughly one
// per four characters. Still an estimate: the web endpoint reports no usage.
const CJK_CHAR = /[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff\uff00-\uffef]/g;

export function estimateTokens(text) {
  const value = String(text || "");
  if (!value) return 0;
  const cjk = value.match(CJK_CHAR)?.length || 0;
  return Math.max(1, cjk + Math.ceil((value.length - cjk) / 4));
}

export class WebTextHttpError extends Error {
  constructor(status, message, code = "upstream_error", retryAfter = null) {
    super(message);
    this.name = "WebTextHttpError";
    this.status = status;
    this.code = code;
    this.retryAfter = retryAfter;
  }
}

export function clearTextCooldowns() {
  textCooldownUntil.clear();
  textWindowUsage.clear();
  stickyTextSlot = "";
  saveCooldowns();
}

export function getWebTextRuntimeSnapshot({
  now = Date.now(),
  windowMs = DEFAULT_TEXT_WINDOW_MS,
  windowMaxCalls = DEFAULT_TEXT_WINDOW_MAX_CALLS,
} = {}) {
  loadCooldowns();
  const slots = {};
  for (const [tag, win] of textWindowUsage) {
    if (win && now - win.start < windowMs && win.count > 0) {
      slots[tag] = {
        windowCalls: win.count,
        windowMaxCalls,
        windowUsedPercent: Math.min(100, Math.round((win.count / Math.max(1, windowMaxCalls)) * 100)),
        windowResetAt: new Date(win.start + windowMs).toISOString(),
        cooldownUntil: null,
      };
    }
  }
  for (const [tag, until] of textCooldownUntil) {
    if (until > now) {
      slots[tag] ??= {
        windowCalls: 0,
        windowMaxCalls,
        windowUsedPercent: 0,
        windowResetAt: null,
      };
      slots[tag].cooldownUntil = new Date(until).toISOString();
    }
  }
  return {
    stickyTextSlot: stickyTextSlot || "",
    windowMaxCalls,
    slots,
  };
}

function getSlotWindowCalls(email, now = Date.now(), windowMs = DEFAULT_TEXT_WINDOW_MS) {
  const tag = accountSlotTag(email);
  if (!tag) return 0;
  loadCooldowns();
  const win = textWindowUsage.get(tag);
  if (!win) return 0;
  if (now - win.start >= windowMs) {
    textWindowUsage.delete(tag);
    return 0;
  }
  return win.count;
}

function recordSlotTextSuccess(email, now = Date.now(), windowMs = DEFAULT_TEXT_WINDOW_MS) {
  const tag = accountSlotTag(email);
  if (!tag) return;
  loadCooldowns();
  const prev = textWindowUsage.get(tag);
  if (!prev || now - prev.start >= windowMs) {
    if (textWindowUsage.size >= MAX_COOLDOWN_ENTRIES) {
      const oldestKey = textWindowUsage.keys().next().value;
      if (oldestKey) textWindowUsage.delete(oldestKey);
    }
    textWindowUsage.set(tag, { start: now, count: 1 });
  } else {
    prev.count += 1;
  }
  saveCooldowns();
}

function getSlotCooldownUntil(email) {
  const tag = accountSlotTag(email);
  if (!tag) return 0;
  loadCooldowns();
  const until = textCooldownUntil.get(tag) || 0;
  if (until && until <= Date.now()) {
    textCooldownUntil.delete(tag);
    return 0;
  }
  return until;
}

function setSlotCooldown(email, retryAfter) {
  const tag = accountSlotTag(email);
  if (!tag) return;
  const secs = Number(retryAfter);
  const waitMs = (Number.isFinite(secs) && secs > 0 ? Math.min(secs, 3600) : DEFAULT_COOLDOWN_SECS) * 1000;
  if (textCooldownUntil.size >= MAX_COOLDOWN_ENTRIES) {
    const oldestKey = textCooldownUntil.keys().next().value;
    if (oldestKey) textCooldownUntil.delete(oldestKey);
  }
  loadCooldowns();
  textCooldownUntil.set(tag, Date.now() + waitMs);
  saveCooldowns();
}

export function buildWebTextModelCatalog(baseURL) {
  const catalog = {};
  for (const spec of WEB_TEXT_MODEL_SPECS) {
    catalog[spec.id] = {
      id: spec.id,
      providerID: "gptfree-pool",
      name: spec.name,
      api: { id: spec.id, url: baseURL, npm: "@ai-sdk/openai" },
      status: "active",
      tool_call: true,
      capabilities: {
        temperature: true,
        reasoning: spec.reasoning,
        attachment: false,
        toolcall: true,
        input: { text: true, audio: false, image: false, video: false, pdf: false },
        output: { text: true, audio: false, image: false, video: false, pdf: false },
        interleaved: false,
      },
      cost: { input: 0, output: 0, cache: { read: 0, write: 0 } },
      limit: { context: 128000, output: 16384 },
      options: {},
      ...(spec.reasoning ? { variants: { low: {}, medium: {}, high: {} } } : {}),
    };
  }
  return catalog;
}

export function normalizeRequestedTextModel(model) {
  let raw = String(model || "").trim();
  if (raw.startsWith("gptfree-pool/")) {
    raw = raw.slice("gptfree-pool/".length).trim();
  }
  return raw || DEFAULT_WEB_TEXT_MODEL;
}

function normalizeThinkingEffort(effort) {
  const val = String(effort || "").trim().toLowerCase();
  if (!val || val === "none" || val === "minimal") return "";
  if (val === "max") return "xhigh";
  if (["low", "medium", "high", "xhigh"].includes(val)) return val;
  return "";
}

function extractMessageContentText(content) {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const parts = [];
  for (const part of content) {
    if (typeof part === "string") {
      if (part) parts.push(part);
      continue;
    }
    if (!part || typeof part !== "object") continue;
    const kind = String(part.type || "").toLowerCase();
    if (
      (kind === "input_text" || kind === "output_text" || kind === "text" || !kind)
      && typeof part.text === "string"
      && part.text
    ) {
      parts.push(part.text);
    }
  }
  return parts.join("\n");
}

function appendTextMessage(messages, role, content) {
  const text = String(content || "");
  if (!text.trim()) return;
  const normalizedRole = ["system", "developer", "user", "assistant"].includes(role)
    ? role
    : "user";
  const prev = messages[messages.length - 1];
  if (prev && prev.role === normalizedRole) {
    prev.content = `${prev.content}\n\n${text}`;
    return;
  }
  messages.push({ role: normalizedRole, content: text });
}

function normalizeWebTools(rawTools, rawToolChoice) {
  if (rawToolChoice === "none" || !Array.isArray(rawTools) || !rawTools.length) {
    return [];
  }
  const normalized = [];
  for (const t of rawTools) {
    if (!t || typeof t !== "object") continue;
    const fn = t.type === "function" && t.function && typeof t.function === "object"
      ? t.function
      : t;
    const name = typeof fn.name === "string" ? fn.name.trim() : "";
    if (!name || name.length > 128) continue;
    normalized.push({
      name,
      description: typeof fn.description === "string" ? fn.description.trim().slice(0, 2000) : "",
      parameters: fn.parameters && typeof fn.parameters === "object" ? fn.parameters : { type: "object", properties: {} },
    });
    if (normalized.length >= 64) break;
  }
  return normalized;
}

function buildToolBridgeSystemPrompt(tools, toolChoice) {
  const specs = tools.map((t) => ({
    name: t.name,
    ...(t.description ? { description: t.description } : {}),
    parameters: t.parameters,
  }));
  const forcedName = toolChoice && typeof toolChoice === "object"
    ? String(toolChoice.name || toolChoice.function?.name || "").trim()
    : "";
  const choiceRule = toolChoice === "required"
    ? "You MUST invoke at least one tool in this turn."
    : forcedName
      ? `You MUST invoke the tool "${forcedName}" in this turn.`
      : "Invoke a tool ONLY when needed; otherwise reply normally.";
  return [
    "## Available Tools",
    "You can call external functions. When calling a function, emit one XML block per call in the exact format:",
    '<tool_call>{"name":"function_name","arguments":{}}</tool_call>',
    "Rules:",
    "- `arguments` MUST be a valid JSON object matching the tool's parameters schema.",
    "- Do not wrap `<tool_call>` inside markdown code fences.",
    `- ${choiceRule}`,
    JSON.stringify(specs),
  ].join("\n");
}

const TOOL_CALL_BLOCK_RE = /<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/g;

export function parseWebToolCalls(rawText, allowedToolNames = null) {
  const source = String(rawText || "");
  if (!source.includes("<tool_call>")) {
    return { text: source, toolCalls: [] };
  }
  const allowed = allowedToolNames instanceof Set ? allowedToolNames : null;
  const toolCalls = [];
  let stripped = "";
  let lastIdx = 0;
  TOOL_CALL_BLOCK_RE.lastIndex = 0;
  let match;
  while ((match = TOOL_CALL_BLOCK_RE.exec(source)) !== null) {
    const inner = match[1].trim();
    let parsed = null;
    try {
      parsed = JSON.parse(inner);
    } catch {
      parsed = null;
    }
    const name = typeof parsed?.name === "string" ? parsed.name.trim() : "";
    if (name && (!allowed || allowed.has(name))) {
      stripped += source.slice(lastIdx, match.index);
      lastIdx = match.index + match[0].length;
      const argsObj = parsed.arguments;
      const argsStr = typeof argsObj === "string"
        ? argsObj
        : JSON.stringify(argsObj && typeof argsObj === "object" ? argsObj : {});
      toolCalls.push({
        id: `fc_${crypto.randomUUID().replace(/-/g, "")}`,
        callId: `call_${crypto.randomUUID().replace(/-/g, "").slice(0, 24)}`,
        name,
        arguments: argsStr,
      });
    }
  }
  stripped += source.slice(lastIdx);
  return {
    text: toolCalls.length ? stripped.trim() : source,
    toolCalls,
  };
}

export function extractWebTextRequest(rawBody) {
  let parsed;
  try {
    parsed = typeof rawBody === "string" || Buffer.isBuffer(rawBody)
      ? JSON.parse(String(rawBody))
      : rawBody;
  } catch {
    throw new WebTextHttpError(400, "Invalid Responses request JSON", "invalid_request");
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new WebTextHttpError(400, "Responses request body must be a JSON object", "invalid_request");
  }

  const requestedModel = String(parsed.model || DEFAULT_WEB_TEXT_MODEL).trim() || DEFAULT_WEB_TEXT_MODEL;
  const model = normalizeRequestedTextModel(requestedModel);
  const stream = parsed.stream === true;
  const thinkingEffort = normalizeThinkingEffort(
    parsed.reasoning?.effort ?? parsed.reasoning_effort ?? "",
  );

  const messages = [];
  const instructions = typeof parsed.instructions === "string" ? parsed.instructions.trim() : "";

  if (typeof parsed.input === "string") {
    appendTextMessage(messages, "user", parsed.input);
  } else if (Array.isArray(parsed.input)) {
    for (const item of parsed.input) {
      if (typeof item === "string") {
        appendTextMessage(messages, "user", item);
        continue;
      }
      if (!item || typeof item !== "object") continue;
      const itemType = String(item.type || "").toLowerCase();
      if (itemType === "reasoning" || itemType === "item_reference") {
        continue;
      }
      if (itemType === "function_call") {
        const fnName = String(item.name || "tool");
        const fnArgs = typeof item.arguments === "string"
          ? item.arguments
          : JSON.stringify(item.arguments ?? {});
        appendTextMessage(messages, "assistant", `[Tool Call: ${fnName}(${fnArgs})]`);
        continue;
      }
      if (itemType === "function_call_output") {
        const callId = String(item.call_id || "").trim();
        const outText = typeof item.output === "string"
          ? item.output
          : JSON.stringify(item.output ?? "");
        appendTextMessage(
          messages,
          "user",
          `[Tool Result${callId ? ` (${callId})` : ""}]:\n${outText}`,
        );
        continue;
      }
      const role = String(item.role || "user").toLowerCase();
      const text = extractMessageContentText(item.content);
      appendTextMessage(messages, role === "tool" ? "user" : role, text);
    }
  }

  if (instructions) {
    const first = messages[0];
    const hasSameSystem = first
      && (first.role === "system" || first.role === "developer")
      && first.content.trim() === instructions;
    if (!hasSameSystem) {
      messages.unshift({ role: "system", content: instructions });
    }
  }

  const tools = normalizeWebTools(parsed.tools, parsed.tool_choice);
  if (tools.length > 0) {
    const bridgePrompt = buildToolBridgeSystemPrompt(tools, parsed.tool_choice);
    if (messages[0] && (messages[0].role === "system" || messages[0].role === "developer")) {
      messages[0] = {
        ...messages[0],
        content: `${messages[0].content}\n\n${bridgePrompt}`,
      };
    } else {
      messages.unshift({ role: "system", content: bridgePrompt });
    }
  }

  if (!messages.length) {
    throw new WebTextHttpError(400, "Request input must contain at least one non-empty message", "invalid_request");
  }

  let estimatedInputTokens = 0;
  for (const m of messages) estimatedInputTokens += estimateTokens(m.content);

  return {
    model,
    requestedModel,
    stream,
    messages,
    thinkingEffort,
    tools,
    estimatedInputTokens: Math.max(1, estimatedInputTokens),
  };
}

function abortError() {
  const err = new Error("Aborted");
  err.name = "AbortError";
  return err;
}

export async function runWebTextWorker(payload, {
  python = workerPython,
  signal,
  timeout = 180_000,
  workerPath = WEB_TEXT_WORKER,
  onDelta,
  daemon = workerPath === WEB_TEXT_WORKER,
} = {}) {
  if (signal?.aborted) throw abortError();
  const command = await python();
  if (daemon) {
    return textDaemonPool.execute(
      command,
      workerPath,
      { ...payload, stream_events: true },
      { signal, timeout, onDelta, mode: "stream" },
    );
  }
  const serialized = JSON.stringify({ ...payload, stream_events: true });

  return new Promise((resolve, reject) => {
    const child = spawn(command, [workerPath, "--stream"], {
      stdio: ["pipe", "pipe", "pipe"],
      shell: false,
      windowsHide: true,
      env: {
        ...process.env,
        PYTHONDONTWRITEBYTECODE: "1",
        PYTHONUNBUFFERED: "1",
        PYTHONIOENCODING: "utf-8",
        PYTHONUTF8: "1",
      },
    });

    let stdoutBuf = "";
    let stderr = "";
    let ended = false;
    let aborted = false;
    let finalResult = null;
    let errorResult = null;
    let accumulatedText = "";

    const cleanup = () => {
      clearTimeout(timer);
      if (signal && onAbort) {
        signal.removeEventListener("abort", onAbort);
      }
    };

    const finishErr = (err) => {
      if (ended) return;
      ended = true;
      cleanup();
      reject(err);
    };

    const finishOk = (value) => {
      if (ended) return;
      ended = true;
      cleanup();
      resolve(value);
    };

    const onAbort = () => {
      aborted = true;
      try { child.kill(); } catch {}
      finishErr(abortError());
    };

    if (signal) {
      signal.addEventListener("abort", onAbort, { once: true });
    }

    const timer = setTimeout(() => {
      try { child.kill(); } catch {}
      finishOk({
        ok: false,
        http_status: 504,
        error_code: "timeout",
        error: "Web text worker timed out",
      });
    }, timeout);

    const processLine = (rawLine) => {
      const line = String(rawLine || "").trim();
      if (!line) return;
      let parsed;
      try {
        parsed = JSON.parse(line);
      } catch {
        return;
      }
      if (!parsed || typeof parsed !== "object") return;
      if (parsed.type === "delta" && typeof parsed.delta === "string" && parsed.delta) {
        accumulatedText += parsed.delta;
        try {
          onDelta?.(parsed.delta, parsed);
        } catch {}
        return;
      }
      if (parsed.type === "done" || parsed.ok === true) {
        finalResult = parsed;
        return;
      }
      if (parsed.type === "error" || parsed.ok === false) {
        errorResult = parsed;
      }
    };

    child.stdout.on("data", (chunk) => {
      stdoutBuf += chunk.toString("utf-8");
      let idx = stdoutBuf.indexOf("\n");
      while (idx !== -1) {
        const line = stdoutBuf.slice(0, idx);
        stdoutBuf = stdoutBuf.slice(idx + 1);
        processLine(line);
        idx = stdoutBuf.indexOf("\n");
      }
    });

    child.stderr.on("data", (chunk) => {
      if (stderr.length < MAX_STDERR_BYTES) {
        stderr += chunk.toString("utf-8").slice(0, MAX_STDERR_BYTES - stderr.length);
      }
    });

    child.on("error", (err) => {
      finishOk({
        ok: false,
        http_status: 502,
        error_code: "worker_spawn_failed",
        error: sanitizeStatsMessage(err.message) || "Failed to launch web text worker",
      });
    });

    child.on("close", (code) => {
      if (ended || aborted) return;
      if (stdoutBuf.trim()) {
        processLine(stdoutBuf);
        stdoutBuf = "";
      }
      if (errorResult) {
        finishOk(errorResult);
        return;
      }
      if (finalResult) {
        finishOk({
          ok: true,
          text: typeof finalResult.text === "string" ? finalResult.text : accumulatedText,
          conversation_id: finalResult.conversation_id || "",
          model: finalResult.model || payload.model || "",
        });
        return;
      }
      finishOk({
        ok: false,
        http_status: 502,
        error_code: "worker_failed",
        error: sanitizeStatsMessage(stderr) || `Web text worker exited with code ${code}`,
      });
    });

    child.stdin.on("error", () => {});
    child.stdin.end(serialized, "utf-8");
  });
}

export function buildResponsesPayload({
  responseId,
  messageId,
  createdAt,
  requestedModel,
  text,
  inputTokens,
  toolCalls = [],
}) {
  const outputText = String(text || "");
  const calls = Array.isArray(toolCalls) ? toolCalls : [];
  let outputTokens = estimateTokens(outputText);
  for (const tc of calls) {
    outputTokens += estimateTokens(tc.name) + estimateTokens(tc.arguments);
  }
  const output = [];
  if (outputText || calls.length === 0) {
    output.push({
      id: messageId,
      type: "message",
      status: "completed",
      role: "assistant",
      content: [
        {
          type: "output_text",
          text: outputText,
          annotations: [],
        },
      ],
    });
  }
  for (const tc of calls) {
    output.push({
      id: tc.id || `fc_${crypto.randomUUID().replace(/-/g, "")}`,
      type: "function_call",
      call_id: tc.callId || `call_${crypto.randomUUID().replace(/-/g, "").slice(0, 24)}`,
      name: tc.name,
      arguments: tc.arguments,
      status: "completed",
    });
  }
  return {
    id: responseId,
    object: "response",
    created_at: createdAt,
    status: "completed",
    model: requestedModel,
    output,
    output_text: outputText,
    usage: {
      input_tokens: inputTokens,
      input_tokens_details: { cached_tokens: 0 },
      output_tokens: outputTokens,
      output_tokens_details: { reasoning_tokens: 0 },
      total_tokens: inputTokens + outputTokens,
    },
  };
}

function errorResponse(status, message, code = "upstream_error", retryAfter = null) {
  const headers = { "content-type": "application/json" };
  if (Number.isFinite(Number(retryAfter)) && Number(retryAfter) > 0) {
    headers["retry-after"] = String(Math.ceil(Number(retryAfter)));
  }
  return new Response(
    JSON.stringify({
      error: {
        message: String(message || "Upstream request failed"),
        type: status === 400 ? "invalid_request_error" : "api_error",
        code,
      },
    }),
    { status, headers },
  );
}

function shouldFailoverResult(res) {
  const status = Number(res?.http_status || 502);
  if ([401, 403, 408, 429, 500, 502, 503, 504].includes(status)) return true;
  if (res?.token_invalid) return true;
  return false;
}

let webTextRoundRobinCursor = 0;

export function orderWebTextAccounts(accounts, {
  preferredSlot = "",
  strategy = "preferred",
  softCapPercent = 80,
  windowMaxCalls = DEFAULT_TEXT_WINDOW_MAX_CALLS,
  windowMs = DEFAULT_TEXT_WINDOW_MS,
  now = Date.now(),
} = {}) {
  const list = Array.isArray(accounts) ? accounts : [];
  if (!list.length) return [];
  const pct = Number(softCapPercent);
  const ratio = Number.isFinite(pct) && pct > 0 && pct <= 100 ? pct / 100 : 0.8;
  const maxCalls = Math.max(1, Math.trunc(Number(windowMaxCalls) || DEFAULT_TEXT_WINDOW_MAX_CALLS));
  const softCapThreshold = ratio >= 1 ? Infinity : Math.max(1, Math.ceil(maxCalls * ratio));

  const healthyFresh = [];
  const healthyStale = [];
  const softCappedFresh = [];
  const softCappedStale = [];
  const cooled = [];

  for (const acc of list) {
    if (!acc) continue;
    if (getSlotCooldownUntil(acc.email) > now) {
      cooled.push(acc);
      continue;
    }
    const calls = getSlotWindowCalls(acc.email, now, windowMs);
    const isFresh = Boolean(acc.accessToken && accessTokenState(acc.accessToken, now).state === "unexpired_unverified");
    if (calls >= softCapThreshold) {
      (isFresh ? softCappedFresh : softCappedStale).push(acc);
    } else {
      (isFresh ? healthyFresh : healthyStale).push(acc);
    }
  }

  const healthy = [...healthyFresh, ...healthyStale];
  const softCapped = [...softCappedFresh, ...softCappedStale];

  const arrangePrimary = (group, updateSticky = false) => {
    if (group.length < 2) {
      if (updateSticky && group.length === 1 && typeof group[0]?.email === "string") {
        stickyTextSlot = accountSlotTag(group[0].email);
      }
      return group;
    }
    if (strategy === "roundRobin") {
      const start = webTextRoundRobinCursor++ % group.length;
      return [...group.slice(start), ...group.slice(0, start)];
    }
    const slotOf = (a) => (typeof a?.email === "string" ? accountSlotTag(a.email) : "");
    let idx = -1;
    if (preferredSlot) {
      const prefIdx = group.findIndex((a) => slotOf(a) === preferredSlot);
      if (prefIdx >= 0 && (!stickyTextSlot || stickyTextSlot === preferredSlot)) {
        idx = prefIdx;
      }
    }
    if (idx < 0 && stickyTextSlot) {
      idx = group.findIndex((a) => slotOf(a) === stickyTextSlot);
    }
    if (idx < 0 && preferredSlot) {
      idx = group.findIndex((a) => slotOf(a) === preferredSlot);
    }
    if (idx < 0) idx = 0;
    if (updateSticky) {
      const nextSticky = slotOf(group[idx]);
      if (nextSticky && nextSticky !== stickyTextSlot) {
        stickyTextSlot = nextSticky;
        saveCooldowns();
      } else if (nextSticky) {
        stickyTextSlot = nextSticky;
      }
    }
    return [...group.slice(idx), ...group.slice(0, idx)];
  };

  if (healthy.length > 0) {
    return [...arrangePrimary(healthy, true), ...softCapped, ...cooled];
  }
  return [...arrangePrimary(softCapped, true), ...cooled];
}

/**
 * ChatGPT Web patches arrive a few hundred characters at a time, roughly
 * every 250ms, so forwarding them as-is makes clients jump paragraph by
 * paragraph. Re-emit each upstream chunk as small pieces spread over less
 * than one upstream interval, like token deltas from a normal API. The
 * first piece goes out immediately, so no latency is added up front.
 */
export function createDeltaPacer(emit, { tickMs = 15, targetTicks = 12, minChars = 2 } = {}) {
  let queue = "";
  let timer = null;
  let waiters = [];
  const settle = () => {
    const list = waiters;
    waiters = [];
    for (const resolve of list) resolve();
  };
  const step = () => {
    timer = null;
    if (!queue) {
      settle();
      return;
    }
    let cut = Math.min(queue.length, Math.max(minChars, Math.ceil(queue.length / targetTicks)));
    // Never split a UTF-16 surrogate pair.
    const code = queue.charCodeAt(cut - 1);
    if (cut < queue.length && code >= 0xd800 && code <= 0xdbff) cut++;
    const piece = queue.slice(0, cut);
    queue = queue.slice(cut);
    emit(piece);
    timer = setTimeout(step, tickMs);
  };
  return {
    push(text) {
      if (!text) return;
      queue += text;
      if (!timer) step();
    },
    /** Resolves once every queued piece has been emitted. */
    drain() {
      if (!queue && !timer) return Promise.resolve();
      return new Promise((resolve) => waiters.push(resolve));
    },
    /** Emit whatever is left right away (errors, cancellation). */
    flush() {
      if (timer) clearTimeout(timer);
      timer = null;
      if (queue) {
        const rest = queue;
        queue = "";
        emit(rest);
      }
      settle();
    },
  };
}

function encodeSseFrame(eventName, payload) {
  return `event: ${eventName}\ndata: ${JSON.stringify(payload)}\n\n`;
}

export async function executeWebTextGeneration(rawBody, accounts, {
  reauthenticate,
  signal,
  worker = runWebTextWorker,
  proxy = "",
  maxSlowFailovers = MAX_SLOW_FAILOVERS,
  maxTotalMs = MAX_TOTAL_PROBE_MS,
  preferredSlot = "",
  strategy = "preferred",
  softCapPercent = 80,
  windowMaxCalls = DEFAULT_TEXT_WINDOW_MAX_CALLS,
  windowMs = DEFAULT_TEXT_WINDOW_MS,
} = {}) {
  let reqSpec;
  try {
    reqSpec = extractWebTextRequest(rawBody);
  } catch (err) {
    if (err instanceof WebTextHttpError) {
      return errorResponse(err.status, err.message, err.code);
    }
    return errorResponse(400, "Invalid Responses request JSON", "invalid_request");
  }

  const ordered = orderWebTextAccounts(accounts, {
    preferredSlot,
    strategy,
    softCapPercent,
    windowMaxCalls,
    windowMs,
  });
  if (!ordered.length) {
    return errorResponse(
      503,
      "GPT号池 没有可用账号，请先批量导入账号。",
      "pool_empty",
    );
  }

  const deadline = Date.now() + maxTotalMs;
  let slowFailovers = 0;
  let lastError = null;
  let sawRateLimit = false;
  let maxRetryAfter = null;

  const attemptAccount = (account, allowReauth) => new Promise((resolveOutcome) => {
    if (signal?.aborted) {
      resolveOutcome({ kind: "abort" });
      return;
    }

    const attemptAbort = new AbortController();
    const onCallerAbort = () => attemptAbort.abort();
    if (signal) {
      signal.addEventListener("abort", onCallerAbort, { once: true });
    }

    const responseId = `resp_${crypto.randomUUID().replace(/-/g, "")}`;
    const messageId = `msg_${crypto.randomUUID().replace(/-/g, "")}`;
    const createdAt = Math.floor(Date.now() / 1000);

    let streamStarted = false;
    let streamClosed = false;
    let sseController = null;
    let seq = 0;
    let streamedText = "";
    const encoder = new TextEncoder();

    const cleanupSignal = () => {
      if (signal) signal.removeEventListener("abort", onCallerAbort);
    };

    const pushSse = (eventName, payload) => {
      if (!sseController || streamClosed) return;
      try {
        sseController.enqueue(encoder.encode(encodeSseFrame(eventName, payload)));
      } catch {}
    };

    const closeSse = () => {
      if (streamClosed) return;
      streamClosed = true;
      cleanupSignal();
      if (sseController) {
        try { sseController.close(); } catch {}
      }
    };

    const hasTools = Array.isArray(reqSpec.tools) && reqSpec.tools.length > 0;
    const allowedToolSet = hasTools ? new Set(reqSpec.tools.map((t) => t.name)) : null;
    let messageItemOpened = false;
    let emittedPlainLength = 0;

    const openMessageItemIfNeeded = () => {
      if (messageItemOpened) return;
      messageItemOpened = true;
      pushSse("response.output_item.added", {
        type: "response.output_item.added",
        sequence_number: seq++,
        output_index: 0,
        item: {
          id: messageId,
          type: "message",
          status: "in_progress",
          role: "assistant",
          content: [],
        },
      });
      pushSse("response.content_part.added", {
        type: "response.content_part.added",
        sequence_number: seq++,
        item_id: messageId,
        output_index: 0,
        content_index: 0,
        part: { type: "output_text", text: "", annotations: [] },
      });
    };

    const ensureStreamResponse = () => {
      if (streamStarted || !reqSpec.stream) return;
      streamStarted = true;

      const stream = new ReadableStream({
        start(controller) {
          sseController = controller;
          const inProgressResponse = {
            id: responseId,
            object: "response",
            created_at: createdAt,
            status: "in_progress",
            model: reqSpec.requestedModel,
            output: [],
          };
          pushSse("response.created", {
            type: "response.created",
            sequence_number: seq++,
            response: inProgressResponse,
          });
          pushSse("response.in_progress", {
            type: "response.in_progress",
            sequence_number: seq++,
            response: inProgressResponse,
          });
          if (!hasTools) {
            openMessageItemIfNeeded();
          }
        },
        cancel() {
          streamClosed = true;
          pacer.flush();
          attemptAbort.abort();
          cleanupSignal();
        },
      });

      resolveOutcome({
        kind: "response",
        response: new Response(stream, {
          status: 200,
          headers: {
            "content-type": "text/event-stream; charset=utf-8",
            "cache-control": "no-cache, no-transform",
            "x-accel-buffering": "no",
          },
        }),
      });
    };

    const pacer = createDeltaPacer((delta) => {
      openMessageItemIfNeeded();
      pushSse("response.output_text.delta", {
        type: "response.output_text.delta",
        sequence_number: seq++,
        item_id: messageId,
        output_index: 0,
        content_index: 0,
        delta,
      });
    });

    const flushSafePlainPrefix = () => {
      if (!hasTools) return;
      // Hold back any `<` tail that could be the start of `<tool_call>` so raw
      // XML tool blocks never leak into user-visible text deltas.
      const tagIdx = streamedText.indexOf("<tool_call>", emittedPlainLength);
      if (tagIdx >= 0) {
        if (tagIdx > emittedPlainLength) {
          pacer.push(streamedText.slice(emittedPlainLength, tagIdx));
        }
        emittedPlainLength = streamedText.length;
        return;
      }
      const lastLt = streamedText.lastIndexOf("<");
      let safeEnd = streamedText.length;
      if (lastLt >= emittedPlainLength) {
        const tail = streamedText.slice(lastLt);
        if ("<tool_call>".startsWith(tail)) {
          safeEnd = lastLt;
        }
      }
      if (safeEnd > emittedPlainLength) {
        pacer.push(streamedText.slice(emittedPlainLength, safeEnd));
        emittedPlainLength = safeEnd;
      }
    };

    const workerPromise = worker(
      {
        access_token: account.accessToken,
        model: reqSpec.model,
        messages: reqSpec.messages,
        thinking_effort: reqSpec.thinkingEffort,
        proxy,
      },
      {
        signal: attemptAbort.signal,
        onDelta(delta) {
          if (!delta || streamClosed) return;
          streamedText += delta;
          if (!reqSpec.stream) return;
          ensureStreamResponse();
          if (!hasTools) {
            pacer.push(delta);
          } else if (!streamedText.includes("<tool_call>")) {
            flushSafePlainPrefix();
          }
        },
      },
    );

    workerPromise.then(
      (res) => {
        if (attemptAbort.signal.aborted && !streamStarted) {
          cleanupSignal();
          resolveOutcome({ kind: "abort" });
          return;
        }

        if (res?.ok) {
          recordSlotTextSuccess(account.email, Date.now(), windowMs);
          const rawFinalText = typeof res.text === "string" ? res.text : streamedText;
          const parsedTools = hasTools
            ? parseWebToolCalls(rawFinalText, allowedToolSet)
            : { text: rawFinalText, toolCalls: [] };
          const finalText = parsedTools.text;
          const toolCalls = parsedTools.toolCalls;

          if (!reqSpec.stream) {
            cleanupSignal();
            const jsonBody = buildResponsesPayload({
              responseId,
              messageId,
              createdAt,
              requestedModel: reqSpec.requestedModel,
              text: finalText,
              inputTokens: reqSpec.estimatedInputTokens,
              toolCalls,
            });
            resolveOutcome({
              kind: "response",
              response: new Response(JSON.stringify(jsonBody), {
                status: 200,
                headers: { "content-type": "application/json" },
              }),
            });
            return;
          }

          ensureStreamResponse();
          if (!hasTools) {
            if (finalText.length > streamedText.length && finalText.startsWith(streamedText)) {
              pacer.push(finalText.slice(streamedText.length));
            }
          } else if (toolCalls.length === 0) {
            if (finalText.length > emittedPlainLength) {
              pacer.push(finalText.slice(emittedPlainLength));
            }
          }
          void pacer.drain().then(() => {
            const completedResponse = buildResponsesPayload({
              responseId,
              messageId,
              createdAt,
              requestedModel: reqSpec.requestedModel,
              text: finalText,
              inputTokens: reqSpec.estimatedInputTokens,
              toolCalls,
            });

            let nextOutputIndex = 0;
            if (finalText || toolCalls.length === 0 || messageItemOpened) {
              openMessageItemIfNeeded();
              pushSse("response.output_text.done", {
                type: "response.output_text.done",
                sequence_number: seq++,
                item_id: messageId,
                output_index: 0,
                content_index: 0,
                text: finalText,
              });
              pushSse("response.content_part.done", {
                type: "response.content_part.done",
                sequence_number: seq++,
                item_id: messageId,
                output_index: 0,
                content_index: 0,
                part: { type: "output_text", text: finalText, annotations: [] },
              });
              const msgItem = completedResponse.output.find((it) => it.type === "message") || {
                id: messageId,
                type: "message",
                status: "completed",
                role: "assistant",
                content: [{ type: "output_text", text: finalText, annotations: [] }],
              };
              pushSse("response.output_item.done", {
                type: "response.output_item.done",
                sequence_number: seq++,
                output_index: 0,
                item: msgItem,
              });
              nextOutputIndex = 1;
            }

            const fcItems = completedResponse.output.filter((it) => it.type === "function_call");
            for (const fcItem of fcItems) {
              const outIdx = nextOutputIndex++;
              pushSse("response.output_item.added", {
                type: "response.output_item.added",
                sequence_number: seq++,
                output_index: outIdx,
                item: { ...fcItem, arguments: "", status: "in_progress" },
              });
              pushSse("response.function_call_arguments.delta", {
                type: "response.function_call_arguments.delta",
                sequence_number: seq++,
                item_id: fcItem.id,
                output_index: outIdx,
                delta: fcItem.arguments,
              });
              pushSse("response.function_call_arguments.done", {
                type: "response.function_call_arguments.done",
                sequence_number: seq++,
                item_id: fcItem.id,
                output_index: outIdx,
                arguments: fcItem.arguments,
              });
              pushSse("response.output_item.done", {
                type: "response.output_item.done",
                sequence_number: seq++,
                output_index: outIdx,
                item: fcItem,
              });
            }

            pushSse("response.completed", {
              type: "response.completed",
              sequence_number: seq++,
              response: completedResponse,
            });
            closeSse();
          });
          return;
        }

        pacer.flush();
        const errStatus = Number(res?.http_status || 502);
        const errMsg = sanitizeStatsMessage(res?.error) || "Upstream text generation failed";
        const errCode = String(res?.error_code || "upstream_error");

        if (streamStarted) {
          pushSse("error", {
            type: "error",
            error: {
              message: errMsg,
              code: errCode,
              type: "api_error",
            },
          });
          closeSse();
          return;
        }

        cleanupSignal();
        resolveOutcome({
          kind: "retryable_or_fatal",
          allowReauth,
          result: {
            http_status: errStatus,
            error: errMsg,
            error_code: errCode,
            token_invalid: Boolean(res?.token_invalid || errStatus === 401),
            retry_after: res?.retry_after ?? null,
          },
        });
      },
      (err) => {
        cleanupSignal();
        pacer.flush();
        if (err?.name === "AbortError") {
          if (streamStarted) {
            closeSse();
            return;
          }
          resolveOutcome({ kind: "abort" });
          return;
        }
        const errMsg = sanitizeStatsMessage(err?.message) || "Web text worker error";
        if (streamStarted) {
          pushSse("error", {
            type: "error",
            error: {
              message: errMsg,
              code: "worker_error",
              type: "api_error",
            },
          });
          closeSse();
          return;
        }
        resolveOutcome({
          kind: "retryable_or_fatal",
          allowReauth,
          result: {
            http_status: Number(err?.status || 502),
            error: errMsg,
            error_code: String(err?.code || "worker_error"),
            token_invalid: Number(err?.status) === 401,
            retry_after: err?.retryAfter ?? null,
          },
        });
      },
    );
  });

  for (const original of ordered) {
    if (signal?.aborted) {
      throw abortError();
    }
    if (Date.now() > deadline) {
      break;
    }

    let account = original;
    const staleOrMissing = !account.accessToken
      || accessTokenState(account.accessToken).state !== "unexpired_unverified";

    if (staleOrMissing) {
      if (slowFailovers >= maxSlowFailovers || typeof reauthenticate !== "function") {
        continue;
      }
      slowFailovers += 1;
      try {
        account = await reauthenticate(original);
      } catch (err) {
        lastError = {
          http_status: 401,
          error_code: "reauth_failed",
          error: sanitizeStatsMessage(err?.message) || "账号重新登录失败",
        };
        continue;
      }
    }

    let attemptedReauthForAccount = false;
    for (let attempt = 0; attempt < 2; attempt++) {
      if (!account?.accessToken) break;
      const outcome = await attemptAccount(account, attempt === 0 && !attemptedReauthForAccount);

      if (outcome.kind === "abort") {
        throw abortError();
      }
      if (outcome.kind === "response") {
        return outcome.response;
      }

      const res = outcome.result;
      lastError = res;
      const status = Number(res.http_status || 502);

      if (
        (status === 401 || res.token_invalid)
        && outcome.allowReauth
        && typeof reauthenticate === "function"
        && slowFailovers < maxSlowFailovers
      ) {
        slowFailovers += 1;
        attemptedReauthForAccount = true;
        try {
          account = await reauthenticate(account);
          continue;
        } catch (err) {
          lastError = {
            http_status: 401,
            error_code: "reauth_failed",
            error: sanitizeStatsMessage(err?.message) || res.error,
          };
          break;
        }
      }

      if (status === 429) {
        sawRateLimit = true;
        const retrySec = Number(res.retry_after);
        if (Number.isFinite(retrySec) && retrySec > 0) {
          maxRetryAfter = Math.max(maxRetryAfter || 0, Math.ceil(retrySec));
        }
        setSlotCooldown(account.email, retrySec);
      }

      if (!shouldFailoverResult(res)) {
        return errorResponse(status, res.error, res.error_code, res.retry_after);
      }
      break;
    }
  }

  if (sawRateLimit) {
    return errorResponse(
      429,
      lastError?.error || "所有 GPT号池 账号当前触发上游限流，请稍后再试或导入更多账号。",
      "rate_limit_exceeded",
      maxRetryAfter,
    );
  }
  const finalStatus = Number(lastError?.http_status || 503);
  return errorResponse(
    [400, 401, 403, 429, 500, 502, 503, 504].includes(finalStatus) ? finalStatus : 503,
    lastError?.error || "所有 GPT号池 账号均不可用；请检查后台 HTTP 登录、目标模型授权及账号权限。",
    lastError?.error_code || "pool_exhausted",
    lastError?.retry_after ?? null,
  );
}
