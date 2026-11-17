/**
 * Export the local token pool as CLIProxyAPI (CPA) Codex auth files.
 *
 * CPA reads one JSON per account under its auth dir (or through
 * POST /v0/management/auth-files). The pool only stores Platform web access
 * tokens, so the exported files carry no refresh_token/id_token: CPA can use
 * them until the access token expires, but cannot renew them by itself.
 */
import { buildZip } from "./zip.mjs";
import { isWebOnlyToken } from "./codex-pool.mjs";

const UNSAFE_NAME = /[^a-z0-9@._-]+/g;

function jwtClaims(token) {
  try {
    const raw = String(token || "").split(".")[1];
    if (!raw || raw.length > 65536) return null;
    const payload = JSON.parse(Buffer.from(raw, "base64url").toString("utf8"));
    return payload && typeof payload === "object" && !Array.isArray(payload) ? payload : null;
  } catch {
    return null;
  }
}

export function cpaFileName(email) {
  const norm = String(email || "").trim().toLowerCase();
  const safe = norm.replace(UNSAFE_NAME, "_").replace(/^_+|_+$/g, "").slice(0, 120);
  return `codex-${safe || "account"}.json`;
}

export function cpaAuthFile(record, { now = new Date() } = {}) {
  const claims = jwtClaims(record?.accessToken);
  const auth = claims?.["https://api.openai.com/auth"];
  const exp = typeof claims?.exp === "number" && Number.isFinite(claims.exp)
    ? new Date(claims.exp * 1000)
    : null;
  return {
    type: "codex",
    email: String(record?.email || "").trim().toLowerCase(),
    access_token: String(record?.accessToken || ""),
    // The pool never persists refresh/id tokens; CPA will not be able to renew.
    refresh_token: "",
    id_token: "",
    account_id: typeof auth?.chatgpt_account_id === "string" ? auth.chatgpt_account_id : "",
    last_refresh: now.toISOString(),
    expired: exp && !Number.isNaN(exp.getTime()) ? exp.toISOString() : "",
    plan_type: typeof auth?.chatgpt_plan_type === "string" && auth.chatgpt_plan_type
      ? auth.chatgpt_plan_type
      : "free",
  };
}

function readmeText(count, webOnly) {
  return [
    "GPT号池 -> CLIProxyAPI (CPA) 导出",
    "",
    `账号数：${count}（另有 ${webOnly} 个 Platform Web token 未导出）`,
    "格式：每个 codex-<邮箱>.json 是一个 CPA Codex 认证文件。",
    "",
    "导入方式（二选一）：",
    "1. 管理页：CLI Proxy API Management Center -> Auth Files -> 上传这些 JSON；",
    "2. 命令行：POST /v0/management/auth-files?name=<文件名>，Body 为该 JSON。",
    "",
    "已验证的限制：",
    "- ChatGPT Web/Platform access_token 只能读取 /codex/models，",
    "  对 /codex/responses 实际生成请求返回 401 Unauthorized；",
    "  这类账号不会出现在本压缩包里。",
    "- 文件里没有 refresh_token，CPA 无法自动续期；token 到期后需要重新导出。",
    "- 文件含明文凭证，请只放在本机或可信的 CPA 实例，不要提交到仓库/网盘。",
    "",
  ].join("\n");
}

export function buildCpaExport(records, { now = new Date(), readme = true } = {}) {
  const files = [];
  const seen = new Set();
  let skipped = 0;
  let webOnly = 0;
  for (const record of Array.isArray(records) ? records : []) {
    if (!record || typeof record.email !== "string" || !record.email.includes("@") ||
        typeof record.accessToken !== "string" || !record.accessToken) {
      skipped++;
      continue;
    }
    // Verified 2026-11-17: Platform web tokens list /codex/models but get a
    // 401 from /codex/responses, so CPA's codex provider cannot serve them.
    if (isWebOnlyToken(record.accessToken)) {
      webOnly++;
      continue;
    }
    const name = cpaFileName(record.email);
    if (seen.has(name)) {
      skipped++;
      continue;
    }
    seen.add(name);
    files.push({ name, data: JSON.stringify(cpaAuthFile(record, { now }), null, 2) + "\n" });
  }
  if (readme) files.push({ name: "README.txt", data: readmeText(files.length, webOnly) });
  return { files, skipped, webOnly };
}

export function buildCpaZip(records, options = {}) {
  const { files, skipped, webOnly } = buildCpaExport(records, options);
  return { zip: buildZip(files, { date: options.now || new Date() }), files, skipped, webOnly };
}
