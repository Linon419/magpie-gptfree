/**
 * Background token renewal for the local pool.
 *
 * The request path already re-logs in an account when a call gets a 401 or the
 * stored token is expired. That is lazy: a token that expires while nobody is
 * using the pool stays dead until the next request. This scheduler closes that
 * gap by scanning the vault, renewing tokens that are expired or about to
 * expire (password + TOTP, same worker as the importer), and syncing the
 * renewed accounts back to Magpie's providers.
 *
 * It is deliberately small: a few accounts per tick, a retry backoff per
 * account, and no work at all when every token is still fresh.
 */
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";

import { accessTokenExpiryMs, accessTokenState, safeAccountName } from "./credentials.mjs";
import { renewAccount } from "./codex-pool.mjs";
import { listPool, poolDir } from "./pool-vault.mjs";

const SETTINGS = "auto-refresh.json";

export async function readAutoRefreshSettings({ root = poolDir() } = {}) {
  try {
    const parsed = JSON.parse(await readFile(join(root, SETTINGS), "utf8"));
    return {
      proxy: typeof parsed?.proxy === "string" ? parsed.proxy : "",
      enabled: parsed?.enabled !== false,
    };
  } catch {
    return { proxy: "", enabled: true };
  }
}

export async function writeAutoRefreshSettings({ proxy, enabled } = {}, { root = poolDir() } = {}) {
  const current = await readAutoRefreshSettings({ root });
  const next = {
    proxy: typeof proxy === "string" ? proxy : current.proxy,
    enabled: typeof enabled === "boolean" ? enabled : current.enabled,
  };
  await mkdir(root, { recursive: true, mode: 0o700 });
  await writeFile(join(root, SETTINGS), JSON.stringify(next, null, 2) + "\n", { mode: 0o600 });
  return next;
}

function isDue(row, now, leadMs) {
  const state = accessTokenState(row.accessToken, now).state;
  if (state !== "unexpired_unverified") return true;
  const exp = accessTokenExpiryMs(row.accessToken);
  return exp > 0 && exp - now <= leadMs;
}

/**
 * Start the scheduler. Returns a handle with `tick()` (for tests/one-off runs)
 * and `stop()`. The interval timer is unref'd so it never keeps a CLI alive.
 */
export function startAutoRefresh({
  read = listPool,
  renew = renewAccount,
  onRenewed = null,
  log = null,
  now = Date.now,
  intervalMs = 5 * 60 * 1000,
  leadMs = 60 * 60 * 1000,
  maxPerTick = 6,
  retryMs = 30 * 60 * 1000,
  getProxy = async () => "",
  immediate = true,
} = {}) {
  let timer = null;
  let running = false;
  let stopped = false;
  const retryAt = new Map();
  const stats = { renewed: 0, failed: 0, lastRun: 0, lastError: "" };

  function dueRows(rows) {
    const due = [];
    for (const row of Array.isArray(rows) ? rows : []) {
      if (!row || typeof row.email !== "string" || typeof row.accessToken !== "string") continue;
      if (!isDue(row, now(), leadMs)) continue;
      if ((retryAt.get(row.email) || 0) > now()) continue;
      const exp = accessTokenExpiryMs(row.accessToken);
      due.push({ row, exp });
    }
    due.sort((a, b) => (a.exp || 0) - (b.exp || 0));
    return due.map((item) => item.row);
  }

  async function tick() {
    if (running || stopped) return stats;
    running = true;
    stats.lastRun = now();
    try {
      const proxy = String((await getProxy()) || "");
      const rows = await read();
      const due = dueRows(rows).slice(0, maxPerTick);
      for (const account of due) {
        if (stopped) break;
        try {
          const updated = await renew(account, proxy ? { proxy } : undefined);
          stats.renewed++;
          retryAt.delete(account.email);
          try { await onRenewed?.(updated, account); } catch {}
        } catch (error) {
          stats.failed++;
          retryAt.set(account.email, now() + retryMs);
          stats.lastError = String(error?.message || error).slice(0, 160);
          try {
            await log?.("warn", `自动续期失败：${safeAccountName(account.email)}（${stats.lastError}）`);
          } catch {}
        }
      }
      return stats;
    } catch (error) {
      stats.lastError = String(error?.message || error).slice(0, 160);
      return stats;
    } finally {
      running = false;
    }
  }

  timer = setInterval(() => { void tick(); }, intervalMs);
  timer.unref?.();
  if (immediate) void tick();

  return {
    tick,
    stats,
    stop() {
      stopped = true;
      if (timer) clearInterval(timer);
      timer = null;
    },
  };
}
