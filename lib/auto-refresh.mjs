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
      intervalMinutes: Number.isFinite(parsed?.intervalMinutes) && parsed.intervalMinutes >= 5
        ? Math.min(24 * 60, Math.trunc(parsed.intervalMinutes))
        : 60,
      leadHours: Number.isFinite(parsed?.leadHours) && parsed.leadHours >= 1
        ? Math.min(24 * 7, Math.trunc(parsed.leadHours))
        : 48,
    };
  } catch {
    return { proxy: "", enabled: true, intervalMinutes: 60, leadHours: 48 };
  }
}

export async function writeAutoRefreshSettings({ proxy, enabled, intervalMinutes, leadHours } = {}, { root = poolDir() } = {}) {
  const current = await readAutoRefreshSettings({ root });
  const next = {
    proxy: typeof proxy === "string" ? proxy : current.proxy,
    enabled: typeof enabled === "boolean" ? enabled : current.enabled,
    intervalMinutes: Number.isFinite(intervalMinutes) ? Math.trunc(intervalMinutes) : current.intervalMinutes,
    leadHours: Number.isFinite(leadHours) ? Math.trunc(leadHours) : current.leadHours,
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
  intervalMs = 60 * 60 * 1000,
  leadMs = 48 * 60 * 60 * 1000,
  maxPerTick = 100,
  concurrency = 3,
  catchUpMs = 5 * 60 * 1000,
  retryMs = 30 * 60 * 1000,
  getProxy = async () => "",
  immediate = true,
} = {}) {
  let timer = null;
  let running = false;
  let stopped = false;
  const retryAt = new Map();
  const stats = { renewed: 0, failed: 0, lastRun: 0, lastError: "", remaining: 0 };

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
      const due = dueRows(rows);
      if (!due.length) return stats;
      // Size the batch from the backlog and the time left before the earliest
      // expiry, so a clustered pool drains instead of trickling 6 per tick.
      const earliest = due.reduce((min, row) => {
        const exp = accessTokenExpiryMs(row.accessToken);
        return exp > 0 ? Math.min(min, exp) : min;
      }, Infinity);
      const ticksLeft = Number.isFinite(earliest)
        ? Math.max(1, Math.floor((earliest - now()) / intervalMs))
        : 1;
      const budget = Math.max(1, Math.min(maxPerTick, Math.ceil(due.length / ticksLeft)));
      const batch = due.slice(0, budget);
      // The per-tick cap can be smaller than what the configured interval and
      // lead allow. Keep the remainder visible so the timer can come back in
      // minutes instead of waiting a whole interval and missing the deadline.
      stats.remaining = due.length - batch.length;
      if (stats.remaining > 0) {
        try {
          await log?.("warn", `自动续期本轮处理 ${batch.length} 个，仍有 ${stats.remaining} 个待续期，${Math.round(catchUpMs / 60000)} 分钟后继续`);
        } catch {}
      }
      let cursor = 0;
      const limit = Math.max(1, Math.min(Math.trunc(Number(concurrency) || 3), batch.length));
      const runners = Array.from({ length: limit }, async () => {
        while (cursor < batch.length && !stopped) {
          const account = batch[cursor++];
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
      });
      await Promise.all(runners);
      return stats;
    } catch (error) {
      stats.lastError = String(error?.message || error).slice(0, 160);
      return stats;
    } finally {
      running = false;
    }
  }

  function schedule(delay) {
    if (stopped) return;
    timer = setTimeout(async () => {
      const result = await tick();
      // Backlog left after hitting the cap: retry soon, not after a full
      // interval that may be longer than the remaining token lifetime.
      schedule(result.remaining > 0 ? Math.min(catchUpMs, intervalMs) : intervalMs);
    }, delay);
    timer.unref?.();
  }

  schedule(immediate ? 0 : intervalMs);

  return {
    tick,
    stats,
    stop() {
      stopped = true;
      if (timer) clearTimeout(timer);
      timer = null;
    },
  };
}
