import { readFile } from "node:fs/promises";
import { parseAccounts, safeAccountName, validateProxyUrl } from "./credentials.mjs";
import { loginHTTPAccount } from "./dsh-http-auth.mjs";
import { probeAccountImageQuota, verifyAccessToken } from "./codex-pool.mjs";
import { accountSlotTag, formatAccountSlotLabel } from "./pool-connection.mjs";
import { defaultStatsFile } from "./pool-stats.mjs";
import { upsertPool } from "./pool-vault.mjs";

/**
 * DSH-style HTTP authentication with a separate Magpie token pool.
 * No Codex CLI command, no browser launch, no DSH service dependency.
 */
export async function runBatch({
  file, consent = false, dryRun = false, proxy = "", skipInvalid = false,
  mode = "pool", onProgress = () => {}, signal,
  read = readFile, login = loginHTTPAccount,
  verifyToken = verifyAccessToken, savePool = upsertPool,
  checkQuota, statsFile, statsStore, concurrency = login === loginHTTPAccount ? 4 : 1,
} = {}) {
  if (!file || typeof file !== "string") throw new Error("请指定 TXT 文件的完整路径");
  if (mode !== "pool") throw new Error("已移除 Codex CLI 原生导入，请使用 GPT号池");
  const raw = await read(file, "utf8");
  const parsed = parseAccounts(raw);
  if (dryRun) {
    return {
      total: parsed.accounts.length, duplicates: parsed.duplicates,
      invalid: parsed.errors, added: 0, failed: [], mode,
      ready: parsed.accounts.map((a) => ({
        line: a.line, email: safeAccountName(a.email),
        has2FA: !!a.totp, kind: a.kind,
        tokenState: a.kind === "access_token" ? a.tokenState : undefined,
      })),
    };
  }
  if (!consent) throw new Error("导入真实账号前必须确认账号的使用授权");
  if (!parsed.accounts.length) throw new Error("TXT 中没有有效账号");
  if (parsed.errors.length && !skipInvalid) throw new Error("账号列表存在格式错误，请先检查导入预览");
  const proxyCheck = validateProxyUrl(proxy);
  if (proxyCheck.error) throw new Error(proxyCheck.error);
  const safeProxy = proxyCheck.proxy;
  const quotaChecker = checkQuota !== undefined
    ? checkQuota
    : (savePool === upsertPool && verifyToken === verifyAccessToken && login === loginHTTPAccount
      ? probeAccountImageQuota
      : null);
  const quotaTarget = (statsFile !== undefined || statsStore !== undefined)
    ? {
        ...(statsFile !== undefined ? { statsFile } : {}),
        ...(statsStore !== undefined ? { statsStore } : {}),
      }
    : (savePool === upsertPool ? { statsFile: defaultStatsFile() } : { statsFile: null });
  const result = {
    total: parsed.accounts.length, added: 0, reused: 0,
    mode, failed: [], duplicates: parsed.duplicates,
    invalid: skipInvalid ? parsed.errors.slice(0, 100).map(({ line, reason }) => ({ line, reason })) : [],
    formats: { standard: 0, mail_link: 0, access_token: 0 },
    tokenNeedsOAuth: 0,
  };
  // Concurrent network login/probe with strictly in-order vault commit so
  // multi-account TXT imports run up to 4x faster without scrambling row order.
  const prepareAccount = async (account) => {
    if (signal?.aborted) throw new Error("批量导入已取消");
    const safe = { email: safeAccountName(account.email), line: account.line };
    const slotMeta = { slotTag: accountSlotTag(account.email), accountLabel: formatAccountSlotLabel(account.email) };
    const preEvents = [];
    if (account.kind === "mail_link") {
      preEvents.push({ state: "mail_link_ignored", ...safe });
    }
    try {
      if (account.kind === "access_token") {
        if (account.tokenState === "unexpired_unverified") {
          const probe = await verifyToken(account.accessToken, { expectedEmail: account.email, signal });
          if (probe.accepted) {
            return {
              kind: "reused",
              account,
              safe,
              slotMeta,
              preEvents,
              savedRecord: {
                email: account.email, password: account.password,
                totp: account.totp, accessToken: account.accessToken,
              },
            };
          }
        }
        preEvents.push({
          state: account.tokenState === "expired" ? "token_expired" : "token_needs_oauth",
          ...safe,
          needsOAuth: true,
        });
      }
      preEvents.push({ state: "authorizing", ...safe });
      const { tokens } = await login(account, { signal, proxy: safeProxy });
      const probe = await verifyToken(tokens.access, { expectedEmail: account.email, signal });
      if (!probe.accepted) throw new Error("DSH HTTP 登录成功，但 Platform Token 未通过 Magpie 模型访问验证");
      return {
        kind: "added",
        account,
        safe,
        slotMeta,
        preEvents,
        savedRecord: {
          email: account.email, password: account.password,
          totp: account.totp, accessToken: tokens.access,
        },
      };
    } catch (err) {
      if (signal?.aborted) throw new Error("批量导入已取消");
      const message = String(err?.message || "");
      const known = /DSH HTTP|Platform Token|没有返回可用|未找到可用 Python|HTTP 登录|认证后端/.test(message);
      const reason = known ? message.slice(0, 180) :
        "后台 HTTP 授权失败；请检查账号、TOTP、网络或代理。不会自动打开登录网页";
      return {
        kind: "failed",
        account,
        safe,
        preEvents,
        reason,
      };
    }
  };

  const commitPrepared = async (outcome) => {
    if (signal?.aborted) throw new Error("批量导入已取消");
    result.formats[outcome.account.kind]++;
    for (const ev of outcome.preEvents) {
      if (ev.needsOAuth) {
        result.tokenNeedsOAuth++;
        const { needsOAuth: _, ...cleanEv } = ev;
        onProgress(cleanEv);
      } else {
        onProgress(ev);
      }
    }
    if (outcome.kind === "reused") {
      await savePool(outcome.savedRecord);
      result.added++;
      result.reused++;
      onProgress({ state: "token_reused", ...outcome.safe, ...outcome.slotMeta });
      if (typeof quotaChecker === "function") {
        await quotaChecker(outcome.savedRecord, {
          proxy: safeProxy, signal, ...quotaTarget, reauthenticate: null,
        }).catch(() => {});
      }
      return;
    }
    if (outcome.kind === "added") {
      await savePool(outcome.savedRecord);
      result.added++;
      onProgress({ state: "added", ...outcome.safe, ...outcome.slotMeta });
      if (typeof quotaChecker === "function") {
        await quotaChecker(outcome.savedRecord, {
          proxy: safeProxy, signal, ...quotaTarget, reauthenticate: null,
        }).catch(() => {});
      }
      return;
    }
    result.failed.push({ line: outcome.account.line, email: outcome.safe.email, reason: outcome.reason });
    onProgress({ state: "failed", ...outcome.safe, reason: outcome.reason });
  };

  if (parsed.accounts.length > 0) {
    // Settle the first account immediately for fast native sign-in & early abort.
    const firstOutcome = await prepareAccount(parsed.accounts[0]);
    await commitPrepared(firstOutcome);

    if (parsed.accounts.length > 1) {
      if (signal?.aborted) throw new Error("批量导入已取消");
      const total = parsed.accounts.length;
      const readyOutcomes = new Array(total);
      let fetchCursor = 1;
      let commitCursor = 1;
      let committing = false;

      const drainCommits = async () => {
        if (committing) return;
        committing = true;
        try {
          while (commitCursor < total && readyOutcomes[commitCursor] !== undefined) {
            const item = readyOutcomes[commitCursor];
            readyOutcomes[commitCursor] = undefined;
            commitCursor++;
            await commitPrepared(item);
          }
        } finally {
          committing = false;
        }
      };

      const limit = Math.max(1, Math.min(Math.trunc(Number(concurrency) || 4), total - 1));
      const workers = Array.from({ length: limit }, async () => {
        while (fetchCursor < total) {
          if (signal?.aborted) throw new Error("批量导入已取消");
          const idx = fetchCursor++;
          readyOutcomes[idx] = await prepareAccount(parsed.accounts[idx]);
          await drainCommits();
        }
      });
      await Promise.all(workers);
      await drainCommits();
    }
  }
  return result;
}
