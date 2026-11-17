import assert from "node:assert/strict";
import { mock, test } from "node:test";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  readAutoRefreshSettings,
  startAutoRefresh,
  writeAutoRefreshSettings,
} from "../lib/auto-refresh.mjs";

function b64(obj) {
  return Buffer.from(JSON.stringify(obj)).toString("base64url");
}

function token(expSeconds) {
  return [b64({ alg: "RS256" }), b64({ exp: expSeconds, client_id: "app_2SKx67EdpoN0G6j64rFvigXD" }), "sig"].join(".");
}

function row(email, expSeconds) {
  return { email, password: "pw", totp: "totp", accessToken: token(expSeconds) };
}

test("auto refresh renews expired and near-expiry tokens, not fresh ones", async () => {
  const now = 1_800_000_000_000;
  const rows = [
    row("fresh@example.com", Math.floor(now / 1000) + 72 * 3600),
    row("near@example.com", Math.floor(now / 1000) + 10 * 60),
    row("expired@example.com", Math.floor(now / 1000) - 60),
  ];
  const calls = [];
  const handle = startAutoRefresh({
    immediate: false,
    now: () => now,
    read: async () => rows,
    renew: async (account, options) => {
      calls.push({ email: account.email, options });
      return { ...account, accessToken: token(Math.floor(now / 1000) + 6 * 3600) };
    },
    getProxy: async () => "http://127.0.0.1:7890",
  });
  try {
    const stats = await handle.tick();
    assert.deepEqual(calls.map((c) => c.email).sort(), ["expired@example.com", "near@example.com"]);
    assert.equal(calls[0].options.proxy, "http://127.0.0.1:7890");
    assert.equal(stats.renewed, 2);
    assert.equal(stats.failed, 0);
  } finally {
    handle.stop();
  }
});

test("auto refresh caps work per tick and retries failures after backoff", async () => {
  let now = 1_800_000_000_000;
  const rows = [
    row("a@example.com", Math.floor(now / 1000) - 60),
    row("b@example.com", Math.floor(now / 1000) - 60),
    row("c@example.com", Math.floor(now / 1000) - 60),
  ];
  const calls = [];
  const handle = startAutoRefresh({
    immediate: false,
    now: () => now,
    read: async () => rows,
    maxPerTick: 2,
    retryMs: 30 * 60 * 1000,
    renew: async (account) => {
      calls.push(account.email);
      if (account.email === "a@example.com") throw new Error("login failed");
      // Simulate the vault write: the renewed row is fresh on the next read.
      account.accessToken = token(Math.floor(now / 1000) + 240 * 3600);
      return account;
    },
  });
  try {
    let stats = await handle.tick();
    assert.equal(calls.length, 2);
    assert.equal(stats.failed, 1);
    assert.equal(stats.renewed, 1);

    // a@ is backing off: the next tick only picks c@, not a@ again.
    calls.length = 0;
    stats = await handle.tick();
    assert.deepEqual(calls, ["c@example.com"]);

    // After the backoff window a@ is retried.
    now += 31 * 60 * 1000;
    calls.length = 0;
    await handle.tick();
    assert.ok(calls.includes("a@example.com"));
  } finally {
    handle.stop();
  }
});

test("auto refresh does nothing when every token is fresh", async () => {
  const now = 1_800_000_000_000;
  let calls = 0;
  const handle = startAutoRefresh({
    immediate: false,
    now: () => now,
    read: async () => [row("fresh@example.com", Math.floor(now / 1000) + 72 * 3600)],
    renew: async () => { calls++; },
  });
  try {
    const stats = await handle.tick();
    assert.equal(calls, 0);
    assert.equal(stats.renewed, 0);
  } finally {
    handle.stop();
  }
});

test("a clustered 310-account pool drains before its tokens expire", async () => {
  let now = 1_800_000_000_000;
  const rows = Array.from({ length: 310 }, (_, i) => row(`u${i}@example.com`, Math.floor(now / 1000) + 6 * 3600));
  const renewed = new Set();
  const expiredBeforeRenew = [];
  const handle = startAutoRefresh({
    immediate: false,
    now: () => now,
    read: async () => rows,
    intervalMs: 60 * 60 * 1000,
    leadMs: 48 * 60 * 60 * 1000,
    maxPerTick: 100,
    renew: async (account) => {
      const exp = JSON.parse(Buffer.from(account.accessToken.split(".")[1], "base64url").toString()).exp * 1000;
      if (now >= exp) expiredBeforeRenew.push(account.email);
      renewed.add(account.email);
      account.accessToken = token(Math.floor(now / 1000) + 240 * 3600);
      return account;
    },
  });
  try {
    for (let tick = 0; tick < 6; tick++) {
      await handle.tick();
      now += 60 * 60 * 1000;
    }
    assert.equal(renewed.size, 310);
    assert.deepEqual(expiredBeforeRenew, []);
  } finally {
    handle.stop();
  }
});

test("a 24h interval with a 48h lead still drains 310 clustered tokens via catch-up ticks", async () => {
  let now = 1_800_000_000_000;
  const rows = Array.from({ length: 310 }, (_, i) => row(`c${i}@example.com`, Math.floor(now / 1000) + 40 * 3600));
  const renewed = new Set();
  const late = [];
  const handle = startAutoRefresh({
    immediate: false,
    now: () => now,
    read: async () => rows,
    // The most relaxed cadence the settings file allows: one scan per day.
    intervalMs: 24 * 60 * 60 * 1000,
    leadMs: 48 * 60 * 60 * 1000,
    maxPerTick: 100,
    catchUpMs: 5 * 60 * 1000,
    renew: async (account) => {
      const exp = JSON.parse(Buffer.from(account.accessToken.split(".")[1], "base64url").toString()).exp * 1000;
      if (now >= exp) late.push(account.email);
      renewed.add(account.email);
      account.accessToken = token(Math.floor(now / 1000) + 240 * 3600);
      return account;
    },
  });
  try {
    for (let i = 0; i < 20 && renewed.size < 310; i++) {
      const stats = await handle.tick();
      now += stats.remaining > 0 ? 5 * 60 * 1000 : 24 * 60 * 60 * 1000;
    }
    assert.equal(renewed.size, 310);
    assert.deepEqual(late, []);
  } finally {
    handle.stop();
  }
});

test("the real timer retries a capped backlog after five minutes, not a full interval", async () => {
  mock.timers.enable({ apis: ["setTimeout"] });
  let now = 1_800_000_000_000;
  const rows = Array.from({ length: 310 }, (_, i) => row(`t${i}@example.com`, Math.floor(now / 1000) + 40 * 3600));
  const calls = [];
  let reads = 0;
  const handle = startAutoRefresh({
    immediate: false,
    now: () => now,
    read: async () => { reads++; return rows; },
    intervalMs: 24 * 60 * 60 * 1000,
    leadMs: 48 * 60 * 60 * 1000,
    maxPerTick: 100,
    catchUpMs: 5 * 60 * 1000,
    renew: async (account) => {
      calls.push(account.email);
      account.accessToken = token(Math.floor(now / 1000) + 240 * 3600);
      return account;
    },
  });
  const flush = async () => {
    for (let i = 0; i < 5; i++) await new Promise((resolve) => setImmediate(resolve));
  };
  try {
    // First scan only happens when the configured 24h interval elapses.
    mock.timers.tick(24 * 60 * 60 * 1000);
    await flush();
    now += 24 * 60 * 60 * 1000;
    assert.equal(calls.length, 100);
    assert.equal(handle.stats.remaining, 210);

    // The backlog is picked up by the timer five minutes later, not in 24h.
    mock.timers.tick(5 * 60 * 1000);
    await flush();
    now += 5 * 60 * 1000;
    assert.equal(calls.length, 200);

    mock.timers.tick(5 * 60 * 1000);
    await flush();
    now += 5 * 60 * 1000;
    assert.equal(calls.length, 300);

    mock.timers.tick(5 * 60 * 1000);
    await flush();
    now += 5 * 60 * 1000;
    assert.equal(calls.length, 310);
    assert.equal(handle.stats.remaining, 0);

    // Backlog drained: the scheduler is back on the 24h interval, so a
    // five-minute tick must not even scan the vault.
    const readsAfterDrain = reads;
    mock.timers.tick(5 * 60 * 1000);
    await flush();
    assert.equal(calls.length, 310);
    assert.equal(reads, readsAfterDrain);

    // The next scan happens when the normal interval elapses.
    mock.timers.tick(24 * 60 * 60 * 1000);
    await flush();
    assert.equal(reads, readsAfterDrain + 1);
  } finally {
    handle.stop();
    mock.timers.reset();
  }
});

test("auto refresh settings round-trip the proxy and stay enabled by default", async () => {
  const dir = await mkdtemp(join(tmpdir(), "gptfree-auto-"));
  try {
    assert.deepEqual(await readAutoRefreshSettings({ root: dir }), {
      proxy: "", enabled: true, intervalMinutes: 60, leadHours: 48,
    });
    await writeAutoRefreshSettings({ proxy: "http://127.0.0.1:7890", intervalMinutes: 120 }, { root: dir });
    assert.deepEqual(await readAutoRefreshSettings({ root: dir }), {
      proxy: "http://127.0.0.1:7890", enabled: true, intervalMinutes: 120, leadHours: 48,
    });
    await writeAutoRefreshSettings({ enabled: false }, { root: dir });
    assert.deepEqual(await readAutoRefreshSettings({ root: dir }), {
      proxy: "http://127.0.0.1:7890", enabled: false, intervalMinutes: 120, leadHours: 48,
    });
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});
