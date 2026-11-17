import assert from "node:assert/strict";
import { test } from "node:test";
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
    row("fresh@example.com", Math.floor(now / 1000) + 5 * 3600),
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
      account.accessToken = token(Math.floor(now / 1000) + 6 * 3600);
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
    read: async () => [row("fresh@example.com", Math.floor(now / 1000) + 7200)],
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

test("auto refresh settings round-trip the proxy and stay enabled by default", async () => {
  const dir = await mkdtemp(join(tmpdir(), "gptfree-auto-"));
  try {
    assert.deepEqual(await readAutoRefreshSettings({ root: dir }), { proxy: "", enabled: true });
    await writeAutoRefreshSettings({ proxy: "http://127.0.0.1:7890" }, { root: dir });
    assert.deepEqual(await readAutoRefreshSettings({ root: dir }), {
      proxy: "http://127.0.0.1:7890", enabled: true,
    });
    await writeAutoRefreshSettings({ enabled: false }, { root: dir });
    assert.deepEqual(await readAutoRefreshSettings({ root: dir }), {
      proxy: "http://127.0.0.1:7890", enabled: false,
    });
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});
