import assert from "node:assert/strict";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import {
  CODEX_BASE,
  clearCodexRuntimeState,
  poolFetch,
  renewAccount,
  rotateAccountsForTest,
} from "../lib/codex-pool.mjs";
import {
  clearTextCooldowns,
  closeTextWorkerPool,
  executeWebTextGeneration,
  parseWebToolCalls,
  runWebTextWorker,
} from "../lib/chatgpt-web-text.mjs";
import { closeImageWorkerPool, runWebImageWorker } from "../lib/chatgpt-web-image.mjs";
import { accountSlotTag, createPoolConnector } from "../lib/pool-connection.mjs";
import {
  createInMemoryStatsStore,
  effectiveAccountImageQuota,
  isAccountImageSoftCapped,
  readPoolStats,
  recordImageCall,
  recordQuotaCheck,
  summarizePoolStats,
} from "../lib/pool-stats.mjs";
import { clearPoolVaultCache, listPool, upsertManyPool } from "../lib/pool-vault.mjs";
import { runBatch } from "../lib/importer.mjs";

const seed = "GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ";

function makeJwt(claims) {
  const b = (o) => Buffer.from(JSON.stringify(o)).toString("base64url");
  return `${b({ alg: "RS256", typ: "JWT" })}.${b(claims)}.sig`;
}

function codexJwt(email, deltaSec = 3600) {
  return makeJwt({
    exp: Math.floor(Date.now() / 1000) + deltaSec,
    email,
    "https://api.openai.com/auth": { chatgpt_account_id: `acct_${email}` },
  });
}

function webJwt(email, deltaSec = 3600) {
  return makeJwt({
    exp: Math.floor(Date.now() / 1000) + deltaSec,
    client_id: "app_2SKx67EdpoN0G6j64rFvigXD",
    "https://api.openai.com/profile": { email },
  });
}

test("cache-first 80% image routing sticks to active slot until 80% quota is used, then switches cleanly", async () => {
  clearCodexRuntimeState();
  const acc1 = { email: "s1@example.com", password: "p", totp: seed, accessToken: codexJwt("s1@example.com") };
  const acc2 = { email: "s2@example.com", password: "p", totp: seed, accessToken: codexJwt("s2@example.com") };
  const acc3 = { email: "s3@example.com", password: "p", totp: seed, accessToken: codexJwt("s3@example.com") };
  const accounts = [acc1, acc2, acc3];
  const slot1 = accountSlotTag(acc1.email);

  const store = createInMemoryStatsStore();
  for (const acc of accounts) {
    await recordQuotaCheck({ email: acc.email, quota: 5, hasImageLimit: true, planType: "free" }, { statsStore: store });
  }

  const usedTokens = [];
  const callOnce = async () => {
    const res = await poolFetch(`${CODEX_BASE}/images/generations`, {
      method: "POST",
      body: JSON.stringify({ model: "gpt-image-2", prompt: "warm cache test" }),
    }, {
      read: async () => accounts,
      preferredSlot: slot1,
      statsStore: store,
      runWorker: async (payload) => {
        usedTokens.push(payload.access_token);
        return {
          ok: true,
          created: 1731150000,
          data: [{ b64_json: Buffer.from("img").toString("base64"), media_type: "image/png" }],
        };
      },
    });
    assert.equal(res.status, 200);
  };

  // First 4 calls stay sticky on acc1 (5 -> 4 -> 3 -> 2 -> 1 remaining = 80% used!).
  for (let i = 0; i < 4; i++) await callOnce();
  assert.deepEqual(usedTokens, [acc1.accessToken, acc1.accessToken, acc1.accessToken, acc1.accessToken]);

  const statsAfter4 = await readPoolStats({ statsStore: store });
  assert.equal(isAccountImageSoftCapped(acc1.email, statsAfter4), true);

  // Calls 5..8 automatically hand off to acc2 and stay sticky on acc2 (preserving acc2's cache)
  // even though Magpie's `preferredSlot` is still pointing at slot1!
  for (let i = 0; i < 4; i++) await callOnce();
  assert.deepEqual(usedTokens.slice(4, 8), [acc2.accessToken, acc2.accessToken, acc2.accessToken, acc2.accessToken]);

  // Calls 9..12 hand off to acc3 and stay sticky on acc3.
  for (let i = 0; i < 4; i++) await callOnce();
  assert.deepEqual(usedTokens.slice(8, 12), [acc3.accessToken, acc3.accessToken, acc3.accessToken, acc3.accessToken]);

  // Call 13: all 3 accounts are now >=80% used (each has 1 image left > 0), so the last 20% headroom is still usable!
  await callOnce();
  assert.equal(usedTokens.length, 13);
});

test("expired restoreAt automatically restores quota in effectiveAccountImageQuota, summarizePoolStats, and recordImageCall", async () => {
  const email = "restored@example.com";
  const tok = codexJwt(email);
  const store = createInMemoryStatsStore();

  await recordQuotaCheck({ email, quota: 5, hasImageLimit: true, planType: "free" }, { statsStore: store });
  const pastIso = new Date(Date.now() - 120_000).toISOString();
  await recordQuotaCheck({ email, quota: 0, restoreAt: pastIso, hasImageLimit: true, planType: "free" }, { statsStore: store });

  const statsBefore = await readPoolStats({ statsStore: store });
  const eff = effectiveAccountImageQuota(email, statsBefore);
  assert.equal(eff.windowExpired, true);
  assert.equal(eff.remaining, 5);
  assert.equal(eff.usedRatio, 0);

  const summary = summarizePoolStats([{ email, accessToken: tok }], statsBefore);
  assert.equal(summary.quotaStats.active, 1);
  assert.equal(summary.quotaStats.limited, 0);
  assert.equal(summary.quotaStats.totalQuota, 5);
  assert.equal(summary.accounts[0].cooldownExpired, true);
  assert.equal(summary.accounts[0].quotaStatus, "available");

  // Next successful call restores from 5 -> 4 and clears expired restoreAt.
  await recordImageCall({ email, success: true, count: 1 }, { statsStore: store });
  const statsAfter = await readPoolStats({ statsStore: store });
  assert.equal(statsAfter.accounts[accountSlotTag(email)].imageQuota, 4);
  assert.equal(statsAfter.accounts[accountSlotTag(email)].restoreAt, null);
});

test("web text generation sticks to one account until 80% of window calls, then switches cleanly", async () => {
  clearTextCooldowns();
  const a1 = { email: "w1@example.com", accessToken: webJwt("w1@example.com") };
  const a2 = { email: "w2@example.com", accessToken: webJwt("w2@example.com") };
  const slot1 = accountSlotTag(a1.email);
  const seen = [];

  // windowMaxCalls = 5, softCapPercent = 80 => threshold = ceil(5 * 0.8) = 4 calls.
  for (let i = 0; i < 6; i++) {
    const res = await executeWebTextGeneration(
      JSON.stringify({ model: "gpt-5.6-luna", input: `turn ${i}`, stream: false }),
      [a1, a2],
      {
        preferredSlot: slot1,
        softCapPercent: 80,
        windowMaxCalls: 5,
        worker: async (payload) => {
          seen.push(payload.access_token);
          return { ok: true, text: "ack" };
        },
      },
    );
    assert.equal(res.status, 200);
  }

  assert.deepEqual(seen, [
    a1.accessToken, a1.accessToken, a1.accessToken, a1.accessToken,
    a2.accessToken, a2.accessToken,
  ]);
  clearTextCooldowns();
});

test("Codex /responses x-codex-primary-used-percent >= 80 switches sticky slot on next request", async () => {
  clearCodexRuntimeState();
  const c1 = { email: "c1@example.com", password: "p", totp: seed, accessToken: codexJwt("c1@example.com") };
  const c2 = { email: "c2@example.com", password: "p", totp: seed, accessToken: codexJwt("c2@example.com") };
  const slot1 = accountSlotTag(c1.email);
  const hits = [];

  // Request 1 hits c1 and receives x-codex-primary-used-percent: 82
  await poolFetch(`${CODEX_BASE}/responses`, {
    method: "POST",
    body: JSON.stringify({ model: "gpt-6", input: "first" }),
  }, {
    read: async () => [c1, c2],
    preferredSlot: slot1,
    fetcher: async (_url, init) => {
      hits.push(init.headers.get("authorization"));
      return new Response("data: ok\n\n", {
        status: 200,
        headers: { "x-codex-primary-used-percent": "82" },
      });
    },
  });

  // Request 2 & 3 automatically switch to c2 (<80%) and stay sticky on c2!
  for (let i = 0; i < 2; i++) {
    await poolFetch(`${CODEX_BASE}/responses`, {
      method: "POST",
      body: JSON.stringify({ model: "gpt-6", input: "next" }),
    }, {
      read: async () => [c1, c2],
      preferredSlot: slot1,
      fetcher: async (_url, init) => {
        hits.push(init.headers.get("authorization"));
        return new Response("data: ok\n\n", {
          status: 200,
          headers: { "x-codex-primary-used-percent": "20" },
        });
      },
    });
  }

  assert.deepEqual(hits, [
    `Bearer ${c1.accessToken}`,
    `Bearer ${c2.accessToken}`,
    `Bearer ${c2.accessToken}`,
  ]);
  clearCodexRuntimeState();
});

test("web-only text tool bridge emits Responses function_call items in both JSON and SSE modes", async () => {
  clearTextCooldowns();
  const acc = { email: "tools@example.com", accessToken: webJwt("tools@example.com") };
  const tools = [{
    type: "function",
    name: "get_weather",
    description: "Fetch weather for a city",
    parameters: { type: "object", properties: { city: { type: "string" } }, required: ["city"] },
  }];

  // 1. Non-streaming JSON mode
  const jsonRes = await executeWebTextGeneration(
    JSON.stringify({ model: "gpt-6-luna", input: "Weather in Sydney?", tools, stream: false }),
    [acc],
    {
      worker: async (payload) => {
        assert.match(payload.messages[0].content, /get_weather/);
        return {
          ok: true,
          text: 'Checking now.\n<tool_call>{"name":"get_weather","arguments":{"city":"Sydney"}}</tool_call>',
        };
      },
    },
  );
  assert.equal(jsonRes.status, 200);
  const body = await jsonRes.json();
  assert.equal(body.output_text, "Checking now.");
  const fc = body.output.find((item) => item.type === "function_call");
  assert.ok(fc);
  assert.equal(fc.name, "get_weather");
  assert.deepEqual(JSON.parse(fc.arguments), { city: "Sydney" });

  // 2. Streaming SSE mode
  const sseRes = await executeWebTextGeneration(
    JSON.stringify({ model: "gpt-6-luna", input: "Weather in Sydney?", tools, stream: true }),
    [acc],
    {
      worker: async (_payload, { onDelta }) => {
        const reply = '<tool_call>{"name":"get_weather","arguments":{"city":"Sydney"}}</tool_call>';
        onDelta?.(reply.slice(0, 18));
        onDelta?.(reply.slice(18));
        return { ok: true, text: reply };
      },
    },
  );
  assert.equal(sseRes.status, 200);
  const sseText = await sseRes.text();
  assert.ok(!sseText.includes("<tool_call>"), "raw XML tag must not leak into text deltas");
  assert.match(sseText, /event: response\.function_call_arguments\.done/);
  assert.match(sseText, /"name":"get_weather"/);

  const parsed = parseWebToolCalls("hello", new Set(["get_weather"]));
  assert.equal(parsed.text, "hello");
  assert.equal(parsed.toolCalls.length, 0);
});

test("pool-vault caches decrypted snapshots, preserves OAuth refreshToken, and batches upsertManyPool", async () => {
  const root = await mkdtemp(join(tmpdir(), "gptfree-vault-opt-"));
  try {
    const batch = Array.from({ length: 25 }, (_, i) => ({
      email: `u${i}@example.com`,
      password: "",
      totp: "",
      accessToken: codexJwt(`u${i}@example.com`),
      refreshToken: `rt_secret_${i}`,
      authType: "oauth",
      planType: "plus",
    }));
    const res = await upsertManyPool(batch, { root });
    assert.equal(res.added, 25);
    assert.equal(res.total, 25);

    const firstRead = await listPool({ root });
    assert.equal(firstRead.length, 25);
    assert.equal(firstRead[0].refreshToken, "rt_secret_0");
    assert.equal(firstRead[0].authType, "oauth");

    // Mutating returned array must not corrupt in-memory vault cache.
    firstRead[0].email = "mutated@example.com";
    const secondRead = await listPool({ root });
    assert.equal(secondRead[0].email, "u0@example.com");

    const encOnDisk = await readFile(join(root, "pool.enc.json"), "utf8");
    assert.ok(!encOnDisk.includes("rt_secret_0"));
  } finally {
    clearPoolVaultCache(root);
    await rm(root, { recursive: true, force: true });
  }
});

test("pool-connection skips redundant client.auth.set calls on unchanged accounts", async () => {
  let setCalls = 0;
  let removeCalls = 0;
  const client = {
    auth: {
      async set() { setCalls++; return { data: true }; },
      async remove() { removeCalls++; return { data: true }; },
    },
  };
  let rows = [
    { email: "a@example.com", password: "p", totp: "", accessToken: "t1" },
    { email: "b@example.com", password: "p", totp: "", accessToken: "t2" },
    { email: "c@example.com", password: "p", totp: "", accessToken: "t3" },
  ];
  const conn = createPoolConnector(client, { poolRead: async () => rows, slotsFile: false });

  await conn.sync();
  assert.equal(setCalls, 3);

  // Token renewals do not change slot metadata: 0 redundant auth.set RPCs!
  rows = rows.map((r, idx) => ({ ...r, accessToken: `renewed_${idx}` }));
  await conn.sync();
  assert.equal(setCalls, 3);
  assert.equal(removeCalls, 0);

  // Adding 1 new account performs only 1 incremental auth.set RPC!
  rows.push({ email: "d@example.com", password: "p", totp: "", accessToken: "t4" });
  await conn.sync();
  assert.equal(setCalls, 4);
});

test("renewAccount and runBatch execute concurrently with bounded parallelism", async () => {
  // 1. renewAccount does not serialize unrelated accounts behind a slow one
  let active = 0;
  let maxActive = 0;
  const accounts = ["k1@example.com", "k2@example.com", "k3@example.com"].map((email) => ({
    email, password: "pw", totp: seed, accessToken: codexJwt(email, -100),
  }));

  await Promise.all(accounts.map((acc) => renewAccount(acc, {
    authenticate: async () => {
      active++;
      maxActive = Math.max(maxActive, active);
      await new Promise((r) => setTimeout(r, 35));
      active--;
      return { tokens: { access: codexJwt(acc.email, 3600) } };
    },
    verify: async () => ({ accepted: true }),
    save: async () => {},
  })));
  assert.ok(maxActive >= 2, `expected concurrent reauth, got maxActive=${maxActive}`);

  // 2. runBatch with concurrency=3 preserves row commit order while parallelizing logins
  const txt = [1, 2, 3, 4].map((n) => `u${n}@example.com----pw----${seed}`).join("\n");
  const committed = [];
  await runBatch({
    file: "batch.txt",
    consent: true,
    concurrency: 3,
    read: async () => txt,
    login: async (acc) => {
      const delay = acc.email.startsWith("u2") ? 30 : 5;
      await new Promise((r) => setTimeout(r, delay));
      return { tokens: { access: codexJwt(acc.email, 3600) } };
    },
    verifyToken: async () => ({ accepted: true }),
    savePool: async (rec) => { committed.push(rec.email); },
  });
  assert.deepEqual(committed, [
    "u1@example.com",
    "u2@example.com",
    "u3@example.com",
    "u4@example.com",
  ]);
});

test("Python daemon pool reuses warm worker process across consecutive calls", async () => {
  try {
    // Missing/invalid payload triggers fast in-worker validation without hitting network,
    // proving the same persistent daemon handles back-to-back IPC requests cleanly.
    const r1 = await runWebImageWorker({ action: "invalid_probe" }, {
      python: async () => "python3",
      timeout: 15_000,
    });
    const r2 = await runWebImageWorker({ action: "check_image_quota", access_token: "" }, {
      python: async () => "python3",
      timeout: 15_000,
    });
    assert.equal(r1.ok, false);
    assert.equal(r1.error_code, "invalid_request");
    assert.equal(r2.ok, false);
    assert.equal(r2.http_status, 401);

    const t1 = await runWebTextWorker({ access_token: "", messages: [{ role: "user", content: "hi" }] }, {
      python: async () => "python3",
      timeout: 15_000,
    });
    const t2 = await runWebTextWorker({ access_token: "", messages: [{ role: "user", content: "hi2" }] }, {
      python: async () => "python3",
      timeout: 15_000,
    });
    assert.equal(t1.ok, false);
    assert.equal(t1.http_status, 401);
    assert.equal(t2.ok, false);
    assert.equal(t2.http_status, 401);
  } finally {
    closeImageWorkerPool();
    closeTextWorkerPool();
  }
});
