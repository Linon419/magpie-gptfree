import assert from "node:assert/strict";
import { mock, test } from "node:test";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  clearTextCooldowns,
  configureTextCooldownStore,
  estimateTokens,
  executeWebTextGeneration,
} from "../lib/chatgpt-web-text.mjs";
import { startGuiServerWithRetry } from "../lib/gui-service.mjs";

function jwt(email) {
  const b = (o) => Buffer.from(JSON.stringify(o)).toString("base64url");
  return [b({ alg: "RS256" }), b({ exp: Math.floor(Date.now() / 1000) + 3600,
    client_id: "app_2SKx67EdpoN0G6j64rFvigXD", "https://api.openai.com/profile": { email } }), "sig"].join(".");
}

test("token estimate counts CJK characters individually", () => {
  assert.equal(estimateTokens(""), 0);
  assert.equal(estimateTokens("abcd"), 1);
  assert.equal(estimateTokens("hello world!"), 3);
  assert.equal(estimateTokens("蒙特卡洛模型"), 6);
  assert.equal(estimateTokens("答案是 C"), 4); // 3 CJK + " C" (2 chars -> 1)
  assert.ok(estimateTokens("一".repeat(400)) >= 400, "Chinese is no longer counted as length / 4");
});

test("text 429 cooldowns survive a reload of the store", async () => {
  const dir = await mkdtemp(join(tmpdir(), "gptfree-cooldown-"));
  const file = join(dir, "text-cooldowns.json");
  const accounts = [
    { email: "a@example.com", accessToken: jwt("a@example.com") },
    { email: "b@example.com", accessToken: jwt("b@example.com") },
  ];
  try {
    configureTextCooldownStore({ file });
    clearTextCooldowns();
    const seen = [];
    const worker = async (payload) => {
      seen.push(payload.access_token);
      return payload.access_token === accounts[0].accessToken
        ? { ok: false, http_status: 429, error_code: "rate_limit_exceeded", error: "slow down", retry_after: 600 }
        : { ok: true, text: "fine" };
    };
    const body = JSON.stringify({ model: "gpt-6-luna", input: "hi" });
    const first = await executeWebTextGeneration(body, accounts, { worker });
    assert.equal(first.status, 200);
    const stored = JSON.parse(await readFile(file, "utf8"));
    assert.equal(Object.keys(stored.until).length, 1);

    // Simulate a plugin-host restart: drop memory, reload from the file.
    configureTextCooldownStore({ file: null });
    clearTextCooldowns();
    configureTextCooldownStore({ file });
    seen.length = 0;
    await executeWebTextGeneration(body, accounts, { worker });
    assert.equal(seen[0], accounts[1].accessToken, "the cooled-down account is tried last");
  } finally {
    configureTextCooldownStore({ file: null });
    clearTextCooldowns();
    await rm(dir, { recursive: true, force: true });
  }
});

test("GUI start retries with backoff until the port is free", async () => {
  mock.timers.enable({ apis: ["setTimeout"] });
  const flush = async () => { for (let i = 0; i < 5; i++) await new Promise((r) => setImmediate(r)); };
  let attempts = 0;
  const waits = [];
  let started = null;
  try {
    const handle = startGuiServerWithRetry({
      start: async () => {
        attempts++;
        if (attempts < 4) throw Object.assign(new Error("port already in use"), { code: "EADDRINUSE" });
        return { ok: true };
      },
      onWaiting: (_e, delay) => waits.push(delay),
      onStarted: (service) => { started = service; },
      initialDelay: 2000, maxDelay: 5000,
    });
    await handle.first;
    assert.equal(attempts, 1, "first attempt is immediate and does not block the caller");
    for (const step of [2000, 4000, 5000]) { mock.timers.tick(step); await flush(); }
    assert.equal(attempts, 4);
    assert.deepEqual(waits, [2000, 4000, 5000], "backoff doubles and is capped");
    assert.deepEqual(started, { ok: true });
    handle.stop();
  } finally {
    mock.timers.reset();
  }
});
