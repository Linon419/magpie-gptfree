import assert from "node:assert/strict";
import { test } from "node:test";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { buildCpaExport, buildCpaZip, cpaAuthFile, cpaFileName } from "../lib/cpa-export.mjs";
import { buildZip, crc32 } from "../lib/zip.mjs";
import { createGuiService } from "../lib/gui-service.mjs";

function b64(obj) {
  return Buffer.from(JSON.stringify(obj)).toString("base64url");
}

function jwt(claims) {
  return [b64({ alg: "RS256", typ: "JWT" }), b64(claims), "signature"].join(".");
}

function record(email, claims = {}) {
  return {
    email,
    password: "pw",
    totp: "totp",
    accessToken: jwt({
      exp: Math.floor(Date.now() / 1000) + 6 * 24 * 3600,
      client_id: "app_EMoamEEZ73f0CkXaXp7hrann",
      "https://api.openai.com/auth": {
        chatgpt_account_id: "acct_123",
        chatgpt_plan_type: "free",
      },
      ...claims,
    }),
  };
}

/** Minimal reader for the store-only archives this plugin writes. */
function readZip(buffer) {
  const end = buffer.lastIndexOf(Buffer.from([0x50, 0x4b, 0x05, 0x06]));
  assert(end >= 0, "end of central directory present");
  const count = buffer.readUInt16LE(end + 10);
  let offset = buffer.readUInt32LE(end + 16);
  const files = [];
  for (let i = 0; i < count; i++) {
    assert.equal(buffer.readUInt32LE(offset), 0x02014b50);
    const method = buffer.readUInt16LE(offset + 10);
    const crc = buffer.readUInt32LE(offset + 16);
    const size = buffer.readUInt32LE(offset + 24);
    const nameLen = buffer.readUInt16LE(offset + 28);
    const extraLen = buffer.readUInt16LE(offset + 30);
    const commentLen = buffer.readUInt16LE(offset + 32);
    const localOffset = buffer.readUInt32LE(offset + 42);
    const name = buffer.slice(offset + 46, offset + 46 + nameLen).toString("utf8");
    assert.equal(method, 0, "stored, no compression");
    assert.equal(buffer.readUInt32LE(localOffset), 0x04034b50);
    const localNameLen = buffer.readUInt16LE(localOffset + 26);
    const localExtraLen = buffer.readUInt16LE(localOffset + 28);
    const dataStart = localOffset + 30 + localNameLen + localExtraLen;
    const data = buffer.slice(dataStart, dataStart + size);
    assert.equal(crc32(data), crc, "crc matches");
    files.push({ name, data });
    offset += 46 + nameLen + extraLen + commentLen;
  }
  return files;
}

test("CPA auth file carries the fields CLIProxyAPI reads", () => {
  const now = new Date("2026-11-17T00:00:00Z");
  const file = cpaAuthFile(record("User@Example.com"), { now });
  assert.equal(file.type, "codex");
  assert.equal(file.email, "user@example.com");
  assert.equal(file.account_id, "acct_123");
  assert.equal(file.plan_type, "free");
  assert.equal(file.refresh_token, "");
  assert.equal(file.id_token, "");
  assert.equal(file.last_refresh, now.toISOString());
  assert.match(file.expired, /^\d{4}-\d{2}-\d{2}T/);
  assert.equal(file.access_token.split(".").length, 3);
});

test("CPA file names are stable and safe for an auth dir", () => {
  assert.equal(cpaFileName("User@Example.com"), "codex-user@example.com.json");
  assert.equal(cpaFileName("a+b@example.com"), "codex-a_b@example.com.json");
  assert.equal(cpaFileName("../../etc/passwd"), "codex-.._.._etc_passwd.json");
  assert.equal(cpaFileName(""), "codex-account.json");
});

test("export skips invalid, duplicate and web-only records and adds a README", () => {
  const webOnly = {
    email: "web@example.com",
    password: "pw",
    totp: "totp",
    accessToken: jwt({
      exp: Math.floor(Date.now() / 1000) + 6 * 24 * 3600,
      client_id: "app_2SKx67EdpoN0G6j64rFvigXD",
      "https://api.openai.com/profile": { email: "web@example.com" },
    }),
  };
  const { files, skipped, webOnly: webOnlyCount } = buildCpaExport([
    record("a@example.com"),
    record("A@Example.com"),
    webOnly,
    { email: "no-token@example.com" },
    null,
  ]);
  assert.equal(webOnlyCount, 1);
  assert.equal(skipped, 3);
  assert.deepEqual(files.map((f) => f.name), ["codex-a@example.com.json", "README.txt"]);
  const parsed = JSON.parse(files[0].data);
  assert.equal(parsed.email, "a@example.com");
  assert.match(files[1].data, /refresh_token/);
  assert.match(files[1].data, /401/);
});

test("ZIP archive round-trips every exported file", () => {
  const { zip, files } = buildCpaZip([record("a@example.com"), record("b@example.com")]);
  const entries = readZip(zip);
  assert.deepEqual(entries.map((e) => e.name), files.map((f) => f.name));
  for (const entry of entries) {
    const source = files.find((f) => f.name === entry.name);
    assert.equal(entry.data.toString("utf8"), source.data);
  }
});

test("GUI export endpoint writes a 0600 ZIP and never returns tokens", async () => {
  const dir = await mkdtemp(join(tmpdir(), "gptfree-cpa-"));
  const rows = [record("a@example.com"), record("b@example.com")];
  const service = createGuiService({
    port: 0,
    poolRead: async () => rows,
    poolRemove: async () => ({ removed: 0, total: 0 }),
    exportDir: dir,
  });
  const addr = await service.listen();
  try {
    const base = "http://127.0.0.1:" + addr.port + "/";
    const html = await (await fetch(base + "ui")).text();
    const token = html.match(/meta name="gptfree-session" content="([0-9a-f]{64})"/)?.[1];
    assert(token);
    const res = await fetch(base + "api/export/cpa", {
      method: "POST",
      headers: { Origin: "null", "X-GPTFree-Session": token, "Content-Type": "application/json" },
      body: "{}",
    });
    assert.equal(res.status, 200);
    const body = await res.json();
    assert.equal(body.ok, true);
    assert.equal(body.accounts, 2);
    assert.equal(body.skipped, 0);
    assert.equal(body.webOnly, 0);
    assert(!JSON.stringify(body).includes(rows[0].accessToken));

    const zip = await readFile(body.path);
    const entries = readZip(zip);
    assert.equal(entries.length, 3); // two accounts + README
    const first = JSON.parse(entries.find((e) => e.name.startsWith("codex-a@")).data);
    assert.equal(first.access_token, rows[0].accessToken);
    assert.equal(first.type, "codex");
  } finally {
    await service.close();
    await rm(dir, { recursive: true, force: true });
  }
});

test("ZIP writer rejects empty names", () => {
  assert.throws(() => buildZip([{ name: "", data: "x" }]), /name is invalid/);
});
