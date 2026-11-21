/**
 * Owner-local, encrypted token pool. The key stays in the owner's home
 * directory; it is NOT a hardware-backed OS keychain. A compromise of the
 * same OS user can read both files, so we never claim otherwise.
 *
 * Never persist id_token or refresh_token acquired by a Codex sign-in that
 * Magpie's native importer may rotate. Pool mode stores only access tokens
 * plus the owner-provided email/password/TOTP needed for browser reauth.
 */
import { createCipheriv, createDecipheriv, randomBytes } from "node:crypto";
import { homedir } from "node:os";
import { join, resolve } from "node:path";
import { chmod, mkdir, readFile, rename, rm, stat, writeFile } from "node:fs/promises";

const NAME = "pool.enc.json";
const KEYFILE = "master.key";
const FORMAT = 1;
const MAX_RECORDS = 1000;

// In-memory decrypted snapshot cache keyed by resolved vault root.
// Eliminates repeated 686KB+ base64 + AES-256-GCM decryption on every API call
// while staying coherent with external edits via file (mtimeMs, ctimeMs, size).
const vaultCache = new Map();
const keyCache = new Map();

function cloneRows(rows) {
  return rows.map((r) => ({ ...r }));
}

export function clearPoolVaultCache(root = null) {
  if (root) {
    const resolved = resolve(root);
    vaultCache.delete(resolved);
    keyCache.delete(resolved);
    return;
  }
  vaultCache.clear();
  keyCache.clear();
}

export function poolDir() {
  return resolve(process.env.MAGPIE_GPTFREE_HOME || join(homedir(), ".magpie-gptfree"));
}

async function machineKey(root) {
  const resolved = resolve(root);
  const file = join(resolved, KEYFILE);
  try {
    const st = await stat(file);
    const cached = keyCache.get(resolved);
    if (cached && cached.mtimeMs === st.mtimeMs && cached.size === st.size) {
      return cached.key;
    }
  } catch (err) {
    if (err.code !== "ENOENT") throw err;
  }
  try {
    await writeFile(file, randomBytes(32), { mode: 0o600, flag: "wx" });
  } catch (err) {
    if (err.code !== "EEXIST") throw err;
  }
  const key = await readFile(file);
  if (key.length !== 32) throw new Error("本地号池加密密钥无效；拒绝读取凭证");
  try { await chmod(file, 0o600); } catch {}
  try {
    const st = await stat(file);
    keyCache.set(resolved, { mtimeMs: st.mtimeMs, size: st.size, key });
  } catch {}
  return key;
}

async function readUnlocked(root, key) {
  let file;
  try { file = await readFile(join(root, NAME), "utf8"); }
  catch (err) {
    if (err.code === "ENOENT") return [];
    throw err;
  }
  let blob;
  try { blob = JSON.parse(file); }
  catch { throw new Error("本地号池数据文件无法解析"); }
  if (blob?.version !== FORMAT || typeof blob?.iv !== "string" ||
      typeof blob?.tag !== "string" || typeof blob?.data !== "string")
    throw new Error("本地号池数据文件格式无效");
  try {
    const decipher = createDecipheriv("aes-256-gcm", key, Buffer.from(blob.iv, "base64"));
    decipher.setAuthTag(Buffer.from(blob.tag, "base64"));
    const plaintext = Buffer.concat([
      decipher.update(Buffer.from(blob.data, "base64")), decipher.final(),
    ]);
    const records = JSON.parse(plaintext.toString("utf8"));
    if (!Array.isArray(records) || records.length > MAX_RECORDS)
      throw new Error("invalid size");
    return records;
  } catch {
    throw new Error("本地号池解密失败；检查当前系统用户/密钥及文件完整性");
  }
}

async function storeUnlocked(root, key, rows) {
  if (rows.length > MAX_RECORDS) throw new Error("号池账号数量超过 1000");
  const iv = randomBytes(12);
  const cipher = createCipheriv("aes-256-gcm", key, iv);
  const ct = Buffer.concat([
    cipher.update(JSON.stringify(rows), "utf8"), cipher.final(),
  ]);
  const blob = JSON.stringify({
    version: FORMAT, iv: iv.toString("base64"),
    tag: cipher.getAuthTag().toString("base64"), data: ct.toString("base64"),
  });
  const target = join(root, NAME);
  const tmp = join(root, "." + NAME + "-" + randomBytes(8).toString("hex") + ".tmp");
  try {
    await writeFile(tmp, blob, { mode: 0o600, flag: "wx" });
    await rename(tmp, target);
    try { await chmod(target, 0o600); } catch {}
    try {
      const st = await stat(target);
      vaultCache.set(resolve(root), {
        mtimeMs: st.mtimeMs,
        ctimeMs: st.ctimeMs,
        size: st.size,
        rows: cloneRows(rows),
      });
    } catch {
      vaultCache.delete(resolve(root));
    }
  } finally {
    await rm(tmp, { force: true }).catch(() => {});
  }
}

// The lock is advisory for this process; importing in more than one CLI
// simultaneously is not supported, and the README warns against it.
let pending = Promise.resolve();
async function exclusive(work) {
  const run = pending.then(work, work);
  pending = run.catch(() => {});
  return run;
}

function normalizeVaultRecord(record, nowIso = new Date().toISOString()) {
  if (!record || typeof record.email !== "string" || !record.email.trim().includes("@") ||
      typeof record.password !== "string" || typeof record.totp !== "string" ||
      typeof record.accessToken !== "string" || !record.accessToken)
    throw new Error("号池记录缺少账号凭证");
  const item = {
    email: record.email.trim().toLowerCase(),
    password: record.password,
    totp: record.totp,
    accessToken: record.accessToken,
    updatedAt: typeof record.updatedAt === "string" && record.updatedAt ? record.updatedAt : nowIso,
  };
  // Preserve optional encrypted fields for browser OAuth accounts (v0.7.0+) so
  // refreshToken / authType / planType survive host restarts inside pool.enc.json.
  if (typeof record.refreshToken === "string" && record.refreshToken) item.refreshToken = record.refreshToken;
  if (typeof record.idToken === "string" && record.idToken) item.idToken = record.idToken;
  if (typeof record.authType === "string" && record.authType) item.authType = record.authType;
  if (typeof record.kind === "string" && record.kind) item.kind = record.kind;
  if (typeof record.planType === "string" && record.planType) item.planType = record.planType;
  return item;
}

export async function listPool({ root = poolDir() } = {}) {
  const resolved = resolve(root);
  const target = join(resolved, NAME);
  let st = null;
  try {
    st = await stat(target);
  } catch (err) {
    if (err.code === "ENOENT") {
      vaultCache.delete(resolved);
      await mkdir(resolved, { recursive: true, mode: 0o700 });
      return [];
    }
    throw err;
  }
  const cached = vaultCache.get(resolved);
  if (cached && cached.mtimeMs === st.mtimeMs && cached.ctimeMs === st.ctimeMs && cached.size === st.size) {
    return cloneRows(cached.rows);
  }
  await mkdir(resolved, { recursive: true, mode: 0o700 });
  const rows = await readUnlocked(resolved, await machineKey(resolved));
  vaultCache.set(resolved, {
    mtimeMs: st.mtimeMs,
    ctimeMs: st.ctimeMs,
    size: st.size,
    rows: cloneRows(rows),
  });
  return cloneRows(rows);
}

export async function upsertPool(record, { root = poolDir() } = {}) {
  return exclusive(async () => {
    const item = normalizeVaultRecord({ ...record, updatedAt: new Date().toISOString() });
    await mkdir(root, { recursive: true, mode: 0o700 });
    const key = await machineKey(root);
    const rows = await readUnlocked(root, key);
    const i = rows.findIndex((r) => r.email === item.email);
    if (i < 0) rows.push(item);
    else rows[i] = item;
    await storeUnlocked(root, key, rows);
    return { updated: i >= 0, total: rows.length };
  });
}

export async function upsertManyPool(records, { root = poolDir() } = {}) {
  if (!Array.isArray(records) || !records.length) {
    const current = await listPool({ root });
    return { added: 0, updated: 0, total: current.length };
  }
  return exclusive(async () => {
    const nowIso = new Date().toISOString();
    const items = records.map((r) => normalizeVaultRecord({ ...r, updatedAt: nowIso }));
    await mkdir(root, { recursive: true, mode: 0o700 });
    const key = await machineKey(root);
    const rows = await readUnlocked(root, key);
    const indexByEmail = new Map();
    for (let idx = 0; idx < rows.length; idx++) {
      indexByEmail.set(rows[idx].email, idx);
    }
    let added = 0;
    let updated = 0;
    for (const item of items) {
      const existingIdx = indexByEmail.get(item.email);
      if (existingIdx === undefined) {
        indexByEmail.set(item.email, rows.length);
        rows.push(item);
        added++;
      } else {
        rows[existingIdx] = item;
        updated++;
      }
    }
    await storeUnlocked(root, key, rows);
    return { added, updated, total: rows.length };
  });
}

export async function removePool(email, { root = poolDir() } = {}) {
  return exclusive(async () => {
    await mkdir(root, { recursive: true, mode: 0o700 });
    const key = await machineKey(root), rows = await readUnlocked(root, key);
    const kept = rows.filter((r) => r.email !== String(email).trim().toLowerCase());
    if (kept.length !== rows.length) await storeUnlocked(root, key, kept);
    return { removed: rows.length - kept.length, total: kept.length };
  });
}

export async function removeManyPool(emails, { root = poolDir() } = {}) {
  const dropSet = new Set(
    (Array.isArray(emails) ? emails : [])
      .map((e) => String(e || "").trim().toLowerCase())
      .filter(Boolean),
  );
  if (!dropSet.size) {
    const current = await listPool({ root });
    return { removed: 0, total: current.length };
  }
  return exclusive(async () => {
    await mkdir(root, { recursive: true, mode: 0o700 });
    const key = await machineKey(root);
    const rows = await readUnlocked(root, key);
    const kept = rows.filter((r) => !dropSet.has(String(r?.email || "").trim().toLowerCase()));
    if (kept.length !== rows.length) await storeUnlocked(root, key, kept);
    return { removed: rows.length - kept.length, total: kept.length };
  });
}
