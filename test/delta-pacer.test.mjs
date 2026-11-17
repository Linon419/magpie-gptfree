import assert from "node:assert/strict";
import { test } from "node:test";

import { createDeltaPacer } from "../lib/chatgpt-web-text.mjs";

test("pacer splits upstream chunks into small ordered pieces", async () => {
  const pieces = [];
  const pacer = createDeltaPacer((p) => pieces.push(p), { tickMs: 1, targetTicks: 10 });
  const chunk = "Correct answer\n\n**What address did the database contain on a given date?**";
  pacer.push(chunk);
  assert.equal(pieces.length, 1, "first piece is emitted immediately");
  pacer.push(" 😀 done");
  await pacer.drain();
  assert.equal(pieces.join(""), chunk + " 😀 done");
  assert.ok(pieces.length >= 10, `expected token-like pieces, got ${pieces.length}`);
  for (const p of pieces) assert.doesNotMatch(p, /^[\uDC00-\uDFFF]|[\uD800-\uDBFF]$/);
});

test("pacer flush emits the remainder at once", () => {
  const pieces = [];
  const pacer = createDeltaPacer((p) => pieces.push(p), { tickMs: 1000 });
  pacer.push("abcdefghijklmnopqrstuvwxyz");
  pacer.flush();
  assert.equal(pieces.join(""), "abcdefghijklmnopqrstuvwxyz");
});
