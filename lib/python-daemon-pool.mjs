/**
 * Bounded pool of long-lived `--daemon` Python workers (NDJSON over stdio).
 *
 * Eliminates the 1.5s–3.5s cold-start cost of spawning Python, importing
 * `curl_cffi`, and re-bootstrapping `https://chatgpt.com/` on every text or
 * image call. Each worker holds at most one active request at a time:
 * - Normal completion returns the warm worker (and its in-process SessionPool)
 *   straight back to the idle list for immediate reuse.
 * - AbortSignal cancellation or request timeout kills ONLY that busy child so
 *   stalled network sockets are torn down immediately without affecting siblings.
 * - Idle workers are `unref()`'d and reaped after `idleMs`, so tests and CLIs
 *   never hang on exit.
 */
import { spawn } from "node:child_process";
import crypto from "node:crypto";

const DEFAULT_MAX_WORKERS = 3;
const DEFAULT_IDLE_MS = 5 * 60 * 1000;
const MAX_STDERR_BYTES = 32 * 1024;

function abortError(message = "Aborted") {
  const err = new Error(message);
  err.name = "AbortError";
  return err;
}

export function createPythonDaemonPool({
  maxWorkers = DEFAULT_MAX_WORKERS,
  idleMs = DEFAULT_IDLE_MS,
  maxLineBytes = 64 * 1024 * 1024,
} = {}) {
  const idle = [];
  const all = new Set();
  const waiters = [];

  function killHandle(handle) {
    if (!handle || handle.dead) return;
    handle.dead = true;
    if (handle.idleTimer) {
      clearTimeout(handle.idleTimer);
      handle.idleTimer = null;
    }
    const idx = idle.indexOf(handle);
    if (idx >= 0) idle.splice(idx, 1);
    all.delete(handle);
    try { handle.child.stdin.end(); } catch {}
    try { handle.child.kill(); } catch {}
  }

  function refHandle(handle) {
    try { handle.child.ref?.(); } catch {}
    try { handle.child.stdout?.ref?.(); } catch {}
    try { handle.child.stderr?.ref?.(); } catch {}
  }

  function unrefHandle(handle) {
    try { handle.child.unref?.(); } catch {}
    try { handle.child.stdout?.unref?.(); } catch {}
    try { handle.child.stderr?.unref?.(); } catch {}
    try { handle.child.stdin?.unref?.(); } catch {}
  }

  function spawnDaemon(command, workerPath) {
    const child = spawn(command, [workerPath, "--daemon"], {
      stdio: ["pipe", "pipe", "pipe"],
      shell: false,
      windowsHide: true,
      env: {
        ...process.env,
        PYTHONDONTWRITEBYTECODE: "1",
        PYTHONUNBUFFERED: "1",
        PYTHONIOENCODING: "utf-8",
        PYTHONUTF8: "1",
      },
    });

    const handle = {
      command,
      workerPath,
      child,
      dead: false,
      activeTask: null,
      idleTimer: null,
      stderr: "",
      lineChunks: [],
      lineBytes: 0,
    };

    all.add(handle);
    unrefHandle(handle);

    const dispatchLine = (rawBuf) => {
      const task = handle.activeTask;
      if (!task) return;
      const line = rawBuf.toString("utf8").trim();
      if (!line) return;
      let parsed;
      try {
        parsed = JSON.parse(line);
      } catch {
        task.invalidOutput = true;
        return;
      }
      if (!parsed || typeof parsed !== "object") return;
      if (parsed.id && parsed.id !== task.id) return;
      task.onEvent(parsed);
    };

    child.stdout.on("data", (chunk) => {
      const buf = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
      let offset = 0;
      while (offset < buf.length) {
        const nl = buf.indexOf(0x0a, offset);
        if (nl === -1) {
          const rem = buf.subarray(offset);
          handle.lineBytes += rem.length;
          if (handle.lineBytes > maxLineBytes) {
            const task = handle.activeTask;
            handle.lineChunks = [];
            handle.lineBytes = 0;
            killHandle(handle);
            task?.onOverflow();
            return;
          }
          handle.lineChunks.push(rem);
          break;
        }
        const piece = buf.subarray(offset, nl);
        offset = nl + 1;
        let fullBuf;
        if (handle.lineChunks.length === 0) {
          if (piece.length > maxLineBytes) {
            const task = handle.activeTask;
            killHandle(handle);
            task?.onOverflow();
            return;
          }
          fullBuf = piece;
        } else {
          handle.lineBytes += piece.length;
          if (handle.lineBytes > maxLineBytes) {
            const task = handle.activeTask;
            handle.lineChunks = [];
            handle.lineBytes = 0;
            killHandle(handle);
            task?.onOverflow();
            return;
          }
          handle.lineChunks.push(piece);
          fullBuf = Buffer.concat(handle.lineChunks, handle.lineBytes);
          handle.lineChunks = [];
          handle.lineBytes = 0;
        }
        dispatchLine(fullBuf);
      }
    });

    child.stderr.on("data", (chunk) => {
      if (handle.stderr.length < MAX_STDERR_BYTES) {
        handle.stderr += String(chunk).slice(0, MAX_STDERR_BYTES - handle.stderr.length);
      }
    });

    child.once("error", (err) => {
      const task = handle.activeTask;
      handle.activeTask = null;
      killHandle(handle);
      pumpWaiters();
      task?.onSpawnError(err);
    });

    child.once("close", (code) => {
      const wasDead = handle.dead;
      const remChunks = handle.lineChunks;
      handle.lineChunks = [];
      handle.lineBytes = 0;
      if (!wasDead && remChunks.length && handle.activeTask) {
        dispatchLine(Buffer.concat(remChunks));
      }
      const task = handle.activeTask;
      handle.activeTask = null;
      killHandle(handle);
      pumpWaiters();
      task?.onUnexpectedClose(code, handle.stderr);
    });

    child.stdin.on("error", () => {});
    return handle;
  }

  function releaseHandle(handle) {
    if (!handle || handle.dead) {
      pumpWaiters();
      return;
    }
    handle.activeTask = null;
    handle.stderr = "";
    unrefHandle(handle);
    if (waiters.length > 0) {
      const matchIdx = waiters.findIndex((w) => w.command === handle.command && w.workerPath === handle.workerPath);
      if (matchIdx >= 0) {
        const waiter = waiters.splice(matchIdx, 1)[0];
        waiter.resolve(handle);
        return;
      }
      // Reap this mismatched idle worker if at capacity so the waiter can spawn its target.
      if (all.size >= maxWorkers) {
        killHandle(handle);
        pumpWaiters();
        return;
      }
    }
    handle.idleTimer = setTimeout(() => killHandle(handle), idleMs);
    handle.idleTimer.unref?.();
    idle.push(handle);
  }

  function pumpWaiters() {
    while (waiters.length > 0) {
      const next = waiters[0];
      if (next.signal?.aborted) {
        waiters.shift();
        next.reject(abortError());
        continue;
      }
      const idleIdx = idle.findIndex((h) => !h.dead && h.command === next.command && h.workerPath === next.workerPath);
      if (idleIdx >= 0) {
        const handle = idle.splice(idleIdx, 1)[0];
        if (handle.idleTimer) {
          clearTimeout(handle.idleTimer);
          handle.idleTimer = null;
        }
        waiters.shift();
        next.resolve(handle);
        continue;
      }
      if (all.size < maxWorkers) {
        waiters.shift();
        next.resolve(spawnDaemon(next.command, next.workerPath));
        continue;
      }
      if (idle.length > 0) {
        killHandle(idle.shift());
        continue;
      }
      break;
    }
  }

  function acquireHandle(command, workerPath, signal) {
    if (signal?.aborted) return Promise.reject(abortError());
    const idleIdx = idle.findIndex((h) => !h.dead && h.command === command && h.workerPath === workerPath);
    if (idleIdx >= 0) {
      const handle = idle.splice(idleIdx, 1)[0];
      if (handle.idleTimer) {
        clearTimeout(handle.idleTimer);
        handle.idleTimer = null;
      }
      return Promise.resolve(handle);
    }
    if (all.size < maxWorkers) {
      return Promise.resolve(spawnDaemon(command, workerPath));
    }
    if (idle.length > 0) {
      killHandle(idle.shift());
      return Promise.resolve(spawnDaemon(command, workerPath));
    }
    return new Promise((resolve, reject) => {
      const waiter = { command, workerPath, signal, resolve, reject };
      const onAbort = () => {
        const i = waiters.indexOf(waiter);
        if (i >= 0) waiters.splice(i, 1);
        reject(abortError());
      };
      signal?.addEventListener("abort", onAbort, { once: true });
      waiter.resolve = (h) => {
        signal?.removeEventListener("abort", onAbort);
        resolve(h);
      };
      waiter.reject = (e) => {
        signal?.removeEventListener("abort", onAbort);
        reject(e);
      };
      waiters.push(waiter);
    });
  }

  async function execute(command, workerPath, payload, {
    signal,
    timeout = 180_000,
    onDelta = null,
    mode = "single", // "single" (image/quota) or "stream" (web text)
  } = {}) {
    const handle = await acquireHandle(command, workerPath, signal);
    const reqId = crypto.randomUUID().replace(/-/g, "");
    const envelope = JSON.stringify({ id: reqId, payload }) + "\n";

    return new Promise((resolve, reject) => {
      let settled = false;
      let accumulatedText = "";
      refHandle(handle);

      const cleanupListeners = () => {
        clearTimeout(timer);
        if (signal && onAbort) signal.removeEventListener("abort", onAbort);
      };

      const settleOk = (result, reusable = true) => {
        if (settled) return;
        settled = true;
        cleanupListeners();
        if (reusable) releaseHandle(handle);
        else {
          killHandle(handle);
          pumpWaiters();
        }
        resolve(result);
      };

      const settleErr = (err) => {
        if (settled) return;
        settled = true;
        cleanupListeners();
        killHandle(handle);
        pumpWaiters();
        reject(err);
      };

      const onAbort = () => {
        settleErr(abortError());
      };
      if (signal) {
        if (signal.aborted) {
          releaseHandle(handle);
          reject(abortError());
          return;
        }
        signal.addEventListener("abort", onAbort, { once: true });
      }

      const timer = setTimeout(() => {
        settleOk(
          {
            ok: false,
            http_status: 504,
            error_code: mode === "stream" ? "timeout" : "worker_timeout",
            error: mode === "stream" ? "Web text worker timed out" : "Image generation worker timed out",
          },
          false,
        );
      }, timeout);
      timer.unref?.();

      handle.activeTask = {
        id: reqId,
        invalidOutput: false,
        onEvent(parsed) {
          if (mode === "stream") {
            if (parsed.type === "delta" && typeof parsed.delta === "string" && parsed.delta) {
              accumulatedText += parsed.delta;
              try { onDelta?.(parsed.delta, parsed); } catch {}
              return;
            }
            if (parsed.type === "done" || parsed.ok === true) {
              settleOk({
                ok: true,
                text: typeof parsed.text === "string" ? parsed.text : accumulatedText,
                conversation_id: parsed.conversation_id || "",
                model: parsed.model || payload?.model || "",
              }, true);
              return;
            }
            if (parsed.type === "error" || parsed.ok === false) {
              const copy = { ...parsed };
              delete copy.id;
              settleOk(copy, true);
              return;
            }
            return;
          }
          const copy = { ...parsed };
          delete copy.id;
          settleOk(copy, true);
        },
        onOverflow() {
          settleOk(
            {
              ok: false,
              http_status: 502,
              error_code: "output_too_large",
              error: "Worker output exceeded maximum size",
            },
            false,
          );
        },
        onSpawnError(err) {
          settleOk(
            {
              ok: false,
              http_status: 502,
              error_code: mode === "stream" ? "worker_spawn_failed" : "worker_start_failed",
              error: String(err?.message || "Python worker could not start").slice(0, 180),
            },
            false,
          );
        },
        onUnexpectedClose(code, stderrText) {
          settleOk(
            {
              ok: false,
              http_status: 502,
              error_code: mode === "stream"
                ? "worker_failed"
                : (code === 0 ? "invalid_worker_output" : "worker_crashed"),
              error: mode === "stream"
                ? (String(stderrText || "").trim().slice(0, 180) || `Web text worker exited with code ${code}`)
                : "Image generation worker did not return valid JSON",
            },
            false,
          );
        },
      };

      try {
        handle.child.stdin.write(envelope, "utf8");
      } catch (err) {
        settleOk(
          {
            ok: false,
            http_status: 502,
            error_code: mode === "stream" ? "worker_spawn_failed" : "worker_start_failed",
            error: String(err?.message || "Failed to write to worker").slice(0, 180),
          },
          false,
        );
      }
    });
  }

  function closeAll() {
    for (const h of [...all]) killHandle(h);
    while (waiters.length) {
      const w = waiters.shift();
      w.reject(abortError("Worker pool closed"));
    }
  }

  return {
    execute,
    closeAll,
    size: () => all.size,
    idleCount: () => idle.length,
  };
}
