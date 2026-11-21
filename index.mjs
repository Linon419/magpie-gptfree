/**
 * Standalone Magpie GUI and token-pool providers (GPT号池).
 * GUI entry runs in the plugin-host and serves its own sandboxed page.
 * Primary import & management live in the left-sidebar page (GPT号池).
 */
import { CODEX_BASE, discoverPoolModels, poolFetch } from "./lib/codex-pool.mjs";
import { GUI_PORT, startGuiServer, startGuiServerWithRetry } from "./lib/gui-service.mjs";
import { configureTextCooldownStore } from "./lib/chatgpt-web-text.mjs";
import { poolDir } from "./lib/pool-vault.mjs";
import { join } from "node:path";
import { createPoolConnector, POOL_MARKER, POOL_PROVIDER } from "./lib/pool-connection.mjs";
import { nativeImportMethod, webOAuthMethod } from "./lib/provider-native-signin.mjs";
import { readAutoRefreshSettings, startAutoRefresh } from "./lib/auto-refresh.mjs";

let sharedClient = null;
let sharedConnector = null;
let sharedActiveKey = null;
let autoRefresh = null;
let guiRetry = null;

function ensurePoolConnector(ctx) {
  if (typeof ctx?.magpie?.account === "function") {
    sharedActiveKey = () => String(ctx.magpie.account(POOL_PROVIDER) || "");
  }
  if (typeof ctx?.client?.auth?.set === "function" &&
      typeof ctx?.client?.auth?.remove === "function") {
    if (!sharedConnector || sharedClient !== ctx.client) {
      sharedClient = ctx.client;
      sharedConnector = createPoolConnector(ctx.client, {
        getActiveKey: () => (sharedActiveKey ? sharedActiveKey() : ""),
      });
    }
  }
  return sharedConnector;
}

function slotFromAuth(auth) {
  const slot = auth?.metadata?.poolSlot;
  return typeof slot === "string" && /^s[0-9a-f]{6,16}$/i.test(slot) ? slot : "";
}

export const GPTFreeGuiPlugin = async (ctx, options) => {
  const raw = options?.port ?? GUI_PORT;
  const port = Number(raw);
  if (!Number.isSafeInteger(port) || port < 1 || port > 65535)
    throw new Error("GPT号池 GUI: invalid service port");
  const connection = ensurePoolConnector(ctx) || createPoolConnector(ctx.client);
  // The GUI port is normally held by the previous host, which is still
  // shutting down: a host restart overlaps the two for a moment, and an
  // older Magpie instance can hold it for longer. Do not await port binding
  // during plugin load so the provider's models hook registers immediately;
  // if the initial 30s retry window still sees EADDRINUSE, fall back to
  // background exponential backoff until the port is freed.
  const log = (level, message) => Promise.resolve(ctx?.client?.app?.log?.({
    body: { service: "gptfree", level, message },
  })).catch(() => {});
  guiRetry?.stop();
  guiRetry = null;
  startGuiServer({ port, onPoolChange: () => connection.sync() }).catch((error) => {
    void log("warn", "GPT号池 GUI service could not start (" +
      String(error?.message || error) + "); the pool provider stays available, " +
      "retrying in background until the port is free");
    if (!process.env.NODE_TEST_CONTEXT) {
      guiRetry = startGuiServerWithRetry({
        port,
        onPoolChange: () => connection.sync(),
        onStarted: () => {
          void log("info", "GPT号池 GUI service started on 127.0.0.1:" + port);
        },
      });
    }
  });
  // Restore Magpie's per-account local-pool slots automatically after
  // restarting. A corrupt vault must not make the GUI inaccessible.
  try {
    await connection.sync();
  } catch {
    await ctx.client.app.log({
      body: { service: "gptfree", level: "warn",
        message: "Could not restore local account-pool state; inspect GUI pool storage" },
    }).catch(() => {});
  }
  // Renew tokens before they expire instead of waiting for the next 401.
  // `autoRefresh: false` in the plugin options turns it off.
  if (options?.autoRefresh !== false && !autoRefresh) {
    const settings = await readAutoRefreshSettings().catch(() => ({ enabled: true }));
    if (settings.enabled) {
      let pendingRenewSync = null;
      const coalescedSync = () => {
        if (pendingRenewSync) return pendingRenewSync;
        pendingRenewSync = new Promise((resolve) => {
          const timer = setTimeout(resolve, 250);
          timer.unref?.();
        })
          .then(() => connection.sync())
          .catch(() => {})
          .finally(() => { pendingRenewSync = null; });
        return pendingRenewSync;
      };
      autoRefresh = startAutoRefresh({
        intervalMs: (settings.intervalMinutes || 60) * 60 * 1000,
        leadMs: (settings.leadHours || 48) * 60 * 60 * 1000,
        getProxy: async () => (await readAutoRefreshSettings().catch(() => ({}))).proxy || "",
        onRenewed: () => coalescedSync(),
        log: (level, message) => ctx?.client?.app?.log?.({
          body: { service: "gptfree", level, message },
        }),
      });
    }
  }
  return {};
};

/**
 * Separate, multi-account access-token pool mode.
 * Every admitted vault account is synchronized to its own Magpie Providers
 * account row, allowing individual selection, preferred routing, and failover.
 */
export const GPTFreePoolPlugin = async (ctx = {}) => {
  const connector = ensurePoolConnector(ctx);
  // Keep text 429 cooldowns across plugin-host restarts. Tests run without
  // it so they never write into the real pool directory.
  if (!process.env.NODE_TEST_CONTEXT)
    configureTextCooldownStore({ file: join(poolDir(), "text-cooldowns.json") });
  return {
    config: async (cfg) => {
      cfg.provider ??= {};
      cfg.provider[POOL_PROVIDER] ??= {
        name: "GPT号池",
        npm: "@ai-sdk/openai",
        api: CODEX_BASE,
        models: {},
      };
    },
    auth: {
      provider: POOL_PROVIDER,
      maxConcurrency: 2,
      // Multi-account management stays enabled so every pooled account gets its
      // own toggleable row in Magpie Providers. Supports both browser OAuth
      // sign-in (open or copy link) and native TXT batch file import.
      methods: [
        webOAuthMethod({
          syncPool: () => sharedConnector?.sync(),
          claimSlotKey: (slot, key) => sharedConnector?.claimKey(slot, key),
          activeKey: () => (sharedActiveKey ? sharedActiveKey() : ""),
        }),
        nativeImportMethod({
          ...(connector ? { server: () => startGuiServer({ port: GUI_PORT, onPoolChange: () => sharedConnector?.sync() }) } : {}),
          syncPool: () => sharedConnector?.sync(),
          claimSlotKey: (slot, key) => sharedConnector?.claimKey(slot, key),
          activeKey: () => (sharedActiveKey ? sharedActiveKey() : ""),
        }),
      ],
      async loader(getAuth) {
        const auth = await getAuth();
        if (auth?.type !== "api") return {};
        const preferredSlot = slotFromAuth(auth);
        return {
          baseURL: CODEX_BASE,
          apiKey: POOL_MARKER,
          fetch: (input, init) => poolFetch(input, init, { preferredSlot }),
        };
      },
    },
    provider: {
      id: POOL_PROVIDER,
      async models(_provider, { auth } = {}) {
        if (auth?.type !== "api") return {};
        return discoverPoolModels({ preferredSlot: slotFromAuth(auth) });
      },
    },
  };
};
