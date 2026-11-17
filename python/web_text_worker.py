#!/usr/bin/env python3
"""Streaming ChatGPT Web text worker reusing OpenAIWebImageClient session & sentinel auth.

Reads one JSON request payload from stdin and either:
- streams NDJSON events (`delta`, `done`, `error`) to stdout when invoked with `--stream`
  or `payload["stream_events"] is True`, or
- prints a single JSON result object `{ok: True, text, ...}` / `{ok: False, ...}`.
"""

from __future__ import annotations

import json
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from image_worker import (
    ImageWorkerError,
    OpenAIWebImageClient,
    UpstreamHTTPError,
    _apply_patch_op,
    _assistant_message_text,
    _is_token_invalid_text,
    _is_visible_assistant_message,
    _iter_sse_payloads,
    _sanitize_error_text,
    _sanitize_output_text,
)

MAX_STDIN_BYTES = 16 * 1024 * 1024
DEFAULT_WEB_TEXT_MODEL = "gpt-5.6-luna"
VALID_THINKING_EFFORTS = {"low", "medium", "high", "xhigh"}
ALLOWED_ROLES = {"system", "developer", "user", "assistant"}


def normalize_web_text_model(model: Any) -> str:
    raw = str(model or "").strip()
    if raw.startswith("gptfree-pool/"):
        raw = raw[len("gptfree-pool/") :].strip()
    return raw or DEFAULT_WEB_TEXT_MODEL


def normalize_thinking_effort(effort: Any) -> str:
    val = str(effort or "").strip().lower()
    if val in ("", "none", "minimal"):
        return ""
    if val == "max":
        return "xhigh"
    if val in VALID_THINKING_EFFORTS:
        return val
    return ""


def normalize_web_messages(
    raw_messages: Any,
) -> tuple[list[Dict[str, Any]], list[str], str, set[str]]:
    """Convert normalized `{role, content}` messages into ChatGPT Web payloads.

    Returns `(web_messages, assistant_history_messages, last_user_text, input_message_ids)`.
    """
    if not isinstance(raw_messages, list):
        raise ImageWorkerError(
            "messages must be a non-empty list",
            http_status=400,
            error_code="invalid_request",
        )

    web_messages: list[Dict[str, Any]] = []
    assistant_history: list[str] = []
    input_message_ids: set[str] = set()
    last_user_text = ""

    for item in raw_messages:
        if not isinstance(item, dict):
            continue
        raw_role = str(item.get("role") or "user").strip().lower()
        if raw_role not in ALLOWED_ROLES:
            raw_role = "user"
        web_role = "system" if raw_role == "developer" else raw_role
        content = str(item.get("content") or "")
        if not content.strip():
            continue

        msg_id = str(uuid.uuid4())
        input_message_ids.add(msg_id)
        web_messages.append(
            {
                "id": msg_id,
                "author": {"role": web_role},
                "create_time": time.time(),
                "content": {"content_type": "text", "parts": [content]},
                "metadata": {
                    "developer_mode_connector_ids": [],
                    "selected_github_repos": [],
                    "selected_all_github_repos": False,
                    "serialization_metadata": {"custom_symbol_offsets": []},
                },
            }
        )
        if web_role == "assistant":
            assistant_history.append(content.strip())
        elif web_role == "user":
            last_user_text = content.strip()

    if not web_messages:
        raise ImageWorkerError(
            "Request must contain at least one non-empty message",
            http_status=400,
            error_code="invalid_request",
        )
    if not last_user_text:
        last_user_text = str(web_messages[-1]["content"]["parts"][0]).strip()
    return web_messages, assistant_history, last_user_text, input_message_ids


def strip_history_prefix(
    text: str,
    assistant_history_messages: Sequence[str],
    *,
    hold_partial_prefix: bool = False,
) -> str:
    """Strip echoed assistant history blocks while preserving legitimate short replies on finalize."""
    clean = str(text or "")
    joined = "\n\n".join(m for m in assistant_history_messages if m)
    candidates: list[str] = []
    if joined:
        candidates.append(joined)
    for msg in assistant_history_messages:
        if msg and msg not in candidates:
            candidates.append(msg)

    for cand in candidates:
        if clean.startswith(cand):
            clean = clean[len(cand) :].lstrip()
        elif hold_partial_prefix and clean and cand.startswith(clean):
            return ""
    return clean


def _is_hidden_text_path(path: str) -> bool:
    if not path:
        return False
    if path == "/message/content/parts/0":
        return False
    return (
        "/thoughts/" in path
        or "reasoning" in path
        or path.startswith("/message/content/")
    )


_DIL_TAG_NAME = re.compile(r"[A-Za-z][\w.-]*")
_MAX_DIL_TAG = 4000
# Components seen in GPT-6 Intelligent UI output and its documented catalog.
# Only these may open right after a word ("Hello<badge>..."); anything else
# glued to a word ("x<y", "a<b>") is ordinary text.
_KNOWN_DIL_TAGS = frozenset({
    "appblock", "asyncimage", "asyncimagegroup", "avatar", "badge", "bold",
    "box", "button", "callout", "caption", "card", "chart", "checkbox", "chip",
    "cite", "code", "col", "column", "divider", "entity", "form", "grid",
    "heading", "icon", "image", "input", "italic", "label", "link", "list",
    "listitem", "map", "metric", "progress", "radio", "row", "select",
    "slider", "spacer", "stack", "stat", "tab", "table", "tabs", "tag",
    "text", "textarea", "title", "toggle", "tooltip",
})


def _scan_dil_tag(s: str, i: int) -> Any:
    """Scan a component tag starting at s[i] == "<".

    Returns (end, name, closing, self_closing), None when the tag is not
    finished yet, or -1 when the "<" is ordinary text. Attribute values may
    hold JS (`onClick={()=>...}`), so ">" only ends the tag outside braces
    and quotes.
    """
    n = len(s)
    # An opening tag never follows a word character, so "x<y" and "f(a)<3"
    # are text; closing tags do follow text ("answer</badge>").
    if i + 1 >= n:
        return None
    after_word = i > 0 and s[i + 1] != "/" and (s[i - 1].isalnum() or s[i - 1] in ")]_")
    j = i + 1
    closing = j < n and s[j] == "/"
    if closing:
        j += 1
    if j >= n:
        return None
    m = _DIL_TAG_NAME.match(s, j)
    if not m:
        return -1
    name = m.group(0)
    j = m.end()
    if j >= n:
        # "Hello<bad" may still become "Hello<badge>": hold only while the
        # name can grow into a known component.
        if after_word and not any(t.startswith(name.lower()) for t in _KNOWN_DIL_TAGS):
            return -1
        return None
    if after_word and name.lower() not in _KNOWN_DIL_TAGS:
        return -1
    if s[j] not in " \t\r\n/>":
        return -1
    depth = 0
    quote = ""
    while j < n:
        if j - i > _MAX_DIL_TAG:
            return -1
        c = s[j]
        if quote:
            if c == "\\":
                j += 2
                continue
            if c == quote:
                quote = ""
        elif c in "\"'`" and (depth > 0 or s[j - 1] == "="):
            quote = c
        elif c == "{":
            depth += 1
        elif c == "}":
            depth = max(0, depth - 1)
        elif depth == 0 and c == ">":
            return j + 1, name, closing, s[j - 1] == "/"
        elif depth == 0 and (c == "<" or s.startswith("\n\n", j)):
            return -1
        j += 1
    return None


_LINE_START_MARKERS = ("```", "~~~", "{@body")


class _DilScan:
    """Resumable scan state: everything before `i` is rendered into `out`.

    Decisions about consumed text never depend on text that arrives later;
    anything still ambiguous stops the scan at `i`, so streaming can resume
    from the same position instead of re-rendering the whole message.
    """

    # `out` holds one character per element so _DilRenderer can find the
    # last non-whitespace character without splitting strings.
    __slots__ = ("i", "out", "in_code", "line_start")

    def __init__(self) -> None:
        self.i = 0
        self.out: list[str] = []
        self.in_code = False
        self.line_start = True

    def copy(self) -> "_DilScan":
        other = _DilScan()
        other.i, other.out = self.i, self.out[:]
        other.in_code, other.line_start = self.in_code, self.line_start
        return other


def _dil_scan(s: str, st: _DilScan, *, final: bool) -> None:
    n = len(s)
    out = st.out
    i, in_code, line_start = st.i, st.in_code, st.line_start

    def trim_indent_after_tag() -> None:
        # "  <box>" lines would otherwise leave indentation that Markdown
        # reads as a code block.
        k = len(out)
        while k and out[k - 1] in " \t":
            k -= 1
        if k == 0 or out[k - 1] == "\n":
            del out[k:]

    while i < n:
        if line_start:
            j = i
            while j < n and s[j] in " \t":
                j += 1
            # A fence or {@body line cannot be recognised until the line has
            # enough characters; wait instead of consuming them as text.
            if not final and any(m.startswith(s[j:]) for m in _LINE_START_MARKERS):
                break
            if s.startswith("```", j) or s.startswith("~~~", j):
                end = s.find("\n", j)
                if end < 0:
                    if not final:
                        break
                    out.extend(s[i:])
                    i = n
                    break
                in_code = not in_code
                out.extend(s[i:end + 1])
                i = end + 1
                continue
            if not in_code and s.startswith("{@body", j):
                end = s.find("\n", j)
                if end < 0:
                    if not final:
                        break
                    i = n
                    break
                i = end + 1
                continue
        c = s[i]
        if in_code:
            out.append(c)
            line_start = c == "\n"
            i += 1
            continue
        if c == "`":
            run = 1
            while i + run < n and s[i + run] == "`":
                run += 1
            fence = "`" * run
            close = s.find(fence, i + run)
            if close < 0:
                if not final:
                    break
                out.extend(s[i:i + run])
                i += run
                line_start = False
                continue
            out.extend(s[i:close + run])
            i = close + run
            line_start = False
            continue
        if c == "{" and (i + 1 >= n or s[i + 1] in "#:/"):
            end = s.find("}", i) if i + 1 < n else -1
            if end < 0:
                if not final:
                    break
                out.append(c)
                i += 1
                line_start = False
                continue
            i = end + 1
            line_start = False
            continue
        if c == "<":
            tag = _scan_dil_tag(s, i)
            if tag is None:
                if not final:
                    break
                tag = -1
            if tag != -1:
                end, name, closing, self_closing = tag
                if name == "AppBlock" and not closing and not self_closing:
                    close = s.find("</AppBlock>", end)
                    if close < 0:
                        if not final:
                            break
                        i = n
                        break
                    end = close + len("</AppBlock>")
                trim_indent_after_tag()
                i = end
                continue
        out.append(c)
        line_start = c == "\n"
        i += 1

    st.i, st.in_code, st.line_start = i, in_code, line_start


_WS_ONLY_LINE = re.compile(r"\n[ \t]+(?=\n)")
_EXTRA_BLANK_LINES = re.compile(r"\n{3,}")
_WS = " \t\r\n"


def _dil_normalize(text: str) -> str:
    # Both rules only match runs of whitespace, so normalising two pieces
    # split right after a non-whitespace character gives the same result
    # as normalising them together. _DilRenderer relies on that.
    return _EXTRA_BLANK_LINES.sub("\n\n", _WS_ONLY_LINE.sub("\n", text))


def _dil_finish(out: list[str], *, final: bool) -> str:
    result = _dil_normalize("".join(out)).lstrip()
    return result.rstrip() if final else result.rstrip(_WS)


class _DilRenderer:
    """Incremental _dil_to_text for an append-only stream.

    Output up to the last non-whitespace character can no longer change:
    later tags only trim trailing indentation, and normalisation never
    crosses a non-whitespace character. That part is normalised once and
    cached, so each event normalises only the text it added.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.state = _DilScan()
        self.source_len = 0
        self.generation = 0
        self.flushed = 0
        self.text = ""

    def render(self, text: str, *, final: bool, generation: int = 0) -> str:
        """`generation` changes whenever the caller replaced (not appended
        to) the source; the cached scan is only valid for one generation."""
        s = str(text or "")
        if generation != self.generation or len(s) < self.source_len:
            self.reset()
            self.generation = generation
        if final:
            st = self.state.copy()
            _dil_scan(s, st, final=True)
            rest = _dil_normalize("".join(st.out[self.flushed:]))
            return (self.text + rest).strip() if self.text else rest.strip()
        _dil_scan(s, self.state, final=False)
        self.source_len = len(s)
        out = self.state.out
        end = len(out)
        while end > self.flushed and out[end - 1] in _WS:
            end -= 1
        if end > self.flushed:
            piece = _dil_normalize("".join(out[self.flushed:end]))
            self.text = self.text + piece if self.text else piece.lstrip()
            self.flushed = end
        return self.text


def _dil_to_text(text: str, *, final: bool = True) -> str:
    """Render DIL (Markdown + component tags + {@body} JS) as plain Markdown.

    With final=False an unfinished tag, control block, fence line or
    trailing whitespace is held back instead of being emitted, so each
    result is a prefix of the next one. Fenced code and inline code are
    copied verbatim; AppBlock apps are dropped whole.
    """
    st = _DilScan()
    _dil_scan(str(text or ""), st, final=final)
    return _dil_finish(st.out, final=final)


def _unescape_fallback(text: str) -> str:
    """Undo the punctuation escaping fallbackMarkdown adds (`\\:`, `\\.`)."""
    return re.sub(r"\\([.:!()\-])", r"\1", str(text or "")).strip()


def _infer_sse_error_status(code: str, message: str) -> int:
    lowered = f"{code} {message}".lower()
    if any(tok in lowered for tok in ("rate_limit", "too many", "limit_exceeded", "quota", "usage_limit")):
        return 429
    if any(tok in lowered for tok in ("unauthorized", "token_revoked", "invalid_token", "session_expired")):
        return 401
    if any(tok in lowered for tok in ("moderation", "content_policy", "safety", "policy")):
        return 400
    if "forbidden" in lowered:
        return 403
    return 502


@dataclass
class WebTextStreamTracker:
    """Stateful parser tracking ChatGPT Web v1 SSE patches into clean assistant text deltas."""

    assistant_history_messages: tuple[str, ...] = ()
    input_message_ids: frozenset[str] = field(default_factory=frozenset)
    conversation_id: str = ""
    active_role: str = "assistant"
    active_channel: str = ""
    active_recipient: str = ""
    active_content_type: str = "text"
    active_hidden: bool = False
    active_is_input_echo: bool = False
    current_msg_id: str = ""
    completed_segments: list[str] = field(default_factory=list)
    current_msg_raw: str = ""
    last_patch_path: str = "/message/content/parts/0"
    emitted_text: str = ""
    dil_fallback_buffer: str = ""
    active_has_dil: bool = False
    pending_delta: str = ""
    turn_complete: bool = False
    # Bumped whenever parts/0 is replaced rather than appended to, so the
    # incremental DIL renderer knows its cached scan is stale.
    raw_generation: int = 0
    dil_renderer: _DilRenderer = field(default_factory=_DilRenderer, repr=False, compare=False)
    blocked: bool = False
    model_slug: str = ""

    def _is_active_visible(self) -> bool:
        if self.active_is_input_echo:
            return False
        if self.active_role != "assistant":
            return False
        if self.active_hidden:
            return False
        if self.active_recipient and self.active_recipient != "all":
            return False
        if self.active_channel and self.active_channel != "final":
            return False
        if self.active_content_type in {"thoughts", "reasoning_recap", "code"}:
            return False
        return True

    def _current_candidates(self, *, final: bool) -> list[tuple[str, bool]]:
        """Text sources for the active message, best first: (text, rendered).

        GPT-6 "Intelligent UI" responses stream DIL (Markdown + <badge>,
        <box>, {@body ...}) in parts/0. That raw text holds the whole answer,
        so it is rendered to Markdown as it streams. The server's
        fallbackMarkdown is escaped differently and omits computed content
        (observed: "(Correct answer)" then a stall), so it is only a final
        completion candidate. Raw parts/0 is the last resort, for a DIL flag
        that arrived after tags were already emitted.

        Every source passes the same visibility gate: hidden, reasoning and
        tool-directed messages can carry model_dil_v2 too.
        """
        if not self._is_active_visible():
            return [("", True)]
        raw = self.current_msg_raw
        if not self.active_has_dil:
            return [(raw, True)]
        candidates = [(self.dil_renderer.render(raw, final=final, generation=self.raw_generation), True)]
        if final and self.dil_fallback_buffer:
            candidates.append((_unescape_fallback(self.dil_fallback_buffer), True))
        candidates.append((raw, False))
        return candidates

    def _clean(self, current: str, *, hold_partial_prefix: bool) -> str:
        raw = "\n\n".join(s for s in [*self.completed_segments, current] if s)
        # Citation markers are rare; a memchr-speed check avoids running the
        # regex over the whole answer on every event.
        sanitized = _sanitize_output_text(raw) if "\ue200" in raw else raw
        if not self.input_message_ids and self.assistant_history_messages:
            return strip_history_prefix(
                sanitized,
                self.assistant_history_messages,
                hold_partial_prefix=hold_partial_prefix,
            )
        return sanitized

    def _pick_current(self, *, final: bool) -> tuple[str, str]:
        """Choose the source that continues what was already emitted.

        Streaming takes the first consistent source, so rendered DIL never
        falls through to raw tags while a tag is still incomplete. At the end
        the longest consistent rendered source wins, letting a fallback
        complete an answer whose raw text stopped early.
        """
        consistent = []
        for text, rendered in self._current_candidates(final=final):
            clean = self._clean(text, hold_partial_prefix=not final)
            if clean.startswith(self.emitted_text):
                consistent.append((text, clean, rendered))
                if not final:
                    break
        if not consistent:
            return "", self.emitted_text
        if final:
            rendered = [c for c in consistent if c[2]]
            if rendered:
                best = max(rendered, key=lambda c: len(c[1]))
                return best[0], best[1]
        return consistent[0][0], consistent[0][1]

    def _next_delta(self, *, hold_partial_prefix: bool) -> str:
        _, clean = self._pick_current(final=not hold_partial_prefix)
        delta = ""
        if len(clean) > len(self.emitted_text):
            delta = clean[len(self.emitted_text):]
            self.emitted_text = clean
        pending, self.pending_delta = self.pending_delta, ""
        return pending + delta

    def _finish_current_message(self) -> None:
        """Flush the active message before switching to the next one."""
        if not self._is_active_visible():
            return
        text, clean = self._pick_current(final=True)
        if len(clean) > len(self.emitted_text):
            self.pending_delta += clean[len(self.emitted_text):]
            self.emitted_text = clean
        if text:
            self.completed_segments.append(text)

    def _handle_message(self, msg: dict[str, Any]) -> None:
        msg_id = str(msg.get("id") or "")
        role = str(((msg.get("author") or {}).get("role")) or "").strip().lower()
        content = msg.get("content")
        content_type = (
            str(content.get("content_type") or "").strip().lower()
            if isinstance(content, dict)
            else ""
        )
        metadata = msg.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}

        if msg_id and msg_id in self.input_message_ids:
            self.active_is_input_echo = True
            self.active_role = role
            self.last_patch_path = ""
            return

        is_new_message = bool(msg_id) and msg_id != self.current_msg_id
        previous_visible = self._is_active_visible()
        if is_new_message:
            if previous_visible:
                self._finish_current_message()
            self.current_msg_id = msg_id
            self.current_msg_raw = ""
            self.active_has_dil = False
            self.dil_fallback_buffer = ""
            self.dil_renderer.reset()

        self.active_is_input_echo = False
        self.active_role = role
        self.active_channel = str(msg.get("channel") or "").strip().lower()
        self.active_recipient = str(msg.get("recipient") or "").strip().lower()
        self.active_content_type = content_type
        self.active_hidden = metadata.get("is_visually_hidden_from_conversation") is True
        dil = metadata.get("model_dil_v2")
        if isinstance(dil, dict):
            self.active_has_dil = True
            if isinstance(dil.get("fallbackMarkdown"), str):
                self.dil_fallback_buffer = dil["fallbackMarkdown"]

        slug = str(metadata.get("model_slug") or "").strip()
        if slug and self.active_role == "assistant":
            self.model_slug = slug

        if (
            self.active_role == "assistant"
            and self.active_content_type not in {"thoughts", "reasoning_recap", "code"}
        ):
            self.last_patch_path = "/message/content/parts/0"
            msg_text = _assistant_message_text(msg)
            if msg_text and msg_text != self.current_msg_raw:
                # A re-sent message carries the full text, which may differ
                # anywhere, not only at the end.
                self.current_msg_raw = msg_text
                self.raw_generation += 1
        else:
            self.last_patch_path = ""

    def _apply_visibility_op(self, op: dict[str, Any]) -> None:
        op_p = op.get("p")
        op_o = op.get("o")
        op_v = op.get("v")
        if not isinstance(op_p, str):
            return

        if op_p == "/message/metadata/is_visually_hidden_from_conversation":
            if op_o == "remove":
                self.active_hidden = False
            elif op_o in ("add", "replace"):
                self.active_hidden = op_v is True
        elif op_p == "/message/channel":
            if op_o == "remove":
                self.active_channel = ""
            elif isinstance(op_v, str):
                self.active_channel = op_v.strip().lower()
        elif op_p == "/message/recipient":
            if op_o == "remove":
                self.active_recipient = ""
            elif isinstance(op_v, str):
                self.active_recipient = op_v.strip().lower()
        elif op_p == "/message/metadata" and isinstance(op_v, dict):
            if "is_visually_hidden_from_conversation" in op_v:
                self.active_hidden = op_v["is_visually_hidden_from_conversation"] is True
            slug = str(op_v.get("model_slug") or "").strip()
            if slug and self.active_role == "assistant":
                self.model_slug = slug
            dil = op_v.get("model_dil_v2")
            if isinstance(dil, dict):
                self.active_has_dil = True
                if isinstance(dil.get("fallbackMarkdown"), str):
                    self._set_dil_fallback(dil["fallbackMarkdown"])

    def _handle_patch_op(self, op: dict[str, Any], *, apply_visibility: bool = True) -> None:
        op_p = op.get("p")
        op_o = op.get("o")
        op_v = op.get("v")
        # A compact string continuation can omit both p and o, including
        # inside a patch list. Normalize it here for every entry point.
        if "p" not in op and "o" not in op and isinstance(op_v, str):
            op_o = "append"
            op = {**op, "o": op_o}

        if op_p == "" and op_o == "add" and isinstance(op_v, dict) and isinstance(op_v.get("message"), dict):
            self._handle_message(op_v["message"])
            return

        if apply_visibility and isinstance(op_p, str) and op_p in (
            "/message/metadata/is_visually_hidden_from_conversation",
            "/message/channel",
            "/message/recipient",
            "/message/metadata",
        ):
            self._apply_visibility_op(op)
            return

        if isinstance(op_p, str) and op_p:
            if op_p == "/message/content/parts/0":
                self.last_patch_path = "/message/content/parts/0"
                # Recorded even while hidden: a provisional message may become
                # visible later; _current_candidates gates the output.
                if not self.active_is_input_echo and isinstance(op_v, (str, int, float)):
                    self._patch_raw(op)
                return
            if op_p.startswith("/message/metadata/model_dil_v2"):
                # Compact patches omit p after an explicit path. Keep DIL
                # constants/code and fallback continuations out of parts/0.
                self.last_patch_path = op_p
                self._apply_dil_op(op_p, op_o, op_v)
                return
            if op_p == "/message/end_turn" and op_v is True:
                # The answer is final here; the upstream keeps the socket
                # open for ~1s of telemetry, which callers should not wait on.
                if self._is_active_visible() and self.current_msg_raw:
                    self.turn_complete = True
                return
            if _is_hidden_text_path(op_p):
                self.last_patch_path = op_p
            return

        if op_p in (None, "") and op_o in ("append", "replace") and isinstance(op_v, str):
            if self.last_patch_path == "/message/metadata/model_dil_v2/fallbackMarkdown":
                self._apply_dil_op(self.last_patch_path, op_o, op_v)
            elif not self.active_is_input_echo and self.last_patch_path == "/message/content/parts/0":
                self._patch_raw(op)

    def _patch_raw(self, op: dict[str, Any]) -> None:
        if op.get("o") == "append":
            self.current_msg_raw += str(op.get("v") or "")
        elif op.get("o") == "replace":
            self.current_msg_raw = str(op.get("v") or "")
            self.raw_generation += 1

    def _set_dil_fallback(self, text: str) -> None:
        # Any model_dil_v2 metadata marks the message as DIL, even when it is
        # patched in after creation (gpt-6-luna does this in the same patch
        # list as the first parts/0 append). The fallback is only recorded;
        # _pick_current decides at the end whether it may complete the answer.
        self.active_has_dil = True
        self.dil_fallback_buffer = str(text or "")

    def _apply_dil_op(self, path: str, op: Any, value: Any) -> None:
        self.active_has_dil = True
        if path.endswith("/fallbackMarkdown"):
            if op == "replace":
                self._set_dil_fallback(value)
            elif op == "append":
                self._set_dil_fallback(self.dil_fallback_buffer + str(value or ""))
        elif path == "/message/metadata/model_dil_v2" and isinstance(value, dict):
            if isinstance(value.get("fallbackMarkdown"), str):
                self._set_dil_fallback(value["fallbackMarkdown"])

    def _handle_patch_list(self, items: list[Any]) -> None:
        dict_items = [item for item in items if isinstance(item, dict)]
        has_new_message = any(
            item.get("p") == "" and item.get("o") == "add" and isinstance(item.get("v"), dict) and "message" in item["v"]
            for item in dict_items
        )
        if not has_new_message:
            # ChatGPT Web often sends parts/0 append BEFORE removing is_visually_hidden_from_conversation in the same patch list.
            for item in dict_items:
                self._apply_visibility_op(item)
            for item in dict_items:
                self._handle_patch_op(item, apply_visibility=False)
            return
        for item in dict_items:
            self._handle_patch_op(item, apply_visibility=True)

    def feed_payload(self, payload: str) -> str:
        if not payload or payload == "[DONE]":
            return ""
        try:
            event = json.loads(payload)
        except Exception:
            return ""
        if not isinstance(event, dict):
            return ""

        conv_id = event.get("conversation_id")
        if conv_id:
            self.conversation_id = str(conv_id)
        val = event.get("v")
        if isinstance(val, dict) and val.get("conversation_id"):
            self.conversation_id = str(val["conversation_id"])

        if event.get("type") == "moderation":
            mod = event.get("moderation_response")
            if isinstance(mod, dict) and mod.get("blocked") is True:
                self.blocked = True

        err_val = event.get("error")
        if err_val:
            if isinstance(err_val, dict):
                err_msg = str(err_val.get("message") or err_val.get("code") or err_val)
                err_code = str(err_val.get("code") or "upstream_error")
            else:
                err_msg = str(err_val)
                err_code = "upstream_error"
            status = _infer_sse_error_status(err_code, err_msg)
            raise ImageWorkerError(
                _sanitize_error_text(err_msg) or "Upstream conversation error",
                http_status=status,
                error_code=err_code,
                conversation_id=self.conversation_id,
            )

        if isinstance(event.get("message"), dict):
            self._handle_message(event["message"])
        elif isinstance(val, dict) and isinstance(val.get("message"), dict):
            self._handle_message(val["message"])
        elif event.get("o") == "patch" and isinstance(val, list):
            self._handle_patch_list(val)
        elif isinstance(event.get("p"), str):
            self._handle_patch_op(event)
        elif isinstance(val, list):
            self._handle_patch_list(val)
        elif isinstance(val, str) and "p" not in event and event.get("o") in (None, "append", "replace"):
            self._handle_patch_op(event)

        return self._next_delta(hold_partial_prefix=True)

    def finalize_delta(self) -> str:
        return self._next_delta(hold_partial_prefix=False)


class OpenAIWebTextClient(OpenAIWebImageClient):
    """Reuses OpenAIWebImageClient cookies, sentinel PoW, Turnstile and headers for text chat."""

    def _prepare_text_conversation(
        self,
        *,
        upstream_model: str,
        partial_query_text: str,
        thinking_effort: str,
        timezone_offset_min: int,
        timezone_name: str,
        requirements: Any,
    ) -> tuple[str, str]:
        path = "/backend-api/f/conversation/prepare"
        current_effort = thinking_effort

        for attempt in range(2):
            payload: Dict[str, Any] = {
                "action": "next",
                "fork_from_shared_post": False,
                "parent_message_id": "client-created-root",
                "model": upstream_model,
                "client_prepare_state": "success",
                "timezone_offset_min": int(timezone_offset_min),
                "timezone": timezone_name,
                "conversation_mode": {"kind": "primary_assistant"},
                "system_hints": [],
                "supports_buffering": True,
                "supported_encodings": ["v1"],
                "client_contextual_info": {"app_name": "chatgpt.com"},
            }
            if partial_query_text:
                payload["partial_query"] = {
                    "id": str(uuid.uuid4()),
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": [partial_query_text]},
                }
            if current_effort:
                payload["thinking_effort"] = current_effort

            response = self.session.post(
                self.base_url + path,
                headers=self._image_headers(path, requirements, accept="application/json"),
                json=payload,
                timeout=60,
            )
            if (
                attempt == 0
                and current_effort
                and response.status_code in (400, 422)
                and "thinking_effort" in str(getattr(response, "text", "") or "").lower()
            ):
                current_effort = ""
                continue
            self._ensure_ok(response, path)
            data = response.json() if hasattr(response, "json") else {}
            return str((data or {}).get("conduit_token") or ""), current_effort

        return "", current_effort

    def stream_text(
        self,
        *,
        messages: Any,
        model: str = DEFAULT_WEB_TEXT_MODEL,
        thinking_effort: str = "",
        timezone_offset_min: int = -480,
        timezone_name: str = "Asia/Shanghai",
        stream_timeout: float = 180.0,
        on_delta: Optional[Callable[[str, str], None]] = None,
    ) -> Dict[str, Any]:
        upstream_model = normalize_web_text_model(model)
        effort = normalize_thinking_effort(thinking_effort)
        web_messages, assistant_history, last_user_text, input_ids = normalize_web_messages(messages)

        self._bootstrap()
        requirements = self._get_chat_requirements()

        conduit_token, effort = self._prepare_text_conversation(
            upstream_model=upstream_model,
            partial_query_text=last_user_text,
            thinking_effort=effort,
            timezone_offset_min=timezone_offset_min,
            timezone_name=timezone_name,
            requirements=requirements,
        )

        path = "/backend-api/f/conversation"
        tracker = WebTextStreamTracker(
            assistant_history_messages=tuple(assistant_history),
            input_message_ids=frozenset(input_ids),
        )
        resp = None

        for attempt in range(2):
            body: Dict[str, Any] = {
                "action": "next",
                "messages": web_messages,
                "parent_message_id": "client-created-root",
                "model": upstream_model,
                "client_prepare_state": "sent",
                "timezone_offset_min": int(timezone_offset_min),
                "timezone": timezone_name,
                "history_and_training_disabled": True,
                "conversation_mode": {"kind": "primary_assistant"},
                "enable_message_followups": True,
                "system_hints": [],
                "supports_buffering": True,
                "supported_encodings": ["v1"],
                "client_contextual_info": {
                    "is_dark_mode": False,
                    "time_since_loaded": 25,
                    "page_height": 900,
                    "page_width": 1200,
                    "pixel_ratio": 1,
                    "screen_height": 1080,
                    "screen_width": 1920,
                    "app_name": "chatgpt.com",
                },
                "paragen_cot_summary_display_override": "allow",
                "force_parallel_switch": "auto",
            }
            if effort:
                body["thinking_effort"] = effort

            candidate = self.session.post(
                self.base_url + path,
                headers=self._image_headers(
                    path,
                    requirements,
                    conduit_token=conduit_token,
                    accept="text/event-stream",
                ),
                json=body,
                timeout=stream_timeout,
                stream=True,
            )
            if (
                attempt == 0
                and effort
                and candidate.status_code in (400, 422)
                and "thinking_effort" in str(getattr(candidate, "text", "") or "").lower()
            ):
                candidate.close()
                effort = ""
                continue
            resp = candidate
            break

        assert resp is not None
        try:
            self._ensure_ok(resp, path)
            for payload in _iter_sse_payloads(resp):
                if payload == "[DONE]":
                    break
                delta = tracker.feed_payload(payload)
                if delta and on_delta is not None:
                    on_delta(delta, tracker.model_slug)
                if tracker.turn_complete:
                    break
        finally:
            try:
                resp.close()
            except Exception:
                pass

        tail = tracker.finalize_delta()
        if tail and on_delta is not None:
            on_delta(tail, tracker.model_slug)

        if tracker.blocked and not tracker.emitted_text:
            raise ImageWorkerError(
                "Response blocked by upstream moderation",
                http_status=400,
                error_code="content_policy_violation",
                conversation_id=tracker.conversation_id,
            )

        return {
            "ok": True,
            "text": tracker.emitted_text,
            "conversation_id": tracker.conversation_id,
            "model": tracker.model_slug or upstream_model,
        }


def _emit_ndjson(event: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(event, ensure_ascii=True) + "\n")
    sys.stdout.flush()


def _error_response_dict(exc: Exception) -> Dict[str, Any]:
    if isinstance(exc, UpstreamHTTPError):
        body_text = _sanitize_error_text(str(exc.body))
        msg = body_text or _sanitize_error_text(str(exc)) or f"Upstream HTTP {exc.status_code}"
        token_invalid = exc.status_code == 401 or _is_token_invalid_text(body_text or str(exc))
        if token_invalid and exc.status_code in (200, 403):
            status = 401
        else:
            status = exc.status_code
        code = (
            "rate_limit_exceeded"
            if status == 429
            else ("invalid_token" if status == 401 else "upstream_http_error")
        )
        return {
            "ok": False,
            "type": "error",
            "http_status": status,
            "error_code": code,
            "error": _sanitize_error_text(msg),
            "token_invalid": token_invalid,
            "retry_after": exc.retry_after,
        }
    if isinstance(exc, ImageWorkerError):
        msg = _sanitize_error_text(str(exc)) or "Upstream text generation error"
        token_invalid = exc.http_status == 401 or _is_token_invalid_text(msg)
        return {
            "ok": False,
            "type": "error",
            "http_status": 401 if token_invalid else exc.http_status,
            "error_code": exc.error_code or "upstream_error",
            "error": msg,
            "token_invalid": token_invalid,
            "retry_after": exc.retry_after,
        }
    msg = _sanitize_error_text(str(exc)) or exc.__class__.__name__
    token_invalid = _is_token_invalid_text(msg)
    return {
        "ok": False,
        "type": "error",
        "http_status": 401 if token_invalid else 502,
        "error_code": "worker_error",
        "error": msg,
        "token_invalid": token_invalid,
        "retry_after": None,
    }


def handle_text_payload(
    data: Any,
    *,
    session: Any = None,
    on_delta: Optional[Callable[[str, str], None]] = None,
) -> Dict[str, Any]:
    try:
        if not isinstance(data, dict):
            raise ImageWorkerError("Payload must be a JSON object", http_status=400, error_code="invalid_request")
        access_token = str(data.get("access_token") or "").strip()
        if not access_token:
            raise ImageWorkerError(
                "Missing access_token for Web text request",
                http_status=401,
                error_code="missing_token",
            )
        proxy = str(data.get("proxy") or "").strip()
        model = str(data.get("model") or DEFAULT_WEB_TEXT_MODEL).strip()
        effort = str(data.get("thinking_effort") or "").strip()
        tz_offset = int(data.get("timezone_offset_min") or -480)
        tz_name = str(data.get("timezone") or "Asia/Shanghai").strip() or "Asia/Shanghai"
        timeout_val = float(data.get("stream_timeout") or 180.0)

        with OpenAIWebTextClient(
            access_token,
            proxy=proxy,
            upstream_model=normalize_web_text_model(model),
            thinking_effort=normalize_thinking_effort(effort),
            session=session,
        ) as client:
            result = client.stream_text(
                messages=data.get("messages"),
                model=model,
                thinking_effort=effort,
                timezone_offset_min=tz_offset,
                timezone_name=tz_name,
                stream_timeout=timeout_val,
                on_delta=on_delta,
            )
            return result
    except Exception as exc:
        return _error_response_dict(exc)


def main() -> None:
    stream_events = "--stream" in sys.argv[1:]
    try:
        raw = sys.stdin.buffer.read(MAX_STDIN_BYTES + 1)
        if len(raw) > MAX_STDIN_BYTES:
            raise ImageWorkerError("Input payload exceeds maximum size", http_status=413, error_code="payload_too_large")
        data = json.loads(raw.decode("utf-8"))
        if isinstance(data, dict) and data.get("stream_events") is True:
            stream_events = True
    except Exception as exc:
        err = _error_response_dict(exc)
        _emit_ndjson(err)
        sys.exit(1)

    def emit_delta(delta: str, model_slug: str) -> None:
        if stream_events and delta:
            ev: Dict[str, Any] = {"type": "delta", "delta": delta}
            if model_slug:
                ev["model"] = model_slug
            _emit_ndjson(ev)

    result = handle_text_payload(data, on_delta=emit_delta if stream_events else None)
    if stream_events:
        if result.get("ok"):
            _emit_ndjson(
                {
                    "ok": True,
                    "type": "done",
                    "text": result.get("text", ""),
                    "conversation_id": result.get("conversation_id", ""),
                    "model": result.get("model", ""),
                }
            )
            sys.exit(0)
        _emit_ndjson(result)
        sys.exit(1)

    _emit_ndjson(result)
    sys.exit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()
