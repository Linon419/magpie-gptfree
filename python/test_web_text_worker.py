import io
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

PYTHON_DIR = Path(__file__).resolve().parent
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

import web_text_worker


class FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        json_data=None,
        text: str = "",
        headers: dict[str, str] | None = None,
        cookies: dict[str, str] | None = None,
        lines: list[bytes] | None = None,
    ) -> None:
        self.status_code = status_code
        self._json_data = json_data
        self._text = text
        self.headers = headers or {}
        self.cookies = cookies or {}
        self._lines = lines or []
        self.closed = False

    def json(self):
        if self._json_data is not None:
            return self._json_data
        return json.loads(self._text or "{}")

    @property
    def text(self) -> str:
        if self._text:
            return self._text
        if self._json_data is not None:
            return json.dumps(self._json_data)
        return ""

    def iter_lines(self):
        yield from self._lines

    def close(self) -> None:
        self.closed = True


class FakeSession:
    def __init__(self, handler) -> None:
        self.handler = handler
        self.calls: list[tuple[str, str, dict]] = []
        self.closed = False

    def get(self, url: str, **kwargs):
        self.calls.append(("GET", url, kwargs))
        return self.handler("GET", url, kwargs)

    def post(self, url: str, **kwargs):
        self.calls.append(("POST", url, kwargs))
        return self.handler("POST", url, kwargs)

    def close(self) -> None:
        self.closed = True


def _sse_lines(events: list[object]) -> list[bytes]:
    out = []
    for ev in events:
        if isinstance(ev, str):
            out.append(f"data: {ev}\n".encode("utf-8"))
        else:
            out.append(f"data: {json.dumps(ev)}\n".encode("utf-8"))
        out.append(b"")
    return out


class WebTextWorkerTests(unittest.TestCase):
    def test_normalization_and_messages_mapping(self) -> None:
        self.assertEqual(web_text_worker.normalize_web_text_model("gptfree-pool/gpt-6-luna"), "gpt-6-luna")
        self.assertEqual(web_text_worker.normalize_thinking_effort("minimal"), "")
        self.assertEqual(web_text_worker.normalize_thinking_effort("MAX"), "xhigh")
        self.assertEqual(web_text_worker.normalize_thinking_effort("high"), "high")

        web_msgs, history, last_user, input_ids = web_text_worker.normalize_web_messages([
            {"role": "developer", "content": "Be brief."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "user", "content": "   "},
        ])
        self.assertEqual([m["author"]["role"] for m in web_msgs], ["system", "user", "assistant", "user"])
        self.assertEqual(history, ["Hello!"])
        self.assertEqual(last_user, "What is 2+2?")
        self.assertEqual(len(input_ids), 4)

    def test_tracker_streams_v1_patches_hides_thoughts_and_strips_citations_and_history(self) -> None:
        tracker = web_text_worker.WebTextStreamTracker(assistant_history_messages=("Okay, understood.",))
        events = [
            "v1",
            {
                "v": {
                    "message": {
                        "id": "msg-thought",
                        "author": {"role": "assistant"},
                        "channel": "commentary",
                        "content": {"content_type": "thoughts", "parts": [""]},
                    },
                    "conversation_id": "conv-123",
                }
            },
            {"p": "/message/content/thoughts/0/content", "o": "append", "v": "Internal reasoning"},
            {"v": " should stay hidden"},
            {
                "v": {
                    "message": {
                        "id": "msg-final",
                        "author": {"role": "assistant"},
                        "channel": "final",
                        "recipient": "all",
                        "content": {"content_type": "text", "parts": [""]},
                    }
                }
            },
            # Partial echo followed by full history echo -> stripped cleanly
            {
                "o": "patch",
                "v": [
                    {"p": "/message/content/parts/0", "o": "append", "v": "Okay, "},
                    {"p": "/message/metadata", "o": "append", "v": {"model_slug": "gpt-6-luna"}},
                ],
            },
            {"v": "understood.\n\nAns\ue200cite"},
            {"v": "turn0\ue201wer"},
            "[DONE]",
        ]

        deltas = []
        for ev in events:
            payload = ev if isinstance(ev, str) else json.dumps(ev)
            d = tracker.feed_payload(payload)
            if d:
                deltas.append(d)
        tail = tracker.finalize_delta()
        if tail:
            deltas.append(tail)

        self.assertEqual("".join(deltas), "Answer")
        self.assertEqual(tracker.conversation_id, "conv-123")
        self.assertEqual(tracker.model_slug, "gpt-6-luna")

        # Short valid reply that happens to start like assistant history must still flush on finalize
        short_tracker = web_text_worker.WebTextStreamTracker(assistant_history_messages=("Okay, understood.",))
        short_tracker.feed_payload(json.dumps({
            "v": {"message": {"id": "a1", "author": {"role": "assistant"}, "content": {"content_type": "text", "parts": ["Okay"]}}}
        }))
        self.assertEqual(short_tracker.emitted_text, "")
        self.assertEqual(short_tracker.finalize_delta(), "Okay")

        # Real ChatGPT Web behavior: echoed prompt assistant IDs + provisional is_visually_hidden_from_conversation removed inside patch list
        echo_tracker = web_text_worker.WebTextStreamTracker(
            assistant_history_messages=("alpha",),
            input_message_ids=frozenset({"echo-assistant-1"}),
        )
        self.assertEqual(
            echo_tracker.feed_payload(json.dumps({
                "v": {"message": {"id": "echo-assistant-1", "author": {"role": "assistant"}, "content": {"content_type": "text", "parts": ["alpha"]}}}
            })),
            "",
        )
        self.assertEqual(
            echo_tracker.feed_payload(json.dumps({
                "v": {
                    "message": {
                        "id": "real-assistant-2",
                        "author": {"role": "assistant"},
                        "metadata": {"is_visually_hidden_from_conversation": True},
                        "content": {"content_type": "text", "parts": [""]},
                    }
                }
            })),
            "",
        )
        self.assertEqual(
            echo_tracker.feed_payload(json.dumps({
                "o": "patch",
                "v": [
                    {"p": "/message/content/parts/0", "o": "append", "v": "alpha-2"},
                    {"p": "/message/metadata/is_visually_hidden_from_conversation", "o": "remove"},
                ],
            })),
            "alpha-2",
        )


    def test_tracker_prefers_markdown_fallback_for_intelligent_ui_dil(self) -> None:
        tracker = web_text_worker.WebTextStreamTracker()
        # GPT-6 Intelligent UI: parts/0 carries DIL markup, metadata carries the
        # plain-Markdown fallback the server compiles for clients that cannot render it.
        tracker.feed_payload(json.dumps({
            "v": {"message": {
                "id": "ui-1",
                "author": {"role": "assistant"},
                "content": {"content_type": "text", "parts": [""]},
                "metadata": {"model_dil_v2": {"fallbackMarkdown": ""}},
            }}
        }))
        emitted = []
        emitted.append(tracker.feed_payload(json.dumps({
            "o": "patch",
            "v": [
                {"p": "/message/content/parts/0", "o": "append", "v": "## 蒙特卡洛\n\n<box border radius=\"lg\" padding={3} gap={2}>"},
                {"p": "/message/metadata/model_dil_v2/fallbackMarkdown", "o": "append", "v": "## 蒙特卡洛\n\n"},
            ],
        })))
        emitted.append(tracker.feed_payload(json.dumps({
            "o": "patch",
            "v": [
                {"p": "/message/content/parts/0", "o": "append", "v": "<text color=\"secondary\">成本、销量</text></box>"},
                {"p": "/message/metadata/model_dil_v2/fallbackMarkdown", "o": "append", "v": "成本、销量"},
            ],
        })))
        emitted.append(tracker.finalize_delta())
        text = "".join(emitted)
        self.assertIn("蒙特卡洛", text)
        self.assertIn("成本、销量", text)
        self.assertNotIn("<box", text)
        self.assertNotIn("<text", text)

        # No fallbackMarkdown at all: finalize degrades DIL to text instead of leaking tags.
        bare = web_text_worker.WebTextStreamTracker()
        bare.feed_payload(json.dumps({
            "v": {"message": {
                "id": "ui-2",
                "author": {"role": "assistant"},
                "content": {"content_type": "text", "parts": [""]},
                "metadata": {"model_dil_v2": {}},
            }}
        }))
        bare.feed_payload(json.dumps({
            "p": "/message/content/parts/0",
            "o": "append",
            "v": "{@body const x = 1}\n<box padding={2}>Hello</box>",
        }))
        bare.finalize_delta()
        self.assertEqual(bare.emitted_text, "Hello")


    def test_tracker_dil_fallback_respects_visibility_filters(self) -> None:
        def dil_message(msg_id, *, content_type="text", metadata=None, **extra):
            return {
                "v": {"message": {
                    "id": msg_id,
                    "author": {"role": "assistant"},
                    "content": {"content_type": content_type, "parts": [""]},
                    "metadata": metadata or {},
                    **extra,
                }}
            }

        def dil_patch(text, fallback):
            return {
                "o": "patch",
                "v": [
                    {"p": "/message/content/parts/0", "o": "append", "v": text},
                    {"p": "/message/metadata/model_dil_v2/fallbackMarkdown", "o": "append", "v": fallback},
                ],
            }

        # Hidden message carrying DIL must stay silent.
        hidden = web_text_worker.WebTextStreamTracker()
        hidden.feed_payload(json.dumps(dil_message(
            "hidden-1",
            metadata={
                "is_visually_hidden_from_conversation": True,
                "model_dil_v2": {"fallbackMarkdown": ""},
            },
        )))
        self.assertEqual(hidden.feed_payload(json.dumps(dil_patch("<box>secret</box>", "secret"))), "")
        self.assertEqual(hidden.finalize_delta(), "")

        # Reasoning (thoughts) message carrying DIL must stay silent.
        reasoning = web_text_worker.WebTextStreamTracker()
        reasoning.feed_payload(json.dumps(dil_message(
            "thought-1",
            content_type="thoughts",
            metadata={"model_dil_v2": {"fallbackMarkdown": ""}},
        )))
        self.assertEqual(reasoning.feed_payload(json.dumps(dil_patch("<box>thinking</box>", "thinking"))), "")
        self.assertEqual(reasoning.finalize_delta(), "")

        # Tool-directed message (recipient/channel not "all"/"final") must stay silent.
        for extra in ({"recipient": "web.run"}, {"channel": "commentary"}):
            tool = web_text_worker.WebTextStreamTracker()
            tool.feed_payload(json.dumps(dil_message(
                "tool-1",
                metadata={"model_dil_v2": {"fallbackMarkdown": ""}},
                **extra,
            )))
            self.assertEqual(tool.feed_payload(json.dumps(dil_patch("<box>tool</box>", "tool"))), "")
            self.assertEqual(tool.finalize_delta(), "")

        # A provisional hidden message that becomes visible may then stream its fallback.
        provisional = web_text_worker.WebTextStreamTracker()
        provisional.feed_payload(json.dumps(dil_message(
            "prov-1",
            metadata={
                "is_visually_hidden_from_conversation": True,
                "model_dil_v2": {"fallbackMarkdown": ""},
            },
        )))
        self.assertEqual(provisional.feed_payload(json.dumps(dil_patch("<box>later</box>", "later"))), "")
        visible_delta = provisional.feed_payload(json.dumps({
            "p": "/message/metadata/is_visually_hidden_from_conversation",
            "o": "remove",
        }))
        self.assertEqual(visible_delta, "later")
        self.assertEqual(provisional.finalize_delta(), "")


    def test_tracker_ignores_late_dil_metadata_on_plain_text(self) -> None:
        # Observed on gpt-6-luna: the message is created as plain text, then
        # model_dil_v2/fallbackMarkdown is patched in mid-stream. Switching to
        # it broke the delta prefix and truncated the answer after a few chars.
        tracker = web_text_worker.WebTextStreamTracker()
        tracker.feed_payload(json.dumps({
            "v": {"message": {
                "id": "plain-1",
                "author": {"role": "assistant"},
                "content": {"content_type": "text", "parts": [""]},
            }}
        }))
        emitted = []
        emitted.append(tracker.feed_payload(json.dumps({
            "p": "/message/content/parts/0", "o": "append", "v": "\\(正确答案：第",
        })))
        emitted.append(tracker.feed_payload(json.dumps({
            "p": "/message/metadata/model_dil_v2/fallbackMarkdown",
            "o": "append", "v": "(正确答案：第一项)",
        })))
        emitted.append(tracker.feed_payload(json.dumps({
            "p": "/message/content/parts/0", "o": "append", "v": "一项)",
        })))
        tail = tracker.finalize_delta()
        if tail:
            emitted.append(tail)
        self.assertEqual("".join(emitted), "\\(正确答案：第一项)")


    def test_tracker_accepts_late_dil_fallback_when_it_extends_prefix(self) -> None:
        # Raw parts stop early and the rest of the answer only arrives through
        # late fallbackMarkdown: ignoring it would truncate the reply.
        tracker = web_text_worker.WebTextStreamTracker()
        tracker.feed_payload(json.dumps({
            "v": {"message": {
                "id": "late-1",
                "author": {"role": "assistant"},
                "content": {"content_type": "text", "parts": [""]},
            }}
        }))
        emitted = []
        emitted.append(tracker.feed_payload(json.dumps({
            "p": "/message/content/parts/0", "o": "append", "v": "(正确答案：第",
        })))
        emitted.append(tracker.feed_payload(json.dumps({
            "p": "/message/metadata/model_dil_v2/fallbackMarkdown",
            "o": "append", "v": "(正确答案：第一项)",
        })))
        tail = tracker.finalize_delta()
        if tail:
            emitted.append(tail)
        self.assertEqual("".join(emitted), "(正确答案：第一项)")


    def test_tracker_accumulates_late_fallback_fragments_until_they_cover_emitted(self) -> None:
        # Late fallback arrives in chunks that are individually shorter than
        # the emitted text. The buffer must keep them, so the complete value
        # can still be adopted once it covers everything already sent.
        tracker = web_text_worker.WebTextStreamTracker()
        tracker.feed_payload(json.dumps({
            "v": {"message": {
                "id": "chunks-1",
                "author": {"role": "assistant"},
                "content": {"content_type": "text", "parts": [""]},
            }}
        }))
        self.assertEqual(tracker.feed_payload(json.dumps({
            "p": "/message/content/parts/0", "o": "append", "v": "Hello World",
        })), "Hello World")
        self.assertEqual(tracker.feed_payload(json.dumps({
            "p": "/message/metadata/model_dil_v2/fallbackMarkdown",
            "o": "append", "v": "Hello",
        })), "")
        self.assertEqual(tracker.dil_fallback_buffer, "Hello")
        self.assertEqual(tracker.feed_payload(json.dumps({
            "p": "/message/metadata/model_dil_v2/fallbackMarkdown",
            "o": "append", "v": " World",
        })), "")
        self.assertEqual(tracker.dil_fallback_buffer, "Hello World")
        self.assertEqual(tracker.finalize_delta(), "")
        self.assertEqual(tracker.emitted_text, "Hello World")

    def test_tracker_completes_partial_fallback_from_raw_dil(self) -> None:
        # A DIL response whose fallbackMarkdown stops early must still return
        # the rest of the answer from the raw DIL instead of truncating.
        tracker = web_text_worker.WebTextStreamTracker()
        tracker.feed_payload(json.dumps({
            "v": {"message": {
                "id": "dil-1",
                "author": {"role": "assistant"},
                "content": {"content_type": "text", "parts": [""]},
                "metadata": {"model_dil_v2": {"fallbackMarkdown": ""}},
            }}
        }))
        tracker.feed_payload(json.dumps({
            "p": "/message/content/parts/0", "o": "append",
            "v": "<box>(正确答案：第一项)</box>",
        }))
        tracker.feed_payload(json.dumps({
            "p": "/message/metadata/model_dil_v2/fallbackMarkdown",
            "o": "append", "v": "(正确答案：第",
        }))
        tracker.finalize_delta()
        self.assertEqual(tracker.emitted_text, "(正确答案：第一项)")

    def test_fallback_buffer_and_prefix_are_scoped_to_each_message(self) -> None:
        tracker = web_text_worker.WebTextStreamTracker()
        output = []
        def feed(event):
            output.append(tracker.feed_payload(json.dumps(event)))
        feed({"message": {
            "id": "first", "author": {"role": "assistant"},
            "content": {"content_type": "text", "parts": ["<box>First</box>"]},
            "metadata": {"model_dil_v2": {"fallbackMarkdown": "First"}},
        }})
        feed({"message": {
            "id": "second", "author": {"role": "assistant"},
            "content": {"content_type": "text", "parts": ["Second"]},
        }})
        self.assertEqual(tracker.dil_fallback_buffer, "")
        self.assertFalse(tracker.active_has_dil)
        for chunk in ["Sec", "ond", " complete"]:
            feed({"p": "/message/metadata/model_dil_v2/fallbackMarkdown",
                  "o": "append", "v": chunk})
        output.append(tracker.finalize_delta())
        self.assertEqual("".join(output), "First\n\nSecond complete")
        self.assertEqual(tracker.dil_fallback_buffer, "Second complete")

    def test_compact_dil_patches_follow_their_own_path(self) -> None:
        for compact in [{"v": " complete"}, {"o": "append", "v": " complete"}]:
            with self.subTest(compact=compact):
                tracker = web_text_worker.WebTextStreamTracker()
                events = [
                    {"message": {"id": "a", "author": {"role": "assistant"},
                     "content": {"content_type": "text", "parts": ["Answer"]}}},
                    {"p": "/message/metadata/model_dil_v2/fallbackMarkdown",
                     "o": "append", "v": "Answer"},
                    compact,
                ]
                output = [tracker.feed_payload(json.dumps(e)) for e in events]
                output.append(tracker.finalize_delta())
                self.assertEqual("".join(output), "Answer complete")
                self.assertEqual(tracker.current_msg_raw, "Answer")

    def test_compact_string_continuations_inside_patch_lists(self) -> None:
        for path, initial in [
            ("/message/content/parts/0", ""),
            ("/message/metadata/model_dil_v2/fallbackMarkdown", "Answer"),
        ]:
            for compact in [{"v": " complete"}, {"o": "append", "v": " complete"}]:
                with self.subTest(path=path, compact=compact):
                    tracker = web_text_worker.WebTextStreamTracker()
                    output = [tracker.feed_payload(json.dumps({"message": {
                        "id": "a", "author": {"role": "assistant"},
                        "content": {"content_type": "text", "parts": [initial]},
                    }}))]
                    output.append(tracker.feed_payload(json.dumps({"o": "patch", "v": [
                        {"p": path, "o": "append", "v": "Answer"}, compact,
                    ]})))
                    output.append(tracker.finalize_delta())
                    self.assertEqual("".join(output), "Answer complete")
                    if "fallbackMarkdown" in path:
                        self.assertEqual(tracker.current_msg_raw, "Answer")
                        self.assertEqual(tracker.dil_fallback_buffer, "Answer complete")

    def test_compact_patch_list_keeps_hidden_text_and_dil_code_silent(self) -> None:
        for path, hidden in [
            ("/message/metadata/model_dil_v2/fallbackMarkdown", True),
            ("/message/metadata/model_dil_v2/code", False),
        ]:
            with self.subTest(path=path):
                tracker = web_text_worker.WebTextStreamTracker()
                output = [tracker.feed_payload(json.dumps({"message": {
                    "id": "a", "author": {"role": "assistant"},
                    "content": {"content_type": "text", "parts": [""]},
                    "metadata": {"is_visually_hidden_from_conversation": hidden},
                }}))]
                output.append(tracker.feed_payload(json.dumps({"o": "patch", "v": [
                    {"p": path, "o": "append", "v": "internal"}, {"v": " What"},
                ]})))
                output.append(tracker.finalize_delta())
                self.assertEqual("".join(output), "")

    def test_dil_constants_do_not_become_answer_text(self) -> None:
        for path in ["constants/0", "code"]:
            with self.subTest(path=path):
                tracker = web_text_worker.WebTextStreamTracker()
                events = [
                    {"message": {"id": "a", "author": {"role": "assistant"},
                     "content": {"content_type": "text", "parts": ["(正确答案：第 3 项)"]}}},
                    {"p": "/message/metadata/model_dil_v2/" + path,
                     "o": "replace", "v": "internal"},
                    {"v": "\n\nWhat"},
                    {"o": "append", "v": " hidden"},
                ]
                output = [tracker.feed_payload(json.dumps(e)) for e in events]
                output.append(tracker.finalize_delta())
                self.assertEqual("".join(output), "(正确答案：第 3 项)")

    def test_real_gpt6_luna_badge_stream_renders_parts_not_fallback(self) -> None:
        # Shape captured from gpt-6-luna on 2026-11-17: a hidden provisional
        # message, then one patch list that appends a <badge>, removes the
        # hidden flag and attaches model_dil_v2 whose fallback is escaped and
        # incomplete. The old tracker emitted "(Correct answer)" and stalled.
        tracker = web_text_worker.WebTextStreamTracker()
        out = [tracker.feed_payload(json.dumps({"v": {"message": {
            "id": "a", "author": {"role": "assistant"},
            "content": {"content_type": "text", "parts": [""]},
            "metadata": {"is_visually_hidden_from_conversation": True},
            "recipient": "all", "channel": "final",
        }}}))]
        out.append(tracker.feed_payload(json.dumps({"o": "patch", "v": [
            {"p": "/message/content/parts/0", "o": "append",
             "v": "<badge color=\"success\">Correct answer</badge>\n\n**"},
            {"p": "/message/metadata/is_visually_hidden_from_conversation", "o": "remove"},
            {"p": "/message/metadata", "o": "append", "v": {"model_dil_v2": {
                "code": "DIL.render()", "fallbackMarkdown": "(Correct answer)\n\n**"}}},
        ]})))
        out.append(tracker.feed_payload(json.dumps({"v": [
            {"p": "/message/content/parts/0", "o": "append",
             "v": "Option C: What address did the database contain?**\n\n**Why:** stored time."},
            {"p": "/message/metadata/model_dil_v2/code", "o": "replace", "v": "DIL.render(x)"},
            {"p": "/message/metadata/model_dil_v2/fallbackMarkdown", "o": "append",
             "v": "\\: stored time\\."},
        ]})))
        out.append(tracker.finalize_delta())
        self.assertEqual("".join(out),
            "Correct answer\n\n**Option C: What address did the database contain?**\n\n**Why:** stored time.")
        self.assertGreaterEqual(sum(1 for c in out if c), 2, "answer must stream, not arrive at the end")

    def test_dil_tags_split_across_chunks_never_leak(self) -> None:
        source = ("Intro\n\n<box border radius=\"lg\" padding={3}>\n"
                  "  <text color=\"secondary\" size=\"sm\">成本、销量</text>\n"
                  "  <button onClick={()=>GenUI.issueNewTurn(\"a>b\")}>Go</button>\n"
                  "</box>\n\nDone")
        tracker = web_text_worker.WebTextStreamTracker()
        tracker.feed_payload(json.dumps({"v": {"message": {
            "id": "m", "author": {"role": "assistant"},
            "content": {"content_type": "text", "parts": [""]},
            "metadata": {"model_dil_v2": {}},
        }}}))
        out = []
        for i in range(0, len(source), 3):
            d = tracker.feed_payload(json.dumps(
                {"p": "/message/content/parts/0", "o": "append", "v": source[i:i + 3]}))
            self.assertNotIn("<", d)
            self.assertNotIn("=>", d)
            out.append(d)
        out.append(tracker.finalize_delta())
        self.assertEqual("".join(out), "Intro\n\n成本、销量\nGo\n\nDone")

    def test_dil_rendering_keeps_code_blocks_and_drops_appblocks(self) -> None:
        source = ("Use:\n\n```html\n<div class=\"x\">{a}</div>\n```\n\n"
                  "<AppBlock title=\"Drum\">\n<div>app</div>\n<script>if(a<b){}</script>\n</AppBlock>\n\nEnd")
        self.assertEqual(web_text_worker._dil_to_text(source),
                         "Use:\n\n```html\n<div class=\"x\">{a}</div>\n```\n\nEnd")
        # Streaming prefixes stay prefixes of the final rendering.
        final = web_text_worker._dil_to_text(source)
        for i in range(len(source) + 1):
            self.assertTrue(final.startswith(web_text_worker._dil_to_text(source[:i], final=False)), i)

    def test_visible_end_turn_completes_the_turn(self) -> None:
        tracker = web_text_worker.WebTextStreamTracker()
        tracker.feed_payload(json.dumps({"v": {"message": {
            "id": "sys", "author": {"role": "system"}, "end_turn": True,
            "content": {"content_type": "text", "parts": [""]},
        }}}))
        self.assertFalse(tracker.turn_complete)
        tracker.feed_payload(json.dumps({"v": {"message": {
            "id": "a", "author": {"role": "assistant"},
            "content": {"content_type": "text", "parts": [""]},
        }}}))
        tracker.feed_payload(json.dumps({"p": "/message/content/parts/0", "o": "append", "v": "Done"}))
        self.assertFalse(tracker.turn_complete)
        tracker.feed_payload(json.dumps({"v": [
            {"p": "/message/status", "o": "replace", "v": "finished_successfully"},
            {"p": "/message/end_turn", "o": "replace", "v": True},
        ]}))
        self.assertTrue(tracker.turn_complete)
        self.assertEqual(tracker.emitted_text, "Done")

    def test_end_to_end_stream_with_prepare_conduit_and_effort_fallback(self) -> None:
        conv_calls: list[dict] = []

        def handler(method: str, url: str, kwargs: dict):
            if method == "GET" and url == "https://chatgpt.com/":
                return FakeResponse(status_code=200, cookies={"oai-did": "did-1"})
            if url.endswith("/backend-api/sentinel/chat-requirements/prepare"):
                return FakeResponse(status_code=200, json_data={"prepare_token": "prep-1"})
            if url.endswith("/backend-api/sentinel/chat-requirements/finalize"):
                return FakeResponse(status_code=200, json_data={"token": "req-tok-1"})
            if url.endswith("/backend-api/f/conversation/prepare"):
                req_json = kwargs.get("json") or {}
                self.assertEqual(req_json.get("system_hints"), [])
                return FakeResponse(status_code=200, json_data={"conduit_token": "conduit-abc"})
            if url.endswith("/backend-api/f/conversation"):
                req_json = kwargs.get("json") or {}
                conv_calls.append(req_json)
                if "thinking_effort" in req_json:
                    return FakeResponse(
                        status_code=400,
                        json_data={"detail": "thinking_effort is not supported for this model"},
                    )
                self.assertEqual(kwargs["headers"].get("X-Conduit-Token"), "conduit-abc")
                self.assertTrue(req_json.get("history_and_training_disabled"))
                return FakeResponse(
                    status_code=200,
                    lines=_sse_lines([
                        "v1",
                        {
                            "v": {
                                "message": {
                                    "id": "m1",
                                    "author": {"role": "assistant"},
                                    "content": {"content_type": "text", "parts": [""]},
                                },
                                "conversation_id": "conv-web-1",
                            }
                        },
                        {
                            "o": "patch",
                            "v": [
                                {"p": "/message/content/parts/0", "o": "append", "v": "Hello"},
                                {"p": "/message/metadata", "o": "append", "v": {"model_slug": "gpt-5.6-luna"}},
                            ],
                        },
                        {"v": " world"},
                        "[DONE]",
                    ]),
                )
            raise AssertionError(f"Unexpected call: {method} {url}")

        deltas: list[str] = []
        session = FakeSession(handler)
        res = web_text_worker.handle_text_payload(
            {
                "access_token": "tok-1",
                "model": "gpt-5.6-luna",
                "thinking_effort": "high",
                "messages": [{"role": "user", "content": "Greet me"}],
            },
            session=session,
            on_delta=lambda d, _m: deltas.append(d),
        )
        self.assertTrue(res["ok"])
        self.assertEqual(deltas, ["Hello", " world"])
        self.assertEqual(res["text"], "Hello world")
        self.assertEqual(len(conv_calls), 2)
        self.assertEqual(conv_calls[0].get("thinking_effort"), "high")
        self.assertNotIn("thinking_effort", conv_calls[1])

    def test_error_mapping_401_and_429_and_sse_rate_limit(self) -> None:
        def rate_limit_handler(method: str, url: str, kwargs: dict):
            if method == "GET" and url == "https://chatgpt.com/":
                return FakeResponse(status_code=200)
            if "/sentinel/chat-requirements/" in url:
                return FakeResponse(status_code=429, headers={"Retry-After": "45"}, json_data={"detail": "slow down"})
            raise AssertionError(url)

        res429 = web_text_worker.handle_text_payload(
            {"access_token": "tok-1", "model": "gpt-6-luna", "messages": [{"role": "user", "content": "hi"}]},
            session=FakeSession(rate_limit_handler),
        )
        self.assertFalse(res429["ok"])
        self.assertEqual(res429["http_status"], 429)
        self.assertEqual(res429["retry_after"], 45)
        self.assertEqual(res429["error_code"], "rate_limit_exceeded")

        # SSE stream-embedded error should also classify cleanly
        tracker = web_text_worker.WebTextStreamTracker()
        with self.assertRaises(web_text_worker.ImageWorkerError) as ctx:
            tracker.feed_payload(json.dumps({"error": {"code": "usage_limit", "message": "Too many requests"}}))
        self.assertEqual(ctx.exception.http_status, 429)


if __name__ == "__main__":
    unittest.main()
