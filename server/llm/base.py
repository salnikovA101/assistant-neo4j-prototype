import asyncio
import base64
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from openai import AsyncOpenAI

from server.llm.stream_events import (
    ContentThinkSplitter,
    StreamEvent,
    ToolCallAssembler,
    assembled_to_openai_tool_calls,
    build_assistant_replay,
    normalize_chunk,
    preview_tool_result,
)
from server.tools.source_registry import tool_history_stub
from server.utils.config import OpenAIProfile

logger = logging.getLogger(__name__)

# Final user-facing strip: generic <think>, Gemma channel thoughts, bare <|think|>.
# Also drop unclosed think blocks (truncated streams / broken tags).
_THINK_TAG_RE = re.compile(
    r"(?s)(?:<think>.*?(?:</think>|$)|"
    r"<\|channel\>thought.*?(?:<channel\|>|$)|"
    r"<\|think\|>)"
)

# Untagged English chain-of-thought that some OpenRouter models dump into content
# after tool calls (seen with qwen3.* when reasoning lands in Completion).
_COT_MARKER_RE = re.compile(
    r"(?im)^(?:The user wants|I have gathered|I need to|"
    r"Drafting the response|Structure of the answer|"
    r"Final check(?: on constraints)?|Let's write\.?|Plan:)\b"
)
_CYRILLIC_LINE_RE = re.compile(r"(?m)^[^\n]*[А-Яа-яЁё]{3,}")


def strip_leaked_cot_preamble(text: str) -> str:
    """
    Drop untagged English planning preamble before a Cyrillic final answer.

    OpenRouter sometimes puts Qwen/DeepSeek "thinking" into Completion/content
    without <think> tags. Our stream then treats it as the answer. If we detect
    multiple CoT markers and a later Cyrillic answer start, keep only the answer.
    """
    raw = (text or "").strip()
    if not raw:
        return ""

    cleaned = _THINK_TAG_RE.sub("", raw).strip()
    if not cleaned:
        return ""

    first_cot = _COT_MARKER_RE.search(cleaned)
    if not first_cot or first_cot.start() > 240:
        return cleaned

    for match in _CYRILLIC_LINE_RE.finditer(cleaned):
        if match.start() <= first_cot.start():
            continue
        preamble = cleaned[: match.start()]
        if len(preamble) < 180:
            continue
        if len(_COT_MARKER_RE.findall(preamble)) < 2:
            continue
        answer = cleaned[match.start() :].strip()
        if answer:
            logger.info(
                "Stripped untagged CoT preamble from final answer (%s chars)",
                len(preamble),
            )
            return answer
    return cleaned



UI_THINK_EFFORTS = ("low", "medium", "xhigh")


def parse_ui_think_effort(value: Any) -> Optional[str]:
    """Accept ChatGPT/Cursor-style UI values: low | medium | xhigh."""
    if not isinstance(value, str):
        return None
    effort = value.strip().lower()
    if effort in UI_THINK_EFFORTS:
        return effort
    return None


def _reasoning_kwargs(
    profile: OpenAIProfile, effort_override: Optional[str] = None
) -> Dict[str, Any]:
    """
    Build request kwargs that enable/disable thinking for the whole agentic loop
    (first turn and every turn after tool results).

    - OpenAI SDK top-level reasoning_effort (LM Studio / OpenAI gateways)
    - OpenRouter: extra_body.reasoning
    - DeepSeek direct: extra_body.thinking
    - Qwen3.8: extra_body.chat_template_kwargs.preserve_thinking (+ body flag)

    Note: Gemma in LM Studio only accepts reasoning on/off (not high/medium).
    Sending an OpenAI effort still works (LMS warns and falls back to on);
    post-tool thinking for Gemma additionally needs think_token on tool content.
    """
    if not profile.think:
        return {
            "reasoning_effort": "none",
            "extra_body": {
                "reasoning": {"enabled": False, "effort": "none"},
                "thinking": {"type": "disabled"},
            },
        }

    effort = (effort_override or profile.think_effort or "high").strip().lower()
    # LM Studio Gemma: on/off only — keep SDK-valid effort for OpenAI/DeepSeek,
    # and always pass enabled=true for local templates.
    template_kwargs: Dict[str, Any] = {"enable_thinking": True}
    extra: Dict[str, Any]
    if effort in {"on", "off"}:
        enabled = effort == "on"
        template_kwargs = {"enable_thinking": enabled}
        # Omit top-level reasoning_effort: LM Studio Gemma only accepts on/off and
        # warns on high/medium; extra_body is enough (verified).
        extra = {
            "reasoning": {"enabled": enabled},
            "thinking": {"type": "enabled" if enabled else "disabled"},
            "chat_template_kwargs": template_kwargs,
        }
        _apply_preserve_thinking(extra, template_kwargs, profile)
        return {"extra_body": extra}

    if effort not in {"low", "medium", "high", "max", "xhigh", "minimal"}:
        effort = "high"
    sdk_effort = "high" if effort == "max" else effort
    extra = {
        "reasoning": {"enabled": True, "effort": effort},
        "thinking": {"type": "enabled"},
        "chat_template_kwargs": template_kwargs,
    }
    _apply_preserve_thinking(extra, template_kwargs, profile)
    return {
        "reasoning_effort": sdk_effort,
        "extra_body": extra,
    }


def _apply_preserve_thinking(
    extra: Dict[str, Any],
    template_kwargs: Dict[str, Any],
    profile: OpenAIProfile,
) -> None:
    """Qwen3.8: keep historical think blocks (vLLM template + Qwen Cloud body)."""
    if not profile.preserve_thinking:
        return
    template_kwargs["preserve_thinking"] = True
    extra["preserve_thinking"] = True


def _sampling_kwargs(profile: OpenAIProfile) -> Dict[str, Any]:
    """OpenAI-standard sampling on the request; vendor extras in extra_body."""
    kwargs: Dict[str, Any] = {
        "temperature": profile.temperature,
        "max_tokens": profile.max_output_tokens,
    }
    if profile.top_p is not None:
        kwargs["top_p"] = profile.top_p
    if profile.presence_penalty is not None:
        kwargs["presence_penalty"] = profile.presence_penalty
    extra: Dict[str, Any] = {}
    if profile.top_k is not None:
        extra["top_k"] = profile.top_k
    if profile.min_p is not None:
        extra["min_p"] = profile.min_p
    if profile.repetition_penalty is not None:
        extra["repetition_penalty"] = profile.repetition_penalty
    if extra:
        kwargs["extra_body"] = extra
    return kwargs


def _merge_request_kwargs(*parts: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow-merge chat kwargs; extra_body dicts are combined."""
    out: Dict[str, Any] = {}
    extra: Dict[str, Any] = {}
    for part in parts:
        body = part.get("extra_body")
        if isinstance(body, dict):
            extra.update(body)
        out.update({k: v for k, v in part.items() if k != "extra_body"})
    if extra:
        out["extra_body"] = extra
    return out


def _with_think_token(system_prompt: str, profile: OpenAIProfile) -> str:
    """
    Gemma/LM Studio: thinking after tool results only stays on if <|think|> is
    present in the rendered prompt on EVERY request of the loop. Putting it in
    system (not only the first user turn) covers post-tool generations.
    """
    token = (profile.think_token or "").strip()
    if not profile.think or not token:
        return system_prompt
    if token in system_prompt:
        return system_prompt
    if system_prompt:
        return f"{token}\n{system_prompt}"
    return token


def _tool_result_content(result: str, profile: OpenAIProfile) -> str:
    """
    Gemma/LM Studio: after tool results the chat template often does not reopen
    the think channel. Prefixing the tool message with think_token forces a new
    reasoning block over the REAL tool output (verified: plain tool → rt=0,
    '<|think|>\\n'+tool → rt>0).
    """
    text = str(result)
    token = (profile.think_token or "").strip()
    if not profile.think or not token:
        return text
    if text.startswith(token):
        return text
    return f"{token}\n{text}"


def _tool_budget_footer(turn: int, max_turns: int) -> str:
    """
    Hard tool-call budget notice appended at the END of the tool payload
    (recency bias; avoids Lost-in-the-Middle when UNIT blocks are long).
    """
    used = max(1, int(turn))
    budget = max(1, int(max_turns))
    if used >= budget:
        return (
            f"\n\n[Tool budget: {used}/{budget} exhausted. "
            "Answer NOW from retrieved evidence. Do not call tools again.]"
        )
    left = budget - used
    calls = "call" if left == 1 else "calls"
    return (
        f"\n\n[Tool budget: {used}/{budget} used. "
        f"{left} {calls} left, or answer now.]"
    )


def _assistant_message_dict(message: Any, profile: OpenAIProfile | None = None) -> Dict[str, Any]:
    """
    Replay assistant turn into the next request, preserving reasoning fields.

    DeepSeek requires reasoning_content after tool calls; OpenRouter needs
    reasoning / reasoning_details. model_dump(exclude_none=True) keeps extras.

    For Gemma/LM Studio (think_token set): if content is empty, put think_token
    into content so the post-tool generation reopens the think channel.
    """
    dumped = message.model_dump(exclude_none=True)
    # Belt-and-suspenders: some SDK builds park extras only in model_extra.
    extra = getattr(message, "model_extra", None) or {}
    for key in ("reasoning_content", "reasoning", "reasoning_details"):
        if key not in dumped and key in extra and extra[key] is not None:
            dumped[key] = extra[key]

    if profile is not None:
        token = (profile.think_token or "").strip()
        if profile.think and token:
            content = dumped.get("content")
            if content is None or (isinstance(content, str) and not content.strip()):
                dumped["content"] = token
    return dumped


class BaseLLMProvider(ABC):
    """
    Базовый провайдер LLM на основе OpenAI-совместимого SDK.

    Содержит конкретную реализацию generate_response — общую для всех
    провайдеров (OpenAI, Gemini и др.).

    Подклассы обязаны реализовать unload() и warmup().
    """

    def __init__(self, profile: OpenAIProfile) -> None:
        self.profile = profile
        self.client = AsyncOpenAI(
            base_url=profile.base_url,
            api_key=profile.api_key or "",
        )
        logger.info(
            f"[{self.__class__.__name__}] Инициализирован: "
            f"model={profile.model}, url={profile.base_url}"
        )

    async def generate_response(
        self,
        user_text: str,
        image_bytes: Optional[bytes] = None,
        prompt: str = "",
        history: Optional[List[Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_map: Optional[Dict[str, Callable]] = None,
        think_effort: Optional[str] = None,
    ) -> str:
        """
        Генерирует текстовый ответ на основе входных данных.

        Collects generate_response_stream events and returns final content.
        """
        final = ""
        try:
            async for event in self.generate_response_stream(
                user_text=user_text,
                image_bytes=image_bytes,
                prompt=prompt,
                history=history,
                tools=tools,
                tool_map=tool_map,
                think_effort=think_effort,
            ):
                if event.type == "content":
                    final += event.data.get("delta") or ""
                elif event.type == "error":
                    return f"Ошибка: {event.data.get('message', 'unknown')}"
                elif event.type == "done":
                    # Prefer assembled final_content if present.
                    if event.data.get("final_content") is not None:
                        final = event.data["final_content"]
            return final
        except Exception as e:
            logger.error(f"[{self.__class__.__name__}] Ошибка generate_response: {e}")
            return f"Ошибка: {e}"

    async def generate_response_stream(
        self,
        user_text: str,
        image_bytes: Optional[bytes] = None,
        prompt: str = "",
        history: Optional[List[Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_map: Optional[Dict[str, Callable]] = None,
        think_effort: Optional[str] = None,
    ) -> AsyncIterator[StreamEvent]:
        """
        Stream thinking / tool_call / tool_result / content events for one user turn.
        Yields a final done event with final_content (think tags stripped).
        """
        start = time.perf_counter()
        messages: List[Dict[str, Any]] = []
        final_content_parts: List[str] = []
        history_tool_messages: List[Dict[str, Any]] = []

        try:
            system_prompt = _with_think_token(prompt, self.profile)
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if history:
                messages.extend(history)

            content: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
            if image_bytes:
                b64 = base64.b64encode(image_bytes).decode()
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    }
                )
            messages.append({"role": "user", "content": content})

            kwargs: Dict[str, Any] = _merge_request_kwargs(
                {
                    "model": self.profile.model,
                    "messages": messages,
                    "stream": True,
                },
                _sampling_kwargs(self.profile),
                _reasoning_kwargs(self.profile, think_effort),
            )
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"

            turns = 0
            max_turns = max(1, int(self.profile.max_turns))
            label = "turn0"

            while True:
                assembler = ToolCallAssembler()
                splitter = ContentThinkSplitter(self.profile)
                content_acc = ""
                reasoning_parts: List[str] = []
                usage_holder: Any = None

                stream = await self._create_chat_stream(kwargs)
                async for chunk in stream:
                    usage_holder = getattr(chunk, "usage", None) or usage_holder
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    norm = normalize_chunk(delta, splitter)
                    if norm.thinking:
                        reasoning_parts.append(norm.thinking)
                        yield StreamEvent("thinking", {"delta": norm.thinking})
                    if norm.content:
                        content_acc += norm.content
                        final_content_parts.append(norm.content)
                        yield StreamEvent("content", {"delta": norm.content})
                    if norm.tool_call_deltas:
                        assembler.push(norm.tool_call_deltas)

                # Flush any held content from the think-tag splitter.
                flush_think, flush_content = splitter.flush()
                if flush_think:
                    reasoning_parts.append(flush_think)
                    yield StreamEvent("thinking", {"delta": flush_think})
                if flush_content:
                    content_acc += flush_content
                    final_content_parts.append(flush_content)
                    yield StreamEvent("content", {"delta": flush_content})

                self._log_stream_usage(label, usage_holder, reasoning_parts)

                tool_calls = assembler.finish()
                if not tool_calls or turns >= max_turns:
                    if tool_calls and turns >= max_turns:
                        logger.warning(
                            "LLM still requested tools after budget exhausted "
                            "(turns=%s max_turns=%s); returning partial content",
                            turns,
                            max_turns,
                        )
                    break

                turns += 1
                label = f"turn{turns}"
                logger.debug(f"Tool loop turn {turns}/{max_turns}")

                # Replay assistant turn (with reasoning) then execute tools.
                # Content emitted during a tool-calling turn is usually empty;
                # do not treat it as final answer — drop from final_content_parts
                # for this turn's content only (already appended). Revert those
                # tokens from the user-facing final answer.
                if content_acc:
                    # Remove this turn's content from final answer assembly:
                    # tool-call turns should not pollute the final reply.
                    joined = "".join(final_content_parts)
                    if joined.endswith(content_acc):
                        final_content_parts = [joined[: -len(content_acc)]] if joined[: -len(content_acc)] else []
                    else:
                        # Fallback: rebuild without last content_acc occurrence.
                        final_content_parts = [joined.replace(content_acc, "", 1)]

                messages.append(
                    build_assistant_replay(
                        content=content_acc or None,
                        tool_calls=tool_calls,
                        reasoning_parts=reasoning_parts,
                        profile=self.profile,
                    )
                )
                history_tool_messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": assembled_to_openai_tool_calls(tool_calls),
                    }
                )
                budget_footer = _tool_budget_footer(turns, max_turns)

                for tc in tool_calls:
                    try:
                        args = json.loads(tc.arguments) if tc.arguments else {}
                    except json.JSONDecodeError:
                        args = {}
                    if not isinstance(args, dict):
                        args = {}

                    yield StreamEvent(
                        "tool_call",
                        {
                            "id": tc.id,
                            "name": tc.name,
                            "arguments": args if args else tc.arguments,
                        },
                    )

                    fn = tool_map.get(tc.name) if tool_map else None
                    ok = True
                    display_result = ""
                    try:
                        if not fn:
                            raise KeyError(f"function '{tc.name}' not found")
                        logger.debug(f"Вызов инструмента '{tc.name}', args={args}")
                        if asyncio.iscoroutinefunction(fn):
                            result = await fn(**args)
                        else:
                            result = await asyncio.to_thread(fn, **args)
                        display_result = str(result).rstrip()
                        payload = f"{display_result}{budget_footer}"
                    except Exception as tool_err:
                        ok = False
                        logger.error(
                            "Инструмент '%s' ошибка: %s", tc.name, tool_err
                        )
                        display_result = f"Error: {tool_err}"
                        payload = f"{display_result}{budget_footer}"

                    yield StreamEvent(
                        "tool_result",
                        {
                            "id": tc.id,
                            "name": tc.name,
                            "ok": ok,
                            # Full tool output for UI (small scrollable window).
                            "result": display_result,
                            # Backward-compatible short preview.
                            "preview": preview_tool_result(display_result),
                        },
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": _tool_result_content(payload, self.profile),
                        }
                    )
                    history_tool_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": tool_history_stub(display_result, ok=ok),
                        }
                    )

                kwargs["messages"] = messages
                kwargs["tool_choice"] = "none" if turns >= max_turns else "auto"

            text = "".join(final_content_parts)
            text = strip_leaked_cot_preamble(text)

            logger.debug(
                f"[{self.__class__.__name__}] Ответ за {time.perf_counter() - start:.2f}s"
            )
            yield StreamEvent(
                "done",
                {
                    "final_content": text,
                    "has_graph": False,
                    "history_tool_messages": history_tool_messages,
                },
            )

        except Exception as e:
            logger.error(
                f"[{self.__class__.__name__}] Ошибка generate_response_stream: {e}"
            )
            yield StreamEvent("error", {"message": str(e)})

    async def _create_chat_stream(self, kwargs: Dict[str, Any]):
        """
        Create a streaming completion. Prefer stream_options.include_usage when
        supported; fall back without it for Ollama / older gateways.
        """
        with_usage = {**kwargs, "stream_options": {"include_usage": True}}
        try:
            return await self.client.chat.completions.create(**with_usage)
        except Exception as e:
            msg = str(e).lower()
            if "stream_options" in msg or "include_usage" in msg or "unexpected" in msg:
                logger.debug("Retrying stream without stream_options: %s", e)
                return await self.client.chat.completions.create(**kwargs)
            # Some servers reject unknown fields with a generic 400 — retry once.
            try:
                return await self.client.chat.completions.create(**kwargs)
            except Exception:
                raise e

    def _log_stream_usage(
        self, label: str, usage: Any, reasoning_parts: List[str]
    ) -> None:
        details = getattr(usage, "completion_tokens_details", None) if usage else None
        reasoning_tokens = (
            getattr(details, "reasoning_tokens", None) if details else None
        )
        logger.info(
            "LLM %s reasoning_tokens=%s has_reasoning_content=%s",
            label,
            reasoning_tokens,
            bool("".join(reasoning_parts)),
        )

    def _log_reasoning_usage(self, label: str, response: Any) -> None:
        usage = getattr(response, "usage", None)
        details = getattr(usage, "completion_tokens_details", None) if usage else None
        reasoning_tokens = (
            getattr(details, "reasoning_tokens", None) if details else None
        )
        msg = response.choices[0].message
        has_rc = bool(
            getattr(msg, "reasoning_content", None)
            or (getattr(msg, "model_extra", None) or {}).get("reasoning_content")
            or (getattr(msg, "model_extra", None) or {}).get("reasoning")
        )
        logger.info(
            "LLM %s reasoning_tokens=%s has_reasoning_content=%s",
            label,
            reasoning_tokens,
            has_rc,
        )

    @abstractmethod
    async def unload(self) -> None:
        """Выгрузить модель из памяти (если поддерживается провайдером)."""

    @abstractmethod
    async def warmup(self) -> None:
        """Прогреть / загрузить модель (если поддерживается провайдером)."""
