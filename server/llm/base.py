import asyncio
import base64
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from openai import AsyncOpenAI

from server.utils.config import OpenAIProfile

logger = logging.getLogger(__name__)

# Final user-facing strip: generic <think>, Gemma channel thoughts, bare <|think|>.
_THINK_TAG_RE = re.compile(
    r"(?s)(?:<think>.*?</think>|"
    r"<\|channel\>thought.*?<channel\|>|"
    r"<\|think\|>)"
)


def _reasoning_kwargs(profile: OpenAIProfile) -> Dict[str, Any]:
    """
    Build request kwargs that enable/disable thinking for the whole agentic loop
    (first turn and every turn after tool results).

    - OpenAI SDK top-level reasoning_effort (LM Studio / OpenAI gateways)
    - OpenRouter: extra_body.reasoning
    - DeepSeek direct: extra_body.thinking

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

    effort = (profile.think_effort or "high").strip().lower()
    # LM Studio Gemma: on/off only — keep SDK-valid effort for OpenAI/DeepSeek,
    # and always pass enabled=true for local templates.
    if effort in {"on", "off"}:
        enabled = effort == "on"
        # Omit top-level reasoning_effort: LM Studio Gemma only accepts on/off and
        # warns on high/medium; extra_body is enough (verified).
        return {
            "extra_body": {
                "reasoning": {"enabled": enabled},
                "thinking": {"type": "enabled" if enabled else "disabled"},
                "chat_template_kwargs": {"enable_thinking": enabled},
            },
        }

    if effort not in {"low", "medium", "high", "max", "xhigh", "minimal"}:
        effort = "high"
    sdk_effort = "high" if effort == "max" else effort
    return {
        "reasoning_effort": sdk_effort,
        "extra_body": {
            "reasoning": {"enabled": True, "effort": effort},
            "thinking": {"type": "enabled"},
            "chat_template_kwargs": {"enable_thinking": True},
        },
    }


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
    ) -> str:
        """
        Генерирует текстовый ответ на основе входных данных.

        Args:
            user_text: Текст запроса пользователя.
            image_bytes: Опциональное изображение.
            prompt: Системный промпт.
            history: История диалога в формате OpenAI messages.
            tools: Список инструментов в формате OpenAI tool schema.
            tool_map: Карта {имя_функции: callable} для вызова инструментов.

        Returns:
            Текстовый ответ модели (think-теги всегда убираются из вывода).
        """
        try:
            start = time.perf_counter()
            messages: List[Dict[str, Any]] = []

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

            # Reasoning kwargs are reused on every tool-loop iteration so
            # thinking stays enabled after tool results (not only pre-tool).
            kwargs: Dict[str, Any] = {
                "model": self.profile.model,
                "messages": messages,
                "temperature": self.profile.temperature,
                "max_tokens": self.profile.max_output_tokens,
                **_reasoning_kwargs(self.profile),
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"

            response = await self.client.chat.completions.create(**kwargs)
            logger.debug(response)
            message = response.choices[0].message
            self._log_reasoning_usage("turn0", response)

            turns = 0
            max_turns = max(1, int(self.profile.max_turns))
            while message.tool_calls and turns < max_turns:
                turns += 1
                logger.debug(f"Tool loop turn {turns}/{max_turns}")
                messages.append(_assistant_message_dict(message, self.profile))
                budget_footer = _tool_budget_footer(turns, max_turns)

                for tc in message.tool_calls:
                    fn = tool_map.get(tc.function.name) if tool_map else None
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}

                    if fn:
                        logger.debug(
                            f"Вызов инструмента '{tc.function.name}', args={args}"
                        )
                        if asyncio.iscoroutinefunction(fn):
                            result = await fn(**args)
                        else:
                            result = await asyncio.to_thread(fn, **args)
                        payload = f"{str(result).rstrip()}{budget_footer}"
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": _tool_result_content(payload, self.profile),
                            }
                        )
                    else:
                        logger.error(
                            f"Инструмент '{tc.function.name}' не найден в tool_map"
                        )
                        payload = (
                            f"Error: function '{tc.function.name}' not found."
                            f"{budget_footer}"
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": _tool_result_content(payload, self.profile),
                            }
                        )

                kwargs["messages"] = messages
                # Last allowed tool round: forbid further tool calls hard.
                kwargs["tool_choice"] = "none" if turns >= max_turns else "auto"
                response = await self.client.chat.completions.create(**kwargs)
                logger.debug(response)
                message = response.choices[0].message
                self._log_reasoning_usage(f"turn{turns}", response)

            if message.tool_calls:
                logger.warning(
                    "LLM still requested tools after budget exhausted "
                    "(turns=%s max_turns=%s); returning empty/partial content",
                    turns,
                    max_turns,
                )

            text = message.content or ""
            text = _THINK_TAG_RE.sub("", text).strip()

            logger.debug(
                f"[{self.__class__.__name__}] Ответ за {time.perf_counter() - start:.2f}s"
            )
            return text

        except Exception as e:
            logger.error(f"[{self.__class__.__name__}] Ошибка generate_response: {e}")
            return f"Ошибка: {e}"

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
