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

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)


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

            if prompt:
                messages.append({"role": "system", "content": prompt})
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

            kwargs: Dict[str, Any] = {
                "model": self.profile.model,
                "messages": messages,
                "temperature": self.profile.temperature,
                "max_tokens": self.profile.max_output_tokens,
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"

            response = await self.client.chat.completions.create(**kwargs)
            logger.debug(response)
            message = response.choices[0].message

            turns = 0
            while message.tool_calls and turns < self.profile.max_turns:
                turns += 1
                logger.debug(f"Tool loop turn {turns}/{self.profile.max_turns}")
                messages.append(message.model_dump(exclude_none=True))

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
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": str(result),
                            }
                        )
                    else:
                        logger.error(
                            f"Инструмент '{tc.function.name}' не найден в tool_map"
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": f"Error: function '{tc.function.name}' not found.",
                            }
                        )

                kwargs["messages"] = messages
                response = await self.client.chat.completions.create(**kwargs)
                logger.debug(response)
                message = response.choices[0].message

            text = message.content or ""
            text = _THINK_TAG_RE.sub("", text).strip()

            logger.debug(
                f"[{self.__class__.__name__}] Ответ за {time.perf_counter() - start:.2f}s"
            )
            return text

        except Exception as e:
            logger.error(f"[{self.__class__.__name__}] Ошибка generate_response: {e}")
            return f"Ошибка: {e}"

    @abstractmethod
    async def unload(self) -> None:
        """Выгрузить модель из памяти (если поддерживается провайдером)."""

    @abstractmethod
    async def warmup(self) -> None:
        """Прогреть / загрузить модель (если поддерживается провайдером)."""
