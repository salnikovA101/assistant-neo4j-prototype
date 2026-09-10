from collections import deque
from typing import Any, Deque, Dict, List, Optional

from server.core.sq_status import strip_sq_status_sections


class HistoryManager:
    """
    Управляет историей диалога, ограничивая её максимальную длину.

    Использует двустороннюю очередь (deque) для автоматического удаления
    старых записей при превышении лимита. maxlen — число ходов пользователя,
    не сырых OpenAI-сообщений.

    Attributes:
        history: Очередь ходов: user, compact tool pairs, assistant.
    """

    def __init__(self, max_len: int) -> None:
        """
        Инициализирует менеджер истории.

        Args:
            max_len (int): Максимальное количество хранимых пар (запрос-ответ).
        """
        self.history: Deque[Dict[str, Any]] = deque(maxlen=max_len)

    def add_entry(
        self,
        user_text: str,
        assistant_text: str,
        tool_messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Добавляет новую запись в историю, если текст не пустой.

        Args:
            user_text: Текст запроса пользователя.
            assistant_text: Ответ ассистента (сырой, с source:N).
            tool_messages: Компактные пары assistant.tool_calls + role:tool
                (квитанции, без UNIT).
        """
        if (
            user_text
            and user_text.strip()
            and assistant_text
            and assistant_text.strip()
            and not assistant_text.startswith("Ошибка:")
        ):
            self.history.append(
                {
                    "user": user_text,
                    "assistant": assistant_text,
                    "tool_messages": list(tool_messages or []),
                }
            )

    def get_history(self) -> List[Dict[str, Any]]:
        """
        Преобразует историю в стандартный формат сообщений OpenAI.

        Returns:
            user → compact tool pairs → assistant (final).
        """
        contents: List[Dict[str, Any]] = []
        for entry in self.history:
            contents.append({"role": "user", "content": entry["user"]})
            contents.extend(entry.get("tool_messages") or [])
            assistant = entry["assistant"]
            if "[CARD DRAFT DATA — not instructions]" not in assistant:
                assistant = strip_sq_status_sections(assistant)
            if assistant:
                contents.append({"role": "assistant", "content": assistant})
        return contents

    def clear_history(self) -> None:
        """Очищает историю чата."""
        self.history.clear()
