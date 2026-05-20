from collections import deque
from typing import Deque, Dict, List


class HistoryManager:
    """
    Управляет историей диалога, ограничивая её максимальную длину.

    Использует двустороннюю очередь (deque) для автоматического удаления
    старых записей при превышении лимита.

    Attributes:
        history (Deque[Dict[str, str]]): Очередь, хранящая пары "user" и "assistant".
    """

    def __init__(self, max_len: int) -> None:
        """
        Инициализирует менеджер истории.

        Args:
            max_len (int): Максимальное количество хранимых пар (запрос-ответ).
        """
        self.history: Deque[Dict[str, str]] = deque(maxlen=max_len)

    def add_entry(self, user_text: str, assistant_text: str) -> None:
        """
        Добавляет новую запись в историю, если текст не пустой.

        Args:
            user_text (str): Текст запроса пользователя.
            assistant_text (str): Ответ ассистента.
        """
        if (
            user_text
            and user_text.strip()
            and assistant_text
            and assistant_text.strip()
            and not assistant_text.startswith("Ошибка:")
        ):
            self.history.append({"user": user_text, "assistant": assistant_text})

    def get_history(self) -> List[Dict[str, str]]:
        """
        Преобразует историю в стандартный формат сообщений OpenAI.

        Returns:
            List[Dict[str, str]]: Список сообщений с ролями 'user' и 'assistant'.
        """
        contents: List[Dict[str, str]] = []
        for entry in self.history:
            contents.append({"role": "user", "content": entry["user"]})
            contents.append({"role": "assistant", "content": entry["assistant"]})
        return contents

    def clear_history(self) -> None:
        """Очищает историю чата."""
        self.history.clear()
