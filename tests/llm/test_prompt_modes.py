from server.llm.manager import LLMManager
from server.llm.prompt_loader import PromptLoader
from server.utils.config import load_config
from server.utils.constants import TTSModes


def test_prompt_loader_routes_auto_staged_and_card() -> None:
    loader = PromptLoader("prompts", TTSModes.QUALITY, audio_enabled=False)

    auto = loader.get_system_prompt("auto")
    staged = loader.get_system_prompt("staged")
    assert "Не останавливайся только потому, что уже сделал один поиск" in auto
    assert "Максимум 2 вызова" not in auto
    assert "Вызов 2 —" not in auto
    assert "не подмешиваются" in auto
    assert "продолжающийся процесс разработки" in staged
    assert "Список исследовательских вопросов пуст" in staged
    assert "open_sq_refs: []" in staged
    assert "Revise" in staged
    assert "### GAPS" in auto
    assert "Никогда не выводи `<SQ_STATUS_JSON>`" in auto
    assert "В режиме «С планом» не пиши `### GAPS`" in staged
    assert "<SQ_STATUS_JSON>" in staged
    assert '"status":"closed"' in staged
    assert "можно писать LaTeX" in auto
    assert "можно писать LaTeX" in staged
    assert "Вызови `submit_card` ровно один раз" in loader.get_system_prompt("card")
    assert "TTS" not in loader.get_system_prompt("card")


def test_qwen_context_preflight_keeps_room_without_rejecting_normal_prompt() -> None:
    manager = LLMManager(load_config())
    provider = manager.provider_for("qwen_cloud")

    assert provider.profile.context_window == 262144
    assert manager._context_fits(
        prompt="system" * 100,
        history=[{"role": "user", "content": "контекст" * 1000}],
        user_text="вопрос",
        provider=provider,
    )


def test_auto_prompt_describes_goal_completion_and_grounded_search():
    loader = PromptLoader("prompts", TTSModes.QUALITY, audio_enabled=False)
    prompt = loader.get_system_prompt("auto")
    assert "Главная цель — довести текущий запрос пользователя до проверяемого результата" in prompt
    assert "Своих знаний не добавляешь" in prompt
    assert "query_graph" in prompt
    assert "Связи направленные" in prompt
    assert "один факт — одно ребро" in prompt
    assert "(a)-[r]->(b)" not in prompt
    assert "get_service_guide" in prompt
    assert "не больше 2 успешных `ask_subgraph`" in prompt
    assert "8 успешных `query_graph`" in prompt
    assert "инструменты кончились, отвечай" in prompt
    assert "n/X" in prompt
    assert "Не повторяй тот же запрос" in prompt
    assert "Технический лимит ходов контролирует сервер" in prompt
    assert "Не выдавай остановку по лимиту или сбою за исчерпание всех направлений поиска" in prompt
    assert "фундамент" in prompt
    assert "без открытых GAPS" in prompt
    assert "Один пустой `ask_subgraph`" in prompt
    assert "только после проверки обоими" in prompt
    assert "не выдавай за найденное по этому объекту" in prompt
    assert "не подтверждает утверждение про объект вопроса" in prompt
    assert "Пользователь уже видит нормальные имена файлов" in prompt
    assert "номер файла, он одинаков" not in prompt
    assert "«из общих знаний» запрещены" in prompt
    assert "Общее объяснение — без подтверждения в найденных материалах" not in prompt
    assert "Четыре основания" not in prompt
    assert "общие знания модели" not in prompt.lower()


def test_ui_reports_autonomous_search_budget_independent_of_model():
    from types import SimpleNamespace
    from server.core.app import build_ui_config

    cfg = load_config()
    cfg.llm.current_profile = "qwen38_flash"
    assert cfg.llm.profiles.qwen38_flash.max_turns == 2
    assert build_ui_config(SimpleNamespace(config=cfg))["max_searches_per_answer"] == 12
