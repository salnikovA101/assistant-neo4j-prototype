from server.llm.manager import LLMManager
from server.llm.prompt_loader import PromptLoader
from server.utils.config import load_config
from server.utils.constants import TTSModes


def test_prompt_loader_routes_auto_staged_and_card() -> None:
    loader = PromptLoader("prompts", TTSModes.QUALITY, audio_enabled=False)

    auto = loader.get_system_prompt("auto")
    staged = loader.get_system_prompt("staged")
    assert "Максимум 2 вызова" in auto
    assert "не подмешиваются" in auto
    assert "продолжающийся процесс разработки" in staged
    assert "Чеклист пуст" in staged
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
