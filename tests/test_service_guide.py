from __future__ import annotations

import pytest

from server.service_guide import SERVICE_GUIDE_FILENAME, load_service_guide
from server.tools.registry import Tools
from server.utils.config import AppConfig


def test_service_guide_loader_rejects_missing_and_empty_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="Service guide not found"):
        load_service_guide(tmp_path)

    (tmp_path / SERVICE_GUIDE_FILENAME).write_text("  \n", encoding="utf-8")
    with pytest.raises(ValueError, match="Service guide is empty"):
        load_service_guide(tmp_path)


def test_service_guide_is_shared_and_available_in_every_mode(tmp_path):
    guide_path = tmp_path / SERVICE_GUIDE_FILENAME
    guide_path.write_text("# Guide\n\nFirst version. (source:N)", encoding="utf-8")
    config = AppConfig()
    config.llm.prompt_folder = str(tmp_path)
    tools = Tools(config)

    first = tools.get_service_guide()
    assert first.startswith("# Guide\n\nFirst version.")
    assert "source:" not in first
    assert tools.source_registry.snapshot() == []
    assert set(tools.get_tool_map("auto")) == {"ask_subgraph", "get_service_guide"}
    assert set(tools.get_tool_map("staged")) == {
        "advance_research",
        "get_service_guide",
    }

    for mode in ("auto", "staged"):
        schemas = {
            item["function"]["name"]: item["function"]
            for item in tools.get_openai_tools(mode)
        }
        help_schema = schemas["get_service_guide"]
        assert help_schema["parameters"] == {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }

    # UI and tool read on demand, so editing the one Markdown source cannot
    # leave either consumer with a stale embedded copy.
    guide_path.write_text("# Guide\n\nSecond version. (source:17)", encoding="utf-8")
    second = tools.get_service_guide()
    assert second.startswith("# Guide\n\nSecond version.")
    assert "source:" not in second
