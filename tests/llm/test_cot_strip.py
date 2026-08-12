"""Tests for leaked CoT stripping in final answers."""

from server.llm.base import strip_leaked_cot_preamble


def test_strip_tagged_think_block() -> None:
    text = "<think>secret plan</think>\n\nОтвет пользователю"
    assert strip_leaked_cot_preamble(text) == "Ответ пользователю"


def test_strip_untagged_english_cot_before_russian_answer() -> None:
    text = """The user wants a starter culture formulation for strawberry kefir.
I have gathered information on:

**Kefir starter composition**: LAB, yeasts, AAB.

I need to synthesize a formulation based on this.
Drafting the response:
Plan:
- list strains
- note gaps

Let's write.

Закваска для клубничного кефира

Традиционная закваска — симбиотический консорциум МКБ и дрожжей.
"""
    out = strip_leaked_cot_preamble(text)
    assert out.startswith("Закваска для клубничного кефира")
    assert "The user wants" not in out
    assert "Drafting the response" not in out


def test_keep_normal_russian_answer() -> None:
    text = "Закваска включает Lactobacillus и Saccharomyces."
    assert strip_leaked_cot_preamble(text) == text


def test_keep_english_answer_without_cot_markers() -> None:
    text = (
        "Starter cultures for kefir typically include lactic acid bacteria "
        "and yeasts. Ratios vary by product."
    )
    assert strip_leaked_cot_preamble(text) == text
