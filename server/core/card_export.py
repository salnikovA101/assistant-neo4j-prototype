"""Portable card snapshot: a reading copy and structured data in one archive."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from html import escape
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from markdown_it import MarkdownIt
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import LongTable, Paragraph, SimpleDocTemplate, Spacer, TableStyle


_FONT_LOCK = Lock()
_WIDTH = A4[0] - 40 * mm


def _fonts() -> None:
    with _FONT_LOCK:
        if "CardSans" in pdfmetrics.getRegisteredFontNames():
            return
        directory = Path(__file__).resolve().parents[1] / "assets" / "fonts"
        pdfmetrics.registerFont(TTFont("CardSans", str(directory / "DejaVuSans.ttf")))
        pdfmetrics.registerFont(TTFont("CardSansBold", str(directory / "DejaVuSans-Bold.ttf")))
        pdfmetrics.registerFontFamily("CardSans", normal="CardSans", bold="CardSansBold", italic="CardSans", boldItalic="CardSansBold")


def _date(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp / 1000, timezone.utc).strftime("%d.%m.%Y %H:%M UTC")


def _inline(tokens: list) -> str:
    """Only emit our own ReportLab markup; card text never becomes raw HTML."""
    parts = []
    links = []
    for token in tokens:
        if token.type in ("text", "code_inline", "html_inline"):
            parts.append(escape(token.content))
        elif token.type in ("softbreak", "hardbreak"):
            parts.append("<br/>")
        elif token.type == "strong_open":
            parts.append("<b>")
        elif token.type == "strong_close":
            parts.append("</b>")
        elif token.type == "link_open":
            href = token.attrGet("href") or ""
            links.append(href if href.startswith(("https://", "http://")) else "")
        elif token.type == "link_close":
            href = links.pop() if links else ""
            if href:
                parts.append(f" ({escape(href)})")
        elif token.type == "image":
            parts.append(escape(token.content or "Изображение"))
    return "".join(parts)


def _markdown(text: str, body: ParagraphStyle, heading: ParagraphStyle) -> list:
    tokens = MarkdownIt("commonmark", {"html": False}).enable("table").parse(text)
    result = []
    index = 0
    lists: list[int | None] = []
    prefix = ""
    in_heading = False
    while index < len(tokens):
        token = tokens[index]
        if token.type == "table_open":
            rows, row = [], []
            index += 1
            while index < len(tokens) and tokens[index].type != "table_close":
                child = tokens[index]
                if child.type == "tr_open":
                    row = []
                elif child.type == "inline":
                    row.append(Paragraph(_inline(child.children or []) or " ", body))
                elif child.type == "tr_close":
                    rows.append(row)
                index += 1
            if rows:
                columns = max(map(len, rows))
                rows = [row + [""] * (columns - len(row)) for row in rows]
                table = LongTable(rows, colWidths=[_WIDTH / columns] * columns, repeatRows=1, splitInRow=1, hAlign="LEFT")
                table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf2f7")),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 7),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]))
                result.extend([table, Spacer(1, 8)])
        elif token.type in ("bullet_list_open", "ordered_list_open"):
            lists.append(int(token.attrGet("start") or 1) if token.type == "ordered_list_open" else None)
        elif token.type in ("bullet_list_close", "ordered_list_close"):
            lists.pop()
        elif token.type == "list_item_open":
            number = lists[-1] if lists else None
            prefix = "• " if number is None else f"{number}. "
            if number is not None:
                lists[-1] = number + 1
        elif token.type == "heading_open":
            in_heading = True
        elif token.type == "heading_close":
            in_heading = False
        elif token.type == "inline":
            result.append(Paragraph(prefix + _inline(token.children or []) or " ", heading if in_heading else body))
            prefix = ""
        elif token.type in ("fence", "code_block", "html_block"):
            # Separate paragraphs allow long blocks to flow across pages.
            result.extend(Paragraph(escape(line) or " ", body) for line in token.content.splitlines())
        index += 1
    return result


def _value(value: Any, body: ParagraphStyle, heading: ParagraphStyle, schema: dict | None = None, unit: str = "") -> list:
    if value is None or value == "" or value == [] or value == {}:
        return [Paragraph("Не заполнено", body)]
    if isinstance(value, dict):
        result = []
        properties = (schema or {}).get("properties", {})
        for key, item in value.items():
            child = properties.get(key, {})
            result.append(Paragraph(escape(str(child.get("title") or key)), heading))
            result.extend(_value(item, body, heading, child))
        return result
    if isinstance(value, list):
        result = []
        for number, item in enumerate(value, 1):
            if isinstance(item, (dict, list)):
                result.append(Paragraph(f"Пункт {number}", heading))
                result.extend(_value(item, body, heading, (schema or {}).get("items", {})))
            else:
                result.extend(_markdown(f"- {item}", body, heading))
        return result
    text = "Да" if value is True else "Нет" if value is False else str(value)
    if unit and isinstance(value, (int, float)) and not isinstance(value, bool):
        text += f" {unit}"
    return _markdown(text, body, heading)


def _pdf(payload: dict[str, Any]) -> bytes:
    _fonts()
    body = ParagraphStyle("body", fontName="CardSans", fontSize=10, leading=15, spaceAfter=7, textColor=colors.HexColor("#243447"), alignment=TA_LEFT)
    heading = ParagraphStyle("heading", parent=body, fontName="CardSansBold", fontSize=11, leading=16, spaceBefore=12, keepWithNext=True)
    title_style = ParagraphStyle("title", parent=heading, fontSize=21, leading=28, spaceAfter=14)
    meta_style = ParagraphStyle("meta", parent=body, fontSize=8, leading=12, textColor=colors.HexColor("#64748b"))
    metadata, template, data = payload["metadata"], payload["template"], payload["data"]
    story = [
        Paragraph("ОТЧЁТ ПО КАРТОЧКЕ · " + escape(template["name"]), meta_style),
        Paragraph(escape(payload["title"]), title_style),
        Paragraph(escape(metadata["authorship"]), body),
        Paragraph(f"Создано: {_date(metadata['createdAt'])}<br/>Изменено: {_date(metadata['updatedAt'])}<br/>Выгружено: {_date(metadata['exportedAt'])}", meta_style),
        Spacer(1, 12),
    ]
    schema, ui = template["schema"], template["ui"]
    properties = schema.get("properties", {})
    title_key = ui.get("titleField") or "title"
    order = list(dict.fromkeys([*ui.get("order", []), *properties, *data]))
    for key in order:
        if key in (title_key, "title") or (key not in properties and key not in data):
            continue
        field = properties.get(key, {})
        story.append(Paragraph(escape(str(field.get("title") or key)), heading))
        unit = (ui.get("fields", {}).get(key) or {}).get("unit", "")
        story.extend(_value(data.get(key), body, heading, field, unit))
    if payload["gaps"]:
        story.append(Paragraph("Недостающие данные", heading))
        story.extend(_value(payload["gaps"], body, heading))
    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#cbd5e1"))
        canvas.line(20 * mm, 17 * mm, A4[0] - 20 * mm, 17 * mm)
        canvas.setFont("CardSans", 8)
        canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.drawString(20 * mm, 12 * mm, "Neo4j Assistant · Отчёт по карточке")
        canvas.drawRightString(A4[0] - 20 * mm, 12 * mm, str(doc.page))
        canvas.restoreState()

    output = BytesIO()
    SimpleDocTemplate(output, pagesize=A4, rightMargin=20 * mm, leftMargin=20 * mm, topMargin=19 * mm, bottomMargin=25 * mm, title=payload["title"], author="Neo4j Assistant").build(story, onFirstPage=footer, onLaterPages=footer)
    return output.getvalue()


def _portable_json(card: dict[str, Any]) -> dict[str, Any]:
    """Share card content without the application's persistence and UI contracts.

    Format markers allow the importer to distinguish this envelope from plain
    field data. Labels (including units) make custom fields useful to another
    assistant without copying the full validation schema.
    """
    template = card["template"]
    properties = template["schema"].get("properties") or {}
    ui = template.get("ui") or {}
    data = {**card["latestRevision"]["data"], "title": card["title"]}
    order = dict.fromkeys(["title", *(ui.get("order") or []), *data])
    data = {key: data[key] for key in order if key in data}
    labels = {}
    for key in data:
        label = str((properties.get(key) or {}).get("title") or key)
        unit = ((ui.get("fields") or {}).get(key) or {}).get("unit")
        labels[key] = f"{label}, {unit}" if unit else label
    return {
        "format": "neo4j-assistant-card",
        "formatVersion": 2,
        "template": template["name"],
        "fieldLabels": labels,
        "data": data,
    }


def export_card(card: dict[str, Any]) -> tuple[str, bytes]:
    revision = card["latestRevision"]
    payload = {
        "title": card["title"],
        "template": card["template"],
        "data": {**revision["data"], "title": card["title"]},
        "gaps": revision["gaps"],
        "metadata": {
            "authorship": card["authorship"],
            "createdAt": card["createdAt"], "updatedAt": card["updatedAt"],
            "exportedAt": int(datetime.now(timezone.utc).timestamp() * 1000),
        },
    }
    name = re.sub(r'[\x00-\x1f\\/:*?"<>|]', "_", card["title"]).strip(" .")[:100] or "Карточка"
    output = BytesIO()
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(f"{name}.pdf", _pdf(payload))
        archive.writestr(f"{name}.json", json.dumps(_portable_json(card), ensure_ascii=False, indent=2).encode("utf-8"))
    return f"{name}.zip", output.getvalue()
