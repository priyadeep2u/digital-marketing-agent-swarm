import json
import re
import operator
from typing import Any, TypedDict, Annotated
from datetime import datetime
from langchain_ollama import ChatOllama
from ddgs import DDGS
from dotenv import load_dotenv
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable, PageBreak,
    Table, TableStyle,
)

load_dotenv()

OLLAMA_BASE_URL="http://localhost:11434"
OLLAMA_MODEL="gemma4:latest"
OLLAMA_REVIEWER_MODEL="llama3.1:8b"
LLM_TEMPERATURE=0.3

CYAN          = "\033[96m"
GREEN         = "\033[92m"
YELLOW        = "\033[93m"
RED           = "\033[91m"
BLUE          = "\033[94m"
MAGENTA       = "\033[95m"
WHITE         = "\033[97m"
DARK_RED      = "\033[31m"
DARK_GREEN    = "\033[32m"
DARK_YELLOW   = "\033[33m"
DARK_BLUE     = "\033[34m"
DARK_MAGENTA  = "\033[35m"
DARK_CYAN     = "\033[36m"
GRAY          = "\033[90m"
ORANGE        = "\033[38;5;214m"
PINK          = "\033[38;5;213m"
PURPLE        = "\033[38;5;129m"
LIME          = "\033[38;5;118m"
TEAL          = "\033[38;5;30m"
GOLD          = "\033[38;5;220m"
CORAL         = "\033[38;5;203m"
LAVENDER      = "\033[38;5;183m"
BOLD          = "\033[1m"
UNDERLINE     = "\033[4m"
RESET         = "\033[0m"

def ddg_search(query: str, max_results: int = 8) -> str:
    try:
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=max_results))
            
        if not hits:
            return "[No results returned]"
            
        return "\n\n".join(
            f"Title  : {r.get('title','')}\n"
            f"URL    : {r.get('href','')}\n"
            f"Snippet: {r.get('body','')}"
            for r in hits
        )
    except Exception as exc:
        return f"[Search error: {exc}]"



def banner(agent: str, msg: str, color: str = CYAN) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"{color}{BOLD}[{ts}][{agent}]{RESET} {msg}")


llm_reviewer = ChatOllama(
    model       = OLLAMA_REVIEWER_MODEL,
    base_url    = OLLAMA_BASE_URL,
    temperature = LLM_TEMPERATURE,
)

llm = ChatOllama(
    model       = OLLAMA_MODEL,
    base_url    = OLLAMA_BASE_URL,
    temperature = LLM_TEMPERATURE,
)

# ══════════════════════════════════════════════════════════════════════════════
# SHARED STATE  (LangGraph TypedDict)
# ══════════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    cycle_count:                  int
    competitor_analyst_decision:  str
    content_strategist_decision:  str
    content_writer_decision:      str
    copywriter_decision:          str
    creative_director_decision:   str
    cro_decision:                 str
    design_agent_decision:        str
    email_marketer_decision:      str
    google_ads_decision:          str
    keyword_research_decision:    str
    lead_generation_decision:     str
    linkedin_specialist_decision: str
    market_researcher_decision:   str
    marketing_director_decision:  str
    meta_ads_decision:            str
    outreach_decision:            str
    paid_ads_decision:            str
    quality_reviewer_decision:    str
    reporting_decision:           str
    seo_specialist_decision:      str
    seo_specialist_decision:      str
    social_media_decision:        str
    trend_analyst_decision:       str
    output_file:                  Annotated[list[str], operator.add]
    messages:                     Annotated[list[str], operator.add]
    task:                         str


_TABLE_SEP_CELL_RE = re.compile(r"^:?-{2,}:?$")


def _split_table_row(line: str) -> list[str]:
    """Split a markdown table row ('| a | b |') into trimmed cell strings."""
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [cell.strip() for cell in line.split("|")]


def _is_separator_row(cells: list[str]) -> bool:
    """True if every non-empty cell looks like a markdown separator (---, :---:, etc.)."""
    non_empty = [c for c in cells if c.strip()]
    return bool(non_empty) and all(_TABLE_SEP_CELL_RE.match(c.strip()) for c in non_empty)


class WriteDocumentInput(BaseModel):
    title:   str = Field(..., description="Document title")
    content: str = Field(..., description="Full document body (Markdown)")

class WriteDocumentTool(BaseTool):
    """Persists agent output as a formatted PDF."""
    name:        str = "write_document"
    description: str = (
        "Writes a well-formatted PDF document to disk. "
        "Supply a title and the full content in Markdown."
    )
    args_schema: type[BaseModel] = WriteDocumentInput

    def _run(self, **kwargs: Any) -> str:
        import html
        title   = kwargs.get("title", "Untitled")
        content = kwargs.get("content", "")

        safe_title = re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "_")[:60]
        ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename   = f"output/{safe_title}_output_{ts}.pdf"

        doc = SimpleDocTemplate(
            filename,
            pagesize=A4,
            rightMargin=2*cm, leftMargin=2*cm,
            topMargin=2*cm,   bottomMargin=2*cm,
            title=title,
        )

        styles = getSampleStyleSheet()
        style_cover_title = ParagraphStyle("CoverTitle",
            parent=styles["Title"], fontSize=28, leading=34, spaceAfter=12,
            textColor=colors.HexColor("#1a1a2e"))
        style_cover_sub = ParagraphStyle("CoverSub",
            parent=styles["Normal"], fontSize=11,
            textColor=colors.HexColor("#555555"), spaceAfter=6)
        style_h1 = ParagraphStyle("H1", parent=styles["Heading1"],
            fontSize=16, leading=20, spaceBefore=16, spaceAfter=6,
            textColor=colors.HexColor("#1a1a2e"))
        style_h2 = ParagraphStyle("H2", parent=styles["Heading2"],
            fontSize=13, leading=16, spaceBefore=12, spaceAfter=4,
            textColor=colors.HexColor("#2d2d5e"))
        style_body = ParagraphStyle("Body", parent=styles["Normal"],
            fontSize=10, leading=15, spaceAfter=6)
        style_bullet = ParagraphStyle("Bullet", parent=styles["Normal"],
            fontSize=10, leading=14, spaceAfter=3, leftIndent=16, bulletIndent=4)
        style_footer = ParagraphStyle("Footer", parent=styles["Normal"],
            fontSize=8, textColor=colors.HexColor("#888888"))
        style_cell = ParagraphStyle("Cell", parent=styles["Normal"],
            fontSize=9, leading=12)
        style_header_cell = ParagraphStyle("HeaderCell", parent=styles["Normal"],
            fontSize=9, leading=12, textColor=colors.white, fontName="Helvetica-Bold")

        def clean(text: str) -> str:
            """Strip all HTML tags, then convert **bold** to <b>bold</b> safely."""
            # 1. Remove any HTML/XML tags the LLM may have emitted
            text = re.sub(r"<[^>]+>", "", text)
            # 2. Escape ampersands and angle brackets for ReportLab XML parser
            text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            # 3. Convert **bold** markdown to ReportLab <b> tags
            text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
            return text.strip()

        story = []

        avail_width = A4[0] - 4 * cm  # page width minus left+right margins (2cm each)

        def make_table(rows: list[list[str]]) -> Table:
            num_cols = max(len(r) for r in rows)
            norm_rows = [r + [""] * (num_cols - len(r)) for r in rows]
            data = []
            for ridx, row in enumerate(norm_rows):
                cstyle = style_header_cell if ridx == 0 else style_cell
                data.append([Paragraph(clean(cell), cstyle) for cell in row])
            col_width = avail_width / num_cols
            t = Table(data, colWidths=[col_width] * num_cols, repeatRows=1)
            t.setStyle(TableStyle([
                ("BACKGROUND",     (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
                ("TEXTCOLOR",      (0, 0), (-1, 0), colors.white),
                ("GRID",           (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
                ("VALIGN",         (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f7")]),
                ("LEFTPADDING",    (0, 0), (-1, -1), 6),
                ("RIGHTPADDING",   (0, 0), (-1, -1), 6),
                ("TOPPADDING",     (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING",  (0, 0), (-1, -1), 4),
            ]))
            return t

        # Some local models emit markdown tables without real newlines between rows
        # (rows end up joined by adjacent pipes, e.g. "...| Goal || :--- |..."").
        # Reinsert the missing line breaks so the parser below can find row boundaries.
        if re.search(r"\|\s*:?-{2,}:?\s*\|", content):
            content = re.sub(r"\|{2,}", lambda m: "|\n" * (len(m.group(0)) - 1) + "|", content)

        # ── Cover page ────────────────────────────────────────────
        story.append(Spacer(1, 4*cm))
        story.append(Paragraph(clean(title), style_cover_title))
        story.append(HRFlowable(width="100%", thickness=2,
                                color=colors.HexColor("#1a1a2e"), spaceAfter=8))
        story.append(Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            style_cover_sub))
        story.append(Paragraph(
            "Produced by Multi-Agent Marketing Pipeline · Ollama · LangGraph",
            style_cover_sub))
        story.append(PageBreak())

        # ── Content ───────────────────────────────────────────────
        lines = content.splitlines()
        i, n = 0, len(lines)
        while i < n:
            stripped = lines[i].strip()

            # Markdown table: a "| ... |" row immediately followed by a
            # "| --- | --- |" separator row.
            if stripped.startswith("|") and stripped.endswith("|") and i + 1 < n:
                next_line = lines[i + 1].strip()
                next_cells = _split_table_row(next_line) if next_line.startswith("|") else []
                if _is_separator_row(next_cells):
                    table_rows = [_split_table_row(stripped)]
                    i += 2
                    while i < n and lines[i].strip().startswith("|") and lines[i].strip().endswith("|"):
                        table_rows.append(_split_table_row(lines[i].strip()))
                        i += 1
                    story.append(Spacer(1, 6))
                    story.append(make_table(table_rows))
                    story.append(Spacer(1, 10))
                    continue

            if not stripped:
                story.append(Spacer(1, 6))
            elif stripped.startswith("### "):
                story.append(Paragraph(clean(stripped[4:]), style_h2))
            elif stripped.startswith("## "):
                story.append(Paragraph(clean(stripped[3:]), style_h1))
            elif stripped.startswith("# "):
                story.append(Paragraph(clean(stripped[2:]), style_h1))
            elif stripped.startswith(("• ", "- ", "* ")):
                story.append(Paragraph(f"• {clean(stripped[2:])}", style_bullet))
            elif stripped.startswith("---"):
                story.append(HRFlowable(width="100%", thickness=0.5,
                                        color=colors.HexColor("#cccccc"),
                                        spaceBefore=4, spaceAfter=4))
            else:
                story.append(Paragraph(clean(stripped), style_body))
            i += 1

        # ── Footer ────────────────────────────────────────────────
        story.append(Spacer(1, 1*cm))
        story.append(HRFlowable(width="100%", thickness=0.5,
                                color=colors.HexColor("#cccccc")))
        story.append(Paragraph(
            "Document produced by the Multi-Agent Pipeline · multi-agent-pipeline",
            style_footer))

        doc.build(story)

        preview = content[:300].replace("\n", " ")
        return json.dumps({"file": filename, "chars": len(content), "preview": preview})
        
write_tool = WriteDocumentTool()