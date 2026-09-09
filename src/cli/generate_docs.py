"""Generate interview PDF from living markdown documentation."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
)

from src.config import ROOT


def _styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="BodyCustom",
            parent=styles["BodyText"],
            fontSize=10,
            leading=13,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CodeBlock",
            parent=styles["Code"],
            fontName="Courier",
            fontSize=8,
            leading=10,
            leftIndent=10,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="H1Custom",
            parent=styles["Heading1"],
            fontSize=16,
            spaceBefore=12,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="H2Custom",
            parent=styles["Heading2"],
            fontSize=13,
            spaceBefore=10,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="H3Custom",
            parent=styles["Heading3"],
            fontSize=11,
            spaceBefore=8,
            spaceAfter=4,
        )
    )
    return styles


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def markdown_to_flowables(md: str, styles) -> list:
    """Minimal Markdown subset renderer for documentation PDFs."""
    flowables: list = []
    lines = md.splitlines()
    i = 0
    bullet_buf: list[str] = []

    def flush_bullets():
        nonlocal bullet_buf
        if not bullet_buf:
            return
        items = [
            ListItem(Paragraph(_escape(b), styles["BodyCustom"])) for b in bullet_buf
        ]
        flowables.append(ListFlowable(items, bulletType="bullet", leftIndent=15))
        bullet_buf = []

    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("```"):
            flush_bullets()
            i += 1
            code_lines = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            code = "\n".join(code_lines)
            flowables.append(Preformatted(code, styles["CodeBlock"]))
            i += 1
            continue

        if line.startswith("# "):
            flush_bullets()
            flowables.append(Paragraph(_escape(line[2:].strip()), styles["H1Custom"]))
        elif line.startswith("## "):
            flush_bullets()
            flowables.append(Paragraph(_escape(line[3:].strip()), styles["H2Custom"]))
        elif line.startswith("### "):
            flush_bullets()
            flowables.append(Paragraph(_escape(line[4:].strip()), styles["H3Custom"]))
        elif re.match(r"^[-*] ", line.strip()):
            bullet_buf.append(re.sub(r"^[-*] ", "", line.strip()))
        elif line.strip() == "":
            flush_bullets()
            flowables.append(Spacer(1, 0.08 * inch))
        else:
            flush_bullets()
            # Inline code/bold-ish handling kept minimal.
            text = _escape(line.strip())
            text = re.sub(r"`([^`]+)`", r"<font face='Courier'>\1</font>", text)
            text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
            flowables.append(Paragraph(text, styles["BodyCustom"]))
        i += 1
    flush_bullets()
    return flowables


def build_pdf_source_markdown() -> str:
    """Assemble PDF study guide from final submission docs."""
    parts = []
    for rel in [
        "reports/final-report.md",
        "reports/final/headline_metric.md",
        "reports/final/comparison_table.md",
        "reports/final/top5_failure_modes.md",
        "reports/final/case_studies.md",
        "reports/final/EXPERIMENT_FREEZE.md",
        "reports/final/performance.json",
        "reports/phase3/judge_calibration.md",
        "reports/decision-log-short.md",
        "reports/SUBMISSION_CHECKLIST.md",
        "reports/project-engineering-guide.md",
        "reports/engineering-notes.md",
    ]:
        path = ROOT / rel
        if not path.exists():
            continue
        title = path.name
        body = path.read_text(encoding="utf-8")
        if path.suffix == ".json":
            body = "```json\n" + body + "\n```"
        parts.append(f"# {title}\n\n{body}")
    return "\n\n---\n\n".join(parts)


def generate_pdf(output_path: Path | None = None) -> Path:
    output_path = output_path or (ROOT / "reports" / "project-engineering-guide.pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    md = build_pdf_source_markdown()
    styles = _styles()
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=LETTER,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title="Hiver AI Support Agent — Project Engineering Guide",
        author="Phase 0 documentation generator",
    )
    story = markdown_to_flowables(md, styles)
    story.append(PageBreak())
    story.append(
        Paragraph(
            "Status legend: IMPLEMENTED / PLANNED / OBSERVED / ASSUMED / NOT YET IMPLEMENTED",
            styles["BodyCustom"],
        )
    )
    doc.build(story)
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate project engineering PDF")
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "reports" / "project-engineering-guide.pdf"),
    )
    args = parser.parse_args(argv)
    path = generate_pdf(Path(args.output))
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
