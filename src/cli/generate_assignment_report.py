"""Generate a max-~6 page assignment report PDF."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import ListFlowable, ListItem, Paragraph, Preformatted, SimpleDocTemplate, Spacer
import re

from src.config import ROOT


def _styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BodyCustom", parent=styles["BodyText"], fontSize=9, leading=11, spaceAfter=4))
    styles.add(ParagraphStyle(name="H1Custom", parent=styles["Heading1"], fontSize=13, spaceBefore=8, spaceAfter=4))
    styles.add(ParagraphStyle(name="H2Custom", parent=styles["Heading2"], fontSize=11, spaceBefore=6, spaceAfter=3))
    styles.add(ParagraphStyle(name="CodeBlock", parent=styles["Code"], fontName="Courier", fontSize=7, leading=9, spaceAfter=4))
    return styles


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def md_to_flowables(md: str, styles):
    flowables = []
    bullet_buf = []

    def flush():
        nonlocal bullet_buf
        if not bullet_buf:
            return
        items = [ListItem(Paragraph(_escape(b), styles["BodyCustom"])) for b in bullet_buf]
        flowables.append(ListFlowable(items, bulletType="bullet", leftIndent=12))
        bullet_buf = []

    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("```"):
            flush()
            i += 1
            code = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            flowables.append(Preformatted("\n".join(code), styles["CodeBlock"]))
            i += 1
            continue
        if line.startswith("# "):
            flush()
            flowables.append(Paragraph(_escape(line[2:]), styles["H1Custom"]))
        elif line.startswith("## "):
            flush()
            flowables.append(Paragraph(_escape(line[3:]), styles["H2Custom"]))
        elif line.startswith("### "):
            flush()
            flowables.append(Paragraph(f"<b>{_escape(line[4:])}</b>", styles["BodyCustom"]))
        elif re.match(r"^[-*] ", line.strip()):
            bullet_buf.append(re.sub(r"^[-*] ", "", line.strip()))
        elif line.strip().startswith("|") and "---" not in line:
            flush()
            flowables.append(Paragraph(_escape(line.strip()), styles["BodyCustom"]))
        elif line.strip() == "" or set(line.strip()) <= {"|", "-", ":", " "}:
            flush()
            flowables.append(Spacer(1, 0.05 * inch))
        else:
            flush()
            text = _escape(line.strip())
            text = re.sub(r"`([^`]+)`", r"<font face='Courier'>\1</font>", text)
            text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
            flowables.append(Paragraph(text, styles["BodyCustom"]))
        i += 1
    flush()
    return flowables


def main():
    md = (ROOT / "reports" / "assignment-report.md").read_text(encoding="utf-8")
    short = (ROOT / "reports" / "decision-log-short.md").read_text(encoding="utf-8")
    md = md + "\n\n" + short
    out = ROOT / "reports" / "assignment-report.pdf"
    doc = SimpleDocTemplate(
        str(out),
        pagesize=LETTER,
        leftMargin=0.65 * inch,
        rightMargin=0.65 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
        title="Hiver Take-Home Assignment Report",
    )
    doc.build(md_to_flowables(md, _styles()))
    print("Wrote", out)


if __name__ == "__main__":
    main()
