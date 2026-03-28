import argparse
import re
from pathlib import Path
from typing import List
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import Image, Paragraph, Preformatted, SimpleDocTemplate, Spacer


IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


def read_text_auto(path: Path) -> str:
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def markdown_to_story(md_path: Path) -> List:
    text = read_text_auto(md_path)
    lines = text.splitlines()

    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "BodyCN",
        parent=styles["BodyText"],
        fontName="STSong-Light",
        fontSize=11,
        leading=16,
        spaceAfter=5,
    )
    h1 = ParagraphStyle(
        "H1CN",
        parent=styles["Heading1"],
        fontName="STSong-Light",
        fontSize=20,
        leading=24,
        spaceAfter=10,
    )
    h2 = ParagraphStyle(
        "H2CN",
        parent=styles["Heading2"],
        fontName="STSong-Light",
        fontSize=15,
        leading=20,
        spaceBefore=10,
        spaceAfter=6,
    )
    quote = ParagraphStyle(
        "QuoteCN",
        parent=body,
        textColor=colors.grey,
        leftIndent=14,
        borderColor=colors.lightgrey,
        borderPadding=4,
        borderLeft=1,
    )
    table_style = ParagraphStyle(
        "TableCN",
        parent=body,
        fontSize=9,
        leading=13,
    )

    story = []
    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()

        if not line:
            story.append(Spacer(1, 0.25 * cm))
            i += 1
            continue

        if line.startswith("# "):
            story.append(Paragraph(escape(line[2:].strip()), h1))
            i += 1
            continue

        if line.startswith("## "):
            story.append(Paragraph(escape(line[3:].strip()), h2))
            i += 1
            continue

        if line.startswith(">"):
            story.append(Paragraph(escape(line[1:].strip()), quote))
            i += 1
            continue

        image_match = IMAGE_RE.search(line)
        if image_match:
            image_rel = image_match.group(1).strip()
            image_path = (md_path.parent / image_rel).resolve()
            if image_path.exists():
                max_width = A4[0] - 4 * cm
                img = Image(str(image_path))
                ratio = img.imageHeight / float(img.imageWidth) if img.imageWidth else 0.6
                img.drawWidth = max_width
                img.drawHeight = max_width * ratio
                story.append(img)
                story.append(Spacer(1, 0.25 * cm))
            else:
                story.append(Paragraph(f"图片未找到: {escape(image_rel)}", quote))
            i += 1
            continue

        if line.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].rstrip())
                i += 1
            block = "\n".join(table_lines)
            story.append(Preformatted(block, table_style))
            continue

        if line.startswith("- "):
            bullet = "• " + line[2:].strip()
            story.append(Paragraph(escape(bullet), body))
            i += 1
            continue

        story.append(Paragraph(escape(line), body))
        i += 1

    return story


def convert_md_to_pdf(input_md: Path, output_pdf: Path) -> None:
    story = markdown_to_story(input_md)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output_pdf),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title=input_md.stem,
    )
    doc.build(story)


def main() -> None:
    parser = argparse.ArgumentParser(description="将 Markdown 分析报告转换为 PDF（支持中文与插图）")
    parser.add_argument(
        "--input-md",
        default="term_structure_tool/output/INE_sc_analysis.md",
        help="输入 Markdown 文件路径",
    )
    parser.add_argument("--output-pdf", default="", help="输出 PDF 文件路径，默认与输入同名")
    args = parser.parse_args()

    input_md = Path(args.input_md).resolve()
    if not input_md.exists():
        raise SystemExit(f"输入文件不存在: {input_md}")

    if args.output_pdf:
        output_pdf = Path(args.output_pdf).resolve()
    else:
        output_pdf = input_md.with_suffix(".pdf")

    convert_md_to_pdf(input_md, output_pdf)
    print(f"PDF 已生成: {output_pdf}")


if __name__ == "__main__":
    main()
