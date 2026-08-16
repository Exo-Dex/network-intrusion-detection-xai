from pathlib import Path
import re

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "docs" / "research_paper"
DRAFT = PAPER_DIR / "ieee_paper_draft.md"
OUT_DOCX = PAPER_DIR / "nids_xai_paper_review_draft.docx"
ARCH_PNG = PAPER_DIR / "nids_xai_architecture_ieee.png"


FIGURE_FILES = {
    "Fig. 1.": ARCH_PNG,
    "Fig. 2.": ROOT / "results" / "graphs" / "attack_category_distribution.png",
    "Fig. 3.": ROOT / "results" / "graphs" / "binary_all_models_comparison.png",
    "Fig. 4.": ROOT / "results" / "graphs" / "multiclass_all_models_comparison.png",
    "Fig. 5.": ROOT / "results" / "xai" / "shap_xgb_summary_beeswarm.png",
    "Fig. 6.": ROOT / "results" / "xai" / "lime_xgb_attack_instance.png",
}


def font(size=24, bold=False):
    candidates = [
        "C:/Windows/Fonts/timesbd.ttf" if bold else "C:/Windows/Fonts/times.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_centered(draw, box, text, fnt, fill=(20, 20, 20), spacing=6):
    x0, y0, x1, y1 = box
    lines = text.split("\n")
    heights = []
    widths = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=fnt)
        widths.append(bbox[2] - bbox[0])
        heights.append(bbox[3] - bbox[1])
    total_h = sum(heights) + spacing * (len(lines) - 1)
    y = y0 + (y1 - y0 - total_h) / 2
    for line, w, h in zip(lines, widths, heights):
        draw.text((x0 + (x1 - x0 - w) / 2, y), line, font=fnt, fill=fill)
        y += h + spacing


def arrow(draw, start, end, fill=(45, 45, 45), width=3):
    draw.line([start, end], fill=fill, width=width)
    x0, y0 = start
    x1, y1 = end
    if abs(x1 - x0) >= abs(y1 - y0):
        direction = 1 if x1 >= x0 else -1
        pts = [(x1, y1), (x1 - 14 * direction, y1 - 7), (x1 - 14 * direction, y1 + 7)]
    else:
        direction = 1 if y1 >= y0 else -1
        pts = [(x1, y1), (x1 - 7, y1 - 14 * direction), (x1 + 7, y1 - 14 * direction)]
    draw.polygon(pts, fill=fill)


def rounded(draw, box, fill, outline=(45, 45, 45)):
    draw.rectangle(box, fill=fill, outline=outline, width=3)


def build_architecture_png():
    img = Image.new("RGB", (1800, 850), "white")
    d = ImageDraw.Draw(img)
    title = font(36, True)
    label = font(26, True)
    small = font(22)
    tiny = font(19)

    d.text((900, 45), "Proposed NIDS-XAI Architecture", font=title, anchor="mm", fill=(10, 10, 10))

    boxes = {
        "dataset": (60, 185, 270, 330),
        "prep": (360, 150, 620, 365),
        "features": (710, 185, 950, 330),
        "ml": (1060, 110, 1340, 330),
        "dl": (1060, 430, 1340, 610),
        "pred": (1460, 230, 1690, 375),
        "xai": (1415, 500, 1760, 690),
        "dash": (560, 640, 870, 785),
        "eval": (1110, 25, 1510, 95),
    }

    rounded(d, boxes["dataset"], "#ffffff")
    draw_centered(d, boxes["dataset"], "NSL-KDD\nNetwork traffic\nrecords", label)

    rounded(d, boxes["prep"], "#f4f7fb")
    draw_centered(d, boxes["prep"], "Preprocessing\nClean data\nEncode categories\nScale features\nBinary and multiclass labels", small)

    rounded(d, boxes["features"], "#ffffff")
    draw_centered(d, boxes["features"], "Feature Matrix\nProcessed train\nand test sets", label)

    rounded(d, boxes["ml"], "#f4f7fb")
    draw_centered(d, boxes["ml"], "ML Models\nRandom Forest\nXGBoost\nDecision Tree\nLogistic Regression", small)

    rounded(d, boxes["dl"], "#f4f7fb")
    draw_centered(d, boxes["dl"], "DL Models\nMLP classifier\nAutoencoder\nAnomaly detection", small)

    rounded(d, boxes["pred"], "#ffffff")
    draw_centered(d, boxes["pred"], "Prediction\nNormal or Attack\nAttack category", label)

    rounded(d, boxes["xai"], "#fff7e6")
    draw_centered(d, boxes["xai"], "Explainability Layer\nSHAP global and local\nLIME instance explanation\nFeature importance", small)

    rounded(d, boxes["dash"], "#eef8f1")
    draw_centered(d, boxes["dash"], "Dashboard\nPrediction, metrics,\nplots and explanations", label)

    rounded(d, boxes["eval"], "#ffffff")
    draw_centered(d, boxes["eval"], "Evaluation: Accuracy, Precision, Recall, F1", tiny)

    arrow(d, (270, 258), (360, 258))
    arrow(d, (620, 258), (710, 258))
    arrow(d, (950, 258), (1060, 220))
    arrow(d, (950, 258), (1060, 520))
    arrow(d, (1340, 220), (1460, 285))
    arrow(d, (1340, 520), (1460, 320))
    arrow(d, (1575, 375), (1575, 500))
    arrow(d, (1415, 595), (870, 710))
    arrow(d, (1575, 230), (1510, 75))

    d.text((490, 405), "Data layer", font=tiny, anchor="mm", fill=(60, 60, 60))
    d.line((380, 420, 600, 420), fill=(120, 120, 120), width=2)
    d.text((1200, 650), "Detection layer", font=tiny, anchor="mm", fill=(60, 60, 60))
    d.line((1080, 665, 1320, 665), fill=(120, 120, 120), width=2)
    d.text((1585, 735), "Interpretability layer", font=tiny, anchor="mm", fill=(60, 60, 60))
    d.line((1425, 750, 1745, 750), fill=(120, 120, 120), width=2)

    img.save(ARCH_PNG)


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_cell_text(cell, text, bold=False):
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER if len(text) < 30 else WD_ALIGN_PARAGRAPH.LEFT
    run = p.add_run(text)
    run.bold = bold
    run.font.name = "Times New Roman"
    run.font.size = Pt(8)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def set_two_columns(section):
    sect_pr = section._sectPr
    cols = sect_pr.xpath("./w:cols")
    cols = cols[0] if cols else OxmlElement("w:cols")
    cols.set(qn("w:num"), "2")
    cols.set(qn("w:space"), "360")
    if not sect_pr.xpath("./w:cols"):
        sect_pr.append(cols)


def set_one_column(section):
    sect_pr = section._sectPr
    cols = sect_pr.xpath("./w:cols")
    cols = cols[0] if cols else OxmlElement("w:cols")
    cols.set(qn("w:num"), "1")
    if not sect_pr.xpath("./w:cols"):
        sect_pr.append(cols)


def style_doc(doc):
    section = doc.sections[0]
    section.page_width = Inches(8.27)
    section.page_height = Inches(11.69)
    section.top_margin = Inches(0.75)
    section.bottom_margin = Inches(0.75)
    section.left_margin = Inches(0.8)
    section.right_margin = Inches(0.8)
    set_one_column(section)

    styles = doc.styles
    styles["Normal"].font.name = "Times New Roman"
    styles["Normal"].font.size = Pt(10)
    styles["Normal"].paragraph_format.space_after = Pt(6)
    styles["Normal"].paragraph_format.line_spacing = 1


def add_paragraph(doc, text, style=None, align=None, size=10, bold=False, italic=False):
    p = doc.add_paragraph(style=style)
    if align:
        p.alignment = align
    else:
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(size)
    r.bold = bold
    r.italic = italic
    return p


def add_caption(doc, caption):
    p = add_paragraph(doc, caption, align=WD_ALIGN_PARAGRAPH.CENTER, size=8, italic=False)
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(6)


def add_picture_if_exists(doc, path, width):
    if path.exists():
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run()
        run.add_picture(str(path), width=width)


def parse_table(lines, start):
    table_lines = []
    i = start
    while i < len(lines) and lines[i].strip().startswith("|"):
        table_lines.append(lines[i].strip())
        i += 1
    rows = []
    for line in table_lines:
        parts = [p.strip() for p in line.strip("|").split("|")]
        if all(re.fullmatch(r":?-+:?", p or "") for p in parts):
            continue
        rows.append(parts)
    return rows, i


def add_markdown_table(doc, rows):
    if not rows:
        return
    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Light Grid Accent 1"
    for ri, row in enumerate(rows):
        for ci, value in enumerate(row):
            cell = table.cell(ri, ci)
            set_cell_text(cell, value, bold=ri == 0)
            if ri == 0:
                set_cell_shading(cell, "EAF2F8")
    doc.add_paragraph()


def build_docx():
    build_architecture_png()
    md = DRAFT.read_text(encoding="utf-8")
    lines = md.splitlines()

    doc = Document()
    style_doc(doc)

    # Title block.
    title = lines[0].lstrip("# ").strip()
    add_paragraph(doc, title, align=WD_ALIGN_PARAGRAPH.CENTER, size=20, bold=True)
    add_paragraph(doc, "Piyush M. Borkar, Varun Gada", align=WD_ALIGN_PARAGRAPH.CENTER, size=10)
    add_paragraph(
        doc,
        "Department of Artificial Intelligence & Data Science / Department of Computer Engineering, "
        "Marathwada Mitramandal's Institute of Technology, Pune, India",
        align=WD_ALIGN_PARAGRAPH.CENTER,
        size=9,
    )
    add_paragraph(
        doc,
        "piyushborkar.official@gmail.com, varungada2004@gmail.com",
        align=WD_ALIGN_PARAGRAPH.CENTER,
        size=9,
    )

    # Abstract and keywords.
    abstract = re.search(r"## Abstract\n\n(.*?)\n\n## Keywords", md, re.S).group(1).strip()
    keywords = re.search(r"## Keywords\n\n(.*?)\n\n## I\. Introduction", md, re.S).group(1).strip()
    add_paragraph(doc, "Abstract", align=WD_ALIGN_PARAGRAPH.LEFT, size=9, bold=True)
    add_paragraph(doc, abstract, size=9, italic=True)
    add_paragraph(doc, "Keywords-" + keywords, size=9, italic=True)

    add_picture_if_exists(doc, ARCH_PNG, Inches(6.2))
    add_caption(doc, "Fig. 1. Proposed NIDS-XAI workflow from NSL-KDD preprocessing to prediction, explanation, and dashboard visualization.")

    body_start = lines.index("## I. Introduction")
    i = body_start
    pending_table_caption = None
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()
        if not line or line.startswith("Suggested figure file:"):
            i += 1
            continue
        if line.startswith("> Draft status") or line.startswith("## Context Needed"):
            break
        if line.startswith("# ") or line in {"## Abstract", "## Keywords"}:
            i += 1
            continue
        if line.startswith("## References"):
            add_paragraph(doc, "REFERENCES", align=WD_ALIGN_PARAGRAPH.CENTER, size=10, bold=True)
            i += 1
            continue
        if line.startswith("## "):
            p = add_paragraph(doc, line[3:].upper(), align=WD_ALIGN_PARAGRAPH.LEFT, size=12, bold=True)
            p.paragraph_format.space_before = Pt(10)
            i += 1
            continue
        if line.startswith("### "):
            p = add_paragraph(doc, line[4:], size=10, bold=True)
            p.paragraph_format.space_before = Pt(4)
            i += 1
            continue
        if re.match(r"TABLE [IVX]+\. ", line):
            pending_table_caption = line
            add_paragraph(doc, line, align=WD_ALIGN_PARAGRAPH.CENTER, size=8, bold=True)
            i += 1
            continue
        if line.startswith("|"):
            rows, i = parse_table(lines, i)
            add_markdown_table(doc, rows)
            pending_table_caption = None
            continue
        if line.startswith("**Fig. "):
            caption = line.strip("*")
            fig_key = caption.split(" ", 2)[0] + " " + caption.split(" ", 2)[1]
            if fig_key == "Fig. 1.":
                i += 1
                continue
            path = FIGURE_FILES.get(fig_key)
            if path:
                width = Inches(5.8) if fig_key in {"Fig. 2.", "Fig. 5."} else Inches(5.2)
                add_picture_if_exists(doc, path, width)
            add_caption(doc, caption)
            i += 1
            continue
        if re.match(r"\d+\. ", line):
            p = doc.add_paragraph(style="List Number")
            r = p.add_run(re.sub(r"^\d+\.\s+", "", line))
            r.font.name = "Times New Roman"
            r.font.size = Pt(10)
            i += 1
            continue
        if line.startswith("[") and "] " in line:
            add_paragraph(doc, line, size=8)
            i += 1
            continue
        add_paragraph(doc, line, size=10)
        i += 1

    doc.save(OUT_DOCX)


if __name__ == "__main__":
    build_docx()
    print(OUT_DOCX)
