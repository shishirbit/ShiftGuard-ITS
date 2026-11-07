from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor


OUTPUT = Path("output/editor_revision_20260918/Author_Response_Sheet.docx")


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=100, start=120, bottom=100, end=120):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{margin}"))
        if node is None:
            node = OxmlElement(f"w:{margin}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_table_borders(table, color="D9D9D9", size="6"):
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.first_child_found_in("w:tblBorders")
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = qn(f"w:{edge}")
        border = borders.find(tag)
        if border is None:
            border = OxmlElement(f"w:{edge}")
            borders.append(border)
        border.set(qn("w:val"), "single")
        border.set(qn("w:sz"), size)
        border.set(qn("w:space"), "0")
        border.set(qn("w:color"), color)


def set_repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def set_run_font(run, name="Arial", size=10.5, bold=None, color=None):
    run.font.name = name
    run._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:ascii"), name)
    run._element.rPr.rFonts.set(qn("w:hAnsi"), name)
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def add_cell_paragraph(cell, text, bold_lead=None):
    paragraph = cell.paragraphs[0] if not cell.paragraphs[0].text else cell.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(3)
    paragraph.paragraph_format.line_spacing = 1.08
    if bold_lead and text.startswith(bold_lead):
        first = paragraph.add_run(bold_lead)
        set_run_font(first, bold=True)
        rest = paragraph.add_run(text[len(bold_lead):])
        set_run_font(rest)
    else:
        run = paragraph.add_run(text)
        set_run_font(run)
    return paragraph


doc = Document()
section = doc.sections[0]
section.top_margin = Cm(1.7)
section.bottom_margin = Cm(1.7)
section.left_margin = Cm(1.65)
section.right_margin = Cm(1.65)

styles = doc.styles
normal = styles["Normal"]
normal.font.name = "Arial"
normal._element.rPr.rFonts.set(qn("w:ascii"), "Arial")
normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Arial")
normal.font.size = Pt(10.5)
normal.paragraph_format.space_after = Pt(6)
normal.paragraph_format.line_spacing = 1.08

title_style = styles["Title"]
title_style.font.name = "Arial"
title_style._element.rPr.rFonts.set(qn("w:ascii"), "Arial")
title_style._element.rPr.rFonts.set(qn("w:hAnsi"), "Arial")
title_style.font.size = Pt(18)
title_style.font.bold = True
title_style.font.color.rgb = RGBColor(0, 0, 0)
title_style.paragraph_format.space_after = Pt(8)
style_ppr = title_style._element.get_or_add_pPr()
style_border = style_ppr.find(qn("w:pBdr"))
if style_border is not None:
    style_ppr.remove(style_border)

title = doc.add_paragraph("Author Response to Editorial Amendments", style="Title")
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
title_ppr = title._p.get_or_add_pPr()
title_border = title_ppr.find(qn("w:pBdr"))
if title_border is not None:
    title_ppr.remove(title_border)
for run in title.runs:
    set_run_font(run, size=18, bold=True, color="000000")

meta = doc.add_paragraph()
meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
meta.paragraph_format.space_after = Pt(10)
for text, bold in [
    ("Manuscript title: ", True),
    ("ShiftGuard: Risk-Controlled Spatiotemporal Traffic Forecasting under Compound Regime Shifts and Sensor Outages\n", False),
    ("Author: ", True),
    ("Shishir Singh Chauhan\n", False),
    ("Date: ", True),
    ("18 September 2026", False),
]:
    run = meta.add_run(text)
    set_run_font(run, size=10.5, bold=bold)

intro = doc.add_paragraph()
intro.paragraph_format.space_after = Pt(9)
run = intro.add_run(
    "Dear Editor,\n\nThank you for identifying these presentation issues. "
    "The revised manuscript addresses both amendments. The responses and exact locations of the changes are provided below."
)
set_run_font(run)

table = doc.add_table(rows=1, cols=4)
table.alignment = WD_TABLE_ALIGNMENT.CENTER
table.autofit = False
widths = [Inches(0.42), Inches(1.78), Inches(3.83), Inches(0.86)]
headers = ["No.", "Editorial amendment", "Author response and revision", "Location"]
for idx, (cell, width, header) in enumerate(zip(table.rows[0].cells, widths, headers)):
    cell.width = width
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    set_cell_shading(cell, "1F4E78")
    set_cell_margins(cell, top=110, bottom=110)
    paragraph = cell.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_after = Pt(0)
    run = paragraph.add_run(header)
    set_run_font(run, size=9.5, bold=True, color="FFFFFF")
set_repeat_table_header(table.rows[0])

rows = [
    (
        "1",
        "The location of the Figure 5 citation is not clear. Ensure the figure is cited and indicate where it should appear.",
        "Response: The manuscript now cites Figure 5 explicitly in Section 6.3 immediately before its insertion marker. "
        "The added sentence reads: \"Figure 5 directly compares the retention and accepted-origin failure rates of "
        "ShiftGuard and STAEformer under the registered risk controller, together with the all-forecast ShiftGuard control.\" "
        "A visible \"[Insert Fig. 5 near here]\" marker now identifies the intended placement.",
        "Section 6.3, page 21; Fig. 5 appears on page 23",
    ),
    (
        "2",
        "Provide the email address of the corresponding author in the manuscript.",
        "Response: The author is now marked as the corresponding author, and the following email address has been added to the title page: shishir.chauhan@jaipur.manipal.edu.",
        "Title page, page 1",
    ),
]

for row_index, values in enumerate(rows, start=1):
    cells = table.add_row().cells
    fill = "FFFFFF" if row_index % 2 else "F2F6FA"
    for cell, width, value in zip(cells, widths, values):
        cell.width = width
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        set_cell_shading(cell, fill)
        set_cell_margins(cell, top=120, bottom=120)
        paragraph = cell.paragraphs[0]
        paragraph.paragraph_format.space_after = Pt(0)
        paragraph.paragraph_format.line_spacing = 1.08
        run = paragraph.add_run(value)
        set_run_font(run, size=9.4)
    cells[0].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    cells[3].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

set_table_borders(table)

closing = doc.add_paragraph()
closing.paragraph_format.space_before = Pt(10)
closing.paragraph_format.space_after = Pt(0)
run = closing.add_run("Sincerely,\nShishir Singh Chauhan\nManipal University Jaipur")
set_run_font(run)

doc.core_properties.title = "Author Response to Editorial Amendments"
doc.core_properties.author = "Shishir Singh Chauhan"
doc.core_properties.subject = "Response to editorial amendments for the ShiftGuard manuscript"

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUTPUT)
print(OUTPUT)
