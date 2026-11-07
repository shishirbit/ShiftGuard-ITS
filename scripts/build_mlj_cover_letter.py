from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


OUT = Path(r"D:\Research\Paper-6_ShiftGuard\output\ShiftGuard_MLJ_Resubmission_Ready_20261001\MLJ_Cover_Letter.docx")


def set_font(style, name, size, bold=False):
    style.font.name = name
    style._element.rPr.rFonts.set(qn("w:ascii"), name)
    style._element.rPr.rFonts.set(qn("w:hAnsi"), name)
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.color.rgb = RGBColor(0, 0, 0)


doc = Document()
section = doc.sections[0]
section.page_width = Inches(8.5)
section.page_height = Inches(11)
section.top_margin = Inches(0.68)
section.bottom_margin = Inches(0.65)
section.left_margin = Inches(0.82)
section.right_margin = Inches(0.82)

normal = doc.styles["Normal"]
set_font(normal, "Times New Roman", 10.5)
normal.paragraph_format.line_spacing = 1.03
normal.paragraph_format.space_after = Pt(6)

title = doc.styles["Title"]
set_font(title, "Times New Roman", 15, bold=True)
title.paragraph_format.space_after = Pt(7)
title_ppr = title._element.get_or_add_pPr()
border = title_ppr.find(qn("w:pBdr"))
if border is not None:
    title_ppr.remove(border)

p = doc.add_paragraph("Cover Letter", style="Title")
p.alignment = WD_ALIGN_PARAGRAPH.CENTER

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
p.paragraph_format.space_after = Pt(7)
r = p.add_run("Shishir Singh Chauhan\n")
r.bold = True
p.add_run(
    "Manipal University Jaipur\n"
    "Dehmi Kalan, Jaipur-Ajmer Expressway\n"
    "Jaipur, Rajasthan 303007, India\n"
    "shishir.chauhan@jaipur.manipal.edu"
)

doc.add_paragraph("1 October 2026")

p = doc.add_paragraph()
p.add_run("Editor-in-Chief\n").bold = True
p.add_run("Machine Learning\nSpringer Nature")

p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(8)
r = p.add_run("Subject  ")
r.bold = True
p.add_run(
    "Resubmission of ShiftGuard Risk-Controlled Spatiotemporal Traffic Forecasting under Compound Regime Shifts and Sensor Outages"
)

doc.add_paragraph("Dear Editor-in-Chief,")

doc.add_paragraph(
    "I am resubmitting the manuscript entitled \"ShiftGuard: Risk-Controlled Spatiotemporal Traffic Forecasting under Compound Regime Shifts and Sensor Outages\" for consideration in Machine Learning. In response to the editorial notice, this resubmission includes the required two-page MLJ Contribution Information Sheet. The introductory material has also been revised so that the main claim, supporting evidence, closest related work, and publication history are stated directly in the manuscript."
)

doc.add_paragraph(
    "The paper studies graph time-series forecasting when traffic regimes change while sensor observations are lost in structured patterns. Its main claim is that this setting should be treated as a selective prediction problem in which the observation process and the release decision are part of the learning system. ShiftGuard combines structured-outage training, mask-aware graph-temporal prediction, prequential node-horizon calibration, and origin-level abstention. This formulation is relevant to machine learning systems that operate on partially observed graph streams, including transportation, energy, environmental, and infrastructure networks."
)

doc.add_paragraph(
    "The evidence comprises 640 confirmatory stress runs on METR-LA and PEMS-BAY, using four outage families, five severities, and five random seeds. Relative to the strongest locally reproduced transformer, STAEformer, ShiftGuard reduces mean stress MAE by 0.347 mph on METR-LA and 0.546 mph on PEMS-BAY. At 30% outage severity, it retains 86% and 98% of forecast origins while limiting empirical failure among accepted origins to 8% and 2%, respectively. The manuscript also reports paired statistical comparisons, subgroup coverage, risk-coverage behaviour, ablations, and efficiency. The claim is deliberately limited: ShiftGuard is statistically tied with STAEformer on clean METR-LA and is 0.027 mph worse on clean PEMS-BAY."
)

doc.add_paragraph(
    "I believe the manuscript fits Machine Learning because it connects structured partial observability, distribution shift, conformal calibration, and selective prediction in a reproducible graph-forecasting study. The contribution is the learning formulation and its empirical validation, rather than a claim of universal clean-benchmark superiority."
)

doc.add_paragraph(
    "The manuscript and its results have not been published in a conference, workshop, journal, or book chapter, are not an extension of a previous paper, and are not under consideration elsewhere. The study uses public benchmark data and involves no human participants or animals. Code, experiment configurations, and dataset access information are provided in the manuscript to support reproducibility."
)

doc.add_paragraph("Thank you for considering this resubmission.")

p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(4)
p.add_run("Sincerely,\n")
r = p.add_run("Shishir Singh Chauhan")
r.bold = True
p.add_run("\nCorresponding Author\nManipal University Jaipur")

doc.core_properties.title = "Cover Letter for Machine Learning"
doc.core_properties.subject = "Resubmission of the ShiftGuard manuscript"
doc.core_properties.author = "Shishir Singh Chauhan"
doc.save(OUT)
print(OUT)
