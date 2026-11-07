from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


OUT = Path(r"D:\Research\Paper-6_ShiftGuard\output\mlj_resubmission_20261001\MLJ_Contribution_Information_Sheet.docx")


def set_cell_margins(cell, top=70, start=90, bottom=70, end=90):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for edge, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{edge}"))
        if node is None:
            node = OxmlElement(f"w:{edge}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def add_page_field(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run("Page ")
    run.font.size = Pt(8)
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    paragraph._p.append(fld)


doc = Document()
sec = doc.sections[0]
sec.page_width = Inches(8.5)
sec.page_height = Inches(11)
sec.top_margin = Inches(0.52)
sec.bottom_margin = Inches(0.52)
sec.left_margin = Inches(0.62)
sec.right_margin = Inches(0.62)
sec.header_distance = Inches(0.25)
sec.footer_distance = Inches(0.22)

styles = doc.styles
normal = styles["Normal"]
normal.font.name = "Arial"
normal._element.rPr.rFonts.set(qn("w:ascii"), "Arial")
normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Arial")
normal.font.size = Pt(9.4)
normal.font.color.rgb = RGBColor(0, 0, 0)
normal.paragraph_format.space_after = Pt(3.2)
normal.paragraph_format.line_spacing = 1.0

title_style = styles["Title"]
title_style.font.name = "Arial"
title_style._element.rPr.rFonts.set(qn("w:ascii"), "Arial")
title_style._element.rPr.rFonts.set(qn("w:hAnsi"), "Arial")
title_style.font.size = Pt(16)
title_style.font.bold = True
title_style.font.color.rgb = RGBColor(0, 0, 0)
title_style.paragraph_format.space_after = Pt(4)
title_ppr = title_style._element.get_or_add_pPr()
title_border = title_ppr.find(qn("w:pBdr"))
if title_border is not None:
    title_ppr.remove(title_border)

h1 = styles["Heading 1"]
h1.font.name = "Arial"
h1._element.rPr.rFonts.set(qn("w:ascii"), "Arial")
h1._element.rPr.rFonts.set(qn("w:hAnsi"), "Arial")
h1.font.size = Pt(10.6)
h1.font.bold = True
h1.font.color.rgb = RGBColor(0, 0, 0)
h1.paragraph_format.space_before = Pt(4.5)
h1.paragraph_format.space_after = Pt(1.5)
h1.paragraph_format.keep_with_next = True

if "Question" not in styles:
    q_style = styles.add_style("Question", WD_STYLE_TYPE.PARAGRAPH)
else:
    q_style = styles["Question"]
q_style.font.name = "Arial"
q_style._element.rPr.rFonts.set(qn("w:ascii"), "Arial")
q_style._element.rPr.rFonts.set(qn("w:hAnsi"), "Arial")
q_style.font.size = Pt(9.4)
q_style.font.italic = True
q_style.font.color.rgb = RGBColor(0, 0, 0)
q_style.paragraph_format.space_after = Pt(2)
q_style.paragraph_format.keep_with_next = True

doc.add_paragraph("MLJ Contribution Information Sheet", style="Title").alignment = WD_ALIGN_PARAGRAPH.CENTER

meta = doc.add_paragraph()
meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
meta.paragraph_format.space_after = Pt(4)
r = meta.add_run("Manuscript  ")
r.bold = True
meta.add_run("ShiftGuard: Risk-Controlled Spatiotemporal Traffic Forecasting under Compound Regime Shifts and Sensor Outages\n")
r = meta.add_run("Author  ")
r.bold = True
meta.add_run("Shishir Singh Chauhan   |   Manipal University Jaipur\n")
r = meta.add_run("Corresponding email  ")
r.bold = True
meta.add_run("shishir.chauhan@jaipur.manipal.edu")

doc.add_heading("Question 1 Main claim and importance", level=1)
doc.add_paragraph(
    "What is the main claim of the paper and why is it an important contribution to the machine learning literature",
    style="Question",
)
doc.add_paragraph(
    "The paper claims that graph time-series forecasting under simultaneous target-regime shift and structured loss of observations should be learned and evaluated as a selective prediction problem, rather than as clean point prediction followed by an independent missing-value repair step. ShiftGuard implements this claim through one leakage-controlled system: it exposes a compact mask-aware graph-temporal predictor to outage geometries during training, estimates heterogeneous predictive scales, updates node-horizon conformal scores only after past outcomes resolve, and releases or abstains from an entire forecast origin using disjoint threshold-construction and certification samples."
)
doc.add_paragraph(
    "The contribution to machine learning is the problem formulation and the evidence that the observation process and the release decision materially change model selection. Standard traffic benchmarks reward low average error on a fully observed chronological test split. That criterion can select a model that performs well when every sensor is available but fails when missingness is spatially correlated, temporally persistent, or concentrated at high-centrality nodes. The same issue arises in graph-valued streams from energy systems, environmental monitoring, and infrastructure telemetry. ShiftGuard therefore connects structured partial observability, subgroup uncertainty calibration, and selective prediction in one prequential evaluation. The paper does not claim universal clean-data superiority or distribution-free validity under arbitrary dependence. Its claim is empirical and bounded to the registered datasets, outage processes, and risk definitions."
)

doc.add_heading("Question 2 Evidence supporting the claim", level=1)
doc.add_paragraph(
    "What evidence is provided to support the claim",
    style="Question",
)
doc.add_paragraph(
    "The evidence comes from a leakage-controlled study on METR-LA and PEMS-BAY. It uses chronological splits, four outage families (independent, temporal burst, contiguous region, and high-centrality hub), five outage severities, five random seeds, and 640 confirmatory stress runs. The comparison spans persistence, ridge and linear models, MLP and GRU predictors, STGCN, DCRNN, Graph WaveNet, AGCRN, STID, and STAEformer. All locally reproduced neural baselines use the same data pipeline and evaluation units. The main results are:"
)
for item in [
    "Under the registered compound-shift protocol, ShiftGuard reduces mean MAE relative to the strongest locally reproduced transformer, STAEformer, from 3.74 to 3.40 mph on METR-LA and from 2.49 to 1.94 mph on PEMS-BAY. The corresponding average advantages are 0.347 and 0.546 mph.",
    "The advantage grows with outage severity. Relative to STAEformer, the MAE difference increases from 0.133 to 0.521 mph between 10% and 50% severity on METR-LA, and from 0.224 to 0.866 mph on PEMS-BAY.",
    "At 30% outage severity, the risk controller retains 86% of METR-LA origins and 98% of PEMS-BAY origins, while empirical failure among accepted origins is 8% and 2%, respectively. The paper also reports marginal and worst node-horizon coverage, interval width, AURC, and risk-coverage curves rather than relying on marginal coverage alone.",
    "The clean-data results limit the claim: ShiftGuard is statistically tied with STAEformer on METR-LA and is 0.027 mph worse on PEMS-BAY. Thus, robustness is not inferred from a clean-data win.",
    "Ablations identify the source of the gain. Removing structured outage augmentation increases MAE by 0.306 mph on METR-LA and 0.703 mph on PEMS-BAY; using independent masks alone loses 0.098 and 0.220 mph. Some architectural removals do not hurt and can improve error, so the paper explicitly avoids claiming that every component is necessary.",
    "ShiftGuard uses about 89,000 to 100,000 parameters versus 1.26 to 1.37 million for STAEformer. On the reported GPU, batch-eight throughput is approximately 7.1 times higher on METR-LA and 12.4 times higher on PEMS-BAY.",
]:
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.left_indent = Inches(0.18)
    p.paragraph_format.first_line_indent = Inches(-0.12)
    p.paragraph_format.space_after = Pt(1.5)
    p.add_run(item)

doc.add_paragraph(
    "Paired hierarchical inference, seed-level variability, outage-family breakdowns, efficiency measurements, and negative ablations are reported so that robustness, calibration, and computational cost can be evaluated separately. Together, these results support the narrower conclusion that structured outage exposure plus prequential selective control improves dependability under the tested compound shifts."
)

doc.add_page_break()
doc.add_heading("Question 3 Closest related contributions", level=1)
doc.add_paragraph(
    "What papers by other authors make the most closely related contributions and how is this paper related to them",
    style="Question",
)
doc.add_paragraph(
    "DCRNN (Li et al., 2018) and Graph WaveNet (Wu et al., 2019) established diffusion-based and learned-adjacency graph forecasting. STID (Shao et al., 2022) showed the strength of identity embeddings, STAEformer (Liu et al., 2023) provided a strong adaptive-embedding transformer, and TESTAM (Lee and Ko, 2024) routed recurring and non-recurring traffic through specialized experts. ShiftGuard uses these works as conceptual or experimental comparators. Its distinction is not a larger point-forecasting backbone; it couples an observation mask, structured outage training, prequential uncertainty calibration, and an origin-level abstention rule, then evaluates that complete system under graph-aware outages."
)
doc.add_paragraph(
    "For missing data, Baggag et al. (2021) and Deng et al. (2023) reconstruct traffic observations using tensor or graph-tensor structure. ShiftGuard does not present imputation accuracy as the endpoint. It conditions prediction on observed masks, tests different outage geometries, and evaluates the decision to release a complete future tensor. For uncertainty, Stankeviciute et al. (2021) developed conformal time-series forecasting; Gibbs and Candes (2024) and Angelopoulos et al. (2024) developed online conformal methods under changing distributions; and Bao et al. (2025) studied online selective conformal prediction with false coverage-rate control. ShiftGuard adapts these ideas to partially observed graph streams through node-horizon calibration groups and a forecast-origin release rule. It does not replace the general theory in those papers and does not claim unconditional validity under arbitrary temporal dependence."
)

doc.add_heading("Question 4 Prior publication", level=1)
doc.add_paragraph(
    "Have parts of the paper been published before",
    style="Question",
)
doc.add_paragraph(
    "No. Neither the manuscript nor any part of its methods, experiments, or results has been published in a conference, workshop, journal, or book chapter. The submission is not an extension of a previous paper by the author. The public code repository and dataset links are supplied only to support reproducibility and do not constitute a prior archival publication. The manuscript is not under consideration by another journal."
)

add_page_field(sec.footer.paragraphs[0])

for p in doc.paragraphs:
    p.paragraph_format.widow_control = True

doc.core_properties.title = "MLJ Contribution Information Sheet"
doc.core_properties.subject = "Contribution information for the ShiftGuard manuscript"
doc.core_properties.author = "Shishir Singh Chauhan"
doc.core_properties.keywords = "Machine Learning journal, contribution information sheet, ShiftGuard"
doc.save(OUT)
print(OUT)
