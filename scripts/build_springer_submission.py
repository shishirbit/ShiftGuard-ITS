from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper" / "main.tex"
TEMPLATE = ROOT / "tmp" / "springer-template-dec2024" / "sn-article-template"
PACKAGE = ROOT / "output" / "springer_submission"
FIGURES = PACKAGE / "figures"
TMP_FIGURES = ROOT / "tmp" / "springer-figures"
TECTONIC = ROOT / "tools" / "tectonic" / "tectonic.exe"


def braced_argument(text: str, command: str) -> str:
    start = text.index(command) + len(command)
    if text[start] != "{":
        raise ValueError(f"Expected braced argument after {command}")
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{" and (index == 0 or text[index - 1] != "\\"):
            depth += 1
        elif text[index] == "}" and (index == 0 or text[index - 1] != "\\"):
            depth -= 1
            if depth == 0:
                return text[start + 1 : index]
    raise ValueError(f"Unclosed argument after {command}")


def clean_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def compile_figure(index: int, tikz: str) -> Path:
    tex = TMP_FIGURES / f"figure{index}.tex"
    tex.write_text(
        "\\documentclass[tikz,border=4pt]{standalone}\n"
        "\\usepackage{amsmath,amssymb,graphicx}\n"
        "\\usepackage{pgfplots}\n"
        "\\usepgfplotslibrary{groupplots}\n"
        "\\usetikzlibrary{arrows.meta,positioning}\n"
        "\\pgfplotsset{compat=1.18}\n"
        "\\begin{document}\n"
        f"{tikz}\n"
        "\\end{document}\n",
        encoding="utf-8",
    )
    subprocess.run(
        [str(TECTONIC), "-X", "compile", tex.name, "--outdir", str(TMP_FIGURES)],
        cwd=TMP_FIGURES,
        check=True,
    )
    pdf = TMP_FIGURES / f"figure{index}.pdf"
    target = FIGURES / pdf.name
    shutil.copy2(pdf, target)
    return target


def externalize_figures(body: str) -> str:
    pattern = re.compile(
        r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}", re.DOTALL
    )
    replacements: list[tuple[str, str]] = []
    for index, match in enumerate(pattern.finditer(body), start=1):
        block = match.group(0)
        tikz_match = re.search(
            r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", block, re.DOTALL
        )
        if not tikz_match:
            raise ValueError(f"Figure {index} does not contain a TikZ picture")
        caption = braced_argument(block, "\\caption")
        label_match = re.search(r"\\label\{([^}]+)\}", block)
        if not label_match:
            raise ValueError(f"Figure {index} has no label")
        label = label_match.group(1)
        tikz = tikz_match.group(0)
        if index == 6:
            tikz = tikz.replace("at={(1.06,-0.20)}", "at={(1.06,-0.34)}")
        compile_figure(index, tikz)
        max_height = "0.46" if index == 1 else "0.34"
        replacement = (
            "\\begin{figure*}[t]\n"
            "\\centering\n"
            f"\\includegraphics[width=\\textwidth,height={max_height}\\textheight,keepaspectratio]"
            f"{{figures/figure{index}.pdf}}\n"
            f"\\caption{{{caption}}}\\label{{{label}}}\n"
            "\\end{figure*}"
        )
        replacements.append((block, replacement))
    for old, new in replacements:
        body = body.replace(old, new, 1)
    return body


def inline_tables(body: str) -> str:
    def replace(match: re.Match[str]) -> str:
        filename = match.group(1)
        path = ROOT / "paper" / filename
        if not path.exists():
            raise FileNotFoundError(path)
        return path.read_text(encoding="utf-8").strip()

    return re.sub(r"\\input\{(generated_[^}]+\.tex)\}", replace, body)


def main() -> None:
    clean_directory(PACKAGE)
    clean_directory(FIGURES)
    clean_directory(TMP_FIGURES)

    source = SOURCE.read_text(encoding="utf-8")
    abstract = (
        "Accurate short-term traffic forecasting is essential for intelligent "
        "transportation systems, but conventional benchmarks largely assume stable "
        "traffic regimes and fully observed sensor networks. This study presents "
        "ShiftGuard, a risk-controlled spatiotemporal framework for compound shifts "
        "in traffic conditions and sensor availability. ShiftGuard jointly produces "
        "point forecasts, calibrated node--horizon prediction intervals, and "
        "origin-level abstention decisions using reliability-conditioned graph--temporal "
        "encoding, structured-outage augmentation, probabilistic scale estimation, and "
        "prequential group-adaptive conformal calibration. A leakage-controlled protocol "
        "evaluates independent, burst, contiguous-region, and hub outages at multiple "
        "severities on METR-LA and PEMS-BAY. Across 640 confirmatory stress runs over five "
        "seeds, ShiftGuard reduces mean absolute error relative to the strongest locally "
        "reproduced transformer from 3.74 to 3.40 mph on METR-LA and from 2.49 to 1.94 mph "
        "on PEMS-BAY. At 30\\% outage severity, risk control retains 86\\% and 98\\% of "
        "forecast origins while limiting empirical origin-level failure to 8\\% and "
        "2\\%, respectively. Although clean PEMS-BAY accuracy remains 0.027 mph behind "
        "STAEformer, the results show that ShiftGuard provides stronger robustness, "
        "calibrated uncertainty, and actionable failure control under compound shifts."
    )
    keyword_text = re.search(
        r"\\begin\{keyword\}(.*?)\\end\{keyword\}", source, re.DOTALL
    ).group(1)
    keywords = re.sub(r"\s+", " ", keyword_text).replace("\\sep", ",").strip()
    start = source.index("\\section{Introduction}")
    end = source.index("\\section*{Data and code availability}")
    body = source[start:end].strip()
    body = inline_tables(body)
    body = body.replace("\\begin{table}[", "\\begin{table*}[")
    body = body.replace("\\end{table}", "\\end{table*}")
    body = body.replace("\\columnwidth", "\\textwidth")
    body = externalize_figures(body)

    preamble = r"""% Springer Nature journal article template, version 3.1 (December 2024)
\documentclass[pdflatex,sn-mathphys-num]{sn-jnl}
\usepackage{graphicx}
\usepackage{multirow}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{booktabs}
\usepackage{siunitx}
\usepackage{cleveref}
\usepackage{xcolor}
\raggedbottom

\begin{document}

\title[ShiftGuard for robust traffic forecasting]{ShiftGuard: Risk-Controlled Spatiotemporal Traffic Forecasting under Compound Regime Shifts and Sensor Outages}

\author[1]{\fnm{Shishir Singh} \sur{Chauhan}}

\affil[1]{\orgname{Manipal University Jaipur},
\orgaddress{\street{Dehmi Kalan, Jaipur-Ajmer Expressway},
\city{Jaipur}, \postcode{303007}, \state{Rajasthan}, \country{India}}}

"""
    backmatter = r"""

\backmatter

\bmhead{Acknowledgements}
Not applicable.

\section*{Declarations}

\bmhead{Funding}
Funding information was not supplied by the author.

\bmhead{Competing interests}
A competing-interests declaration was not supplied by the author.

\bmhead{Ethics approval and consent to participate}
Not applicable. This study uses previously published traffic-sensor benchmark data and does not involve human participants, human samples, or animals.

\bmhead{Consent for publication}
Not applicable.

\bmhead{Data availability}
The METR-LA and PEMS-BAY benchmark files are available through the official DCRNN data-preparation repository at \url{https://github.com/liyaguang/DCRNN#data-preparation} and its linked dataset archive at \url{https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX}. PEMS-BAY originates from the California Department of Transportation Performance Measurement System (PeMS), \url{https://pems.dot.ca.gov/}. The raw benchmark files are not redistributed in this repository or submission package.

\bmhead{Code availability}
The source code, preprocessing utilities, experiment configurations, fixed seeds, stress-mask generators, training and evaluation scripts, and table-generation utilities are openly available at \url{https://github.com/shishirbit/ShiftGuard-ITS}. Large generated checkpoints and prediction tensors are excluded from version control.

\bmhead{Author contributions}
Shishir Singh Chauhan conceived the study, developed the methodology and software, conducted the experiments, analyzed the results, and prepared the manuscript.

\bibliography{references}

\end{document}
"""
    manuscript = (
        preamble
        + "\\abstract{" + abstract + "}\n\n"
        + "\\keywords{" + keywords + "}\n\n"
        + "\\maketitle\n\n"
        + body
        + backmatter
    )
    (PACKAGE / "manuscript.tex").write_text(manuscript, encoding="utf-8")
    shutil.copy2(ROOT / "paper" / "references.bib", PACKAGE / "references.bib")
    shutil.copy2(TEMPLATE / "sn-jnl.cls", PACKAGE / "sn-jnl.cls")
    shutil.copy2(TEMPLATE / "bst" / "sn-mathphys-num.bst", PACKAGE / "sn-mathphys-num.bst")
    shutil.copytree(TEMPLATE / "bst", PACKAGE / "bst")
    (PACKAGE / "README.txt").write_text(
        "Springer Nature submission package\n"
        "Template: sn-jnl version 3.1, December 2024\n"
        "Main source: manuscript.tex\n"
        "Bibliography: references.bib\n"
        "Figures: figures/figure1.pdf through figures/figure6.pdf\n"
        "Compile with pdfLaTeX, BibTeX, pdfLaTeX, pdfLaTeX.\n"
        "Before submission, the author must confirm the Funding and Competing interests declarations and add an email address if required by the target journal.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
