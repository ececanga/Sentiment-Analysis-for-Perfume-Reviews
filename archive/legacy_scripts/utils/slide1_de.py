# Date Created: 2026-03-02 11:07:58+03:00
# Date Last Modified: 2026-05-15

"""
Deterministic slide-image generator for the worked example (no generative models, no fees, no API).

Outputs
- Folder: docs/diagrams
- Files: workedExample_slide1.png ... workedExample_slide11.png

Install
- pip install matplotlib pillow numpy

Run
- python archive/legacy_scripts/utils/slide1_de.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D projection)

from src.utils.paths import docs_path


# -----------------------------
# Output configuration
# -----------------------------

OUTPUT_DIR = docs_path("diagrams")


# -----------------------------
# Data
# -----------------------------

@dataclass(frozen=True)
class Review:
    perfume: str
    review_identifier: str
    final_transcript: str


def worked_example_reviews() -> List[Review]:
    return [
        Review(
            perfume="Citrus Dawn",
            review_identifier="R1",
            final_transcript=(
                "I love this bright citrus opening. Lemon and bergamot feel fresh and uplifting, like sunshine. "
                "It is clean, playful, and makes me happy. Longevity is decent and the drydown is soft musk and vanilla."
            ),
        ),
        Review(
            perfume="Citrus Dawn",
            review_identifier="R2",
            final_transcript=(
                "Citrus Dawn is cheerful and cozy. The orange blossom and vanilla are sweet but not cloying. "
                "It feels elegant, comforting, and confident. I adore how radiant it stays."
            ),
        ),
        Review(
            perfume="Smoky Noir",
            review_identifier="R3",
            final_transcript=(
                "Smoky Noir starts dark and smoky with leather and tar. At first I hate the harsh chemical edge; "
                "it feels overwhelming and weird. Later it becomes mysterious, intense, and strangely beautiful, "
                "but still a bit sad and brooding."
            ),
        ),
    ]


DISCRETE_CATEGORIES = [
    "Anger",
    "Anticipation",
    "Disgust",
    "Fear",
    "Joy",
    "Sadness",
    "Surprise",
    "Trust",
    "Positive sentiment",
    "Negative sentiment",
]

# Concrete anchors for charts in the worked example.
CITRUS_DISCRETE = {
    "Anger": 0,
    "Anticipation": 5,
    "Disgust": 1,
    "Fear": 0,
    "Joy": 12,
    "Sadness": 0,
    "Surprise": 0,
    "Trust": 7,
    "Positive sentiment": 15,
    "Negative sentiment": 1,
}
SMOKY_DISCRETE = {
    "Anger": 3,
    "Anticipation": 2,
    "Disgust": 6,
    "Fear": 7,
    "Joy": 1,
    "Sadness": 3,
    "Surprise": 2,
    "Trust": 1,
    "Positive sentiment": 1,
    "Negative sentiment": 12,
}

CITRUS_VALENCE_AROUSAL_DOMINANCE = {"Valence": 0.79, "Arousal": 0.58, "Dominance": 0.60}
SMOKY_VALENCE_AROUSAL_DOMINANCE = {"Valence": 0.30, "Arousal": 0.61, "Dominance": 0.58}


# -----------------------------
# Figure helpers
# -----------------------------

def make_slide_figure() -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a 16:9 slide canvas as a Matplotlib figure.
    """
    matplotlib.rcParams["font.size"] = 16
    matplotlib.rcParams["font.family"] = "DejaVu Sans"

    fig = plt.figure(figsize=(16, 9), dpi=140)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    # white background
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor="white", edgecolor="none"))
    return fig, ax


def add_title(ax: plt.Axes, title: str) -> None:
    ax.text(
        0.05,
        0.94,
        title,
        transform=ax.transAxes,
        fontsize=26,
        fontweight="bold",
        va="top",
        ha="left",
        color="black",
    )


def add_box(ax: plt.Axes, x: float, y: float, w: float, h: float, title: str, body_lines: List[str]) -> None:
    ax.add_patch(Rectangle((x, y), w, h, transform=ax.transAxes, fill=False, linewidth=2.0, edgecolor="black"))
    ax.text(x + 0.015, y + h - 0.03, title, transform=ax.transAxes, fontsize=18, fontweight="bold", va="top")
    text = "\n".join(body_lines)
    ax.text(x + 0.015, y + h - 0.08, text, transform=ax.transAxes, fontsize=14, va="top")


def add_arrow(ax: plt.Axes, x1: float, y1: float, x2: float, y2: float) -> None:
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=1.8,
        color="black",
    )
    ax.add_patch(arrow)


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def shorten(text: str, max_chars: int = 90) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


# -----------------------------
# Chart helpers
# -----------------------------

def radar_chart(fig: plt.Figure, rect: List[float], values_a: List[float], values_b: List[float], labels: List[str]) -> None:
    """
    Add a radar chart in a given rectangle (axes fraction coords).
    """
    ax = fig.add_axes(rect, polar=True)
    ax.set_facecolor("white")

    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    values_a = values_a + values_a[:1]
    values_b = values_b + values_b[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=10)
    ax.set_rlabel_position(0)
    ax.grid(True)

    ax.plot(angles, values_a, linewidth=2, label="Citrus Dawn")
    ax.fill(angles, values_a, alpha=0.10)
    ax.plot(angles, values_b, linewidth=2, label="Smoky Noir")
    ax.fill(angles, values_b, alpha=0.10)

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=11)


def bar_chart(fig: plt.Figure, rect: List[float], labels: List[str], values: List[float], title: str) -> None:
    ax = fig.add_axes(rect)
    ax.bar(range(len(values)), values)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=8)
    ax.set_title(title, fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)


def color_from_valence_arousal_dominance(valence: float, arousal: float, dominance: float) -> Tuple[int, int, int]:
    """
    Deterministic mapping for a color swatch:
    - Hue: valence mapped from red (0 degrees) to green (120 degrees)
    - Saturation: arousal mapped to 0..1
    - Lightness: dominance mapped to 0.30..0.70
    """
    hue_deg = float(valence) * 120.0
    sat = float(arousal)
    light = 0.30 + float(dominance) * 0.40

    # Convert HSL to RGB
    c = (1 - abs(2 * light - 1)) * sat
    h = hue_deg / 60.0
    x = c * (1 - abs(h % 2 - 1))
    if 0 <= h < 1:
        r1, g1, b1 = c, x, 0
    elif 1 <= h < 2:
        r1, g1, b1 = x, c, 0
    elif 2 <= h < 3:
        r1, g1, b1 = 0, c, x
    elif 3 <= h < 4:
        r1, g1, b1 = 0, x, c
    elif 4 <= h < 5:
        r1, g1, b1 = x, 0, c
    else:
        r1, g1, b1 = c, 0, x
    m = light - c / 2
    r, g, b = (r1 + m), (g1 + m), (b1 + m)
    return (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))


# -----------------------------
# Slide renderers
# -----------------------------

def render_slide_1() -> plt.Figure:
    fig, ax = make_slide_figure()
    add_title(ax, "Worked Example Setup (Randomly Generated Artificial Intelligence Perfume Reviews)")

    add_box(
        ax,
        x=0.05,
        y=0.58,
        w=0.44,
        h=0.28,
        title="Goal",
        body_lines=[
            "Demonstrate the complete pipeline end to end",
            "on a small, concrete worked example.",
        ],
    )

    add_box(
        ax,
        x=0.05,
        y=0.30,
        w=0.44,
        h=0.25,
        title="Input schema",
        body_lines=[
            "Columns include: Perfume, Final Transcript,",
            "and optionally Transcript.",
            "",
            "The file named YouTube_Perfume_Reviews_FullyFormatted.csv",
            "is treated as the immutable reference file that all emotion engines read from.",
        ],
    )

    reviews = worked_example_reviews()
    table_data = [[r.perfume, r.review_identifier, shorten(r.final_transcript, 80)] for r in reviews]

    ax_table = fig.add_axes([0.53, 0.30, 0.42, 0.56])
    ax_table.set_axis_off()
    tbl = ax_table.table(
        cellText=table_data,
        colLabels=["Perfume", "Review Identifier", "Final Transcript"],
        cellLoc="left",
        colLoc="left",
        loc="upper left",
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)

    ax.text(
        0.05,
        0.12,
        "Worked example uses three randomly generated artificial intelligence perfume reviews for two perfumes.",
        transform=ax.transAxes,
        fontsize=14,
        va="top",
    )
    return fig


def render_slide_2() -> plt.Figure:
    fig, ax = make_slide_figure()
    add_title(ax, "Data Source and Preprocessing")

    reviews = worked_example_reviews()
    table_data = [[r.perfume, r.review_identifier, shorten(r.final_transcript, 95)] for r in reviews]

    ax_table = fig.add_axes([0.05, 0.52, 0.90, 0.34])
    ax_table.set_axis_off()
    tbl = ax_table.table(
        cellText=table_data,
        colLabels=["Perfume", "Review Identifier", "Final Transcript"],
        cellLoc="left",
        colLoc="left",
        loc="upper left",
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)

    # Flow diagram
    y = 0.22
    w = 0.20
    h = 0.16
    xs = [0.05, 0.30, 0.55, 0.80]
    labels = [
        "Raw captions and metadata",
        "Preprocessing using format_transcripts.py\nFlatten line breaks\nNormalize whitespace\nStrip leading and trailing spaces",
        "Structured comma-separated values dataset",
        "Output file\nYouTube_Perfume_Reviews_FullyFormatted.csv",
    ]
    for i, x in enumerate(xs):
        ax.add_patch(Rectangle((x, y), w, h, transform=ax.transAxes, fill=False, linewidth=2))
        ax.text(x + 0.01, y + h - 0.02, labels[i], transform=ax.transAxes, fontsize=12, va="top")
        if i < 3:
            add_arrow(ax, x + w, y + h / 2, xs[i + 1], y + h / 2)

    ax.text(
        0.05,
        0.08,
        "Note: Tokenization and lowercasing happen later inside emotion extraction scripts.",
        transform=ax.transAxes,
        fontsize=14,
        va="top",
    )
    return fig


def render_slide_3() -> plt.Figure:
    fig, ax = make_slide_figure()
    add_title(ax, "Lexicon-Based Emotion Extraction (National Research Council Emotion Lexicon)")

    # Steps
    add_box(
        ax,
        0.05,
        0.60,
        0.28,
        0.22,
        "Step 1: Input final transcript excerpt",
        ["Citrus Dawn, Review R1:", shorten(worked_example_reviews()[0].final_transcript, 150)],
    )
    add_box(
        ax,
        0.36,
        0.60,
        0.28,
        0.22,
        "Step 2: Tokenization and lowercasing",
        ["Example tokens:", "love, bright, citrus, lemon, bergamot,", "fresh, uplifting, clean, playful, happy"],
    )
    add_box(
        ax,
        0.67,
        0.60,
        0.28,
        0.22,
        "Step 3: Lexicon match and counting",
        ["Match tokens against the National Research Council Emotion Lexicon", "Increment category counters for matched tokens"],
    )

    add_arrow(ax, 0.33, 0.71, 0.36, 0.71)
    add_arrow(ax, 0.64, 0.71, 0.67, 0.71)

    # Category list + bar chart
    ax.text(0.05, 0.52, "Discrete categories (ten):", transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")
    ax.text(0.05, 0.49, "\n".join(DISCRETE_CATEGORIES), transform=ax.transAxes, fontsize=13, va="top")

    # Bar chart for one review (illustrative)
    values = [1, 2, 0, 0, 6, 0, 0, 2, 7, 0]  # illustrative per-review counts
    bar_chart(fig, [0.45, 0.15, 0.50, 0.33], DISCRETE_CATEGORIES, values, "Discrete emotion counts for one review")

    ax.text(
        0.05,
        0.10,
        "Script file name: apply_emotion_vad_lexicons.py. Output file name: NRC_Emotion_Labels_Perfume.csv.",
        transform=ax.transAxes,
        fontsize=13,
        va="top",
    )
    return fig


def render_slide_4() -> plt.Figure:
    fig, ax = make_slide_figure()
    add_title(ax, "National Research Council Emotion Profile Comparison")

    # Radar chart values scaled
    citrus_vals = [CITRUS_DISCRETE[c] for c in DISCRETE_CATEGORIES]
    smoky_vals = [SMOKY_DISCRETE[c] for c in DISCRETE_CATEGORIES]
    max_val = max(max(citrus_vals), max(smoky_vals), 1)
    citrus_scaled = [v / max_val for v in citrus_vals]
    smoky_scaled = [v / max_val for v in smoky_vals]

    radar_chart(fig, [0.05, 0.18, 0.52, 0.70], citrus_scaled, smoky_scaled, DISCRETE_CATEGORIES)

    add_box(
        ax,
        0.62,
        0.30,
        0.33,
        0.55,
        "Worked example intuition",
        [
            "Citrus Dawn language includes love, bright, fresh, uplifting, happy",
            "leading to a joy, positive sentiment, and trust skew.",
            "",
            "Smoky Noir language includes dark, hate, harsh, chemical, sad, brooding",
            "leading to a negative sentiment, fear, disgust, and sadness skew.",
        ],
    )
    return fig


def render_slide_5() -> plt.Figure:
    fig, ax = make_slide_figure()
    add_title(ax, "Lexicon-Based Emotion Extraction (Valence, Arousal, Dominance)")

    add_box(
        ax,
        0.05,
        0.58,
        0.28,
        0.28,
        "Panel 1: Input final transcript excerpt",
        ["Smoky Noir, Review R3:", shorten(worked_example_reviews()[2].final_transcript, 170)],
    )
    add_box(
        ax,
        0.36,
        0.58,
        0.28,
        0.28,
        "Panel 2: Lexicon",
        [
            "Valence, Arousal, Dominance lexicon",
            "Match tokens that appear in the lexicon.",
            "Retrieve the three scores per matched token.",
        ],
    )
    add_box(
        ax,
        0.67,
        0.58,
        0.28,
        0.28,
        "Panel 3: Mean pooling in words",
        [
            "Valence score equals the mean of valence scores for all matched tokens.",
            "Arousal score equals the mean of arousal scores for all matched tokens.",
            "Dominance score equals the mean of dominance scores for all matched tokens.",
            "",
            "If no tokens match, assign zero to avoid missing values.",
        ],
    )
    add_arrow(ax, 0.33, 0.72, 0.36, 0.72)
    add_arrow(ax, 0.64, 0.72, 0.67, 0.72)

    ax.text(
        0.05,
        0.18,
        "Script file name: apply_emotion_vad_lexicons.py. Output file name: VAD_Scores_Perfume.csv.",
        transform=ax.transAxes,
        fontsize=14,
        va="top",
    )
    return fig


def render_slide_6() -> plt.Figure:
    fig, ax = make_slide_figure()
    add_title(ax, "Valence, Arousal, Dominance Space (Vector View and Centroid Per Perfume)")

    # 3D scatter as inset
    ax3d = fig.add_axes([0.05, 0.16, 0.60, 0.70], projection="3d")

    # Review points (illustrative)
    r1 = (0.83, 0.63, 0.61)
    r2 = (0.75, 0.53, 0.58)
    r3 = (0.30, 0.61, 0.58)

    ax3d.scatter([r1[0], r2[0]], [r1[1], r2[1]], [r1[2], r2[2]], s=80)
    ax3d.scatter([r3[0]], [r3[1]], [r3[2]], s=80)

    # Centroids
    cd = (CITRUS_VALENCE_AROUSAL_DOMINANCE["Valence"], CITRUS_VALENCE_AROUSAL_DOMINANCE["Arousal"], CITRUS_VALENCE_AROUSAL_DOMINANCE["Dominance"])
    sn = (SMOKY_VALENCE_AROUSAL_DOMINANCE["Valence"], SMOKY_VALENCE_AROUSAL_DOMINANCE["Arousal"], SMOKY_VALENCE_AROUSAL_DOMINANCE["Dominance"])
    ax3d.scatter([cd[0]], [cd[1]], [cd[2]], s=220, marker="X")
    ax3d.scatter([sn[0]], [sn[1]], [sn[2]], s=220, marker="X")

    ax3d.set_xlabel("Valence")
    ax3d.set_ylabel("Arousal")
    ax3d.set_zlabel("Dominance")
    ax3d.set_xlim(0, 1)
    ax3d.set_ylim(0, 1)
    ax3d.set_zlim(0, 1)
    ax3d.grid(True)

    # Point labels
    ax3d.text(r1[0], r1[1], r1[2], "Review R1", fontsize=10)
    ax3d.text(r2[0], r2[1], r2[2], "Review R2", fontsize=10)
    ax3d.text(r3[0], r3[1], r3[2], "Review R3", fontsize=10)
    ax3d.text(cd[0], cd[1], cd[2], "Citrus Dawn centroid", fontsize=10)
    ax3d.text(sn[0], sn[1], sn[2], "Smoky Noir centroid", fontsize=10)

    # Side table of centroid coordinates
    add_box(
        ax,
        0.70,
        0.50,
        0.25,
        0.36,
        "Centroid coordinates",
        [
            f"Citrus Dawn: Valence {cd[0]:.2f}, Arousal {cd[1]:.2f}, Dominance {cd[2]:.2f}",
            f"Smoky Noir: Valence {sn[0]:.2f}, Arousal {sn[1]:.2f}, Dominance {sn[2]:.2f}",
        ],
    )

    ax.text(
        0.70,
        0.36,
        "Caption: Each review is mapped to a three-dimensional vector; the perfume coordinate is the mean of its review vectors.",
        transform=ax.transAxes,
        fontsize=12,
        va="top",
        ha="left",
        wrap=True,
    )
    return fig


def render_slide_7() -> plt.Figure:
    fig, ax = make_slide_figure()
    add_title(ax, "Contextual Emotion Extraction (Transformer Ensemble)")

    # Left: input snippet
    add_box(
        ax,
        0.05,
        0.60,
        0.28,
        0.28,
        "Input final transcript excerpt",
        ["Smoky Noir, Review R3:", shorten(worked_example_reviews()[2].final_transcript, 170)],
    )

    # Middle: model boxes
    model_texts = [
        "Model one:\nDistilled Robustly Optimized Bidirectional Encoder Representations from Transformers model\nfor emotion classification by J Hartmann.",
        "Model two:\nDistilled Bidirectional Encoder Representations from Transformers model\nby Savani for emotion classification.",
        "Model three:\nBidirectional Encoder Representations from Transformers model\nfine-tuned on the GoEmotions multi-label emotion dataset.",
        "Model four:\nRobustly Optimized Bidirectional Encoder Representations from Transformers model\nby Cardiff Natural Language Processing optimized for social media text.",
    ]
    y0 = 0.66
    for i, t in enumerate(model_texts):
        y = y0 - i * 0.14
        ax.add_patch(Rectangle((0.36, y), 0.30, 0.12, transform=ax.transAxes, fill=False, linewidth=2))
        ax.text(0.37, y + 0.10, t, transform=ax.transAxes, fontsize=10, va="top")

    ax.text(
        0.36,
        0.83,
        "Constraint: The input text is truncated to the first five hundred and twelve characters.",
        transform=ax.transAxes,
        fontsize=12,
        va="bottom",
    )

    # Right: probability bars for each model (illustrative)
    probs = [
        [0.05, 0.08, 0.25, 0.30, 0.04, 0.15, 0.03, 0.05, 0.03, 0.02],
        [0.10, 0.06, 0.28, 0.20, 0.03, 0.18, 0.02, 0.05, 0.04, 0.04],
        [0.06, 0.07, 0.32, 0.18, 0.02, 0.21, 0.03, 0.04, 0.03, 0.04],
        [0.08, 0.05, 0.26, 0.22, 0.03, 0.24, 0.02, 0.05, 0.03, 0.02],
    ]
    titles = ["Model one output", "Model two output", "Model three output", "Model four output"]
    for i in range(4):
        rect = [0.70, 0.68 - i * 0.16, 0.25, 0.12]
        bar_chart(fig, rect, [""] * 10, probs[i], titles[i])

    ax.text(
        0.05,
        0.12,
        "Script file names: analyze_emotions_distilroberta_jhartmann.py, analyze_emotions_distilbert_savani.py, "
        "analyze_emotions_goemotions_multi.py, analyze_emotions_roberta_cardiffnlp.py.",
        transform=ax.transAxes,
        fontsize=12,
        va="top",
        wrap=True,
    )
    return fig


def render_slide_8() -> plt.Figure:
    fig, ax = make_slide_figure()
    add_title(ax, "Contextual Term and Scent Analysis (Explaining Why)")

    sentence = "I love this bright citrus opening. Lemon and bergamot feel fresh and uplifting, like sunshine."
    add_box(
        ax,
        0.05,
        0.65,
        0.62,
        0.22,
        "Example sentence (Citrus Dawn, Review R1)",
        [sentence],
    )

    add_box(
        ax,
        0.05,
        0.42,
        0.30,
        0.20,
        "Scent terms (example)",
        ["lemon", "bergamot", "citrus", "musk", "vanilla"],
    )
    add_box(
        ax,
        0.37,
        0.42,
        0.30,
        0.20,
        "Emotion anchor terms (example)",
        ["love", "bright", "fresh", "uplifting", "happy", "playful"],
    )

    add_box(
        ax,
        0.05,
        0.18,
        0.62,
        0.20,
        "Window-based extraction rule",
        [
            "Extract up to four words before and four words after an anchor term.",
            "This creates context keywords that help explain why an emotion is detected.",
        ],
    )

    add_box(
        ax,
        0.70,
        0.38,
        0.25,
        0.49,
        "Context keywords per perfume",
        [
            "Citrus Dawn keywords:",
            "lemon, bergamot, orange blossom, vanilla,",
            "fresh, uplifting, cheerful, comforting.",
            "",
            "Smoky Noir keywords:",
            "smoky, leather, tar, harsh, chemical,",
            "overwhelming, sad, brooding, mysterious.",
        ],
    )

    ax.text(
        0.70,
        0.28,
        "Scripts: extract_context_terms.py and analyze_top_terms.py.\nOutput file: perfume_context_keywords.csv.",
        transform=ax.transAxes,
        fontsize=12,
        va="top",
    )
    return fig


def render_slide_9() -> plt.Figure:
    fig, ax = make_slide_figure()
    add_title(ax, "Hybrid Fusion (From Per-Review Features to Per-Perfume Features)")

    # Three streams
    add_box(
        ax,
        0.05,
        0.62,
        0.27,
        0.26,
        "Stream one",
        [
            "Discrete emotion counts from the National Research Council Emotion Lexicon.",
            "Aggregation: sum counts across all reviews of the perfume.",
        ],
    )
    add_box(
        ax,
        0.36,
        0.62,
        0.27,
        0.26,
        "Stream two",
        [
            "Valence, Arousal, Dominance scores from a lexicon.",
            "Aggregation: mean Valence, mean Arousal, mean Dominance across reviews.",
        ],
    )
    add_box(
        ax,
        0.67,
        0.62,
        0.28,
        0.26,
        "Stream three",
        [
            "Transformer model probability distributions.",
            "Aggregation: mean probability per label across reviews.",
        ],
    )

    add_arrow(ax, 0.18, 0.62, 0.50, 0.45)
    add_arrow(ax, 0.50, 0.62, 0.50, 0.45)
    add_arrow(ax, 0.81, 0.62, 0.50, 0.45)

    add_box(
        ax,
        0.18,
        0.20,
        0.64,
        0.22,
        "Unified emotion profile per perfume (output)",
        [
            "Two rows: Citrus Dawn and Smoky Noir.",
            "Contains discrete sums, continuous means, and averaged transformer probabilities.",
        ],
    )

    ax.text(
        0.05,
        0.12,
        "Script file names: aggregate_and_merge_emotions.py and merge_emotion_profiles.py.",
        transform=ax.transAxes,
        fontsize=12,
        va="top",
    )
    return fig


def render_slide_10() -> plt.Figure:
    fig, ax = make_slide_figure()
    add_title(ax, "Hybrid Fusion Architecture (Concrete Worked Example Outputs)")

    add_box(ax, 0.05, 0.70, 0.26, 0.18, "Input block one", ["National Research Council discrete emotion counts"])
    add_box(ax, 0.05, 0.48, 0.26, 0.18, "Input block two", ["Valence, Arousal, Dominance continuous scores"])
    add_box(ax, 0.05, 0.26, 0.26, 0.18, "Input block three", ["Transformer probability distributions"])

    add_box(
        ax,
        0.36,
        0.32,
        0.36,
        0.56,
        "Fusion rules",
        [
            "Discrete counts are grouped by perfume and summed, then renamed with a National Research Council prefix.",
            "Continuous Valence, Arousal, Dominance values are grouped by perfume and averaged,",
            "then renamed with a Valence Arousal Dominance prefix.",
            "Transformer outputs are grouped by perfume and averaged, then merged into the unified profile.",
        ],
    )

    add_arrow(ax, 0.31, 0.79, 0.36, 0.75)
    add_arrow(ax, 0.31, 0.57, 0.36, 0.57)
    add_arrow(ax, 0.31, 0.35, 0.36, 0.40)

    add_box(ax, 0.77, 0.44, 0.18, 0.32, "Output", ["Unified emotion profile matrix"])

    add_arrow(ax, 0.72, 0.60, 0.77, 0.60)

    ax.text(
        0.05,
        0.12,
        "Output file name: Unified_Emotion_Profile_Perfume.csv.",
        transform=ax.transAxes,
        fontsize=12,
        va="top",
    )
    return fig


def render_slide_11() -> plt.Figure:
    fig, ax = make_slide_figure()
    add_title(ax, "From Emotion Vector to Generative Visualization")

    add_box(
        ax,
        0.05,
        0.62,
        0.30,
        0.24,
        "Input",
        ["Unified emotion profile per perfume", "Input file: Unified_Emotion_Profile_Perfume.csv"],
    )

    add_box(
        ax,
        0.40,
        0.62,
        0.28,
        0.24,
        "Mapping rules",
        [
            "Valence controls hue.",
            "Arousal controls saturation.",
            "Dominance controls lightness.",
        ],
    )

    add_arrow(ax, 0.35, 0.74, 0.40, 0.74)
    add_arrow(ax, 0.68, 0.74, 0.75, 0.74)

    # Color swatches
    cd = CITRUS_VALENCE_AROUSAL_DOMINANCE
    sn = SMOKY_VALENCE_AROUSAL_DOMINANCE

    cd_rgb = color_from_valence_arousal_dominance(cd["Valence"], cd["Arousal"], cd["Dominance"])
    sn_rgb = color_from_valence_arousal_dominance(sn["Valence"], sn["Arousal"], sn["Dominance"])

    # Draw swatches
    ax.add_patch(Rectangle((0.75, 0.66), 0.20, 0.12, transform=ax.transAxes, facecolor=np.array(cd_rgb) / 255.0, edgecolor="black", linewidth=2))
    ax.text(0.75, 0.80, "Citrus Dawn", transform=ax.transAxes, fontsize=14, fontweight="bold", va="bottom")
    ax.text(0.75, 0.64, "High valence, moderate arousal, moderate dominance.", transform=ax.transAxes, fontsize=12, va="top")

    ax.add_patch(Rectangle((0.75, 0.46), 0.20, 0.12, transform=ax.transAxes, facecolor=np.array(sn_rgb) / 255.0, edgecolor="black", linewidth=2))
    ax.text(0.75, 0.60, "Smoky Noir", transform=ax.transAxes, fontsize=14, fontweight="bold", va="bottom")
    ax.text(0.75, 0.44, "Low valence, moderate arousal, moderate dominance.", transform=ax.transAxes, fontsize=12, va="top")

    add_box(
        ax,
        0.05,
        0.20,
        0.63,
        0.30,
        "Legend (plain language)",
        [
            "Hue indicates pleasantness direction.",
            "Saturation indicates activation intensity.",
            "Lightness indicates sense of control.",
        ],
    )
    return fig


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    renderers = {
        1: render_slide_1,
        2: render_slide_2,
        3: render_slide_3,
        4: render_slide_4,
        5: render_slide_5,
        6: render_slide_6,
        7: render_slide_7,
        8: render_slide_8,
        9: render_slide_9,
        10: render_slide_10,
        11: render_slide_11,
    }

    for slide_number in range(1, 12):
        fig = renderers[slide_number]()
        out_path = OUTPUT_DIR / f"workedExample_slide{slide_number}.png"
        print(f"Saving {out_path}")
        save_figure(fig, out_path)

    print(f"Done. Images saved under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
