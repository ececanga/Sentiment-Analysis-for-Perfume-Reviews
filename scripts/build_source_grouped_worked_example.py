"""Rebuild the Chapter III worked example from accepted source-grouped artifacts."""

from __future__ import annotations

import csv
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd

from src.labeling.strict_schema import normalize_perfume_id
from src.training.source_grouped_folds import OFFICIAL_LABELS
from src.utils.paths import PROJECT_ROOT


PERFUME_ID = "vetiver-nocturne-absolu-extrait-de-parfum"
PERFUME_NAME = "vetiver nocturne absolu extrait de parfum"
THRESHOLD = 0.3

RAW_CSV = PROJECT_ROOT / "data/raw/YouTube_Perfume_Reviews_FullyFormatted.csv"
FEATURE_CSV = PROJECT_ROOT / "data/perfume_level/Unified_Emotion_Profile_Perfume.csv"
LABEL_CSV = PROJECT_ROOT / "labels/gemini_browser_og784/gemini_browser_labels_og784_v1.csv"
OOF_CSV = (
    PROJECT_ROOT
    / "outputs/gemini_browser_og784_source_grouped_v1/model_benchmark/predictions/logistic_regression_oof.csv"
)
SUMMARY_CSV = PROJECT_ROOT / "docs/thesis/worked_example_vetiver_nocturne_summary.csv"
TRACE_PNG = PROJECT_ROOT / "docs/thesis/mermaid diagrams/figure3.8_artifact_trace.png"
TRACE_READABLE_PNG = (
    PROJECT_ROOT / "docs/thesis/mermaid diagrams/figure3.8_worked_example_artifact_trace_readable.png"
)
PROBABILITY_PNG = PROJECT_ROOT / "docs/thesis/mermaid diagrams/figure3.9_probability_profile.png"


EXCERPT_ROWS = [
    ("Worked excerpt", "slightly sweet medicinal herbs"),
    ("Worked excerpt tokens", "slightly, sweet, medicinal, herbs"),
    ("Excerpt NRC match", "sweet -> anticipation, joy, positive, surprise, trust"),
    ("Excerpt VAD slightly", "valence -0.060, arousal -0.452, dominance -0.490"),
    ("Excerpt VAD sweet", "valence 0.336, arousal -0.164, dominance -0.236"),
    ("Excerpt VAD medicinal", "valence 0.334, arousal -0.166, dominance 0.228"),
    ("Excerpt VAD herbs", "valence 0.240, arousal -0.402, dominance -0.384"),
    ("Excerpt VAD mean", "valence 0.213, arousal -0.296, dominance -0.221"),
]


def _one_row(frame: pd.DataFrame, mask: pd.Series, description: str) -> pd.Series:
    selected = frame.loc[mask]
    if len(selected) != 1:
        raise ValueError(f"Expected one {description} row, found {len(selected)}")
    return selected.iloc[0]


def load_example() -> dict[str, Any]:
    raw = pd.read_csv(RAW_CSV)
    raw_ids = raw["Perfume"].map(normalize_perfume_id)
    raw_rows = int((raw_ids == PERFUME_ID).sum())
    if raw_rows != 10:
        raise ValueError(f"Expected 10 raw transcript rows, found {raw_rows}")

    features = pd.read_csv(FEATURE_CSV)
    feature = _one_row(
        features,
        features["Perfume"].map(normalize_perfume_id) == PERFUME_ID,
        "feature",
    )
    labels = pd.read_csv(LABEL_CSV)
    label = _one_row(labels, labels["perfume_id"] == PERFUME_ID, "label")
    oof = pd.read_csv(OOF_CSV)
    prediction = _one_row(oof, oof["perfume_id"] == PERFUME_ID, "OOF prediction")

    for emotion in OFFICIAL_LABELS:
        if int(label[emotion]) != int(prediction[f"true_{emotion}"]):
            raise ValueError(f"Label/OOF truth mismatch for {emotion}")
        expected_prediction = int(float(prediction[f"prob_{emotion}"]) >= THRESHOLD)
        if expected_prediction != int(prediction[f"pred_{emotion}"]):
            raise ValueError(f"Stored threshold prediction mismatch for {emotion}")

    return {
        "raw_rows": raw_rows,
        "feature": feature,
        "label": label,
        "prediction": prediction,
    }


def _relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def write_summary(example: dict[str, Any]) -> None:
    feature = example["feature"]
    label = example["label"]
    prediction = example["prediction"]
    positives = [emotion for emotion in OFFICIAL_LABELS if int(prediction[f"pred_{emotion}"]) == 1]
    rows = [
        ("Perfume", PERFUME_NAME),
        ("Perfume ID", PERFUME_ID),
        ("Raw transcript artifact", _relative(RAW_CSV)),
        ("Feature artifact", _relative(FEATURE_CSV)),
        ("Reference-label artifact", _relative(LABEL_CSV)),
        ("Accepted OOF artifact", _relative(OOF_CSV)),
        ("Raw transcript rows", example["raw_rows"]),
        ("Cross-validation fold", int(prediction["fold_id"])),
        ("Source component ID", prediction["source_group_id"]),
        ("Source component perfume count", int(prediction["source_group_size"])),
        *EXCERPT_ROWS,
        (
            "Gemini positive labels",
            ", ".join(emotion for emotion in OFFICIAL_LABELS if int(label[emotion]) == 1),
        ),
        (
            "Gemini VAD labels",
            f"valence {int(label['valence'])}, arousal {int(label['arousal'])}, "
            f"dominance {int(label['dominance'])}",
        ),
        (
            f"Logistic Regression positive predictions at threshold {THRESHOLD}",
            ", ".join(positives),
        ),
        ("NRC_positive", int(feature["NRC_positive"])),
        ("NRC_trust", int(feature["NRC_trust"])),
        ("NRC_joy", int(feature["NRC_joy"])),
        ("NRC_anticipation", int(feature["NRC_anticipation"])),
        ("NRC_negative", int(feature["NRC_negative"])),
        ("VAD_valence", f"{float(feature['VAD_valence']):.3f}"),
        ("VAD_arousal", f"{float(feature['VAD_arousal']):.3f}"),
        ("VAD_dominance", f"{float(feature['VAD_dominance']):.3f}"),
        ("JHartmann_Label", feature["JHartmann_Label"]),
        ("Savani_Label", feature["Savani_Label"]),
        ("GoEmotions_Label", feature["GoEmotions_Label"]),
        ("Cardiff_Label", feature["Cardiff_Label"]),
        *[
            (f"LR probability {emotion}", f"{float(prediction[f'prob_{emotion}']):.6f}")
            for emotion in OFFICIAL_LABELS
        ],
    ]
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["Item", "Value"])
        writer.writerows(rows)


def _add_card(
    axis,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    number: str,
    title: str,
    lines: list[str],
    accent: str,
) -> None:
    card = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1,
        edgecolor="#CFD8E6",
        facecolor="#F7F9FC",
    )
    axis.add_patch(card)
    axis.add_patch(plt.Rectangle((x, y), 0.012, height, color=accent, linewidth=0))
    axis.text(
        x + 0.038,
        y + height - 0.038,
        number,
        ha="center",
        va="center",
        fontsize=13,
        weight="bold",
        color="white",
        bbox={"boxstyle": "circle,pad=0.45", "facecolor": accent, "edgecolor": accent},
    )
    axis.text(x + 0.073, y + height - 0.032, title, fontsize=14, weight="bold", va="center")
    axis.text(
        x + 0.073,
        y + height - 0.072,
        "\n".join(lines),
        fontsize=10.2,
        color="#5B6B83",
        va="top",
        linespacing=1.35,
    )


def build_trace_figure(example: dict[str, Any]) -> None:
    feature = example["feature"]
    label = example["label"]
    prediction = example["prediction"]
    probability_lines = [
        "Threshold: 0.3",
        "Positive predictions: joy, trust, anticipation",
        f"joy {prediction['prob_joy']:.3f}; trust {prediction['prob_trust']:.3f}",
        f"anticipation {prediction['prob_anticipation']:.3f}; surprise {prediction['prob_surprise']:.3f}",
        f"sadness {prediction['prob_sadness']:.3f}; fear {prediction['prob_fear']:.3f}; disgust {prediction['prob_disgust']:.3f}",
    ]
    cards = [
        (
            "1",
            "Benchmark row",
            [
                f"Perfume: {PERFUME_NAME}",
                f"{example['raw_rows']} usable YouTube transcript rows",
                f"Exact-source outer fold {int(prediction['fold_id'])}",
                f"Source component: {int(prediction['source_group_size'])} perfumes",
            ],
            "#2563EB",
        ),
        (
            "2",
            "Scent-focused excerpt",
            [
                'Excerpt: "slightly sweet medicinal herbs"',
                "Tokens: slightly, sweet, medicinal, herbs",
                "NRC: sweet -> anticipation, joy, positive, surprise, trust",
            ],
            "#059669",
        ),
        (
            "3",
            "Excerpt-level VAD",
            [
                "Values: valence, arousal, dominance",
                "slightly (-0.060, -0.452, -0.490)",
                "sweet (0.336, -0.164, -0.236)",
                "medicinal (0.334, -0.166, 0.228)",
                "herbs (0.240, -0.402, -0.384)",
                "Excerpt mean: 0.213, -0.296, -0.221",
            ],
            "#D17A00",
        ),
        (
            "4",
            "Full engineered feature row",
            [
                f"NRC: positive {int(feature['NRC_positive'])}; trust {int(feature['NRC_trust'])}; joy {int(feature['NRC_joy'])}",
                f"NRC: anticipation {int(feature['NRC_anticipation'])}; negative {int(feature['NRC_negative'])}",
                f"Full-row VAD: {feature['VAD_valence']:.3f}, {feature['VAD_arousal']:.3f}, {feature['VAD_dominance']:.3f}",
                f"Transformer labels: Hartmann {feature['JHartmann_Label']}; Savani {feature['Savani_Label']}",
                f"GoEmotions {feature['GoEmotions_Label']}; Cardiff {feature['Cardiff_Label']}",
            ],
            "#7C3AED",
        ),
        (
            "5",
            "Gemini browser labels",
            [
                "Positive labels: joy, trust, anticipation",
                "Negative labels: sadness, fear, surprise, disgust",
                f"Anger recorded as {int(label['anger'])}",
                f"Gemini VAD: valence {int(label['valence'])}; arousal {int(label['arousal'])}; dominance {int(label['dominance'])}",
            ],
            "#009688",
        ),
        ("6", "Source-grouped Logistic Regression output", probability_lines, "#334155"),
    ]

    figure, axis = plt.subplots(figsize=(12, 9.8), dpi=150)
    figure.patch.set_facecolor("white")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.text(0.04, 0.955, "Worked Example Artifact Trace", fontsize=27, weight="bold", color="#111827")
    axis.text(
        0.04,
        0.922,
        "One repaired source-grouped OOF row traced from scent text to features, labels, and output",
        fontsize=13.5,
        color="#5B6B83",
    )
    axis.plot([0.04, 0.96], [0.902, 0.902], color="#DCE3ED", linewidth=1.5)

    positions = [(0.04, 0.68), (0.52, 0.68), (0.04, 0.43), (0.52, 0.43), (0.04, 0.18), (0.52, 0.18)]
    for (number, title, lines, accent), (x, y) in zip(cards, positions, strict=True):
        _add_card(
            axis,
            x=x,
            y=y,
            width=0.44,
            height=0.205,
            number=number,
            title=title,
            lines=lines,
            accent=accent,
        )
    axis.text(
        0.04,
        0.045,
        f"Accepted OOF source: {_relative(OOF_CSV)}",
        fontsize=9.5,
        color="#5B6B83",
    )
    axis.text(
        0.04,
        0.022,
        f"Detailed numeric trace: {_relative(SUMMARY_CSV)}",
        fontsize=9.5,
        color="#5B6B83",
    )
    figure.savefig(TRACE_READABLE_PNG, dpi=150, bbox_inches="tight", facecolor="white")
    figure.savefig(TRACE_PNG, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def build_probability_figure(example: dict[str, Any]) -> None:
    prediction = example["prediction"]
    ordered_labels = sorted(
        OFFICIAL_LABELS,
        key=lambda emotion: float(prediction[f"prob_{emotion}"]),
        reverse=True,
    )
    probabilities = [float(prediction[f"prob_{emotion}"]) for emotion in ordered_labels]
    colors = ["#07877F" if value >= THRESHOLD else "#9AAAC0" for value in probabilities]

    figure, axis = plt.subplots(figsize=(12, 8.4), dpi=150)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    y_positions = list(range(len(ordered_labels)))
    bars = axis.barh(y_positions, probabilities, color=colors, height=0.56)
    axis.invert_yaxis()
    axis.axvline(THRESHOLD, color="#F05252", linestyle=(0, (3, 3)), linewidth=1.8)
    axis.text(THRESHOLD + 0.008, -0.48, "threshold 0.3", color="#E53935", fontsize=11.5)
    axis.set_xlim(0, 0.75)
    axis.set_yticks(y_positions, [emotion.title() for emotion in ordered_labels])
    axis.tick_params(axis="y", labelsize=14, length=0, pad=18)
    axis.tick_params(axis="x", labelsize=10, colors="#5B6B83")
    axis.set_xlabel("Logistic Regression probability", fontsize=12, color="#5B6B83", labelpad=12)
    axis.xaxis.grid(True, color="#DCE3ED", linewidth=1)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_visible(False)
    for bar, value in zip(bars, probabilities, strict=True):
        axis.text(
            max(value + 0.010, 0.018),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=12,
            color="#111827",
        )

    figure.suptitle(
        "Worked Example Probability Profile",
        x=0.08,
        y=0.98,
        ha="left",
        fontsize=27,
        weight="bold",
        color="#111827",
    )
    figure.text(
        0.08,
        0.925,
        "Accepted exact-source-grouped OOF Logistic Regression probabilities",
        fontsize=13.5,
        color="#5B6B83",
    )
    figure.text(
        0.5,
        0.035,
        "Positive predictions at threshold 0.3: joy, trust, anticipation",
        ha="center",
        fontsize=15,
        weight="bold",
        color="#111827",
    )
    figure.text(
        0.5,
        0.010,
        f"Example row: {PERFUME_NAME} · outer fold {int(prediction['fold_id'])} · source component size {int(prediction['source_group_size'])}",
        ha="center",
        fontsize=10.5,
        color="#5B6B83",
    )
    figure.subplots_adjust(left=0.22, right=0.94, top=0.84, bottom=0.14)
    figure.savefig(PROBABILITY_PNG, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    example = load_example()
    write_summary(example)
    build_trace_figure(example)
    build_probability_figure(example)
    prediction = example["prediction"]
    print("Rebuilt source-grouped Chapter III worked example")
    print(f"- perfume_id={PERFUME_ID}")
    print(f"- fold_id={int(prediction['fold_id'])}")
    print(f"- source_group_size={int(prediction['source_group_size'])}")
    print(f"- OOF source={_relative(OOF_CSV)}")


if __name__ == "__main__":
    main()
