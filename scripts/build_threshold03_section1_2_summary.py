#!/usr/bin/env python3
"""Build a compact SVG with only Report 7 Sections 1 and 2.0."""

from __future__ import annotations

import build_threshold03_combination_report as report


OUT = (
    report.ROOT
    / "docs/synthetic_review_experiment_report/table_images/08_threshold03_section1_2_summary.svg"
)


def main() -> None:
    review_count_rows = report.load_review_count_compact_rows()
    label_distribution_balance_tables = report.group_label_distribution_balancing_by_method()
    section20_rows = report.build_section20_method_snapshot(label_distribution_balance_tables)

    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{report.PAGE_W}" height="2400" '
            f'viewBox="0 0 {report.PAGE_W} 2400">'
        ),
        '<rect width="100%" height="100%" fill="#f4f7fb"/>',
        report.text_element(
            x=report.MARGIN,
            y=46,
            text="Review Count and Balancing Summary",
            size=23,
            weight="700",
            color=report.TEXT,
        ),
        report.text_element(
            x=report.MARGIN,
            y=74,
            text="Threshold 0.3",
            size=12,
            weight="700",
            color=report.MUTED,
        ),
    ]
    y = 100

    count_headers = [
        "Corpus",
        "Rows",
        "Perfumes",
        "LR F1",
        "XGBoost F1",
    ]
    count_widths = [170, 75, 80, 150, 150]
    table_parts, y = report.render_table(
        title="1. Review Count Effect: No Balancing Method",
        headers=count_headers,
        rows=review_count_rows,
        widths=count_widths,
        y_start=y,
        accent=report.BLUE,
        center_cols={1, 2, 3, 4},
        separator_after_cols={3},
        merge_cols={0},
    )
    parts.extend(table_parts)
    note_parts, y = report.render_method_note(
        "No balancing method; mixed rows are evaluated on YouTube rows.",
        y,
        report.MUTED,
    )
    parts.extend(note_parts)
    y += report.SECTION_GAP

    heading_parts, y = report.render_section_heading(
        "Emotion-count balancing",
        "Main comparison: no balancing, hybrid, and SMOTE.",
        y,
        report.AMBER,
    )
    parts.extend(heading_parts)

    table_parts, y = report.render_section20_table(
        title="2.0 Summary: No Balancing vs Hybrid vs SMOTE",
        rows=section20_rows,
        widths=[170, 112, 112, 112, 135, 135, 135],
        y_start=y,
        accent=report.AMBER,
    )
    parts.extend(table_parts)
    note_parts, y = report.render_method_note("Best F1 in each row/model is shaded.", y, report.MUTED)
    parts.extend(note_parts)
    y += 36

    parts[0] = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{report.PAGE_W}" height="{y}" '
        f'viewBox="0 0 {report.PAGE_W} {y}">'
    )
    parts.append("</svg>\n")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
