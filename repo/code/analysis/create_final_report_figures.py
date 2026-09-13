from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT = Path(__file__).resolve().parents[2]
ANALYSIS = PROJECT / "results" / "final_subject_level"
FIGURES = PROJECT / "results" / "figures"

COLORS = {
    "blue": (44, 96, 154),
    "orange": (213, 120, 48),
    "green": (55, 133, 103),
    "red": (181, 74, 72),
    "purple": (122, 91, 153),
    "ink": (28, 32, 38),
    "muted": (87, 95, 106),
    "grid": (216, 221, 228),
    "panel": (247, 249, 251),
    "white": (255, 255, 255),
}


def font(size: int, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


TITLE = font(36, True)
SUBTITLE = font(22)
HEADING = font(22, True)
LABEL = font(22)
SMALL = font(21)
TINY = font(20)


def read_csv(name: str) -> list[dict[str, str]]:
    with (ANALYSIS / name).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def canvas(width: int = 1600, height: int = 940):
    image = Image.new("RGB", (width, height), COLORS["white"])
    return image, ImageDraw.Draw(image)


def text(draw, xy, value, *, fill=None, font_obj=None, anchor=None):
    draw.text(xy, value, fill=fill or COLORS["ink"], font=font_obj or LABEL, anchor=anchor)


def title(draw, heading: str, subtitle: str):
    text(draw, (70, 34), heading, font_obj=TITLE)
    text(draw, (70, 86), subtitle, fill=COLORS["muted"], font_obj=SUBTITLE)


def save(image: Image.Image, name: str):
    FIGURES.mkdir(parents=True, exist_ok=True)
    image.save(FIGURES / name, dpi=(300, 300))


def y_value(value: float, top: int, bottom: int, lower: float, upper: float) -> int:
    return bottom - round((value - lower) / (upper - lower) * (bottom - top))


def axes(draw, box, lower: float, upper: float, label: str, chance: float | None = None):
    left, top, right, bottom = box
    draw.rectangle(box, outline=COLORS["grid"], width=2)
    for index in range(6):
        value = lower + (upper - lower) * index / 5
        y = y_value(value, top, bottom, lower, upper)
        draw.line((left, y, right, y), fill=COLORS["grid"], width=1)
        text(draw, (left - 12, y), f"{value:.3f}", fill=COLORS["muted"], font_obj=TINY, anchor="rm")
    text(draw, (left, top - 30), label, fill=COLORS["muted"], font_obj=SMALL)
    if chance is not None:
        y = y_value(chance, top, bottom, lower, upper)
        draw.line((left, y, right, y), fill=COLORS["red"], width=2)
        text(draw, (right - 8, y - 12), f"chance {chance:.1f}", fill=COLORS["red"], font_obj=TINY, anchor="rs")


def legend(draw, items, x: int, y: int):
    cursor = x
    for label, color in items:
        draw.rectangle((cursor, y + 3, cursor + 24, y + 18), fill=color)
        text(draw, (cursor + 32, y), label, font_obj=SMALL)
        cursor += 55 + draw.textlength(label, font=SMALL)


def error_bar(draw, x: int, low: float, high: float, box, lower: float, upper: float):
    left, top, right, bottom = box
    y_low = y_value(low, top, bottom, lower, upper)
    y_high = y_value(high, top, bottom, lower, upper)
    draw.line((x, y_low, x, y_high), fill=COLORS["ink"], width=2)
    draw.line((x - 7, y_low, x + 7, y_low), fill=COLORS["ink"], width=2)
    draw.line((x - 7, y_high, x + 7, y_high), fill=COLORS["ink"], width=2)


def fig1_pipeline():
    image, draw = canvas(1600, 840)
    title(draw, "Study design", "Transfer to unseen subjects and robustness to channel loss are the two primary analyses.")
    stages = [
        ("Dataset", "ZuCo 2.0 Task 1\n18 subjects\nNatural reading"),
        ("Word-level EEG", "All fixations\n512 x 105\nFirst fixation check"),
        ("Targets", "Content/function\nCoarse POS"),
        ("Evaluation", "LOSO primary\nGrouped validation\nSentence separation\ncheck"),
        ("Models", "Linear EEG\nEEGNet / TCN\nNon-EEG controls"),
        ("Channel loss", "Persistent random\nmasks\nSpatial blocks"),
        ("Inference", "Subject-level CIs\nPaired sign-flip\ntests"),
    ]
    box_w, box_h, gap = 190, 170, 24
    start_x, y = 58, 205
    for index, (heading, body) in enumerate(stages):
        x = start_x + index * (box_w + gap)
        draw.rectangle((x, y, x + box_w, y + box_h), fill=COLORS["panel"], outline=COLORS["grid"], width=2)
        text(draw, (x + 16, y + 18), heading, font_obj=HEADING)
        for line_no, line in enumerate(body.splitlines()):
            text(draw, (x + 16, y + 62 + line_no * 28), line, fill=COLORS["muted"], font_obj=font(20))
        if index < len(stages) - 1:
            mid = y + box_h // 2
            draw.line((x + box_w + 4, mid, x + box_w + gap - 4, mid), fill=COLORS["blue"], width=3)
            draw.polygon([(x + box_w + gap - 4, mid), (x + box_w + gap - 15, mid - 7), (x + box_w + gap - 15, mid + 7)], fill=COLORS["blue"])
    text(
        draw,
        (70, 500),
        "Primary questions: transfer to unseen subjects and performance under missing channels.",
        font_obj=font(23, True),
    )
    text(
        draw,
        (70, 548),
        "Supporting analyses: comparisons of data splits, representations and models, and non-EEG controls.",
        fill=COLORS["muted"],
        font_obj=SUBTITLE,
    )
    save(image, "fig1_experimental_pipeline.png")


def fig3_subject_shift(main_rows: list[dict[str, str]]):
    image, draw = canvas()
    title(draw, "EEGNet across subject separation settings", "All fixations, no channel loss; means across five seeds.")
    panels = [
        ("content_function", "Content/function", 0.57, 0.62, 0.5, (115, 175, 745, 730), COLORS["blue"]),
        ("coarse_pos", "Coarse POS", 0.225, 0.247, 0.2, (875, 175, 1505, 730), COLORS["green"]),
    ]
    splits = [("loso", "LOSO"), ("subject_dependent", "Subject-dependent"), ("mixed_subject", "Mixed-subject")]
    for target_name, panel_name, lower, upper, chance, box, color in panels:
        axes(draw, box, lower, upper, panel_name, chance)
        left, top, right, bottom = box
        width = (right - left) / len(splits)
        for index, (split_name, split_label) in enumerate(splits):
            row = next(row for row in main_rows if row["target"] == target_name and row["split"] == split_name and row["model"] == "EEGNet")
            value = float(row["balanced_accuracy"])
            cx = round(left + width * (index + 0.5))
            y = y_value(value, top, bottom, lower, upper)
            if row.get("ci95_low") and row.get("ci95_high"):
                error_bar(draw, cx, float(row["ci95_low"]), float(row["ci95_high"]), box, lower, upper)
                label_y = y_value(float(row["ci95_high"]), top, bottom, lower, upper) - 20
            else:
                label_y = y - 20
            draw.ellipse((cx - 9, y - 9, cx + 9, y + 9), fill=color, outline=COLORS["white"], width=2)
            text(draw, (cx, label_y), f"{value:.3f}", font_obj=SMALL, anchor="mm")
            text(draw, (cx, bottom + 24), split_label, fill=COLORS["muted"], font_obj=TINY, anchor="mm")
    text(draw, (70, 830), "Error bars show 95% subject-bootstrap intervals where available; mixed-subject values are split-repeat means.", fill=COLORS["muted"], font_obj=SMALL)
    save(image, "fig3_subject_shift_eegnet.png")


def fig4_fixed_session_dropout(rows: list[dict[str, str]]):
    image, draw = canvas()
    title(draw, "Persistent random channel loss", "LOSO with grouped inner validation; each sampled mask is fixed across one participant's test set.")
    panels = [
        ("content_function", "Content/function", 0.558, 0.61, 0.5, (115, 175, 745, 730), COLORS["blue"]),
        ("coarse_pos", "Coarse POS", 0.22, 0.245, 0.2, (875, 175, 1505, 730), COLORS["green"]),
    ]
    for target_name, panel_name, lower, upper, chance, box, color in panels:
        axes(draw, box, lower, upper, panel_name, chance)
        left, top, right, bottom = box
        target_rows = sorted((row for row in rows if row["target"] == target_name), key=lambda row: float(row["loss_rate"]))
        points = []
        for index, row in enumerate(target_rows):
            x = round(left + (right - left) * index / (len(target_rows) - 1))
            value = float(row["balanced_accuracy_mean"])
            y = y_value(value, top, bottom, lower, upper)
            error_bar(draw, x, float(row["balanced_accuracy_ci95_low"]), float(row["balanced_accuracy_ci95_high"]), box, lower, upper)
            points.append((x, y))
            text(draw, (x, bottom + 24), row["n_removed_channels"], fill=COLORS["muted"], font_obj=TINY, anchor="mm")
        for first, second in zip(points, points[1:]):
            draw.line((*first, *second), fill=color, width=4)
        for x, y in points:
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color, outline=COLORS["white"], width=2)
        text(draw, ((left + right) // 2, bottom + 58), "Channels removed (of 105)", fill=COLORS["muted"], font_obj=SMALL, anchor="mm")
    text(draw, (180, 830), "Error bars are 95% subject-bootstrap intervals.", fill=COLORS["muted"], font_obj=SMALL)
    save(image, "fig4_channel_dropout_eegnet.png")


def fig5_per_class(rows: list[dict[str, str]]):
    image, draw = canvas()
    title(draw, "F1 score by class under LOSO", "Mean across five seeds within each held-out subject; 95% subject-bootstrap intervals.")
    panels = [
        ("content_function", "Content/function", 0.0, 0.75, (115, 200, 745, 730), COLORS["blue"]),
        ("coarse_pos", "Coarse POS", 0.0, 0.45, (875, 200, 1505, 730), COLORS["green"]),
    ]
    for target_name, panel_name, lower, upper, box, color in panels:
        axes(draw, box, lower, upper, panel_name)
        left, top, right, bottom = box
        target_rows = [row for row in rows if row["target"] == target_name]
        width = (right - left) / len(target_rows)
        for index, row in enumerate(target_rows):
            cx = round(left + width * (index + 0.5))
            value = float(row["f1_mean"])
            y = y_value(value, top, bottom, lower, upper)
            draw.rectangle((cx - 34, y, cx + 34, bottom), fill=color)
            error_bar(draw, cx, float(row["f1_ci95_low"]), float(row["f1_ci95_high"]), box, lower, upper)
            ci_high_y = y_value(float(row["f1_ci95_high"]), top, bottom, lower, upper)
            text(draw, (cx, ci_high_y - 20), f"{value:.3f}", font_obj=TINY, anchor="mm")
            text(draw, (cx, bottom + 24), row["label"], fill=COLORS["muted"], font_obj=TINY, anchor="mm")
    save(image, "fig5_loso_per_class_f1.png")


def fig6_representation(condition_rows: list[dict[str, str]], paired_rows: list[dict[str, str]]):
    image, draw = canvas()
    title(draw, "First fixation versus all fixations", "Matched 512 x 105 inputs and EEGNet capacity; LOSO with grouped inner validation.")
    panels = [
        ("content_function", "Content/function", 0.50, 0.61, 0.5, (115, 175, 745, 730), COLORS["blue"]),
        ("coarse_pos", "Coarse POS", 0.20, 0.245, 0.2, (875, 175, 1505, 730), COLORS["green"]),
    ]
    for target_name, panel_name, lower, upper, chance, box, color in panels:
        axes(draw, box, lower, upper, panel_name, chance)
        left, top, right, bottom = box
        paired = next(row for row in paired_rows if row["target"] == target_name and row["comparison"] == "all_fixations_minus_first_fixation")
        first = next(row for row in condition_rows if row["target"] == target_name and row["representation"] == "first_fixation")
        all_fixations = next(row for row in condition_rows if row["target"] == target_name and row["representation"] == "all_fixations")
        plotted = []
        for cx, label, row, shade in [
            (left + (right - left) // 3, "First fixation", first, COLORS["orange"]),
            (left + 2 * (right - left) // 3, "All fixations", all_fixations, color),
        ]:
            value = float(row["balanced_accuracy_mean"])
            y = y_value(value, top, bottom, lower, upper)
            plotted.append((cx, y, shade))
            error_bar(
                draw,
                cx,
                float(row["balanced_accuracy_ci95_low"]),
                float(row["balanced_accuracy_ci95_high"]),
                box,
                lower,
                upper,
            )
            ci_high_y = y_value(float(row["balanced_accuracy_ci95_high"]), top, bottom, lower, upper)
            text(draw, (cx, ci_high_y - 20), f"{value:.3f}", font_obj=SMALL, anchor="mm")
            text(draw, (cx, bottom + 24), label, fill=COLORS["muted"], font_obj=TINY, anchor="mm")
        draw.line((plotted[0][0], plotted[0][1], plotted[1][0], plotted[1][1]), fill=COLORS["muted"], width=3)
        for cx, y, shade in plotted:
            draw.ellipse((cx - 10, y - 10, cx + 10, y + 10), fill=shade, outline=COLORS["white"], width=2)
        text(
            draw,
            ((left + right) // 2, bottom + 86),
            f"paired difference = +{float(paired['balanced_accuracy_delta_mean']):.3f}; 95% CI "
            f"[{float(paired['balanced_accuracy_delta_ci95_low']):.3f}, "
            f"{float(paired['balanced_accuracy_delta_ci95_high']):.3f}]",
            font_obj=SMALL,
            anchor="mm",
        )
    save(image, "fig6_first_vs_all_fixations_loso.png")


def fig7_structured(rows: list[dict[str, str]]):
    image, draw = canvas(1700, 1040)
    title(draw, "Structured and matched random channel loss", "Change from no-loss balanced accuracy; mean across five seeds within each held-out subject.")
    labels = [
        ("official_frontal_left_9pairs", "Frontal\nleft\n9 channels"),
        ("official_frontal_right_9pairs", "Frontal\nright\n9 channels"),
        ("official_frontal_bilateral_9pairs", "Frontal\nbilateral\n18 channels"),
        ("official_left_homologue_set", "Left\nhomologues\n48 channels"),
        ("official_right_homologue_set", "Right\nhomologues\n48 channels"),
    ]
    panels = [
        ("content_function", "Content/function", -0.012, 0.004, (125, 190, 790, 790)),
        ("coarse_pos", "Coarse POS", -0.012, 0.004, (930, 190, 1595, 790)),
    ]
    for target_name, panel_name, lower, upper, box in panels:
        axes(draw, box, lower, upper, panel_name)
        left, top, right, bottom = box
        zero_y = y_value(0.0, top, bottom, lower, upper)
        draw.line((left, zero_y, right, zero_y), fill=COLORS["ink"], width=2)
        width = (right - left) / len(labels)
        for index, (condition, label) in enumerate(labels):
            structured = next(row for row in rows if row["target"] == target_name and row["condition"] == condition)
            matched = next(row for row in rows if row["target"] == target_name and row["condition"] == structured["matched_random"])
            cx = left + width * (index + 0.5)
            for offset, row, color in [
                (-19, structured, COLORS["purple"]),
                (19, matched, COLORS["orange"]),
            ]:
                value = float(row["delta_vs_none"])
                x1, x2 = round(cx + offset - 15), round(cx + offset + 15)
                y = y_value(value, top, bottom, lower, upper)
                draw.rectangle((x1, min(y, zero_y), x2, max(y, zero_y)), fill=color)
                error_bar(
                    draw,
                    round(cx + offset),
                    float(row["delta_vs_none_ci95_low"]),
                    float(row["delta_vs_none_ci95_high"]),
                    box,
                    lower,
                    upper,
                )
            for line_index, line in enumerate(label.splitlines()):
                text(draw, (round(cx), bottom + 20 + 18 * line_index), line, fill=COLORS["muted"], font_obj=TINY, anchor="mm")
    legend(draw, [("Structured block", COLORS["purple"]), ("Matched random", COLORS["orange"])], 620, 900)
    text(draw, (160, 955), "Error bars are 95% subject-bootstrap intervals; all ten matched contrasts have Holm-adjusted p >= 0.246.", fill=COLORS["muted"], font_obj=SMALL)
    save(image, "fig7_structured_channel_dropout.png")


def main() -> int:
    main_rows = read_csv("main_no_dropout.csv")
    random_rows = read_csv("workflow15_fixed_session_channel_loss_summary.csv")
    matched_condition_rows = read_csv("workflow16_matched_fixation_conditions.csv")
    matched_paired_rows = read_csv("workflow16_matched_fixation_paired_comparisons.csv")
    per_class_rows = read_csv("per_class.csv")
    structured_rows = read_csv("structured_dropout.csv")
    fig1_pipeline()
    fig3_subject_shift(main_rows)
    fig4_fixed_session_dropout(random_rows)
    fig5_per_class(per_class_rows)
    fig6_representation(matched_condition_rows, matched_paired_rows)
    fig7_structured(structured_rows)
    for path in sorted(FIGURES.glob("fig*.png")):
        print(path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
