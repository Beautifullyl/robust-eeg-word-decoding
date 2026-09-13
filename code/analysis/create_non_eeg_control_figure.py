from __future__ import annotations

import csv
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from create_final_report_figures import (
    COLORS as REPORT_COLORS,
    SMALL as REPORT_SMALL,
    TINY as REPORT_TINY,
    axes as report_axes,
    canvas as report_canvas,
    error_bar as report_error_bar,
    text as report_text,
    title as report_title,
)


PROJECT = Path(__file__).resolve().parents[2]
MAIN_RESULTS = PROJECT / "results" / "final_subject_level" / "main_no_dropout.csv"
CONTROLS = PROJECT / "results" / "final_subject_level" / "non_eeg_controls.csv"
OUTPUT = PROJECT / "results" / "figures" / "fig8_eeg_vs_non_eeg_controls.png"

TARGETS = (
    ("content_function", "Content/function", 0.5, 0.9),
    ("coarse_pos", "Coarse POS", 0.2, 0.45),
)
METHODS = (
    ("EEGNet", None, "#36648B"),
    ("Duration /\nfixation", "duration_fixation", "#D49A3A"),
    ("Word length /\nposition", "lexical_position", "#4E8B57"),
    ("Combined\nnon-EEG", "combined_non_eeg", "#8C5A77"),
)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("C:/Windows/Fonts") / ("arialbd.ttf" if bold else "arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu") / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def draw_centered_multiline(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    text_font: ImageFont.FreeTypeFont,
    fill: str,
    spacing: int = 4,
) -> None:
    box = draw.multiline_textbbox((0, 0), text, font=text_font, spacing=spacing, align="center")
    width = box[2] - box[0]
    draw.multiline_text((xy[0] - width / 2, xy[1]), text, font=text_font, fill=fill, spacing=spacing, align="center")


def main() -> int:
    with MAIN_RESULTS.open("r", encoding="utf-8", newline="") as handle:
        main_rows = list(csv.DictReader(handle))
    with CONTROLS.open("r", encoding="utf-8", newline="") as handle:
        control_rows = list(csv.DictReader(handle))
    controls = {
        (row["target"], row["feature_set"]): {
            "mean": float(row["mean"]),
            "ci95_low": float(row["ci95_low"]),
            "ci95_high": float(row["ci95_high"]),
        }
        for row in control_rows
    }
    eegnet = {
        row["target"]: row
        for row in main_rows
        if row["split"] == "loso" and row["model"] == "EEGNet"
    }

    image, draw = report_canvas()
    report_title(draw, "EEGNet and non-EEG controls", "All fixations under LOSO; 95% subject-bootstrap intervals.")
    panels = (
        ("content_function", "Content/function", 0.0, 0.9, 0.5, (115, 175, 745, 730)),
        ("coarse_pos", "Coarse POS", 0.0, 0.45, 0.2, (875, 175, 1505, 730)),
    )
    method_colours = (
        REPORT_COLORS["blue"],
        REPORT_COLORS["orange"],
        REPORT_COLORS["green"],
        REPORT_COLORS["purple"],
    )

    for target, panel_title, lower, upper, chance, box in panels:
        report_axes(draw, box, lower, upper, panel_title)
        left, top, right, bottom = box
        chance_y = bottom - round((chance - lower) / (upper - lower) * (bottom - top))
        draw.line((left, chance_y, right, chance_y), fill=REPORT_COLORS["red"], width=2)
        group_width = (right - left) / len(METHODS)
        for index, ((label, feature_set, _), colour) in enumerate(zip(METHODS, method_colours)):
            row = eegnet[target] if feature_set is None else controls[(target, feature_set)]
            mean = float(row["mean"])
            low = float(row["ci95_low"])
            high = float(row["ci95_high"])
            centre = round(left + group_width * (index + 0.5))
            bar_top = bottom - round((mean - lower) / (upper - lower) * (bottom - top))
            draw.rectangle((centre - 34, bar_top, centre + 34, bottom), fill=colour)
            report_error_bar(draw, centre, low, high, box, lower, upper)
            report_text(draw, (centre, bar_top - 20), f"{mean:.3f}", font_obj=REPORT_TINY, anchor="mm")
            draw.multiline_text(
                (centre, bottom + 20),
                label,
                fill=REPORT_COLORS["muted"],
                font=REPORT_TINY,
                anchor="ma",
                align="center",
                spacing=3,
            )

    report_text(
        draw,
        (70, 842),
        "Balanced accuracy; red lines show theoretical chance.",
        fill=REPORT_COLORS["muted"],
        font_obj=REPORT_SMALL,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, format="PNG", optimize=True, dpi=(300, 300))
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
