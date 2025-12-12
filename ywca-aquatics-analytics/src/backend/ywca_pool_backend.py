#!/usr/bin/env python3
"""
YWCA Aquatics — backend library (final)

- prepare_datasets(csv_path)
- make_bar_chart(...), make_stacked_percent_chart(...)
- PlotStyle / Layout dataclasses
- shorten_labels(...)

Key change: Always reserve bottom space for the footer with a GridSpec row when
constrained_layout=True, so footers never collide with the plot area.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# -------------------------------
# Paths
# -------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "outputs"
OUT_PDF = OUT_DIR / "pdf"
OUT_SLIDE = OUT_DIR / "slides"
OUT_WEB = OUT_DIR / "web"
for _d in (OUT_PDF, OUT_SLIDE, OUT_WEB):
    _d.mkdir(parents=True, exist_ok=True)

# -------------------------------
# Constants / reference orders
# -------------------------------
PERSIMMON = "#FA4616"
GREY = "#929191"

INCOME_ORDER = ["Under $40,000", "$40,000–79,999", "$80,000+", "Not reported"]
ENG_ORDER = ["1 visit", "2 visits", "3 visits", "4–6 visits", "7–10 visits", "10+ visits"]
AGE_BINS = [-np.inf, 4, 9, 14, 19, 29, 44, 64, np.inf]
AGE_LABELS = ["0–4", "5–9", "10–14", "15–19", "20–29", "30–44", "45–64", "65+"]

# -------------------------------
# Styling dataclasses
# -------------------------------
@dataclass
class PlotStyle:
    bar_color: str = PERSIMMON        # bar fill
    edge_color: str = "black"         # bar edge
    grid_alpha: float = 0.35          # grid transparency
    title_color: str = PERSIMMON      # title color
    font_title: int = 12              # title font size
    font_axis: int = 8                # axis label font size
    font_tick: int = 6                # tick label font size
    font_annot: int = 8               # annotation size
    legend_font: int = 8              # legend font
    legend_title_font: int = 9        # legend title font
    footer_font: int = 8
    footer_color: str = "#929191"   # your GREY

@dataclass
class Layout:
    figsize: Tuple[float, float] = (10, 6)     # inches
    fig_scale: float = 1.00                    # global scale
    dpi: int = 300                             # output DPI
    title_pad: float = 20.0                    # title padding

    # Footer handling
    show_footer: bool = True
    footer_x: float = 0.995
    footer_y: float = 0.02
    footer_reserved: float = 0.09              # fraction of height reserved for footer
    use_footer_slot: bool = True               # reserve a bottom row for footer when constrained

    # X ticks
    rotate_x: Optional[int] = 0
    wrap_width: Optional[int] = None           # None disables wrapping
    truncate_after: Optional[int] = None       # None disables truncation
    tick_align_right_when_rotated: bool = True

    # Y limit
    ylim: Optional[Tuple[float, float]] = None
    ylim_pad: float = 1.25

    # Legend (stacked charts)
    legend_outside: bool = True
    legend_loc: str = "upper left"
    legend_bbox_to_anchor: Tuple[float, float] = (1.02, 1.0)

    # Layout engine
    use_constrained_layout: bool = True
    suppress_tight_warnings: bool = True
    bbox_tight: bool = True

    # Only used when use_constrained_layout=False
    margin_left: Optional[float] = None
    margin_right: Optional[float] = None
    margin_bottom: Optional[float] = None
    margin_top: Optional[float] = None

# -------------------------------
# CSV loader (robust)
# -------------------------------
def _read_csv_robust(path: Union[str, Path]) -> pd.DataFrame:
    path = Path(path)
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")

# -------------------------------
# Cleaning helpers
# -------------------------------
def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.replace({u"\xa0": " ", u"\u202f": " "}, regex=True, inplace=True)
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = (
                df[c].astype(str)
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
                .replace({"nan": np.nan, "None": np.nan})
            )
    return df

def clean_gender(x: str) -> Optional[str]:
    if pd.isna(x): return np.nan
    s = str(x).lower()
    if "female" in s: return "Female"
    if "male" in s: return "Male"
    if "non" in s: return "Non-binary"
    return np.nan

def clean_income_raw(x: str) -> Optional[str]:
    if pd.isna(x) or x == "": return np.nan
    s = str(x).replace(" ", "")
    if s.startswith("<"): return "$0-10,000"
    return str(x)

def income_bucket(h: str) -> str:
    if pd.isna(h): return "Not reported"
    low = {"$0-10,000", "$10,001-20,000", "$20,001-30,000", "$30,001-40,000"}
    mid = {"$40,001-50,000", "$50,001-60,000", "$60,001-70,000", "$70,001-80,000"}
    high = {"$80,001-90,000", "$90,001+"}
    if h in low: return "Under $40,000"
    if h in mid: return "$40,000–79,999"
    if h in high: return "$80,000+"
    return "Not reported"

def map_program_category(name: str) -> str:
    if pd.isna(name): return "Other / Unclassified"
    s = str(name).lower().strip()
    if s.startswith("family swim"): return "Family Swim (Drop-In)"
    if "single gender swim" in s or "sgs" in s: return "Single Gender Swim"
    if "companion" in s: return "Swim Lessons — Companion"
    if "parent & child" in s or "parent and child" in s: return "Swim Lessons — Parent & Child"
    if "prek" in s: return "Swim Lessons — PreK"
    if "childcare" in s: return "Swim Lessons — Childcare"
    if "adult" in s: return "Swim Lessons — Adult"
    if any(l in s for l in ["level 1", "level 2", "level 3", "level 4", "level 5"]):
        return "Swim Lessons — Youth Levels"
    return "Other / Unclassified"

def map_race_clean(x: str) -> str:
    if pd.isna(x) or x == "": return "Some other race (write-in)"
    s = str(x).lower()
    if "white/european" in s or ("white" in s and "european" in s): return "White"
    if "black" in s or "african-american" in s or "african american" in s: return "Black or African American"
    if "asian" in s: return "Asian"
    if "hispanic" in s or "latino" in s or "latinx" in s: return "Hispanic or Latino"
    if "middle eastern" in s or "north african" in s: return "Middle Eastern / North African"
    if ("american indian" in s or "alaskan" in s or "native american" in s
        or "native hawaiian" in s or "pacific islander" in s or "first nations" in s):
        return "American Indian / Alaska Native / Native Hawaiian / Pacific Islander"
    if "multiracial" in s or "two or more" in s: return "Two or more races"
    return "Some other race (write-in)"

def collapse_race_for_model(races: pd.Series) -> str:
    uniq = set(races.dropna())
    if len(uniq) == 0: return "Other / Small N"
    if "Black or African American" in uniq: return "Black"
    if "Hispanic or Latino" in uniq: return "Hispanic or Latino"
    if "Asian" in uniq: return "Asian"
    if "White" in uniq: return "White"
    return "Other / Small N"

def engagement_bin(n: int) -> str:
    if n == 1: return "1 visit"
    if n == 2: return "2 visits"
    if n == 3: return "3 visits"
    if 4 <= n <= 6: return "4–6 visits"
    if 7 <= n <= 10: return "7–10 visits"
    return "10+ visits"

def classify_pathway(first_group: str, ever_lessons: bool) -> str:
    if first_group != "Family Swim (Drop-In)" and ever_lessons: return "Entered via Lessons"
    if first_group == "Family Swim (Drop-In)" and ever_lessons: return "Converted from Family Swim"
    if first_group == "Family Swim (Drop-In)" and not ever_lessons: return "Stayed Family Swim Only"
    return "Lessons Only (never drop-in)"

# -------------------------------
# Label utilities
# -------------------------------
def shorten_labels(
    labels: Iterable[str],
    mapping: Optional[Dict[str, str]] = None,
    truncate_after: Optional[int] = None,
    wrap_width: Optional[int] = None,
) -> List[str]:
    """Map → truncate → wrap; wrap skipped if width is None."""
    def _wrap(s: str, width: int) -> str:
        if width is None or width <= 0: return s
        words = s.split()
        lines, line = [], []
        curr = 0
        for w in words:
            add = (1 if line else 0) + len(w)
            if curr + add > width:
                lines.append(" ".join(line)); line = [w]; curr = len(w)
            else:
                line.append(w); curr += add
        if line: lines.append(" ".join(line))
        return "\n".join(lines)

    out = []
    for lbl in labels:
        s = str(lbl)
        if mapping and s in mapping:
            s = mapping[s]
        if truncate_after and len(s) > truncate_after:
            s = s[:truncate_after - 1] + "…"
        s = _wrap(s, wrap_width) if wrap_width else s
        out.append(s)
    return out

def _auto_ylim(max_height: float, pad: float) -> float:
    return max_height * (pad if pad > 1.0 else (1.0 + pad))

# -------------------------------
# Data prep (main)
# -------------------------------
def prepare_datasets(csv_path: Union[str, Path]):
    """
    Returns:
      df          - enrollment-level (one row per registration)
      df_people   - person-level (dedup by [Age, GenderClean, RaceModel, Household])
      people_conv - person-level + LessonFlag and indicator
      pathway     - per-person pathway categorization
    """
    csv_path = Path(csv_path)
    df_raw = _read_csv_robust(csv_path)
    df = standardize_dataframe(df_raw)

    if "House income" in df.columns and "HouseIncomeRaw" not in df.columns:
        df = df.rename(columns={"House income": "HouseIncomeRaw"})

    df["Age"] = pd.to_numeric(df.get("Age"), errors="coerce")
    df["Household"] = pd.to_numeric(df.get("Household"), errors="coerce")

    df["GenderClean"] = df.get("Gender", pd.Series([np.nan]*len(df))).apply(clean_gender)

    df["HouseIncome"] = df.get("HouseIncomeRaw", pd.Series([np.nan]*len(df))).apply(clean_income_raw)
    income_order_detail = [
        "$0-10,000", "$10,001-20,000", "$20,001-30,000", "$30,001-40,000",
        "$40,001-50,000", "$50,001-60,000", "$60,001-70,000", "$70,001-80,000",
        "$80,001-90,000", "$90,001+",
    ]
    df["HouseIncome"] = pd.Categorical(df["HouseIncome"], categories=income_order_detail, ordered=True)
    df["IncomeBucket"] = df["HouseIncome"].apply(income_bucket)

    df["ProgramCategory"] = df.get("Program", pd.Series([""]*len(df))).apply(map_program_category)
    df["ProgramGroup"] = np.where(df["ProgramCategory"] == "Family Swim (Drop-In)",
                                  "Family Swim (Drop-In)", "Swim Lessons (All)")

    df["AgeBand"] = pd.cut(df["Age"], bins=AGE_BINS, labels=AGE_LABELS, right=True)
    df["YouthAdult"] = pd.NA
    df.loc[df["Age"] <= 17, "YouthAdult"] = "Youth (0–17)"
    df.loc[df["Age"] >= 18, "YouthAdult"] = "Adults (18+)"

    race_long = df[["Race"]].copy()
    race_long["Race"] = race_long["Race"].fillna("").astype(str).str.split(",")
    race_long = race_long.explode("Race")
    race_long["Race"] = race_long["Race"].str.replace(r"\s+", " ", regex=True).str.strip()
    race_long = race_long[race_long["Race"] != ""]
    race_long["RaceClean"] = race_long["Race"].apply(map_race_clean)
    race_model = race_long.groupby(race_long.index)["RaceClean"].apply(collapse_race_for_model)
    df["RaceModel"] = race_model.reindex(df.index)

    person_cols = ["Age", "GenderClean", "RaceModel", "Household"]
    df["PersonKey"] = df[person_cols].astype(str).agg("|".join, axis=1)
    dup_counts = df["PersonKey"].value_counts()
    df_people = df.drop_duplicates(subset="PersonKey").copy()

    df_people["EngagementBin"] = df_people["PersonKey"].map(lambda k: engagement_bin(dup_counts.get(k, 1)))

    df_people["POC"] = df_people["RaceModel"].ne("White").astype(int)
    df_people["LowIncome"] = df_people["IncomeBucket"].eq("Under $40,000").astype(int)
    df_people["IsYouth"] = df_people["YouthAdult"].eq("Youth (0–17)").astype(int)
    df_people["Female"] = df_people["GenderClean"].eq("Female").astype(int)
    df_people["HighEng"] = df_people["EngagementBin"].isin({"4–6 visits", "7–10 visits", "10+ visits"}).astype(int)

    conv_lookup = df.groupby("PersonKey")["ProgramGroup"].apply(
        lambda g: "Lesson Participant" if (g == "Swim Lessons (All)").any() else "Family Swim Only"
    )
    people_conv = df_people.copy()
    people_conv["LessonFlag"] = people_conv["PersonKey"].map(conv_lookup)
    people_conv["LessonParticipant"] = people_conv["LessonFlag"].eq("Lesson Participant").astype(int)

    df_sorted = df.sort_index()
    first_touch = df_sorted.groupby("PersonKey")["ProgramGroup"].first()
    ever_lessons = df_sorted.groupby("PersonKey")["ProgramGroup"].apply(lambda g: (g == "Swim Lessons (All)").any())
    pathway = pd.DataFrame({"FirstGroup": first_touch, "EverLessons": ever_lessons})
    pathway["PathwayType"] = pathway.apply(lambda r: classify_pathway(r["FirstGroup"], r["EverLessons"]), axis=1)

    return df, df_people, people_conv, pathway

# -------------------------------
# Save helper
# -------------------------------
def _save_fig_all(fig, fname_base: str, layout: Layout) -> None:
    if not layout.use_constrained_layout:
        if all(v is not None for v in [layout.margin_left, layout.margin_right, layout.margin_bottom, layout.margin_top]):
            fig.subplots_adjust(
                left=layout.margin_left,
                right=1 - layout.margin_right,
                bottom=layout.margin_bottom,
                top=layout.margin_top,
            )
        if layout.suppress_tight_warnings:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                plt.tight_layout()
        else:
            plt.tight_layout()

    bbox = "tight" if layout.bbox_tight else None
    (OUT_PDF / f"{fname_base}.pdf").parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF / f"{fname_base}.pdf", dpi=layout.dpi, bbox_inches=bbox)
    fig.savefig(OUT_SLIDE / f"{fname_base}.png", dpi=layout.dpi, bbox_inches=bbox)
    fig.savefig(OUT_WEB / f"{fname_base}_web.png", dpi=max(150, layout.dpi // 2), bbox_inches=bbox)
    plt.close(fig)

# -------------------------------
# Footers & titles
# -------------------------------
def _footer(fig, level: str, extra: Optional[str], x: float, y: float, style: PlotStyle) -> None:
    line1 = "Source: YWCA Lewiston pool programs (2025)."
    line2 = "Approx. unique participants (person-level)." if level == "people" else "Enrollment-level (all sign-ups)."
    lines = [line1, line2] + ([extra] if extra else [])
    fig.text(
        x, y, "\n".join(lines),
        ha="right", va="bottom",
        fontsize=style.footer_font,
        color=style.footer_color,
    )

def _title(ax, line1: str, line2: str, level_label: str, color: str, size: int, pad: float) -> None:
    ax.set_title(f"{line1}\n{line2}\nYWCA Lewiston Pool Programs, 2025 — {level_label}",
                 fontsize=size, fontweight="bold", color=color, pad=pad)

# -------------------------------
# Plotters (with footer slot reservation)
# -------------------------------
def _make_fig_axes_with_footer(layout: Layout):
    """Create a figure and main axis, reserving a bottom slot for the footer if requested."""
    fig_w = layout.figsize[0] * layout.fig_scale
    fig_h = layout.figsize[1] * layout.fig_scale

    if layout.use_constrained_layout and layout.use_footer_slot and layout.footer_reserved > 0:
        fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)
        # 2-row grid: top=plot (weight 1.0), bottom=footer slot (small)
        footer_weight = max(0.01, min(0.3, layout.footer_reserved))
        gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1.0, footer_weight])
        ax_main = fig.add_subplot(gs[0, 0])
        ax_footer = fig.add_subplot(gs[1, 0])
        ax_footer.axis("off")  # just a spacer; footer text uses fig.text
        return fig, ax_main
    else:
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=layout.use_constrained_layout)
        return fig, ax

def make_bar_chart(
    series: Union[pd.Series, pd.Index, List[Union[str, float, int]], Dict[str, float]],
    fname_base: str,
    line1: str,
    line2: str,
    level_label: str,
    *,
    # data mode
    use_value_counts: bool = True,
    value_is_percent: bool = False,
    # style/layout
    style: PlotStyle = PlotStyle(),
    layout: Layout = Layout(),
    # axis/labels
    ylabel: str = "Number of People",
    show_percent: bool = True,
    show_values: bool = True,
    annot_offset: float = 6.0,
    # label processing
    label_map: Optional[Dict[str, str]] = None,
) -> None:
    # Build series of heights
    if isinstance(series, dict):
        s = pd.Series(series)
        use_value_counts = False
    elif isinstance(series, (list, pd.Index)):
        s = pd.Series(series)
    else:
        s = series

    if use_value_counts:
        counts = s.value_counts(dropna=False)
        labels = counts.index.astype(str).tolist()
        heights = counts.values.astype(float)
        perc = (counts / counts.sum() * 100).round(1)
    else:
        s = pd.Series(s).dropna()
        labels = s.index.astype(str).tolist()
        heights = s.values.astype(float)
        if value_is_percent:
            perc = pd.Series(heights, index=labels).round(1)
        else:
            total = heights.sum() if heights.sum() != 0 else 1.0
            perc = pd.Series(heights / total * 100, index=labels).round(1)

    # Shorten labels (map → truncate → wrap)
    labels = shorten_labels(labels,
                            mapping=label_map,
                            truncate_after=layout.truncate_after,
                            wrap_width=layout.wrap_width)

    # Figure & axis (with footer slot)
    fig, ax = _make_fig_axes_with_footer(layout)

    x = np.arange(len(labels))
    bars = ax.bar(x, heights, color=style.bar_color, edgecolor=style.edge_color)

    # Y limits (auto or explicit)
    ymax = max(heights) if len(heights) else 1.0
    if layout.ylim is not None:
        ax.set_ylim(*layout.ylim)
    else:
        ax.set_ylim(0, _auto_ylim(ymax, layout.ylim_pad))

    # Value annotations
    if show_values:
        for i, b in enumerate(bars):
            if value_is_percent:
                txt = f"{heights[i]:.1f}%"
            else:
                txt = f"{int(heights[i])}"
                if show_percent:
                    txt += f"\n({perc.iloc[i]:.1f}%)"
            ax.annotate(txt,
                        (b.get_x() + b.get_width() / 2.0, b.get_height()),
                        ha="center", va="bottom",
                        fontsize=style.font_annot,
                        xytext=(0, annot_offset),
                        textcoords="offset points")

    # Axes labels, ticks
    ax.set_ylabel(ylabel, fontsize=style.font_axis)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=style.font_tick,
                       rotation=(layout.rotate_x or 0),
                       ha=("right" if layout.rotate_x and layout.tick_align_right_when_rotated else "center"))

    # Grid, title, footer
    ax.grid(axis="y", linestyle="--", alpha=style.grid_alpha)
    _title(ax, line1, line2, level_label, style.title_color, style.font_title, layout.title_pad)
    if layout.show_footer:
        level = "people" if "People" in ylabel or "Person" in ylabel else "enrollment"
        _footer(fig, level=level, extra=None, x=layout.footer_x, y=layout.footer_y, style=style)


    _save_fig_all(fig, fname_base, layout)

def make_stacked_percent_chart(
    table: pd.DataFrame,
    fname_base: str,
    line1: str,
    line2: str,
    level_label: str,
    *,
    style: PlotStyle = PlotStyle(),
    layout: Layout = Layout(),
    ylabel: str = "Percent of People within Group",
    legend_title: str = "",
) -> None:
    table = table.copy()
    if table.empty:
        print(f"[WARN] {fname_base}: empty table, skipping.")
        return

    # Colors: gradient from brand color → lighter tints
    base_rgb = np.array(mcolors.to_rgb(PERSIMMON))
    whites = np.ones(3)
    alphas = np.linspace(0.0, 0.6, table.shape[1])
    colors = [mcolors.to_hex((1 - a) * base_rgb + a * whites) for a in alphas]

    # Figure & axis (with footer slot)
    fig, ax = _make_fig_axes_with_footer(layout)

    x = np.arange(len(table.index))
    bottom = np.zeros(len(table.index))
    for col, color in zip(table.columns, colors):
        vals = table[col].values
        bars = ax.bar(x, vals, bottom=bottom, color=color, edgecolor=style.edge_color, label=col)
        for j, (bar, v) in enumerate(zip(bars, vals)):
            if v >= 7:
                ax.text(bar.get_x() + bar.get_width() / 2, bottom[j] + v / 2,
                        f"{v:.0f}%", ha="center", va="center", fontsize=style.font_annot)
        bottom += vals

    ax.set_ylim(0, 100)
    labels = shorten_labels(table.index.astype(str),
                            truncate_after=layout.truncate_after,
                            wrap_width=layout.wrap_width)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=style.font_tick,
                       rotation=(layout.rotate_x or 0),
                       ha=("right" if layout.rotate_x and layout.tick_align_right_when_rotated else "center"))

    ax.set_ylabel(ylabel, fontsize=style.font_axis)
    ax.grid(axis="y", linestyle="--", alpha=style.grid_alpha)

    # Legend
    if layout.legend_outside:
        ax.legend(title=legend_title, bbox_to_anchor=layout.legend_bbox_to_anchor,
                  loc=layout.legend_loc, borderaxespad=0.0,
                  fontsize=style.legend_font, title_fontsize=style.legend_title_font)
    else:
        ax.legend(title=legend_title, fontsize=style.legend_font, title_fontsize=style.legend_title_font)

    _title(ax, line1, line2, level_label, style.title_color, style.font_title, layout.title_pad)
    if layout.show_footer:
        _footer(fig, level="people", extra=None, x=layout.footer_x, y=layout.footer_y, style=style)


    _save_fig_all(fig, fname_base, layout)
