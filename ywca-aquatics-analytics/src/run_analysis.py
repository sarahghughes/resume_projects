#!/usr/bin/env python3
"""
YWCA Aquatics Program Analytics

- Preps datasets from data/raw/PoolDemographics_2025.csv
- Produces 10 figures to outputs/{pdf,slides,web}
- Prints per-figure numeric tables
- Every plot call lists ALL parameters (with defaults echoed in comments)
- Set up for user to make visuals on their own with minimal difficulty 
"""

from pathlib import Path
import pandas as pd
from dataclasses import replace

from backend.ywca_pool_backend import (
    prepare_datasets,
    make_bar_chart,
    make_stacked_percent_chart,
    PlotStyle,
    Layout,
    INCOME_ORDER,
    ENG_ORDER,
    shorten_labels,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "raw" / "PoolDemographics_2025.csv"

BASE_STYLE = PlotStyle(
    bar_color="#FA4616",
    edge_color="black",
    grid_alpha=0.35,
    title_color="#FA4616",
    font_title=18,
    font_axis=11,
    font_tick=12,
    font_annot=12,
    legend_font=14,
    legend_title_font=12,
    footer_font=10,
    footer_color="#929191"
)

BASE_LAYOUT = Layout(
    figsize=(10, 6),
    fig_scale=1.00,
    dpi=300,
    title_pad=16,
    show_footer=True,
    footer_x=0.995,
    footer_y=0.02,
    footer_reserved=0.1,         # baseline bottom pad
    use_footer_slot=True,         # keep space for footer
    rotate_x=0,
    wrap_width=None,
    truncate_after=None,
    tick_align_right_when_rotated=True,
    ylim=None,
    ylim_pad=1.19,
    legend_outside=True,
    legend_loc="upper left",
    legend_bbox_to_anchor=(1.02, 1.0),
    use_constrained_layout=True,
    suppress_tight_warnings=True,
    bbox_tight=True,
    # margins only used if use_constrained_layout=False
    margin_left=None,
    margin_right=None,
    margin_bottom=None,
    margin_top=None,
)

FIG_NUM = 1
def next_fig_name(slug: str) -> str:
    global FIG_NUM
    name = f"Figure{FIG_NUM:02d}_{slug}"
    FIG_NUM += 1
    return name

def main() -> None:
    df, df_people, people_conv, pathway = prepare_datasets(CSV_PATH)

    print("\n=== Enrollment-level (first 5 cols) ===")
    print(df.iloc[:, :5].head())
    print("\n=== Person-level head ===")
    print(df_people.head())

    # 01
    make_bar_chart(
        series=df["ProgramCategory"],
        fname_base=next_fig_name("program_mix"),
        line1="What Brings Families to the Pool?",
        line2="Program Mix by Category",
        level_label="Enrollment-Level",
        use_value_counts=True,
        value_is_percent=False,
        style=BASE_STYLE,
        layout=replace(
            BASE_LAYOUT,
            rotate_x=25,        # x-tick rotation (degrees)
        ),
        ylabel="Number of Enrollments",
        show_percent=True,
        show_values=True,
        annot_offset=6.0,
        label_map=None,
    )

    # 02
    make_bar_chart(
        series=df_people["AgeBand"],
        fname_base=next_fig_name("age_distribution"),
        line1="Who Reaches the Pool?",
        line2="Participant Age Profile",
        level_label="Person-Level",
        use_value_counts=True,
        value_is_percent=False,
        style=BASE_STYLE,
        layout=BASE_LAYOUT,
        ylabel="Number of Approx-Unique People",
        show_percent=True,
        show_values=True,
        annot_offset=6.0,
        label_map=None,
    )

    # 03
    inc_order = [b for b in INCOME_ORDER if b in df_people["IncomeBucket"].unique()]
    s_income = df_people["IncomeBucket"].astype("category").cat.set_categories(inc_order)
    make_bar_chart(
        series=s_income,
        fname_base=next_fig_name("income_buckets"),
        line1="Who Can Afford to Swim?",
        line2="Household Income Distribution of Participants",
        level_label="Person-Level",
        use_value_counts=True,
        value_is_percent=False,
        style=BASE_STYLE,
        layout=BASE_LAYOUT,
        ylabel="Number of Approx-Unique People",
        show_percent=True,
        show_values=True,
        annot_offset=6.0,
        label_map=None,
    )

    # 04
    make_bar_chart(
        series=df_people["RaceModel"],
        fname_base=next_fig_name("race_model"),
        line1="Are We Serving a Racially Diverse Community?",
        line2="Participant Racial Composition (Collapsed Categories)",
        level_label="Person-Level",
        use_value_counts=True,
        value_is_percent=False,
        style=BASE_STYLE,
        layout=BASE_LAYOUT,
        ylabel="Number of Approx-Unique People",
        show_percent=True,
        show_values=True,
        annot_offset=6.0,
        label_map=None,
    )

    # 05
    df_people_eng = df_people.copy()
    df_people_eng["EngagementBin"] = pd.Categorical(df_people_eng["EngagementBin"],
                                                    categories=ENG_ORDER, ordered=True)
    make_bar_chart(
        series=df_people_eng["EngagementBin"],
        fname_base=next_fig_name("engagement_frequency"),
        line1="How Often Do Families Return?",
        line2="Distribution of Repeat Engagement",
        level_label="Person-Level",
        use_value_counts=True,
        value_is_percent=False,
        style=BASE_STYLE,
        layout=BASE_LAYOUT,
        ylabel="Approx-Unique People",
        show_percent=True,
        show_values=True,
        annot_offset=6.0,
        label_map=None,
    )

    # 06
    tab_inc = (pd.crosstab(df_people["IncomeBucket"], df_people["EngagementBin"], normalize="index") * 100)
    tab_inc = tab_inc.reindex([b for b in INCOME_ORDER if b in tab_inc.index])
    tab_inc = tab_inc[[c for c in ENG_ORDER if c in tab_inc.columns]]
    make_stacked_percent_chart(
        table=tab_inc,
        fname_base=next_fig_name("repeat_by_income"),
        line1="Do Higher-Income Families Engage More Often?",
        line2="Repeat Engagement Distribution by Household Income",
        level_label="Person-Level",
        style=BASE_STYLE,
        layout=BASE_LAYOUT,
        ylabel="Percent of People within Income Segment",
        legend_title="Total Visits Across the Year",
    )

    # 07
    tab_race = (pd.crosstab(df_people["RaceModel"], df_people["EngagementBin"], normalize="index") * 100)
    tab_race = tab_race[[c for c in ENG_ORDER if c in tab_race.columns]]
    make_stacked_percent_chart(
        table=tab_race,
        fname_base=next_fig_name("repeat_by_race"),
        line1="Does Retention Differ by Race?",
        line2="Repeat Engagement Distribution by Race Category",
        level_label="Person-Level",
        style=BASE_STYLE,
        layout=replace(
            BASE_LAYOUT,
            rotate_x=25,        # x-tick rotation (degrees)
        ),
        ylabel="Percent of People within Race Segment",
        legend_title="Total Visits Across the Year",
    )

    # 08
    make_bar_chart(
        series=pathway["PathwayType"],
        fname_base=next_fig_name("entry_pathways"),
        line1="How Do Families Move Through the Pool System?",
        line2="Entry & Pathways Between Drop-In and Lessons",
        level_label="Person-Level",
        use_value_counts=True,
        value_is_percent=False,
        style=BASE_STYLE,
        layout=BASE_LAYOUT,
        ylabel="Approx-Unique People",
        show_percent=True,
        show_values=True,
        annot_offset=6.0,
        label_map=None,
    )

    # 09
    conv_race_tab = (pd.crosstab(people_conv["RaceModel"], people_conv["LessonFlag"], normalize="index") * 100)
    if "Lesson Participant" in conv_race_tab.columns:
        lesson_rates_race = conv_race_tab["Lesson Participant"].round(1)
        make_bar_chart(
            series=lesson_rates_race,
            fname_base=next_fig_name("lesson_by_race"),
            line1="Do All Racial Groups Access Lessons Equally?",
            line2="Lesson Access by Race Category",
            level_label="Person-Level",
            use_value_counts=False,
            value_is_percent=True,
            style=BASE_STYLE,
            layout=BASE_LAYOUT,
            ylabel="Percent Who Ever Took Lessons",
            show_percent=False,
            show_values=True,
            annot_offset=6.0,
            label_map=None,
        )

    # 10
    conv_inc_tab = (pd.crosstab(people_conv["IncomeBucket"], people_conv["LessonFlag"], normalize="index") * 100)
    if "Lesson Participant" in conv_inc_tab.columns:
        lr_inc = conv_inc_tab["Lesson Participant"].reindex([b for b in INCOME_ORDER if b in conv_inc_tab.index]).round(1)
        make_bar_chart(
            series=lr_inc,
            fname_base=next_fig_name("lesson_by_income"),
            line1="Do Families With Higher Income Access More Lessons?",
            line2="Lesson Participation by Household Income",
            level_label="Person-Level",
            use_value_counts=False,
            value_is_percent=True,
            style=BASE_STYLE,
            layout=BASE_LAYOUT,
            ylabel="Percent Who Ever Took Lessons",
            show_percent=False,
            show_values=True,
            annot_offset=6.0,
            label_map=None,
        )

    # tables (concise)
    print("\n=== Tables for report ===")
    print("Figure 01 – Program mix")
    print(df["ProgramCategory"].value_counts().to_frame("count").assign(
        percent=lambda x: (x["count"]/x["count"].sum()*100).round(1)), "\n")

    print("Figure 02 – Age distribution")
    fig02 = df_people["AgeBand"].value_counts().sort_index().to_frame("count")
    fig02["percent"] = (fig02["count"]/fig02["count"].sum()*100).round(1)
    print(fig02, "\n")

    print("Figure 03 – Income buckets")
    fig03 = df_people["IncomeBucket"].value_counts().to_frame("count")
    fig03["percent"] = (fig03["count"]/fig03["count"].sum()*100).round(1)
    print(fig03, "\n")

    print("Figure 04 – RaceModel")
    fig04 = df_people["RaceModel"].value_counts().to_frame("count")
    fig04["percent"] = (fig04["count"]/fig04["count"].sum()*100).round(1)
    print(fig04, "\n")

    print("Figure 05 – Engagement frequency")
    fig05 = df_people_eng["EngagementBin"].value_counts().reindex(ENG_ORDER).to_frame("count")
    fig05["percent"] = (fig05["count"]/fig05["count"].sum()*100).round(1)
    print(fig05, "\n")

    print("Figure 06 – Repeat engagement by income (%)")
    print(tab_inc.round(1), "\n")

    print("Figure 07 – Repeat engagement by race (%)")
    print(tab_race.round(1), "\n")

    print("Figure 08 – Pathways")
    fig08 = pathway["PathwayType"].value_counts().to_frame("count")
    fig08["percent"] = (fig08["count"]/fig08["count"].sum()*100).round(1)
    print(fig08, "\n")

    if "Lesson Participant" in conv_race_tab.columns:
        print("Figure 09 – Lesson by race (%)")
        print(conv_race_tab["Lesson Participant"].round(1).to_frame("percent_with_≥1_lesson"), "\n")

    if "Lesson Participant" in conv_inc_tab.columns:
        print("Figure 10 – Lesson by income (%)")
        print(conv_inc_tab["Lesson Participant"].reindex([b for b in INCOME_ORDER if b in conv_inc_tab.index])
              .round(1).to_frame("percent_with_≥1_lesson"), "\n")

if __name__ == "__main__":
    main()
