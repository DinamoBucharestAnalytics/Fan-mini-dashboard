import base64
import json
import math
import re
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st
from wordcloud import WordCloud


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "fan_survey_dashboard.xlsx"
SOCIAL_MEDIA_DIR = BASE_DIR / "data" / "social_media"
SOCIAL_MEDIA_GEO_PATH = SOCIAL_MEDIA_DIR / "Power BI source.xlsx"
SOCIAL_MEDIA_DEMO_PATH = SOCIAL_MEDIA_DIR / "Age and sex.xlsx"
RO_GEOJSON_PATH = BASE_DIR / "romania.geojson"
LOGO_PATH = BASE_DIR / "data" / "img" / "dinamo-data-analysis-red.ico"

SURVEY_SOURCE = "Survey (May 2026)"
PLATFORM_SOURCES = [
    "Club mobile app",
    "Club Facebook",
    "Fan Facebook (RD1948)",
    "Fan Youtube (RD1948)",
    "Merchandise",
]
SOURCE_OPTIONS = [SURVEY_SOURCE] + PLATFORM_SOURCES
SURVEY_MENUS = ["Demographics", "Sentiment", "Club"]
PLATFORM_MENUS = ["Demographics"]
CUSTOM_ANALYSIS_SHEETS = {
    f"col_{idx}_{suffix}"
    for idx in range(1, 5)
    for suffix in ["keywords", "bigrams", "topics", "classified", "classified_RESULTS"]
}

DINAMO_RED = "#e30613"
DARK_RED = "#9d0208"
BLACK = "#111111"
GREY = "#666666"
LIGHT_GREY = "#f5f5f5"
RED_SCALE = [
    (0.0, "#fff5f5"),
    (0.2, "#ffd6d6"),
    (0.4, "#ffadad"),
    (0.6, "#ff7070"),
    (0.8, "#ff3333"),
    (1.0, DINAMO_RED),
]
PIE_COLORS = [
    DINAMO_RED,
    "#e65f5f",
    "#f08f8f",
    "#f4c2c2",
    BLACK,
    "#444444",
    "#777777",
    "#aaaaaa",
    "#dddddd",
]

VERTICAL_CHART_HEIGHT = 460
MAP_CHART_HEIGHT = 600
TREEMAP_CHART_HEIGHT = 600
BAR_CHART_WIDTH = 900


def image_data_uri(path: Path, mime_type: str) -> str:
    return f"data:{mime_type};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def horizontal_chart_height(row_count: int, minimum: int = 420, per_row: int = 34, maximum: int = 900) -> int:
    return min(max(minimum, row_count * per_row), maximum)


def render_bar_chart(fig, width: int = BAR_CHART_WIDTH):
    fig.update_layout(width=width)
    st.plotly_chart(fig, use_container_width=False)

COUNTRY_NORMALIZE = {
    "romania": "Romania",
    "r moldova": "Moldova",
    "republica moldova": "Moldova",
    "moldova": "Moldova",
    "tarile de jos": "Netherlands",
    "olanda": "Netherlands",
    "italia": "Italy",
    "spania": "Spain",
    "germania": "Germany",
    "franta": "France",
    "franţa": "France",
    "belgia": "Belgium",
    "irlanda": "Ireland",
    "marea britanie": "United Kingdom",
    "regatul unit": "United Kingdom",
    "anglia": "United Kingdom",
    "sua": "United States",
    "statele unite": "United States",
    "statele unite ale americii": "United States",
    "canada": "Canada",
    "austria": "Austria",
    "suedia": "Sweden",
    "elvetia": "Switzerland",
    "danemarca": "Denmark",
    "norvegia": "Norway",
    "portugalia": "Portugal",
    "grecia": "Greece",
    "cipru": "Cyprus",
    "turcia": "Turkey",
    "ucraina": "Ukraine",
    "cehia": "Czechia",
    "luxemburg": "Luxembourg",
    "thailanda": "Thailand",
    "malta": "Malta",
    "lituania": "Lithuania",
    "islanda": "Iceland",
    "noua zeelanda": "New Zealand",
    "polonia": "Poland",
    "filipine": "Philippines",
    "finlanda": "Finland",
    "japonia": "Japan",
    "japan": "Japan",
    "reunion": "Réunion",
    "réunion": "Réunion",
    "serbia": "Serbia",
    "gibraltar": "Gibraltar",
    "emiratele arabe unite": "United Arab Emirates",
    "eau": "United Arab Emirates",
    "united arab emirates": "United Arab Emirates",
    "australia": "Australia",
    "afganistan": "Afghanistan",
    "afghanistan": "Afghanistan",
}

ISO3 = {
    "Romania": "ROU",
    "Moldova": "MDA",
    "Netherlands": "NLD",
    "Italy": "ITA",
    "Spain": "ESP",
    "Germany": "DEU",
    "France": "FRA",
    "Belgium": "BEL",
    "Ireland": "IRL",
    "United Kingdom": "GBR",
    "United States": "USA",
    "Canada": "CAN",
    "Austria": "AUT",
    "Sweden": "SWE",
    "Switzerland": "CHE",
    "Denmark": "DNK",
    "Norway": "NOR",
    "Portugal": "PRT",
    "Greece": "GRC",
    "Cyprus": "CYP",
    "Turkey": "TUR",
    "Ukraine": "UKR",
    "Czechia": "CZE",
    "Luxembourg": "LUX",
    "Thailand": "THA",
    "Malta": "MLT",
    "Lithuania": "LTU",
    "Iceland": "ISL",
    "New Zealand": "NZL",
    "Poland": "POL",
    "Philippines": "PHL",
    "Finland": "FIN",
    "Japan": "JPN",
    "Réunion": "REU",
    "Serbia": "SRB",
    "Gibraltar": "GIB",
    "United Arab Emirates": "ARE",
    "Australia": "AUS",
    "Afghanistan": "AFG",
}


st.set_page_config(
    page_title="Dinamo Fan Survey Dashboard",
    page_icon="data/img/dinamo-data-analysis-red.ico",
    layout="wide",
)


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    text = text.replace("ş", "s").replace("ţ", "t").replace("ș", "s").replace("ț", "t")
    text = "".join(
        char for char in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(char)
    )
    text = re.sub(r"[^A-Za-z0-9\s]", " ", text).lower().strip()
    return re.sub(r"\s+", " ", text)


def find_col(df: pd.DataFrame, contains: str) -> str:
    target = normalize_text(contains)
    for col in df.columns:
        if target in normalize_text(col):
            return col
    raise KeyError(f"Column containing '{contains}' not found")


@st.cache_data
def load_workbook(path: Path, modified_ns: int, file_size: int) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    _ = modified_ns, file_size
    xl = pd.ExcelFile(path)
    df = pd.read_excel(path, sheet_name="responses_with_judet")
    summaries = {
        sheet: pd.read_excel(path, sheet_name=sheet)
        for sheet in xl.sheet_names
        if sheet != "responses_with_judet"
    }
    return df, summaries


@st.cache_data
def load_geojson(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["age_numeric"] = pd.to_numeric(df["Vârstă"], errors="coerce")
    bins = [0, 13, 17, 24, 34, 44, 54, 64, math.inf]
    labels = ["0-13", "14-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    df["age_band"] = pd.cut(df["age_numeric"], bins=bins, labels=labels, right=True)
    df["country_display"] = df["Țară de reședință"].fillna("Unknown").astype(str).str.strip()
    df["country_norm"] = df["country_display"].map(lambda x: COUNTRY_NORMALIZE.get(normalize_text(x), x))
    return df


def _fraction(value: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(value, errors="coerce")
    return numeric / 100 if numeric.max(skipna=True) > 1 else numeric


def _expand_social_rows(df: pd.DataFrame) -> pd.DataFrame:
    country_splits = {
        "benelux": [("Belgium", 0.5), ("Netherlands", 0.5)],
    }
    county_splits = {
        "timisoara": [("Timis", 0.75), ("Caras-Severin", 0.25)],
        "cluj alba": [("Cluj", 0.66), ("Alba", 0.34)],
        "galati braila": [("Galati", 0.5), ("Braila", 0.5)],
    }
    expanded = []
    for _, row in df.iterrows():
        key = normalize_text(row["geography"])
        splits = country_splits.get(key) if row["level"] == "country" else county_splits.get(key)
        if not splits:
            expanded.append(row)
            continue
        for geography, fraction in splits:
            new_row = row.copy()
            new_row["geography"] = geography
            new_row["followers"] = row["followers"] * fraction
            new_row["percent_on_platform"] = row["percent_on_platform"] * fraction
            expanded.append(new_row)
    return pd.DataFrame(expanded)


@st.cache_data
def load_social_media_geo(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = [str(col).strip().lower() for col in df.columns]
    df = df.rename(columns={"percent on the platform": "percent_on_platform"})
    required = {"geography", "level", "platform", "followers", "percent_on_platform"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing social-media geography columns: {', '.join(sorted(missing))}")
    df["geography"] = df["geography"].astype(str).str.replace("\u00a0", " ", regex=False).str.strip()
    df["level"] = df["level"].astype(str).str.lower().str.strip()
    df["platform"] = df["platform"].astype(str).str.strip()
    df["followers"] = pd.to_numeric(df["followers"], errors="coerce").fillna(0)
    df["percent_on_platform"] = pd.to_numeric(df["percent_on_platform"], errors="coerce").fillna(0)
    df = _expand_social_rows(df)
    df["country_norm"] = df["geography"].map(lambda value: COUNTRY_NORMALIZE.get(normalize_text(value), value))
    df["county_norm"] = df["geography"].map(lambda value: normalize_text(value).title().replace(" ", "-"))
    return df[df["platform"].isin(PLATFORM_SOURCES)].copy()


@st.cache_data
def load_social_media_demographics(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = df.rename(
        columns={
            "Age": "age",
            "Percent on platform": "pct_on_platform",
            "Total followers in age group": "total_in_age",
            "Men (percent)": "men_pct",
            "Women (percent)": "women_pct",
            "Men (absolute value)": "men_abs",
            "Women (absolute value)": "women_abs",
            "Platform": "platform",
        }
    )
    required = {"age", "pct_on_platform", "total_in_age", "men_pct", "women_pct", "men_abs", "women_abs", "platform"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing social-media demographics columns: {', '.join(sorted(missing))}")
    df["platform"] = df["platform"].astype(str).str.strip()
    df["age"] = df["age"].astype(str).str.strip()
    for col in ["pct_on_platform", "total_in_age", "men_pct", "women_pct", "men_abs", "women_abs"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["pct_on_platform"] = _fraction(df["pct_on_platform"])
    df["men_pct"] = _fraction(df["men_pct"])
    df["women_pct"] = _fraction(df["women_pct"])
    return df[df["platform"].isin(PLATFORM_SOURCES)].copy()


def social_platform_geo(df: pd.DataFrame, platform: str, level: str) -> pd.DataFrame:
    work = df[(df["platform"].eq(platform)) & (df["level"].eq(level))].copy()
    if level == "country":
        group_col = "country_norm"
        work = work[~work[group_col].map(normalize_text).isin({"alte tari", "benelux"})]
    else:
        group_col = "county_norm"
    if work.empty:
        return pd.DataFrame(columns=[group_col, "followers", "percentage"])
    out = work.groupby(group_col, as_index=False).agg({"followers": "sum", "percent_on_platform": "sum"})
    out["percentage"] = _fraction(out["percent_on_platform"])
    if out["percentage"].fillna(0).sum() <= 0:
        total = out["followers"].sum()
        out["percentage"] = out["followers"] / total if total else 0
    out["platform"] = platform
    return out.sort_values("followers", ascending=False).reset_index(drop=True)


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    def multiselect_filter(label: str, col: str):
        values = sorted([v for v in work[col].dropna().astype(str).unique() if v and v != "nan"])
        selected = st.multiselect(label, values)
        return selected

    filter_cols = st.columns(3)
    with filter_cols[0]:
        gender = multiselect_filter("Gender", "Gen")
        if gender:
            work = work[work["Gen"].astype(str).isin(gender)]

        country = multiselect_filter("Country", "Țară de reședință")
        if country:
            work = work[work["Țară de reședință"].astype(str).isin(country)]

    with filter_cols[1]:
        age_values = [str(v) for v in work["age_band"].dropna().astype(str).unique()]
        age_order = ["0-13", "14-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
        age_values = [v for v in age_order if v in age_values]
        age = st.multiselect("Age band", age_values)
        if age:
            work = work[work["age_band"].astype(str).isin(age)]

        county = multiselect_filter("County", "Județ atribuit")
        if county:
            work = work[work["Județ atribuit"].astype(str).isin(county)]

    with filter_cols[2]:
        region = multiselect_filter("Region", "Regiune atribuită")
        if region:
            work = work[work["Regiune atribuită"].astype(str).isin(region)]

        if "Mediu atribuit" in work.columns:
            mediu = multiselect_filter("Urban / rural", "Mediu atribuit")
            if mediu:
                work = work[work["Mediu atribuit"].astype(str).isin(mediu)]

        tenure = multiselect_filter("Supporter tenure", "De cât timp ești suporter Dinamo?")
        if tenure:
            work = work[work["De cât timp ești suporter Dinamo?"].astype(str).isin(tenure)]

    st.metric("Respondents after filters", f"{len(work):,}")
    return work


def percent_count(df: pd.DataFrame, col: str, order: list[str] | None = None) -> pd.DataFrame:
    counts = df[col].dropna().astype(str).str.strip()
    counts = counts[counts.ne("") & counts.ne("nan")]
    out = counts.value_counts().reset_index()
    out.columns = [col, "count"]
    total = out["count"].sum()
    out["percentage"] = out["count"] / total if total else 0
    if order:
        out[col] = pd.Categorical(out[col], categories=order, ordered=True)
        out = out.sort_values(col)
    return out


def bar_count(
    df: pd.DataFrame,
    col: str,
    title: str,
    order: list[str] | None = None,
    horizontal: bool = False,
    fixed_width: bool = True,
):
    data = percent_count(df, col, order=order)
    if data.empty:
        st.info("No data for the current filters.")
        return
    data["percentage_label"] = data["percentage"].map(lambda value: f"{value:.1%}")
    if horizontal:
        plot_data = data.sort_values("percentage")
        fig = px.bar(
            plot_data,
            x="percentage",
            y=col,
            orientation="h",
            text="percentage_label",
            title=title,
            custom_data=["count", "percentage"],
        )
        fig.update_traces(hovertemplate="%{y}<br>Count: %{customdata[0]}<br>Percentage: %{customdata[1]:.1%}<extra></extra>")
    else:
        fig = px.bar(
            data,
            x=col,
            y="percentage",
            text="percentage_label",
            title=title,
            custom_data=["count", "percentage"],
        )
        fig.update_traces(hovertemplate="%{x}<br>Count: %{customdata[0]}<br>Percentage: %{customdata[1]:.1%}<extra></extra>")
    fig.update_traces(marker_color=DINAMO_RED, textposition="outside")
    chart_height = horizontal_chart_height(len(data)) if horizontal else VERTICAL_CHART_HEIGHT
    axis_tickformat = {"xaxis_tickformat": ".0%"} if horizontal else {"yaxis_tickformat": ".0%"}
    fig.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, t=50, b=0),
        height=chart_height,
        **axis_tickformat,
    )
    if fixed_width:
        render_bar_chart(fig)
    else:
        st.plotly_chart(fig, use_container_width=True)


def donut(df: pd.DataFrame, col: str, title: str):
    data = percent_count(df, col)
    if data.empty:
        st.info("No data for the current filters.")
        return
    fig = px.pie(
        data,
        names=col,
        values="count",
        hole=0.55,
        title=title,
        color_discrete_sequence=[DINAMO_RED, BLACK, "#bbbbbb", DARK_RED],
    )
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0), height=VERTICAL_CHART_HEIGHT)
    st.plotly_chart(fig, use_container_width=True)


def pie_count(
    df: pd.DataFrame,
    col: str,
    title: str,
    order: list[str] | None = None,
    top_n: int | None = None,
):
    data = percent_count(df, col, order=order)
    if data.empty:
        st.info("No data for the current filters.")
        return

    if top_n and len(data) > top_n:
        top = data.head(top_n).copy()
        other_count = int(data.iloc[top_n:]["count"].sum())
        total = int(data["count"].sum())
        other = pd.DataFrame([{col: "Other", "count": other_count, "percentage": other_count / total if total else 0}])
        data = pd.concat([top, other], ignore_index=True)

    fig = px.pie(
        data,
        names=col,
        values="count",
        title=title,
        color_discrete_sequence=PIE_COLORS,
        custom_data=["count", "percentage"],
    )
    fig.update_traces(
        textinfo="label+percent",
        textposition="outside",
        hovertemplate="%{label}<br>Count: %{customdata[0]}<br>Percentage: %{customdata[1]:.1%}<extra></extra>",
        marker=dict(line=dict(color="white", width=1)),
        automargin=True,
    )
    fig.update_layout(
        showlegend=False,
        margin=dict(l=10, r=10, t=50, b=10),
        height=VERTICAL_CHART_HEIGHT,
        uniformtext_minsize=10,
        uniformtext_mode="hide",
    )
    st.plotly_chart(fig, use_container_width=True)


def country_romania_other_pie(df: pd.DataFrame):
    countries = df["country_norm"].dropna().astype(str).str.strip()
    countries = countries[countries.ne("") & countries.ne("nan")]
    if countries.empty:
        st.info("No country data for the current filters.")
        return

    romania_count = int(countries.eq("Romania").sum())
    other_count = int((~countries.eq("Romania")).sum())
    total = romania_count + other_count
    data = pd.DataFrame(
        [
            {"Country group": "România", "count": romania_count, "percentage": romania_count / total if total else 0},
            {"Country group": "Others", "count": other_count, "percentage": other_count / total if total else 0},
        ]
    )
    fig = px.pie(
        data,
        names="Country group",
        values="count",
        title="România vs Others",
        color="Country group",
        color_discrete_map={"România": DINAMO_RED, "Others": "#dddddd"},
        custom_data=["count", "percentage"],
    )
    fig.update_traces(
        textinfo="label+percent",
        textposition="outside",
        hovertemplate="%{label}<br>Count: %{customdata[0]}<br>Percentage: %{customdata[1]:.1%}<extra></extra>",
        marker=dict(line=dict(color="white", width=1)),
        automargin=True,
    )
    fig.update_layout(
        showlegend=False,
        margin=dict(l=10, r=10, t=50, b=10),
        height=VERTICAL_CHART_HEIGHT,
        uniformtext_minsize=10,
        uniformtext_mode="hide",
    )
    st.plotly_chart(fig, use_container_width=True)


def other_countries_pie(df: pd.DataFrame):
    country_series = df["country_norm"].fillna("").astype(str).str.strip()
    other_df = df[country_series.ne("") & country_series.ne("nan") & country_series.ne("Romania")]
    data = percent_count(other_df, "country_norm").head(10)
    if data.empty:
        st.info("No other country data for the current filters.")
        return

    total = int(data["count"].sum())
    data["percentage"] = data["count"] / total if total else 0
    fig = px.pie(
        data,
        names="country_norm",
        values="count",
        title="Top 10 other countries",
        color_discrete_sequence=PIE_COLORS,
        custom_data=["count", "percentage"],
    )
    fig.update_traces(
        textinfo="label+percent",
        textposition="outside",
        hovertemplate="%{label}<br>Count: %{customdata[0]}<br>Percentage: %{customdata[1]:.1%}<extra></extra>",
        marker=dict(line=dict(color="white", width=1)),
        automargin=True,
    )
    fig.update_layout(
        showlegend=False,
        margin=dict(l=10, r=10, t=50, b=10),
        height=VERTICAL_CHART_HEIGHT,
        uniformtext_minsize=10,
        uniformtext_mode="hide",
    )
    st.plotly_chart(fig, use_container_width=True)


def top_bar(df: pd.DataFrame, col: str, title: str, n: int = 20):
    data = percent_count(df, col).head(n)
    if data.empty:
        st.info("No data for the current filters.")
        return
    data["percentage_label"] = data["percentage"].map(lambda value: f"{value:.1%}")
    fig = px.bar(
        data.sort_values("percentage"),
        x="percentage",
        y=col,
        orientation="h",
        text="percentage_label",
        title=title,
        custom_data=["count", "percentage"],
    )
    fig.update_traces(hovertemplate="%{y}<br>Count: %{customdata[0]}<br>Percentage: %{customdata[1]:.1%}<extra></extra>")
    fig.update_traces(marker_color=DINAMO_RED, textposition="outside")
    fig.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, t=50, b=0),
        height=horizontal_chart_height(len(data), minimum=500, per_row=30),
        xaxis_tickformat=".0%",
    )
    render_bar_chart(fig)


def split_multiselect_counts(df: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for value in df[col].dropna().astype(str):
        for item in value.split("|"):
            item = item.strip()
            if item:
                rows.append(item)
    if not rows:
        return pd.DataFrame(columns=[col, "count", "percentage"])
    out = pd.Series(rows).value_counts().reset_index()
    out.columns = [col, "count"]
    out["percentage"] = out["count"] / len(df) if len(df) else 0
    return out


def bar_from_counts(data: pd.DataFrame, label_col: str, title: str, horizontal: bool = True, fixed_width: bool = True):
    if data.empty:
        st.info("No data for the current filters.")
        return
    plot_data = data.copy()
    plot_data["percentage_label"] = plot_data["percentage"].map(lambda value: f"{value:.1%}")
    if horizontal:
        plot_data = plot_data.sort_values("percentage")
        fig = px.bar(
            plot_data,
            x="percentage",
            y=label_col,
            orientation="h",
            text="percentage_label",
            title=title,
            custom_data=["count", "percentage"],
        )
        fig.update_traces(hovertemplate="%{y}<br>Count: %{customdata[0]}<br>Percentage: %{customdata[1]:.1%}<extra></extra>")
    else:
        fig = px.bar(
            plot_data,
            x=label_col,
            y="percentage",
            text="percentage_label",
            title=title,
            custom_data=["count", "percentage"],
        )
        fig.update_traces(hovertemplate="%{x}<br>Count: %{customdata[0]}<br>Percentage: %{customdata[1]:.1%}<extra></extra>")
    fig.update_traces(marker_color=DINAMO_RED, textposition="outside")
    chart_height = horizontal_chart_height(len(plot_data)) if horizontal else VERTICAL_CHART_HEIGHT
    axis_tickformat = {"xaxis_tickformat": ".0%"} if horizontal else {"yaxis_tickformat": ".0%"}
    fig.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, t=50, b=0),
        height=chart_height,
        **axis_tickformat,
    )
    if fixed_width:
        render_bar_chart(fig)
    else:
        st.plotly_chart(fig, use_container_width=True)


def analysis_count(value: object) -> int:
    if pd.isna(value):
        return 0
    if isinstance(value, (int, float)):
        number = float(value)
        if 0 < number < 100 and not math.isclose(number, round(number)):
            return int(round(number * 1000))
        return int(round(number))

    text = str(value).strip()
    if not text or "%" in text:
        return 0
    text = re.sub(r"\s+", "", text)
    if re.fullmatch(r"\d{1,3}(\.\d{3})+", text):
        return int(text.replace(".", ""))
    try:
        number = float(text.replace(",", "."))
    except ValueError:
        return 0
    if 0 < number < 100 and not math.isclose(number, round(number)):
        return int(round(number * 1000))
    return int(round(number))


def analysis_percentage(value: object) -> float:
    if pd.isna(value):
        return 0.0
    if isinstance(value, (int, float)):
        number = float(value)
        return number if 0 <= number <= 1 else number / 100

    text = str(value).strip().replace(" ", "")
    if not text:
        return 0.0
    has_percent = "%" in text
    text = text.replace("%", "").replace(",", ".")
    try:
        number = float(text)
    except ValueError:
        return 0.0
    return number / 100 if has_percent or number > 1 else number


def extract_analysis_explanation(sheet: pd.DataFrame, include_named_columns: bool = False) -> str:
    texts = []
    for col in sheet.columns:
        if not include_named_columns and not str(col).startswith("Unnamed"):
            continue
        for value in sheet[col].dropna():
            text = str(value).strip()
            if len(text) < 50:
                continue
            if text not in texts:
                texts.append(text)
    return "\n\n".join(texts)


def analysis_text_box(text: str):
    if text:
        st.info(text)


def keyword_table(summaries: dict[str, pd.DataFrame], prefix: str) -> pd.DataFrame:
    sheet = summaries.get(f"{prefix}_keywords", pd.DataFrame())
    if sheet.empty:
        return pd.DataFrame(columns=["keyword normalizat", "frequency agregat", "percentage"])
    keyword_col = sheet.columns[5] if len(sheet.columns) > 6 else sheet.columns[0]
    count_col = sheet.columns[6] if len(sheet.columns) > 6 else sheet.columns[1]
    out = sheet[[keyword_col, count_col]].copy()
    out.columns = ["keyword normalizat", "frequency agregat"]
    out["keyword normalizat"] = out["keyword normalizat"].fillna("").astype(str).str.strip()
    out["frequency agregat"] = out["frequency agregat"].map(analysis_count)
    out = out[out["keyword normalizat"].ne("") & out["frequency agregat"].gt(0)]
    out = out.groupby("keyword normalizat", as_index=False)["frequency agregat"].sum()
    total = out["frequency agregat"].sum()
    out["percentage"] = out["frequency agregat"] / total if total else 0
    return out.sort_values("frequency agregat", ascending=False).reset_index(drop=True)


def bigram_table(summaries: dict[str, pd.DataFrame], prefix: str) -> pd.DataFrame:
    sheet = summaries.get(f"{prefix}_bigrams", pd.DataFrame())
    if sheet.empty:
        return pd.DataFrame(columns=["bigram", "count", "percentage"])
    term_col = "term" if "term" in sheet.columns else sheet.columns[0]
    count_col = "frequency" if "frequency" in sheet.columns else sheet.columns[1]
    out = sheet[[term_col, count_col]].copy()
    out.columns = ["bigram", "count"]
    out["bigram"] = out["bigram"].fillna("").astype(str).str.strip()
    out["count"] = out["count"].map(analysis_count)
    out = out[out["bigram"].ne("") & out["count"].gt(0)]
    out = out.groupby("bigram", as_index=False)["count"].sum()
    total = out["count"].sum()
    out["percentage"] = out["count"] / total if total else 0
    return out.sort_values("count", ascending=False).reset_index(drop=True)


def bigram_theme_table(summaries: dict[str, pd.DataFrame], prefix: str) -> pd.DataFrame:
    sheet = summaries.get(f"{prefix}_bigrams", pd.DataFrame())
    empty = pd.DataFrame(columns=["theme", "subthemes", "count", "percentage"])
    if sheet.empty:
        return empty

    if "Tema principală" in sheet.columns and "Total mențiuni" in sheet.columns:
        out = sheet[["Tema principală", "Sub-teme incluse", "Total mențiuni"]].copy()
        out.columns = ["theme", "subthemes", "count"]
        out["subthemes"] = out["subthemes"].fillna("").astype(str).str.strip()
    else:
        theme_col = "Topic Definition.1" if "Topic Definition.1" in sheet.columns else None
        count_col = "Responses.1" if "Responses.1" in sheet.columns else None
        if not theme_col or not count_col:
            return empty
        cols = [theme_col, count_col]
        pct_col = "% din teme identificate" if "% din teme identificate" in sheet.columns else None
        if pct_col:
            cols.append(pct_col)
        out = sheet[cols].copy()
        out.columns = ["theme", "count"] + (["percentage"] if pct_col else [])
        out["subthemes"] = ""

    out["theme"] = out["theme"].fillna("").astype(str).str.strip()
    out["count"] = out["count"].map(analysis_count)
    if "percentage" in out.columns:
        out["percentage"] = out["percentage"].map(analysis_percentage)
    else:
        total = out["count"].sum()
        out["percentage"] = out["count"] / total if total else 0
    out = out[out["theme"].ne("") & out["count"].gt(0)]
    return out[["theme", "subthemes", "count", "percentage"]].sort_values("count", ascending=False).reset_index(drop=True)


def topic_table(summaries: dict[str, pd.DataFrame], prefix: str) -> pd.DataFrame:
    sheet = summaries.get(f"{prefix}_topics", pd.DataFrame())
    if sheet.empty:
        return pd.DataFrame(columns=["topic", "count", "percentage"])
    topic_col = "Topic Definition.1" if "Topic Definition.1" in sheet.columns else "Topic Definition"
    count_col = "Responses.1" if "Responses.1" in sheet.columns else "Responses"
    pct_col = "%" if "%" in sheet.columns else None
    out = sheet[[topic_col, count_col] + ([pct_col] if pct_col else [])].copy()
    out.columns = ["topic", "count"] + (["percentage"] if pct_col else [])
    out["topic"] = out["topic"].fillna("").astype(str).str.strip()
    out["count"] = out["count"].map(analysis_count)
    if "percentage" in out.columns:
        out["percentage"] = out["percentage"].map(analysis_percentage)
    else:
        total = out["count"].sum()
        out["percentage"] = out["count"] / total if total else 0
    out = out[out["topic"].ne("") & out["count"].gt(0)]
    return out.sort_values("count", ascending=False).reset_index(drop=True)


def results_tables(summaries: dict[str, pd.DataFrame], prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    sheet = summaries.get(f"{prefix}_classified_RESULTS", pd.DataFrame())
    empty = pd.DataFrame(columns=["label", "count", "percentage"])
    if sheet.empty or len(sheet.columns) < 3:
        return empty, empty.copy()

    sentiment = sheet.iloc[:, :3].copy()
    sentiment.columns = ["label", "count", "percentage"]
    sentiment["label"] = sentiment["label"].fillna("").astype(str).str.strip()
    sentiment["count"] = sentiment["count"].map(analysis_count)
    sentiment["percentage"] = sentiment["percentage"].map(analysis_percentage)
    sentiment = sentiment[sentiment["label"].ne("") & sentiment["count"].gt(0)]

    tone_cols = [idx for idx, col in enumerate(sheet.columns) if normalize_text(col) == "ton"]
    if tone_cols and tone_cols[0] + 2 < len(sheet.columns):
        idx = tone_cols[0]
        tone = sheet.iloc[:, idx:idx + 3].copy()
        tone.columns = ["label", "count", "percentage"]
        tone["label"] = tone["label"].fillna("").astype(str).str.strip()
        tone["count"] = tone["count"].map(analysis_count)
        tone["percentage"] = tone["percentage"].map(analysis_percentage)
        tone = tone[tone["label"].ne("") & tone["count"].gt(0)]
    else:
        tone = empty.copy()

    return sentiment.reset_index(drop=True), tone.reset_index(drop=True)


def classified_table(summaries: dict[str, pd.DataFrame], prefix: str) -> pd.DataFrame:
    sheet = summaries.get(f"{prefix}_classified", pd.DataFrame()).copy()
    if sheet.empty:
        return sheet
    first_col = sheet.columns[0]
    sheet = sheet[sheet[first_col].notna() & sheet[first_col].astype(str).str.strip().ne("")]
    renamed = {}
    for col in sheet.columns:
        col_text = str(col)
        if col == first_col:
            renamed[col] = "Answer"
        elif col_text.endswith("__sentiment_score"):
            renamed[col] = "Sentiment score"
        elif col_text.endswith("__sentiment"):
            renamed[col] = "Sentiment"
        elif col_text.endswith("__emotion"):
            renamed[col] = "Tone / emotion"
        elif col_text.endswith("__word_count"):
            renamed[col] = "Word count"
    return sheet.rename(columns=renamed).reset_index(drop=True)


def analysis_count_bar(data: pd.DataFrame, label_col: str, count_col: str, title: str, top_n: int | None = None):
    if data.empty:
        st.info("No data in this analysis table.")
        return
    plot_data = data.copy()
    if top_n:
        plot_data = plot_data.head(top_n)
    plot_data["count_label"] = plot_data[count_col].map(lambda value: f"{int(value):,}")
    fig = px.bar(
        plot_data.sort_values(count_col),
        x=count_col,
        y=label_col,
        orientation="h",
        text="count_label",
        title=title,
        custom_data=[count_col, "percentage"] if "percentage" in plot_data.columns else [count_col],
    )
    if "percentage" in plot_data.columns:
        fig.update_traces(hovertemplate="%{y}<br>Count: %{customdata[0]:,.0f}<br>Share: %{customdata[1]:.1%}<extra></extra>")
    else:
        fig.update_traces(hovertemplate="%{y}<br>Count: %{customdata[0]:,.0f}<extra></extra>")
    fig.update_traces(marker_color=DINAMO_RED, textposition="outside")
    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=50, b=0), height=horizontal_chart_height(len(plot_data)))
    st.plotly_chart(fig, use_container_width=True)


def analysis_percentage_bar(data: pd.DataFrame, label_col: str, title: str):
    if data.empty:
        st.info("No data in this analysis table.")
        return
    plot_data = data.copy()
    plot_data["percentage_label"] = plot_data["percentage"].map(lambda value: f"{value:.1%}")
    fig = px.bar(
        plot_data.sort_values("percentage"),
        x="percentage",
        y=label_col,
        orientation="h",
        text="percentage_label",
        title=title,
        custom_data=["count", "percentage"],
    )
    fig.update_traces(
        marker_color=DINAMO_RED,
        textposition="outside",
        hovertemplate="%{y}<br>Count: %{customdata[0]:,.0f}<br>Percentage: %{customdata[1]:.1%}<extra></extra>",
    )
    fig.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, t=50, b=0),
        height=horizontal_chart_height(len(plot_data)),
        xaxis_tickformat=".0%",
    )
    st.plotly_chart(fig, use_container_width=True)


def open_answer_analysis(summaries: dict[str, pd.DataFrame], prefix: str, title: str):
    required = [f"{prefix}_{suffix}" for suffix in ["keywords", "bigrams", "topics", "classified", "classified_RESULTS"]]
    missing = [sheet for sheet in required if sheet not in summaries]
    if missing:
        st.error(
            "The custom analysis sheets are not loaded yet. Refresh the app after the latest workbook update. "
            f"Missing sheets: {', '.join(missing)}"
        )
        return

    st.caption("This open-answer analysis uses the full survey analysis sheets and does not change with sidebar filters.")

    keywords = keyword_table(summaries, prefix)
    keyword_sheet = summaries.get(f"{prefix}_keywords", pd.DataFrame())
    st.markdown("#### Keyword signals")
    left, right = st.columns([2, 1])
    with left:
        analysis_count_bar(keywords, "keyword normalizat", "frequency agregat", "Top normalized keywords", top_n=20)
        with st.expander("Keyword table", expanded=False):
            st.dataframe(keywords, use_container_width=True, hide_index=True)
    with right:
        analysis_text_box(extract_analysis_explanation(keyword_sheet))

    bigrams = bigram_table(summaries, prefix)
    bigram_themes = bigram_theme_table(summaries, prefix)
    bigram_sheet = summaries.get(f"{prefix}_bigrams", pd.DataFrame())
    st.markdown("#### Bigram signals")
    left, right = st.columns([2, 1])
    with left:
        analysis_count_bar(bigrams, "bigram", "count", "Top bigrams", top_n=20)
        with st.expander("Bigram table", expanded=False):
            st.dataframe(bigrams, use_container_width=True, hide_index=True)
        if not bigram_themes.empty:
            with st.expander("Bigram theme aggregation table", expanded=False):
                st.dataframe(bigram_themes, use_container_width=True, hide_index=True)
    with right:
        analysis_text_box(extract_analysis_explanation(bigram_sheet))

    topics = topic_table(summaries, prefix)
    topic_sheet = summaries.get(f"{prefix}_topics", pd.DataFrame())
    st.markdown("#### Topics")
    left, right = st.columns([2, 1])
    with left:
        analysis_percentage_bar(topics.rename(columns={"topic": "label"}), "label", "Topic distribution")
        with st.expander("Topic table", expanded=False):
            st.dataframe(topics, use_container_width=True, hide_index=True)
    with right:
        analysis_text_box(extract_analysis_explanation(topic_sheet))

    sentiment_table, tone_table = results_tables(summaries, prefix)
    results_sheet = summaries.get(f"{prefix}_classified_RESULTS", pd.DataFrame())
    st.markdown("#### Sentiment and tone")
    chart_col1, chart_col2, text_col = st.columns([1.2, 1.2, 1])
    with chart_col1:
        analysis_percentage_bar(sentiment_table, "label", "Sentiment")
    with chart_col2:
        analysis_percentage_bar(tone_table, "label", "Tone")
    with text_col:
        analysis_text_box(extract_analysis_explanation(results_sheet, include_named_columns=True))

    classified = classified_table(summaries, prefix)
    with st.expander("Raw classified answers", expanded=False):
        query = st.text_input("Search answers", key=f"{prefix}_classified_search")
        view = classified
        if query and not classified.empty:
            mask = classified.astype(str).apply(lambda col: col.str.contains(query, case=False, na=False, regex=False)).any(axis=1)
            view = classified[mask]
        st.dataframe(view, use_container_width=True, hide_index=True)


def logo_mention_counts(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    mentioned = df["Siglă menționată"].fillna("").astype(str).str.strip().eq("Da").sum()
    data = pd.DataFrame(
        {
            "Siglă menționată": ["Da", "Nu"],
            "count": [mentioned, total - mentioned],
        }
    )
    data["percentage"] = data["count"] / total if total else 0
    return data


def logo_sentiment_counts(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    order = ["sigla_pozitiv", "sigla_negativ", "sigla_mixt"]
    counts = df["Siglă sentiment"].dropna().astype(str).str.strip().value_counts()
    data = pd.DataFrame({"Siglă sentiment": order, "count": [int(counts.get(item, 0)) for item in order]})
    data["percentage"] = data["count"] / total if total else 0
    return data


def romania_county_map(df: pd.DataFrame):
    geojson = load_geojson(RO_GEOJSON_PATH)
    county_rows = []
    lookup = {}
    for feature in geojson.get("features", []):
        name = feature.get("properties", {}).get("name", "")
        lookup[normalize_text(name)] = name
        county_rows.append({"geojson_name": name, "map_county": name})

    counts = percent_count(df, "Județ atribuit")
    base = pd.DataFrame(county_rows)
    if not counts.empty:
        counts["geojson_name"] = counts["Județ atribuit"].map(lambda x: lookup.get(normalize_text(x)))
        counts = counts[counts["geojson_name"].notna()].copy()
        base = base.merge(counts[["geojson_name", "Județ atribuit", "count"]], on="geojson_name", how="left")
    else:
        base["Județ atribuit"] = None
        base["count"] = 0
    base["Județ atribuit"] = base["Județ atribuit"].fillna(base["map_county"])
    base["count"] = base["count"].fillna(0).astype(int)
    total = base["count"].sum()
    base["share"] = base["count"] / total if total else 0
    base["color_value"] = base["count"].map(lambda value: math.log10(value + 1))
    max_color = float(base["color_value"].max())
    fig = px.choropleth(
        base,
        geojson=geojson,
        locations="geojson_name",
        featureidkey="properties.name",
        color="color_value",
        hover_name="Județ atribuit",
        hover_data={"count": True, "share": ":.1%", "color_value": False, "geojson_name": False},
        color_continuous_scale=RED_SCALE,
        range_color=(0, max_color if max_color > 0 else 1),
        title="Respondents by county",
    )
    fig.update_traces(marker_line_color="#b8b8b8", marker_line_width=0.6)
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0), height=MAP_CHART_HEIGHT, coloraxis_colorbar_title="Log respondents")
    st.plotly_chart(fig, use_container_width=True)


def world_country_map(df: pd.DataFrame):
    counts = percent_count(df, "country_norm")
    if counts.empty:
        st.info("No country data for the current filters.")
        return
    counts["iso3"] = counts["country_norm"].map(ISO3)
    mappable = counts[counts["iso3"].notna()].copy()
    if mappable.empty:
        st.info("No countries matched the world map.")
        return
    mappable["share"] = mappable["count"] / mappable["count"].sum()
    mappable["color_value"] = mappable["count"].map(lambda value: math.log10(value + 1))
    fig = px.choropleth(
        mappable,
        locations="iso3",
        color="color_value",
        hover_name="country_norm",
        hover_data={"count": True, "share": ":.1%", "color_value": False, "iso3": False},
        color_continuous_scale=RED_SCALE,
        title="Respondents by country",
    )
    fig.update_geos(
        showcoastlines=True,
        coastlinecolor="#b8b8b8",
        showcountries=True,
        countrycolor="#b8b8b8",
        showland=True,
        landcolor="#f7f7f7",
        showocean=True,
        oceancolor="#ffffff",
        lonaxis_range=(-170, 45),
        lataxis_range=(10, 75),
    )
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0), height=MAP_CHART_HEIGHT, coloraxis_colorbar_title="Log respondents")
    st.plotly_chart(fig, use_container_width=True)


def social_country_data(df: pd.DataFrame, platform: str) -> pd.DataFrame:
    countries = social_platform_geo(df, platform, "country")
    if countries["country_norm"].eq("Romania").any():
        return countries
    counties = social_platform_geo(df, platform, "county")
    if counties.empty:
        return countries
    romania = pd.DataFrame(
        [
            {
                "country_norm": "Romania",
                "followers": counties["followers"].sum(),
                "percentage": counties["percentage"].sum(),
                "platform": platform,
            }
        ]
    )
    return pd.concat([countries, romania], ignore_index=True).sort_values("followers", ascending=False)


def social_metric_label(platform: str) -> str:
    if platform == "Merchandise":
        return "RON spent"
    if platform == "Fan Youtube (RD1948)":
        return "Views"
    return "Followers"


def social_log_metric_label(platform: str) -> str:
    if platform == "Merchandise":
        return "Log RON spent"
    if platform == "Fan Youtube (RD1948)":
        return "Log views"
    return "Log followers"


def social_top_bar(data: pd.DataFrame, label_col: str, title: str):
    if data.empty:
        st.info("No data for the current source.")
        return
    platform = data["platform"].iloc[0] if "platform" in data.columns and not data.empty else ""
    metric_label = social_metric_label(platform)
    plot_data = data.head(20).copy()
    plot_data["percentage_label"] = plot_data["percentage"].map(lambda value: f"{value:.1%}")
    fig = px.bar(
        plot_data.sort_values("percentage"),
        x="percentage",
        y=label_col,
        orientation="h",
        text="percentage_label",
        title=title,
        custom_data=["followers", "percentage"],
    )
    fig.update_traces(
        marker_color=DINAMO_RED,
        textposition="outside",
        hovertemplate=f"%{{y}}<br>{metric_label}: %{{customdata[0]:,.0f}}<br>Percentage: %{{customdata[1]:.1%}}<extra></extra>",
    )
    fig.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, t=50, b=0),
        height=horizontal_chart_height(len(plot_data), minimum=500, per_row=30),
        xaxis_tickformat=".0%",
        yaxis_title="",
    )
    render_bar_chart(fig)


def social_world_map(df: pd.DataFrame, platform: str):
    data = social_country_data(df, platform)
    if data.empty:
        st.info("No country data for this source.")
        return
    metric_label = social_metric_label(platform)
    data["iso3"] = data["country_norm"].map(ISO3)
    mappable = data[data["iso3"].notna()].copy()
    if mappable.empty:
        st.info("No countries matched the world map.")
        return
    mappable["color_value"] = mappable["followers"].map(lambda value: math.log10(value + 1))
    fig = px.choropleth(
        mappable,
        locations="iso3",
        color="color_value",
        hover_name="country_norm",
        custom_data=["followers", "percentage"],
        hover_data={"color_value": False, "iso3": False},
        color_continuous_scale=RED_SCALE,
        title=f"{metric_label} by country - {platform}",
    )
    fig.update_traces(
        hovertemplate=f"<b>%{{hovertext}}</b><br>{metric_label}: %{{customdata[0]:,.0f}}<br>Percentage: %{{customdata[1]:.1%}}<extra></extra>"
    )
    fig.update_geos(
        showcoastlines=True,
        coastlinecolor="#b8b8b8",
        showcountries=True,
        countrycolor="#b8b8b8",
        showland=True,
        landcolor="#f7f7f7",
        showocean=True,
        oceancolor="#ffffff",
        lonaxis_range=(-170, 45),
        lataxis_range=(10, 75),
    )
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0), height=MAP_CHART_HEIGHT, coloraxis_colorbar_title=social_log_metric_label(platform))
    st.plotly_chart(fig, use_container_width=True)


def social_county_map(df: pd.DataFrame, platform: str):
    geojson = load_geojson(RO_GEOJSON_PATH)
    metric_label = social_metric_label(platform)
    county_rows = []
    lookup = {}
    for feature in geojson.get("features", []):
        name = feature.get("properties", {}).get("name", "")
        lookup[normalize_text(name)] = name
        county_rows.append({"geojson_name": name, "county_norm": name})
    data = social_platform_geo(df, platform, "county")
    base = pd.DataFrame(county_rows)
    if not data.empty:
        data["geojson_name"] = data["county_norm"].map(lambda value: lookup.get(normalize_text(value)))
        mappable = data[data["geojson_name"].notna()].copy()
        base = base.merge(mappable[["geojson_name", "followers", "percentage"]], on="geojson_name", how="left")
    else:
        data["geojson_name"] = None
        base["followers"] = 0
        base["percentage"] = 0
    base["followers"] = base["followers"].fillna(0)
    base["percentage"] = base["percentage"].fillna(0)
    base["color_value"] = base["followers"].map(lambda value: math.log10(value + 1))
    max_color = float(base["color_value"].max())
    fig = px.choropleth(
        base,
        geojson=geojson,
        locations="geojson_name",
        featureidkey="properties.name",
        color="color_value",
        hover_name="county_norm",
        custom_data=["followers", "percentage"],
        hover_data={"color_value": False, "geojson_name": False},
        color_continuous_scale=RED_SCALE,
        range_color=(0, max_color if max_color > 0 else 1),
        title=f"{metric_label} by county/region - {platform}",
    )
    fig.update_traces(
        hovertemplate=f"<b>%{{hovertext}}</b><br>{metric_label}: %{{customdata[0]:,.0f}}<br>Percentage: %{{customdata[1]:.1%}}<extra></extra>",
        marker_line_color="#b8b8b8",
        marker_line_width=0.6,
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0), height=MAP_CHART_HEIGHT, coloraxis_colorbar_title=social_log_metric_label(platform))
    st.plotly_chart(fig, use_container_width=True)
    unmapped = data[data["geojson_name"].isna()].copy()
    if not unmapped.empty:
        with st.expander("Unmapped county/region labels"):
            st.dataframe(unmapped[["county_norm", "followers", "percentage"]], use_container_width=True, hide_index=True)


def social_age_data(demo_df: pd.DataFrame, platform: str) -> pd.DataFrame:
    view = demo_df[demo_df["platform"].eq(platform)].copy()
    if view.empty:
        return view
    age_order = ["13-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    view["age"] = pd.Categorical(view["age"], categories=age_order, ordered=True)
    view = view.sort_values("age")
    if view["pct_on_platform"].fillna(0).sum() <= 0:
        total = view["total_in_age"].sum()
        view["pct_on_platform"] = view["total_in_age"] / total if total else 0
    return view


def social_age_bar(demo_df: pd.DataFrame, platform: str):
    data = social_age_data(demo_df, platform)
    if data.empty:
        st.info(f"No age demographic data available for {platform}.")
        return
    metric_label = social_metric_label(platform)
    data["percentage_label"] = data["pct_on_platform"].map(lambda value: f"{value:.1%}")
    fig = px.bar(
        data,
        x="age",
        y="pct_on_platform",
        text="percentage_label",
        title=f"Age distribution - {platform}",
        custom_data=["total_in_age", "pct_on_platform"],
    )
    fig.update_traces(
        marker_color=DINAMO_RED,
        textposition="outside",
        hovertemplate=f"%{{x}}<br>{metric_label}: %{{customdata[0]:,.0f}}<br>Percentage: %{{customdata[1]:.1%}}<extra></extra>",
    )
    fig.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, t=50, b=0),
        height=VERTICAL_CHART_HEIGHT,
        yaxis_tickformat=".0%",
        xaxis_title="Age",
        yaxis_title="Percentage",
    )
    st.plotly_chart(fig, use_container_width=True)


def social_age_pie(demo_df: pd.DataFrame, platform: str):
    data = social_age_data(demo_df, platform)
    if data.empty:
        st.info(f"No age demographic data available for {platform}.")
        return
    metric_label = social_metric_label(platform)
    fig = px.pie(
        data,
        names="age",
        values="total_in_age",
        title=f"Age share - {platform}",
        color_discrete_sequence=PIE_COLORS,
        custom_data=["total_in_age", "pct_on_platform"],
    )
    fig.update_traces(
        textinfo="label+percent",
        textposition="outside",
        hovertemplate=f"%{{label}}<br>{metric_label}: %{{customdata[0]:,.0f}}<br>Percentage: %{{customdata[1]:.1%}}<extra></extra>",
        marker=dict(line=dict(color="white", width=1)),
        automargin=True,
    )
    fig.update_layout(
        showlegend=False,
        margin=dict(l=10, r=10, t=50, b=10),
        height=VERTICAL_CHART_HEIGHT,
        uniformtext_minsize=10,
        uniformtext_mode="hide",
    )
    st.plotly_chart(fig, use_container_width=True)


def social_sex_data(demo_df: pd.DataFrame, platform: str) -> pd.DataFrame:
    view = demo_df[demo_df["platform"].eq(platform)].copy()
    if view.empty:
        return pd.DataFrame(columns=["Sex", "followers", "percentage"])
    data = pd.DataFrame(
        [
            {"Sex": "Men", "followers": view["men_abs"].sum()},
            {"Sex": "Women", "followers": view["women_abs"].sum()},
        ]
    )
    total = data["followers"].sum()
    data["percentage"] = data["followers"] / total if total else 0
    return data


def social_sex_charts(demo_df: pd.DataFrame, platform: str):
    data = social_sex_data(demo_df, platform)
    if data.empty or data["followers"].sum() <= 0:
        st.info(f"No sex demographic data available for {platform}.")
        return
    metric_label = social_metric_label(platform)
    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(
            data,
            names="Sex",
            values="followers",
            hole=0.55,
            title=f"Sex distribution - {platform}",
            color="Sex",
            color_discrete_map={"Men": DINAMO_RED, "Women": "#dddddd"},
            custom_data=["followers", "percentage"],
        )
        fig.update_traces(
            textinfo="label+percent",
            hovertemplate=f"%{{label}}<br>{metric_label}: %{{customdata[0]:,.0f}}<br>Percentage: %{{customdata[1]:.1%}}<extra></extra>",
        )
        fig.update_layout(margin=dict(l=0, r=0, t=50, b=0), height=VERTICAL_CHART_HEIGHT)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        data["percentage_label"] = data["percentage"].map(lambda value: f"{value:.1%}")
        fig = px.bar(
            data,
            x="Sex",
            y="percentage",
            text="percentage_label",
            title=f"Sex share - {platform}",
            custom_data=["followers", "percentage"],
        )
        fig.update_traces(
            marker_color=[DINAMO_RED, "#dddddd"],
            textposition="outside",
            hovertemplate=f"%{{x}}<br>{metric_label}: %{{customdata[0]:,.0f}}<br>Percentage: %{{customdata[1]:.1%}}<extra></extra>",
        )
        fig.update_layout(
            showlegend=False,
            margin=dict(l=0, r=0, t=50, b=0),
            height=VERTICAL_CHART_HEIGHT,
            yaxis_tickformat=".0%",
            yaxis_title="Percentage",
        )
        st.plotly_chart(fig, use_container_width=True)


def platform_demographics(demo_df: pd.DataFrame, geo_df: pd.DataFrame, platform: str):
    age_data = social_age_data(demo_df, platform)
    sex_data = social_sex_data(demo_df, platform)
    country_data = social_country_data(geo_df, platform)
    county_data = social_platform_geo(geo_df, platform, "county")

    tab_specs = []
    if not age_data.empty and age_data["total_in_age"].sum() > 0:
        tab_specs.append(("Age", "age"))
    if not sex_data.empty and sex_data["followers"].sum() > 0:
        tab_specs.append(("Sex", "sex"))
    if not country_data.empty and country_data["followers"].sum() > 0:
        tab_specs.append(("Countries", "countries"))
    if not county_data.empty and county_data["followers"].sum() > 0:
        tab_specs.append(("Counties/Regions", "counties"))

    if not tab_specs:
        st.info(f"No demographic data available for {platform}.")
        return

    tabs = st.tabs([label for label, _ in tab_specs])
    for tab, (_, key) in zip(tabs, tab_specs):
        with tab:
            if key == "age":
                col1, col2 = st.columns(2)
                with col1:
                    social_age_bar(demo_df, platform)
                with col2:
                    social_age_pie(demo_df, platform)
            elif key == "sex":
                social_sex_charts(demo_df, platform)
            elif key == "countries":
                social_world_map(geo_df, platform)
                social_top_bar(country_data, "country_norm", "Top countries")
            elif key == "counties":
                _, map_col, _ = st.columns([1, 2, 1])
                with map_col:
                    social_county_map(geo_df, platform)
                social_top_bar(county_data, "county_norm", "Top counties/regions")


def platform_club(geo_df: pd.DataFrame, platform: str):
    tabs = st.tabs(["Countries", "Counties/Regions"])
    with tabs[0]:
        social_world_map(geo_df, platform)
        social_top_bar(social_country_data(geo_df, platform), "country_norm", "Top countries")
    with tabs[1]:
        _, map_col, _ = st.columns([1, 2, 1])
        with map_col:
            social_county_map(geo_df, platform)
        social_top_bar(social_platform_geo(geo_df, platform, "county"), "county_norm", "Top counties/regions")


def interactive_word_frequency(df: pd.DataFrame):
    data = percent_count(df, "Dinamo un cuvânt - normalizat").head(80)
    if data.empty:
        st.info("No words for the current filters.")
        return
    fig = px.treemap(
        data,
        path=["Dinamo un cuvânt - normalizat"],
        values="count",
        color="count",
        color_continuous_scale=RED_SCALE,
        title="Interactive word frequency map",
        custom_data=["count", "percentage"],
    )
    fig.update_traces(
        texttemplate="<b>%{label}</b><br>%{customdata[1]:.1%}",
        hovertemplate="<b>%{label}</b><br>Count: %{customdata[0]}<br>Percentage: %{customdata[1]:.1%}<extra></extra>",
    )
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0), height=TREEMAP_CHART_HEIGHT, coloraxis_colorbar_title="Count")
    st.plotly_chart(fig, use_container_width=True)


def word_cloud(df: pd.DataFrame):
    words = df["Dinamo un cuvânt - normalizat"].dropna().astype(str).str.strip()
    words = words[words.ne("") & words.ne("nan")]
    frequencies = words.value_counts().to_dict()
    if not frequencies:
        st.info("No words for the current filters.")
        return
    wc = WordCloud(
        width=1400,
        height=800,
        background_color="white",
        colormap="Reds",
        prefer_horizontal=0.95,
        max_words=250,
        collocations=False,
    ).generate_from_frequencies(frequencies)
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    left, right = st.columns([1, 1])
    with left:
        st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def ordered_likert_chart(df: pd.DataFrame, col: str, title: str, order: list[str], fixed_width: bool = True):
    bar_count(df, col, title, order=order, fixed_width=fixed_width)


def demographics(df: pd.DataFrame):
    tabs = st.tabs(["Age", "Gender", "County", "Region", "Urban / rural", "Country", "Education", "Children", "Children at matches"])
    age_order = ["0-13", "14-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    education_order = ["Școală generală", "Liceu", "Studii postliceale / Școală postliceală", "Studii universitare", "Studii postuniversitare", "Prefer să nu răspund"]
    children_match_order = ["Frecvent", "Uneori", "Rar", "Niciodată", "Copiii sunt prea mici momentan"]
    with tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x="age_numeric", nbins=30, title="Age distribution")
            fig.update_traces(marker_color=DINAMO_RED)
            fig.update_layout(margin=dict(l=0, r=0, t=50, b=0), height=VERTICAL_CHART_HEIGHT, xaxis_title="Age", yaxis_title="Respondents")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            bar_count(df, "age_band", "Age bands", order=age_order, fixed_width=False)
        pie_count(df, "age_band", "Age band share", order=age_order)
    with tabs[1]:
        col1, col2 = st.columns(2)
        with col1:
            donut(df, "Gen", "Gender")
        with col2:
            pie_count(df, "Gen", "Gender share")
    with tabs[2]:
        _, map_col, _ = st.columns([1, 2, 1])
        with map_col:
            romania_county_map(df)
        col1, col2 = st.columns(2)
        with col1:
            top_bar(df, "Județ atribuit", "Top counties")
        with col2:
            pie_count(df, "Județ atribuit", "County share", top_n=12)
    with tabs[3]:
        col1, col2 = st.columns(2)
        with col1:
            top_bar(df, "Regiune atribuită", "Regions", n=10)
        with col2:
            pie_count(df, "Regiune atribuită", "Region share")
    with tabs[4]:
        if "Mediu atribuit" in df.columns:
            col1, col2 = st.columns(2)
            with col1:
                donut(df, "Mediu atribuit", "Urban / rural")
            with col2:
                pie_count(df, "Mediu atribuit", "Urban / rural share")
        else:
            st.info("No urban/rural data in the workbook.")
    with tabs[5]:
        world_country_map(df)
        top_bar(df, "Țară de reședință", "Top countries", n=20)
        col1, col2 = st.columns(2)
        with col1:
            country_romania_other_pie(df)
        with col2:
            other_countries_pie(df)
    with tabs[6]:
        col1, col2 = st.columns(2)
        with col1:
            ordered_likert_chart(df, "Educație", "Education", education_order)
        with col2:
            pie_count(df, "Educație", "Education share", order=education_order)
    with tabs[7]:
        col1, col2 = st.columns(2)
        with col1:
            donut(df, "Ai copii?", "Children")
        with col2:
            pie_count(df, "Ai copii?", "Children share")
    with tabs[8]:
        child_df = df[df["Ai copii?"].astype(str).str.lower().eq("da")]
        col1, col2 = st.columns(2)
        with col1:
            ordered_likert_chart(child_df, "Vii cu copiii la meci?", "Children at matches", children_match_order)
        with col2:
            pie_count(child_df, "Vii cu copiii la meci?", "Children at matches share", order=children_match_order)


def sentiment(df: pd.DataFrame, summaries: dict[str, pd.DataFrame]):
    tabs = st.tabs([
        "Emotional connection",
        "One-word emotion",
        "Supporter tenure",
        "What Dinamo does well",
        "What fans would change",
        "One message for Dinamo",
        "Final suggestions",
    ])
    with tabs[0]:
        score = pd.to_numeric(df["Cât de conectat te simți emoțional cu Dinamo?"], errors="coerce")
        c1, c2 = st.columns(2)
        c1.metric("Mean", f"{score.mean():.2f}" if score.notna().any() else "N/A")
        c2.metric("Median", f"{score.median():.0f}" if score.notna().any() else "N/A")
        ordered_likert_chart(df, "Cât de conectat te simți emoțional cu Dinamo?", "Emotional connection", ["1", "2", "3", "4", "5"])
    with tabs[1]:
        bar_count(df, "Dinamo un cuvânt - categorie", "One-word emotion categories", horizontal=True)
        top_bar(df, "Dinamo un cuvânt - normalizat", "Top normalized words/phrases", n=25)
        interactive_word_frequency(df)
        word_cloud(df)
    with tabs[2]:
        supporter_tenure_order = ["Am început recent", "De câțiva ani", "De peste 10 ani", "De mic, am crescut cu Dinamo"]
        col1, col2 = st.columns(2)
        with col1:
            ordered_likert_chart(
                df,
                "De cât timp ești suporter Dinamo?",
                "Supporter tenure",
                supporter_tenure_order,
                fixed_width=False,
            )
        with col2:
            pie_count(df, "De cât timp ești suporter Dinamo?", "Supporter tenure share", order=supporter_tenure_order)
    with tabs[3]:
        open_answer_analysis(summaries, "col_1", "What Dinamo does well")
    with tabs[4]:
        open_answer_analysis(summaries, "col_2", "What fans would change")
    with tabs[5]:
        open_answer_analysis(summaries, "col_3", "One message for Dinamo")
    with tabs[6]:
        open_answer_analysis(summaries, "col_4", "Final suggestions")


def club(df: pd.DataFrame):
    tabs = st.tabs([
        "Off-field evaluation",
        "Logo / identity",
        "Season ticket drivers",
        "Brand conflict sensitivity",
        "Sponsor purchase lift",
    ])
    with tabs[0]:
        off_field_order = ["În regres vizibil", "În ușor regres", "Stabil", "În ușoară creștere", "În creștere vizibilă", "Nu am o părere formată"]
        col1, col2 = st.columns(2)
        with col1:
            ordered_likert_chart(
                df,
                "Cum evaluezi clubul Dinamo în afara terenului, în ultima perioadă?",
                "Off-field evaluation",
                off_field_order,
                fixed_width=False,
            )
        with col2:
            pie_count(df, "Cum evaluezi clubul Dinamo în afara terenului, în ultima perioadă?", "Off-field evaluation share", order=off_field_order)
    with tabs[1]:
        col1, col2 = st.columns(2)
        with col1:
            mention_data = logo_mention_counts(df)
            fig = px.pie(
                mention_data,
                names="Siglă menționată",
                values="count",
                hole=0.55,
                title="Logo mentioned, % of all respondents",
                color="Siglă menționată",
                color_discrete_map={"Da": DINAMO_RED, "Nu": "#dddddd"},
                custom_data=["count", "percentage"],
            )
            fig.update_traces(
                textinfo="percent+label",
                hovertemplate="%{label}<br>Count: %{customdata[0]}<br>Percentage: %{customdata[1]:.1%}<extra></extra>",
            )
            fig.update_layout(margin=dict(l=0, r=0, t=50, b=0), height=VERTICAL_CHART_HEIGHT)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            sentiment_data = logo_sentiment_counts(df)
            bar_from_counts(sentiment_data, "Siglă sentiment", "Logo sentiment, % of all respondents", fixed_width=False)
        sources = []
        for value in df["Coloană mențiune siglă"].dropna().astype(str):
            for source in value.split(";"):
                source = source.strip()
                if source:
                    sources.append(source)
        if sources:
            source_df = pd.DataFrame({"Source column": sources})
            top_bar(source_df, "Source column", "Where logo was mentioned", n=15)
    with tabs[2]:
        data = split_multiselect_counts(df, "Ce te-ar determina să îți faci abonament pentru sezonul viitor?")
        if data.empty:
            st.info("No season ticket driver data.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                data["percentage_label"] = data["percentage"].map(lambda value: f"{value:.1%}")
                fig = px.bar(
                    data.sort_values("percentage"),
                    x="percentage",
                    y="Ce te-ar determina să îți faci abonament pentru sezonul viitor?",
                    orientation="h",
                    text="percentage_label",
                    title="Season ticket drivers",
                    custom_data=["count", "percentage"],
                )
                fig.update_traces(hovertemplate="%{y}<br>Count: %{customdata[0]}<br>Percentage: %{customdata[1]:.1%}<extra></extra>")
                fig.update_traces(marker_color=DINAMO_RED, textposition="outside")
                fig.update_layout(
                    margin=dict(l=0, r=0, t=50, b=0),
                    height=horizontal_chart_height(len(data)),
                    xaxis_tickformat=".0%",
                    yaxis_title="",
                )
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                pie_fig = px.pie(
                    data,
                    names="Ce te-ar determina să îți faci abonament pentru sezonul viitor?",
                    values="count",
                    title="Season ticket drivers share",
                    color_discrete_sequence=PIE_COLORS,
                    custom_data=["count", "percentage"],
                )
                pie_fig.update_traces(
                    textinfo="label+percent",
                    textposition="outside",
                    hovertemplate="%{label}<br>Count: %{customdata[0]}<br>Respondent percentage: %{customdata[1]:.1%}<extra></extra>",
                    marker=dict(line=dict(color="white", width=1)),
                    automargin=True,
                )
                pie_fig.update_layout(
                    showlegend=False,
                    margin=dict(l=10, r=10, t=50, b=10),
                    height=VERTICAL_CHART_HEIGHT,
                    uniformtext_minsize=10,
                    uniformtext_mode="hide",
                )
                st.plotly_chart(pie_fig, use_container_width=True)
    with tabs[3]:
        brand_order = ["Deloc", "Puțin", "Destul de mult", "Foarte mult"]
        col1, col2 = st.columns(2)
        with col1:
            ordered_likert_chart(df, "Cât de mult contează pentru tine dacă un brand pe care îl cumperi se asociază cu un alt club de fotbal?", "Brand conflict sensitivity", brand_order, fixed_width=False)
        with col2:
            pie_count(df, "Cât de mult contează pentru tine dacă un brand pe care îl cumperi se asociază cu un alt club de fotbal?", "Brand conflict sensitivity share", order=brand_order)
    with tabs[4]:
        brand_order = ["Deloc", "Puțin", "Destul de mult", "Foarte mult"]
        col1, col2 = st.columns(2)
        with col1:
            ordered_likert_chart(df, "Dacă un brand sponsorizează Dinamo, cât de mult îți crește intenția de cumpărare?", "Sponsor purchase lift", brand_order, fixed_width=False)
        with col2:
            pie_count(df, "Dacă un brand sponsorizează Dinamo, cât de mult îți crește intenția de cumpărare?", "Sponsor purchase lift share", order=brand_order)


def main():
    st.markdown(
        """
        <style>
        .stApp { background: white; color: #111111; }
        h1, h2, h3 { color: #111111; }
        [data-testid="stMetricValue"] { color: #e30613; }
        section[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #eeeeee;
            width: 190px !important;
            min-width: 190px !important;
            max-width: 190px !important;
        }
        section[data-testid="stSidebar"] > div {
            padding-top: 1.25rem;
            display: flex;
            flex-direction: column;
            min-height: 100vh;
            width: 190px !important;
        }
        .sidebar-brand {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 0.6rem;
            width: 148px;
            padding: 1rem 0 0.25rem;
            margin: auto auto 0;
            border-top: 1px solid #eeeeee;
        }
        .sidebar-brand img {
            width: 84px;
            height: 84px;
            object-fit: contain;
            flex: 0 0 auto;
        }
        .sidebar-brand-title {
            color: #111111;
            font-size: 1.05rem;
            font-weight: 800;
            letter-spacing: 0;
            line-height: 1.1;
            text-align: center;
        }
        section[data-testid="stSidebar"] [data-testid="stRadio"] {
            width: 100%;
            display: flex;
            justify-content: center;
        }
        section[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 0.5rem;
            width: 148px;
            margin: 0 auto;
        }
        section[data-testid="stSidebar"] [data-testid="stRadio"] label {
            width: 148px;
            min-height: 44px;
            padding: 0.62rem 0.78rem;
            border: 1px solid #e5e5e5;
            border-radius: 8px;
            background: #ffffff;
            color: #111111;
            transition: background 120ms ease, border-color 120ms ease, color 120ms ease;
        }
        section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
            background: #fff1f2;
            border-color: #f3a3a8;
        }
        section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
            background: #e30613;
            border-color: #e30613;
            color: #ffffff;
            box-shadow: 0 8px 18px rgba(227, 6, 19, 0.18);
        }
        section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) * {
            color: #ffffff !important;
        }
        section[data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child {
            display: none;
        }
        section[data-testid="stSidebar"] [data-testid="stRadio"] p {
            font-weight: 700;
            font-size: 0.95rem;
            text-align: center;
            width: 100%;
        }
        section[data-testid="stSidebar"] [data-testid="stSelectbox"] {
            width: 148px;
            margin: 1rem auto 1rem;
        }
        section[data-testid="stSidebar"] [data-testid="stSelectbox"] p {
            font-weight: 700;
            color: #111111;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    current_source = st.session_state.get("source_selector", SURVEY_SOURCE)
    if current_source not in SOURCE_OPTIONS:
        current_source = SURVEY_SOURCE
    menu_options = SURVEY_MENUS if current_source == SURVEY_SOURCE else PLATFORM_MENUS

    menu = st.sidebar.radio(
        "Dashboard section",
        menu_options,
        label_visibility="collapsed",
    )
    source = st.sidebar.selectbox(
        "Source",
        SOURCE_OPTIONS,
        index=SOURCE_OPTIONS.index(current_source),
        key="source_selector",
    )
    logo_uri = image_data_uri(LOGO_PATH, "image/x-icon")
    st.sidebar.markdown(
        f"""
        <div class="sidebar-brand">
            <img src="{logo_uri}" alt="Dinamo Data Analysis logo">
            <div class="sidebar-brand-title">Fan Analytics</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if source == SURVEY_SOURCE:
        workbook_stat = DATA_PATH.stat()
        df, summaries = load_workbook(DATA_PATH, workbook_stat.st_mtime_ns, workbook_stat.st_size)
        df = prepare_data(df)

        st.title("Dinamo Fan Survey Dashboard")
        st.subheader(menu)
        with st.expander("Filters", expanded=False):
            filtered = apply_filters(df)

        st.caption(f"Showing {len(filtered):,} of {len(df):,} respondents")
        if menu == "Demographics":
            demographics(filtered)
        elif menu == "Sentiment":
            sentiment(filtered, summaries)
        else:
            club(filtered)
    else:
        geo_df = load_social_media_geo(SOCIAL_MEDIA_GEO_PATH)
        demo_df = load_social_media_demographics(SOCIAL_MEDIA_DEMO_PATH)

        st.title("Dinamo Fan Analytics")
        st.subheader(f"{source} - {menu}")
        if menu == "Demographics":
            platform_demographics(demo_df, geo_df, source)
        else:
            platform_club(geo_df, source)


if __name__ == "__main__":
    main()
