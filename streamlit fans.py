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
RO_GEOJSON_PATH = BASE_DIR / "romania.geojson"
LOGO_PATH = BASE_DIR / "data" / "img" / "dinamo-data-analysis-red.ico"

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
def load_workbook(path: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
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
    lookup = {}
    for feature in geojson.get("features", []):
        name = feature.get("properties", {}).get("name", "")
        lookup[normalize_text(name)] = name

    counts = percent_count(df, "Județ atribuit")
    if counts.empty:
        st.info("No county data for the current filters.")
        return
    counts["geojson_name"] = counts["Județ atribuit"].map(lambda x: lookup.get(normalize_text(x)))
    mappable = counts[counts["geojson_name"].notna()].copy()
    if mappable.empty:
        st.info("No counties matched the Romania map.")
        return
    mappable["share"] = mappable["count"] / mappable["count"].sum()
    mappable["color_value"] = mappable["count"].map(lambda value: math.log10(value + 1))
    fig = px.choropleth(
        mappable,
        geojson=geojson,
        locations="geojson_name",
        featureidkey="properties.name",
        color="color_value",
        hover_name="Județ atribuit",
        hover_data={"count": True, "share": ":.1%", "color_value": False, "geojson_name": False},
        color_continuous_scale=RED_SCALE,
        title="Respondents by county",
    )
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
    fig.update_geos(showcoastlines=True, showcountries=True, lonaxis_range=(-170, 45), lataxis_range=(10, 75))
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0), height=MAP_CHART_HEIGHT, coloraxis_colorbar_title="Log respondents")
    st.plotly_chart(fig, use_container_width=True)


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
        col1, col2 = st.columns(2)
        with col1:
            top_bar(df, "Țară de reședință", "Top countries", n=20)
        with col2:
            pie_count(df, "country_norm", "Country share", top_n=12)
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


def sentiment(df: pd.DataFrame):
    tabs = st.tabs(["Emotional connection", "Off-field evaluation", "One-word emotion", "Message tone", "Message theme"])
    with tabs[0]:
        score = pd.to_numeric(df["Cât de conectat te simți emoțional cu Dinamo?"], errors="coerce")
        c1, c2 = st.columns(2)
        c1.metric("Mean", f"{score.mean():.2f}" if score.notna().any() else "N/A")
        c2.metric("Median", f"{score.median():.0f}" if score.notna().any() else "N/A")
        ordered_likert_chart(df, "Cât de conectat te simți emoțional cu Dinamo?", "Emotional connection", ["1", "2", "3", "4", "5"])
    with tabs[1]:
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
    with tabs[2]:
        bar_count(df, "Dinamo un cuvânt - categorie", "One-word emotion categories", horizontal=True)
        top_bar(df, "Dinamo un cuvânt - normalizat", "Top normalized words/phrases", n=25)
        interactive_word_frequency(df)
        word_cloud(df)
    with tabs[3]:
        bar_count(df, "Mesaj pentru Dinamo - ton", "Message tone")
    with tabs[4]:
        bar_count(df, "Mesaj pentru Dinamo - temă", "Message theme", horizontal=True)


def club(df: pd.DataFrame):
    tabs = st.tabs([
        "Supporter tenure",
        "What Dinamo does well",
        "What fans would change",
        "Logo / identity",
        "Season ticket drivers",
        "Brand conflict sensitivity",
        "Sponsor purchase lift",
        "Suggestions",
    ])
    with tabs[0]:
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
    with tabs[1]:
        bar_count(df, "Ce face Dinamo BINE - categorie", "What Dinamo does well", horizontal=True)
    with tabs[2]:
        bar_count(df, "Dacă ai putea schimba un singur lucru - categorie", "What fans would change", horizontal=True)
    with tabs[3]:
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
    with tabs[4]:
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
    with tabs[5]:
        brand_order = ["Deloc", "Puțin", "Destul de mult", "Foarte mult"]
        col1, col2 = st.columns(2)
        with col1:
            ordered_likert_chart(df, "Cât de mult contează pentru tine dacă un brand pe care îl cumperi se asociază cu un alt club de fotbal?", "Brand conflict sensitivity", brand_order, fixed_width=False)
        with col2:
            pie_count(df, "Cât de mult contează pentru tine dacă un brand pe care îl cumperi se asociază cu un alt club de fotbal?", "Brand conflict sensitivity share", order=brand_order)
    with tabs[6]:
        brand_order = ["Deloc", "Puțin", "Destul de mult", "Foarte mult"]
        col1, col2 = st.columns(2)
        with col1:
            ordered_likert_chart(df, "Dacă un brand sponsorizează Dinamo, cât de mult îți crește intenția de cumpărare?", "Sponsor purchase lift", brand_order, fixed_width=False)
        with col2:
            pie_count(df, "Dacă un brand sponsorizează Dinamo, cât de mult îți crește intenția de cumpărare?", "Sponsor purchase lift share", order=brand_order)
    with tabs[7]:
        bar_count(df, "Sugestie suplimentară - categorie", "Suggestions", horizontal=True)


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
        </style>
        """,
        unsafe_allow_html=True,
    )
    df, _ = load_workbook(DATA_PATH)
    df = prepare_data(df)

    menu = st.sidebar.radio(
        "Dashboard section",
        ["Demographics", "Sentiment", "Club"],
        label_visibility="collapsed",
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

    st.title("Dinamo Fan Survey Dashboard")
    st.subheader(menu)
    with st.expander("Filters", expanded=False):
        filtered = apply_filters(df)

    st.caption(f"Showing {len(filtered):,} of {len(df):,} respondents")
    if menu == "Demographics":
        demographics(filtered)
    elif menu == "Sentiment":
        sentiment(filtered)
    else:
        club(filtered)


if __name__ == "__main__":
    main()
