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
}


st.set_page_config(
    page_title="Dinamo Fan Survey Dashboard",
    page_icon="🔴",
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


def bar_count(df: pd.DataFrame, col: str, title: str, order: list[str] | None = None, horizontal: bool = False):
    data = percent_count(df, col, order=order)
    if data.empty:
        st.info("No data for the current filters.")
        return
    if horizontal:
        fig = px.bar(data.sort_values("count"), x="count", y=col, orientation="h", text="count", title=title)
    else:
        fig = px.bar(data, x=col, y="count", text="count", title=title)
    fig.update_traces(marker_color=DINAMO_RED, textposition="outside")
    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=50, b=0))
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
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
    st.plotly_chart(fig, use_container_width=True)


def top_bar(df: pd.DataFrame, col: str, title: str, n: int = 20):
    data = percent_count(df, col).head(n)
    if data.empty:
        st.info("No data for the current filters.")
        return
    fig = px.bar(data.sort_values("count"), x="count", y=col, orientation="h", text="count", title=title)
    fig.update_traces(marker_color=DINAMO_RED, textposition="outside")
    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=50, b=0))
    st.plotly_chart(fig, use_container_width=True)


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
    fig = px.choropleth(
        mappable,
        geojson=geojson,
        locations="geojson_name",
        featureidkey="properties.name",
        color="count",
        hover_name="Județ atribuit",
        hover_data={"count": True, "geojson_name": False},
        color_continuous_scale=RED_SCALE,
        title="Respondents by county",
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0), coloraxis_colorbar_title="Respondents")
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
    fig = px.choropleth(
        mappable,
        locations="iso3",
        color="count",
        hover_name="country_norm",
        hover_data={"count": True, "iso3": False},
        color_continuous_scale=RED_SCALE,
        title="Respondents by country",
    )
    fig.update_geos(showcoastlines=True, showcountries=True, lonaxis_range=(-170, 45), lataxis_range=(10, 75))
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0), coloraxis_colorbar_title="Respondents")
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
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def ordered_likert_chart(df: pd.DataFrame, col: str, title: str, order: list[str]):
    bar_count(df, col, title, order=order)


def heatmap_theme_tone(df: pd.DataFrame):
    table = pd.crosstab(df["Mesaj pentru Dinamo - temă"], df["Mesaj pentru Dinamo - ton"])
    if table.empty:
        st.info("No message data for the current filters.")
        return
    fig = px.imshow(
        table,
        text_auto=True,
        color_continuous_scale=RED_SCALE,
        title="Message theme x tone",
        aspect="auto",
    )
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
    st.plotly_chart(fig, use_container_width=True)


def demographics(df: pd.DataFrame):
    tabs = st.tabs(["Age", "Gender", "County", "Region", "Country", "Education", "Children", "Children at matches"])
    with tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x="age_numeric", nbins=30, title="Age distribution")
            fig.update_traces(marker_color=DINAMO_RED)
            fig.update_layout(margin=dict(l=0, r=0, t=50, b=0), xaxis_title="Age", yaxis_title="Respondents")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            bar_count(df, "age_band", "Age bands", order=["0-13", "14-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"])
    with tabs[1]:
        donut(df, "Gen", "Gender")
    with tabs[2]:
        romania_county_map(df)
        top_bar(df, "Județ atribuit", "Top counties")
    with tabs[3]:
        top_bar(df, "Regiune atribuită", "Regions", n=10)
    with tabs[4]:
        world_country_map(df)
        top_bar(df, "Țară de reședință", "Top countries", n=20)
    with tabs[5]:
        ordered_likert_chart(
            df,
            "Educație",
            "Education",
            ["Școală generală", "Liceu", "Studii postliceale / Școală postliceală", "Studii universitare", "Studii postuniversitare", "Prefer să nu răspund"],
        )
    with tabs[6]:
        donut(df, "Ai copii?", "Children")
    with tabs[7]:
        child_df = df[df["Ai copii?"].astype(str).str.lower().eq("da")]
        ordered_likert_chart(child_df, "Vii cu copiii la meci?", "Children at matches", ["Frecvent", "Uneori", "Rar", "Niciodată", "Copiii sunt prea mici momentan"])


def sentiment(df: pd.DataFrame):
    tabs = st.tabs(["Emotional connection", "Off-field evaluation", "One-word emotion", "Message tone", "Message theme", "Message theme x tone"])
    with tabs[0]:
        score = pd.to_numeric(df["Cât de conectat te simți emoțional cu Dinamo?"], errors="coerce")
        c1, c2 = st.columns(2)
        c1.metric("Mean", f"{score.mean():.2f}" if score.notna().any() else "N/A")
        c2.metric("Median", f"{score.median():.0f}" if score.notna().any() else "N/A")
        ordered_likert_chart(df, "Cât de conectat te simți emoțional cu Dinamo?", "Emotional connection", ["1", "2", "3", "4", "5"])
    with tabs[1]:
        ordered_likert_chart(
            df,
            "Cum evaluezi clubul Dinamo în afara terenului, în ultima perioadă?",
            "Off-field evaluation",
            ["În regres vizibil", "În ușor regres", "Stabil", "În ușoară creștere", "În creștere vizibilă", "Nu am o părere formată"],
        )
    with tabs[2]:
        bar_count(df, "Dinamo un cuvânt - categorie", "One-word emotion categories", horizontal=True)
        top_bar(df, "Dinamo un cuvânt - normalizat", "Top normalized words/phrases", n=25)
        word_cloud(df)
    with tabs[3]:
        bar_count(df, "Mesaj pentru Dinamo - ton", "Message tone")
    with tabs[4]:
        bar_count(df, "Mesaj pentru Dinamo - temă", "Message theme", horizontal=True)
    with tabs[5]:
        heatmap_theme_tone(df)


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
        ordered_likert_chart(
            df,
            "De cât timp ești suporter Dinamo?",
            "Supporter tenure",
            ["Am început recent", "De câțiva ani", "De peste 10 ani", "De mic, am crescut cu Dinamo"],
        )
    with tabs[1]:
        bar_count(df, "Ce face Dinamo BINE - categorie", "What Dinamo does well", horizontal=True)
    with tabs[2]:
        bar_count(df, "Dacă ai putea schimba un singur lucru - categorie", "What fans would change", horizontal=True)
    with tabs[3]:
        col1, col2 = st.columns(2)
        with col1:
            donut(df, "Siglă menționată", "Logo mentioned")
        with col2:
            bar_count(df[df["Siglă sentiment"].notna()], "Siglă sentiment", "Logo sentiment")
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
            fig = px.bar(data.sort_values("count"), x="count", y="Ce te-ar determina să îți faci abonament pentru sezonul viitor?", orientation="h", text="count", title="Season ticket drivers")
            fig.update_traces(marker_color=DINAMO_RED, textposition="outside")
            fig.update_layout(margin=dict(l=0, r=0, t=50, b=0), yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
    with tabs[5]:
        ordered_likert_chart(df, "Cât de mult contează pentru tine dacă un brand pe care îl cumperi se asociază cu un alt club de fotbal?", "Brand conflict sensitivity", ["Deloc", "Puțin", "Destul de mult", "Foarte mult"])
    with tabs[6]:
        ordered_likert_chart(df, "Dacă un brand sponsorizează Dinamo, cât de mult îți crește intenția de cumpărare?", "Sponsor purchase lift", ["Deloc", "Puțin", "Destul de mult", "Foarte mult"])
    with tabs[7]:
        bar_count(df, "Sugestie suplimentară - categorie", "Suggestions", horizontal=True)


def main():
    st.markdown(
        """
        <style>
        .stApp { background: white; color: #111111; }
        h1, h2, h3 { color: #111111; }
        [data-testid="stMetricValue"] { color: #e30613; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    df, _ = load_workbook(DATA_PATH)
    df = prepare_data(df)

    st.sidebar.title("Menu")
    menu = st.sidebar.radio(
        "Dashboard section",
        ["Demographics", "Sentiment", "Club"],
        label_visibility="collapsed",
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
