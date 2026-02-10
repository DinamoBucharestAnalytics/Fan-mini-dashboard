import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ============================
# HARD-CODED PATHS (EDIT HERE)
# ============================
# Put the FULL absolute path to your Excel file here.
# Example Windows:
# DATA_PATH = r"C:\Dinamo\SocialMediaMaps\Power BI source.xlsx"
# Example macOS:
# DATA_PATH = r"/Users/razvan/Desktop/SocialMediaMaps/Power BI source.xlsx"
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Power BI source.xlsx"
RO_GEOJSON_PATH = BASE_DIR / "romania.geojson"
DEMOGRAPHICS_PATH = BASE_DIR / "Age and sex.xlsx"
COLOR_CLIP_Q = 0.80
WORLD_LON_RANGE = (-170, 40)
WORLD_LAT_RANGE = (15, 75)
NORMAL_RED_SCALE = [
    (0.0, "#ffffff"),
    (1.0, "#ff0000"),
]
INTENSE_RED = "#a50f15"
STRONG_RED = "#ff0000"
COLOR_GAMMA = 0.6
BIN_COLORS = [
    "#ffffff",
    "#ffe6e6",
    "#ffcccc",
    "#ffb3b3",
    "#ff9999",
    "#ff8080",
    "#ff6666",
    "#ff4d4d",
    "#ff0000",
]
BIN_COUNT = 9
TOP_N_BARS = 20
PLATFORM_CLIP_Q = {
    "Club mobile app": 0.85,
    "Fan Facebook (RD1948)": 0.90,
    "Club Facebook": 0.90,
    "Fan Youtube (RD1948)": 0.95,
    "Merchandise": 0.95,
    "All platforms": 0.95,
}

def _safe_col(name: str) -> str:
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "")


def _rank_bins(values: pd.Series, bin_count: int) -> pd.Series:
    values = values.fillna(0)
    bins = pd.Series(0, index=values.index, dtype=int)
    pos = values > 0
    n = int(pos.sum())
    if n <= 1:
        return bins
    ranks = values[pos].rank(method="first").astype(int)
    bins[pos] = ((ranks - 1) * (bin_count - 1) // (n - 1)).clip(0, bin_count - 1).astype(int)
    return bins

# ----------------------------
# EDIT HERE (easy customization)
# ----------------------------

# 1) Country name normalization (RO -> EN) for better mapping
COUNTRY_NORMALIZE = {
    "Franţa": "France",
    "Germania": "Germany",
    "Irlanda": "Ireland",
    "Italia": "Italy",
    "Spania": "Spain",
    "România": "Romania",
    "R. Moldova": "Moldova",
    "Belgia": "Belgium",
}

# 2) Non-country aggregates that cannot be mapped to a single country polygon
NON_MAPPABLE_COUNTRIES = {"Alte ţări", "Alte Å£Äƒri", "Benelux"}

# 3) Romania "county/region" labels -> (lat, lon)
# Your dataset includes combined regions (e.g., "Cluj&Alba") and macro-regions (e.g., "Oltenia").
# This map uses representative centroids so it's interactive and easy to maintain.
RO_LOCATION_CENTROIDS = {
    "Bucureşti": (44.4268, 26.1025),
    "Cluj": (46.7712, 23.6236),
    "Alba": (46.0740, 23.5800),  # Alba Iulia
    "Cluj&Alba": ((46.7712 + 46.0740) / 2, (23.6236 + 23.5800) / 2),
    "Braşov": (45.6570, 25.6012),
    "Argeş": (44.8565, 24.8692),  # Pitești
    "Bacău": (46.5670, 26.9146),
    "Bihor": (47.0465, 21.9189),  # Oradea
    "Botoşani": (47.7486, 26.6690),
    "Buzău": (45.1517, 26.8236),
    "Constanţa": (44.1598, 28.6348),
    "Călăraşi": (44.2000, 27.3333),
    "Dâmboviţa": (44.9250, 25.4560),  # Târgoviște
    "Galaţi, Brăila": ((45.4353 + 45.2719) / 2, (28.0080 + 27.9575) / 2),
    "Hunedoara": (45.7670, 22.9200),  # Deva
    "Iaşi": (47.1585, 27.6014),
    "Ilfov": (44.5355, 26.2320),  # approx (Buftea area)
    "Maramureş": (47.6580, 23.5840),  # Baia Mare
    "Mureş": (46.5425, 24.5575),  # Târgu Mureș
    "Neamţ": (46.9270, 26.3700),  # Piatra Neamț
    "Oltenia": (44.3302, 23.7949),  # proxy centroid (Craiova)
    "Prahova": (44.9400, 26.0260),  # Ploiești
    "Sibiu": (45.7930, 24.1350),
    "Suceava": (47.6514, 26.2556),
    "Timişoara": (45.7489, 21.2087),
    "Tulcea": (45.1716, 28.7914),
    "Vaslui": (46.6400, 27.7276),
    "Arad": (46.1866, 21.3123),
}

# ----------------------------
# Helpers
# ----------------------------

def _normalize_ro_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    table = str.maketrans(
        {
            "ă": "a",
            "â": "a",
            "î": "i",
            "ș": "s",
            "ş": "s",
            "ț": "t",
            "ţ": "t",
            "Ă": "A",
            "Â": "A",
            "Î": "I",
            "Ș": "S",
            "Ş": "S",
            "Ț": "T",
            "Ţ": "T",
        }
    )
    return name.translate(table).strip().lower()


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)

    # Normalize column names from your file to stable internal names
    # Expected user columns:
    # Geography, Level, Platform, Followers, Percent on the platform
    col_map = {
        "geography": "geography",
        "level": "level",
        "platform": "platform",
        "followers": "followers",
        "percent on the platform": "percent_on_platform",
    }

    # Strip spaces from incoming columns
    df.columns = [c.strip() for c in df.columns]
    normalized_to_actual = {c.strip().lower(): c for c in df.columns}

    missing = [c for c in col_map.keys() if c not in normalized_to_actual]
    if missing:
        raise ValueError(
            "Missing expected columns in Excel: "
            + ", ".join(missing)
            + "\nFound columns: "
            + ", ".join(df.columns)
        )

    rename_map = {normalized_to_actual[k]: v for k, v in col_map.items()}
    df = df.rename(columns=rename_map)

    # Clean types/strings
    df["level"] = df["level"].astype(str).str.lower().str.strip()
    df["platform"] = df["platform"].astype(str).str.strip()
    df["geography"] = (
        df["geography"]
        .astype(str)
        .str.replace("\u00a0", " ", regex=False)
        .str.strip()
    )

    # Keep numeric as numeric (if Excel has commas, handle robustly)
    df["followers"] = pd.to_numeric(df["followers"], errors="coerce").fillna(0)

    if "percent_on_platform" in df.columns:
        df["percent_on_platform"] = pd.to_numeric(df["percent_on_platform"], errors="coerce")

    # Split aggregate country rows into component countries
    split_countries = {
        _normalize_ro_name("Benelux"): [
            ("Belgium", 0.50),
            ("Netherlands", 0.50),
        ],
    }

    expanded_rows = []
    for _, row in df.iterrows():
        if row["level"] != "country":
            expanded_rows.append(row)
            continue
        key = _normalize_ro_name(row["geography"])
        if key not in split_countries:
            expanded_rows.append(row)
            continue
        for name, frac in split_countries[key]:
            new_row = row.copy()
            new_row["geography"] = name
            new_row["followers"] = row["followers"] * frac
            if "percent_on_platform" in row:
                new_row["percent_on_platform"] = row["percent_on_platform"] * frac
            expanded_rows.append(new_row)

    df = pd.DataFrame(expanded_rows)

    # Split aggregate county rows into component counties
    split_map = {
        _normalize_ro_name("Timişoara"): [
            ("Timiş", 0.75),
            ("Caraş-Severin", 0.25),
        ],
        _normalize_ro_name("Cluj&Alba"): [
            ("Cluj", 0.66),
            ("Alba", 0.33),
        ],
        _normalize_ro_name("Cluj, Alba"): [
            ("Cluj", 0.66),
            ("Alba", 0.33),
        ],
        _normalize_ro_name("Galaţi, Brăila"): [
            ("Galaţi", 0.50),
            ("Brăila", 0.50),
        ],
        _normalize_ro_name("Oltenia"): [
            ("Dolj", 0.20),
            ("Olt", 0.20),
            ("Mehedinţi", 0.20),
            ("Vâlcea", 0.20),
            ("Gorj", 0.20),
        ],
    }

    expanded_rows = []
    for _, row in df.iterrows():
        if row["level"] != "county":
            expanded_rows.append(row)
            continue
        key = _normalize_ro_name(row["geography"])
        if key not in split_map:
            expanded_rows.append(row)
            continue
        for name, frac in split_map[key]:
            new_row = row.copy()
            new_row["geography"] = name
            new_row["followers"] = row["followers"] * frac
            if "percent_on_platform" in row:
                new_row["percent_on_platform"] = row["percent_on_platform"] * frac
            expanded_rows.append(new_row)

    df = pd.DataFrame(expanded_rows)

    # Country name normalization (only for countries)
    df["geography_norm"] = df["geography"]
    df.loc[df["level"] == "country", "geography_norm"] = (
        df.loc[df["level"] == "country", "geography"].replace(COUNTRY_NORMALIZE)
    )

    return df


@st.cache_data
def load_demographics(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    rename = {
        "Age": "age",
        "Percent on platform": "pct_on_platform",
        "Total followers in age group": "total_in_age",
        "Men (percent)": "men_pct",
        "Women (percent)": "women_pct",
        "Men (absolute value)": "men_abs",
        "Women (absolute value)": "women_abs",
        "Platform": "platform",
    }
    df = df.rename(columns=rename)
    df["platform"] = df["platform"].astype(str).str.strip()
    df["age"] = df["age"].astype(str).str.strip()
    for col in ["pct_on_platform", "total_in_age", "men_pct", "women_pct", "men_abs", "women_abs"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def to_iso3(country_name: str) -> str | None:
    """
    Convert country display name to ISO-3 code.
    Keep it explicit/hardcoded for stability and easy edits.
    Add new entries here as you see missing countries in the "Non-mappable rows" expander.
    """
    name = str(country_name).replace("\u00a0", " ").strip()
    manual = {
        "Romania": "ROU",
        "Moldova": "MDA",
        "United States": "USA",
        "United Kingdom": "GBR",
        "Ukraine": "UKR",
        "Netherlands": "NLD",
        "Belgium": "BEL",
        "Germany": "DEU",
        "France": "FRA",
        "Italy": "ITA",
        "Spain": "ESP",
        "Sweden": "SWE",
        "Ireland": "IRL",
        "Austria": "AUT",
        "Canada": "CAN",
    }

    if name in manual:
        return manual[name]

    aliases = {
        "Rusia": "Russia",
        "USA": "United States",
        "UK": "United Kingdom",
    }
    if name in aliases and aliases[name] in manual:
        return manual[aliases[name]]

    return None


@st.cache_data
def load_ro_geojson(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _add_romania_from_counties(work: pd.DataFrame, platform_name: str) -> pd.DataFrame:
    if work.empty:
        return work
    platform_mask = work["platform"] == platform_name
    if not platform_mask.any():
        return work
    if ((work["level"] == "country") & platform_mask & (work["geography_norm"] == "Romania")).any():
        return work
    county_mask = (work["level"] == "county") & platform_mask
    if not county_mask.any():
        return work
    total = float(work.loc[county_mask, "followers"].sum())
    romania_row = {
        "geography": "Romania",
        "geography_norm": "Romania",
        "level": "country",
        "platform": platform_name,
        "followers": total,
    }
    return pd.concat([work, pd.DataFrame([romania_row])], ignore_index=True)


def add_platform_breakdown(df_map: pd.DataFrame, df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Adds per-platform values and % of platform total to df_map.
    Percent is share of each platform's total across the current level.
    """
    platforms = sorted(df_all["platform"].unique().tolist())
    agg_map = {"followers": "sum"}
    if "percent_on_platform" in df_all.columns:
        agg_map["percent_on_platform"] = "sum"
    pivot = (
        df_all.groupby(["level", "geography_norm", "platform"], as_index=False)
        .agg(agg_map)
    )
    totals = (
        df_all.groupby(["level", "platform"], as_index=False)
        .agg({"followers": "sum"})
        .rename(columns={"followers": "platform_total"})
    )
    pivot = pivot.merge(totals, on=["level", "platform"], how="left")
    if "percent_on_platform" in pivot.columns:
        pct_src = pivot["percent_on_platform"].astype(float)
        if pct_src.max() > 1:
            pct_src = pct_src / 100.0
    else:
        pct_src = pd.Series(0.0, index=pivot.index)
    computed = np.where(
        pivot["platform_total"] > 0,
        pivot["followers"] / pivot["platform_total"],
        0.0,
    )
    pivot["platform_pct"] = np.where(pct_src > 0, pct_src, computed)
    wide_val = pivot.pivot_table(
        index=["level", "geography_norm"],
        columns="platform",
        values="followers",
        fill_value=0,
        aggfunc="sum",
    )
    wide_pct = pivot.pivot_table(
        index=["level", "geography_norm"],
        columns="platform",
        values="platform_pct",
        fill_value=0,
        aggfunc="max",
    )
    wide_val.columns = [f"val_{_safe_col(c)}" for c in wide_val.columns]
    wide_pct.columns = [f"pct_{_safe_col(c)}" for c in wide_pct.columns]
    wide = pd.concat([wide_val, wide_pct], axis=1).reset_index()
    return df_map.merge(wide, on=["level", "geography_norm"], how="left")

def world_choropleth(df_country: pd.DataFrame, title: str, value_label: str, hover_extra: list[str] | None = None):
    df_country = df_country.copy()
    df_country["iso3"] = df_country["geography_norm"].apply(to_iso3)
    df_country["color_value"] = np.log1p(df_country["followers"])

    mappable = df_country[df_country["iso3"].notna()].copy()
    non_mappable = df_country[df_country["iso3"].isna()][
        ["geography_norm", "platform", "followers"]
    ].copy()
    rank_src = mappable.sort_values("followers", ascending=False).copy()
    top1 = rank_src.head(1)
    others = mappable.drop(top1.index).copy()

    clip_q = PLATFORM_CLIP_Q.get(df_country["platform"].iloc[0] if len(df_country) else "All platforms", COLOR_CLIP_Q)
    vmax = float(others["color_value"].quantile(clip_q)) if len(others) else 0.0
    if vmax <= 0:
        vmax = float(others["color_value"].max() if len(others) else 0.0)
    clipped = others["color_value"].clip(0, vmax)
    if len(others) > 1:
        others["color_norm"] = clipped.rank(pct=True).pow(COLOR_GAMMA)
    else:
        others["color_norm"] = 1.0

    hover_data = {"platform": True, "iso3": False}
    if "Followers" in mappable.columns:
        hover_data["Followers"] = True
    elif "followers_int" in mappable.columns:
        hover_data["followers_int"] = True
    else:
        hover_data["followers"] = True
    if "Percent on platform" in mappable.columns:
        hover_data["Percent on platform"] = True
    if hover_extra:
        for col in hover_extra:
            hover_data[col] = True

    others["color_bin"] = _rank_bins(others["followers"], BIN_COUNT)
    n_bins = BIN_COUNT
    bin_labels = [str(i) for i in range(n_bins - 1, -1, -1)]
    others["color_bin"] = others["color_bin"].astype(str)
    others["color_bin"] = pd.Categorical(others["color_bin"], categories=bin_labels, ordered=True)
    color_map = {str(i): BIN_COLORS[i] for i in range(n_bins)}

    fig = px.choropleth(
        others,
        locations="iso3",
        color="color_bin",
        hover_name="geography_norm",
        hover_data=hover_data,
        color_discrete_map=color_map,
        color_discrete_sequence=BIN_COLORS[:n_bins],
        category_orders={"color_bin": bin_labels},
        title=title,
        labels={"color_bin": "Intensity"},
    )

    if len(top1) > 0:
        fig.add_trace(
            go.Choropleth(
                locations=top1["iso3"],
                z=[1] * len(top1),
                text=top1["geography_norm"],
                hovertemplate="<b>%{text}</b><br>Followers: %{customdata[0]}<br>Platform: %{customdata[1]}<extra></extra>",
                customdata=top1[["followers", "platform"]].to_numpy(),
                colorscale=[[0, INTENSE_RED], [1, INTENSE_RED]],
                showscale=False,
                name="Highest",
            )
        )
    fig.update_geos(
        showcoastlines=True,
        showcountries=True,
        lonaxis_range=WORLD_LON_RANGE,
        lataxis_range=WORLD_LAT_RANGE,
    )
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))

    return fig, non_mappable


def romania_choropleth(df_county: pd.DataFrame, title: str, value_label: str, hover_extra: list[str] | None = None):
    df_county = df_county.copy()
    df_county["color_value"] = np.log1p(df_county["followers"])

    geojson = load_ro_geojson(RO_GEOJSON_PATH)
    name_lookup = {}
    geojson_names = []
    for feature in geojson.get("features", []):
        props = feature.get("properties", {})
        raw_name = props.get("name", "")
        if raw_name:
            geojson_names.append(raw_name)
        norm = _normalize_ro_name(raw_name)
        if norm:
            name_lookup[norm] = raw_name

    df_county["geojson_name"] = df_county["geography_norm"].map(
        lambda x: name_lookup.get(_normalize_ro_name(x))
    )

    data_rows = df_county[df_county["geojson_name"].notna()].copy()
    non_mappable = df_county[df_county["geojson_name"].isna()][
        ["geography_norm", "platform", "followers"]
    ].copy()
    rank_src = data_rows.sort_values("followers", ascending=False).copy()
    top1 = rank_src.head(1)
    others_data = data_rows.drop(top1.index).copy()

    clip_q = PLATFORM_CLIP_Q.get(df_county["platform"].iloc[0] if len(df_county) else "All platforms", COLOR_CLIP_Q)
    vmax = float(others_data["color_value"].quantile(clip_q)) if len(others_data) else 0.0
    if vmax <= 0:
        vmax = float(others_data["color_value"].max() if len(others_data) else 0.0)
    clipped = others_data["color_value"].clip(0, vmax)
    if len(others_data) > 1:
        others_data["color_norm"] = clipped.rank(pct=True).pow(COLOR_GAMMA)
    else:
        others_data["color_norm"] = 1.0

    all_counties = pd.DataFrame({"geojson_name": sorted(set(geojson_names))})
    mappable = all_counties.merge(
        data_rows,
        on="geojson_name",
        how="left",
    )
    mappable["followers"] = mappable["followers"].fillna(0)
    mappable["color_value"] = mappable["color_value"].fillna(0)
    mappable["platform"] = mappable["platform"].fillna("No data")
    mappable["geography_norm"] = mappable["geography_norm"].fillna(mappable["geojson_name"])
    mappable["color_norm"] = 0.0
    if len(others_data) > 0:
        norm_map = (
            others_data.groupby("geojson_name", as_index=True)["color_norm"]
            .max()
        )
        mappable.loc[mappable["geojson_name"].isin(norm_map.index), "color_norm"] = (
            mappable.loc[mappable["geojson_name"].isin(norm_map.index), "geojson_name"]
            .map(norm_map)
            .fillna(0.0)
        )

    hover_data = {"platform": True}
    if "Followers" in mappable.columns:
        hover_data["Followers"] = True
    elif "followers_int" in mappable.columns:
        hover_data["followers_int"] = True
    else:
        hover_data["followers"] = True
    if "Percent on platform" in mappable.columns:
        hover_data["Percent on platform"] = True
    if hover_extra:
        for col in hover_extra:
            hover_data[col] = True

    mappable["color_bin"] = _rank_bins(mappable["followers"], BIN_COUNT)
    n_bins = BIN_COUNT
    bin_labels = [str(i) for i in range(n_bins - 1, -1, -1)]
    mappable["color_bin"] = mappable["color_bin"].astype(str)
    mappable["color_bin"] = pd.Categorical(mappable["color_bin"], categories=bin_labels, ordered=True)
    color_map = {str(i): BIN_COLORS[i] for i in range(n_bins)}

    fig = px.choropleth(
        mappable,
        geojson=geojson,
        locations="geojson_name",
        featureidkey="properties.name",
        color="color_bin",
        hover_name="geography_norm",
        hover_data=hover_data,
        color_discrete_map=color_map,
        color_discrete_sequence=BIN_COLORS[:n_bins],
        category_orders={"color_bin": bin_labels},
        title=title,
        labels={"color_bin": "Intensity"},
    )

    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))

    if len(top1) > 0:
        fig.add_trace(
            go.Choropleth(
                geojson=geojson,
                locations=top1["geojson_name"],
                featureidkey="properties.name",
                z=[1] * len(top1),
                text=top1["geography_norm"],
                hovertemplate="<b>%{text}</b><br>Followers: %{customdata[0]}<br>Platform: %{customdata[1]}<extra></extra>",
                customdata=top1[["followers", "platform"]].to_numpy(),
                colorscale=[[0, INTENSE_RED], [1, INTENSE_RED]],
                showscale=False,
                name="Highest",
            )
        )

    fig.update_traces(
        marker_line_color="#666666",
        marker_line_width=1.0,
        selector=dict(type="choropleth"),
    )

    return fig, non_mappable


def render_demographics():
    demo = load_demographics(DEMOGRAPHICS_PATH)
    demo_platforms = sorted(demo["platform"].unique().tolist())
    demo_choice = st.selectbox("Platform", demo_platforms, key="demo_platform")
    view = demo[demo["platform"] == demo_choice].copy()

    st.markdown("**Age Distribution**")
    c1, c2 = st.columns(2)
    with c1:
        fig_age_pct = px.bar(
            view,
            x="age",
            y="pct_on_platform",
            title="Age distribution (percent of platform)",
        )
        fig_age_pct.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
        fig_age_pct.update_layout(yaxis_title="Percent")
        st.plotly_chart(fig_age_pct, width="stretch")
    with c2:
        fig_age_abs = px.bar(
            view,
            x="age",
            y="total_in_age",
            title="Age distribution (absolute values)",
        )
        fig_age_abs.update_traces(texttemplate="%{y:.0f}", textposition="outside")
        fig_age_abs.update_layout(yaxis_title="Followers")
        st.plotly_chart(fig_age_abs, width="stretch")

    st.markdown("**Sex Distribution**")
    has_sex = view[["men_pct", "women_pct", "men_abs", "women_abs"]].notna().any().any()
    if not has_sex:
        st.info("No sex breakdown available for this platform.")
        return

    sex_pct = pd.DataFrame(
        {
            "Sex": ["Men", "Women"],
            "Percent": [view["men_pct"].mean(), view["women_pct"].mean()],
        }
    )
    sex_abs = pd.DataFrame(
        {
            "Sex": ["Men", "Women"],
            "Count": [view["men_abs"].sum(), view["women_abs"].sum()],
        }
    )
    c3, c4 = st.columns(2)
    with c3:
        fig_sex_pie = px.pie(sex_pct, names="Sex", values="Percent", title="Sex distribution (percent)")
        fig_sex_pie.update_traces(textinfo="percent+label")
        st.plotly_chart(fig_sex_pie, width="stretch")
    with c4:
        fig_sex_bar = px.bar(sex_abs, x="Sex", y="Count", title="Sex distribution (absolute)")
        fig_sex_bar.update_traces(texttemplate="%{y:.0f}", textposition="outside")
        fig_sex_bar.update_layout(yaxis_title="Followers")
        st.plotly_chart(fig_sex_bar, width="stretch")


def render_geography():

    # Load data (hardcoded)
    df = load_data(DATA_PATH)

    # Sidebar controls
    st.sidebar.header("Filters")

    platforms = sorted(df["platform"].unique().tolist())
    platform_choice = st.sidebar.selectbox("Platform", ["All platforms"] + platforms)
    highlight_platform = None
    if platform_choice == "All platforms":
        highlight_platform = st.sidebar.selectbox(
            "Highlight platform",
            ["(none)"] + platforms,
            index=0,
        )

    level_choice = st.sidebar.radio(
        "Geography level",
        ["Countries (World map)", "Counties/Regions (Romania map)"],
        index=0,
    )

    # Prepare filtered data
    work = df.copy()

    metric_label = "Followers"
    if platform_choice.lower() == "merchandise":
        metric_label = "Merchandise revenue (RON)"

    if platform_choice != "All platforms":
        work = work[work["platform"] == platform_choice]
    elif platform_choice == "All platforms" and highlight_platform and highlight_platform != "(none)":
        work = work[work["platform"] == highlight_platform]

    # Club mobile app: sum Romania counties into a country row for world map visibility
    work = _add_romania_from_counties(work, "Club mobile app")

    # Aggregate safely by (level, geography_norm, platform)
    agg_map = {"followers": "sum"}
    if "percent_on_platform" in work.columns:
        agg_map["percent_on_platform"] = "sum"
    agg = (
        work.groupby(["level", "geography_norm", "platform"], as_index=False)
        .agg(agg_map)
        .copy()
    )
    agg["followers_int"] = agg["followers"].round(0).astype("Int64")
    agg["Followers"] = agg["followers_int"].map(lambda x: f"{int(x):,}" if pd.notna(x) else "")
    if "percent_on_platform" in agg.columns:
        pct = agg["percent_on_platform"].astype(float)
        if pct.max() > 1:
            pct = pct / 100.0
        agg["Percent on platform"] = (pct * 100).round(1).astype(str) + "%"

    if platform_choice == "All platforms" and highlight_platform and highlight_platform != "(none)":
        agg_total = (
            df.groupby(["level", "geography_norm"], as_index=False)
            .agg({"followers": "sum"})
            .rename(columns={"followers": "total_followers"})
        )
        agg_total["total_followers_int"] = agg_total["total_followers"].round(0).astype("Int64")
        agg_total["Total followers"] = agg_total["total_followers_int"].map(lambda x: f"{int(x):,}" if pd.notna(x) else "")
        agg = agg_total.merge(
            agg,
            on=["level", "geography_norm"],
            how="left",
        )
        agg["followers"] = agg["followers"].fillna(0)
        agg["followers_int"] = agg["followers"].round(0).astype("Int64")
        agg["Followers"] = agg["followers_int"].map(lambda x: f"{int(x):,}" if pd.notna(x) else "")
        agg["platform"] = highlight_platform
        metric_label = f"{highlight_platform} (highlight)"

    hover_extra_cols = []
    if platform_choice == "All platforms" and (not highlight_platform or highlight_platform == "(none)"):
        agg_total = (
            work.groupby(["level", "geography_norm"], as_index=False)
            .agg({"followers": "sum"})
        )
        agg_total["platform"] = "All platforms"
        agg_total["followers_int"] = agg_total["followers"].round(0).astype("Int64")
        agg_total["Followers"] = agg_total["followers_int"].map(lambda x: f"{int(x):,}" if pd.notna(x) else "")
        agg = agg_total
        agg = add_platform_breakdown(agg, work)
        for p in sorted(df["platform"].unique().tolist()):
            val_raw = f"val_{_safe_col(p)}"
            pct_raw = f"pct_{_safe_col(p)}"
            val_pretty = f"Value - {p}"
            pct_pretty = f"Percentage of followers on {p} from Total"
            if val_raw in agg.columns:
                agg[val_pretty] = agg[val_raw].round(0).astype("Int64").map(lambda x: f"{int(x):,}" if pd.notna(x) else "")
            if pct_raw in agg.columns:
                agg[pct_pretty] = (agg[pct_raw] * 100).round(1).astype(str) + "%"
            hover_extra_cols.append(val_pretty)
            hover_extra_cols.append(pct_pretty)

    if level_choice == "Countries (World map)":
        df_map = agg[agg["level"] == "country"].copy()
        df_map = df_map[~df_map["geography_norm"].isin(NON_MAPPABLE_COUNTRIES)].copy()

        title = f"World Map – {metric_label}"
        if platform_choice != "All platforms":
            title += f" – {platform_choice}"

        fig, non_mappable = world_choropleth(df_map, title, metric_label, hover_extra=hover_extra_cols)
        st.plotly_chart(fig, width="stretch")

        # Bar charts
        st.subheader("Country Breakdown")
        if platform_choice == "All platforms" and (not highlight_platform or highlight_platform == "(none)"):
            for p in platforms:
                bar_src = df[df["platform"] == p].copy()
                if p == "Club mobile app":
                    bar_src = _add_romania_from_counties(bar_src, "Club mobile app")
                bar_src = bar_src[bar_src["level"] == "country"].copy()
                bar_src = bar_src[~bar_src["geography_norm"].isin(NON_MAPPABLE_COUNTRIES)].copy()
                bar_src = bar_src.sort_values("followers", ascending=False).head(TOP_N_BARS)
                if "percent_on_platform" in bar_src.columns:
                    pct = bar_src["percent_on_platform"].astype(float)
                    if pct.max() > 1:
                        pct = pct / 100.0
                    total = bar_src["followers"].sum()
                    computed = bar_src["followers"] / total if total > 0 else 0
                    bar_src["pct"] = pct.fillna(computed)
                else:
                    total = bar_src["followers"].sum()
                    bar_src["pct"] = bar_src["followers"] / total if total > 0 else 0

                fig_abs = px.bar(
                    bar_src,
                    x="geography_norm",
                    y="followers",
                    title=f"{p} — Absolute values (top countries)",
                    text="pct",
                )
                fig_abs.update_traces(texttemplate="%{text:.1%}", textposition="outside")
                st.plotly_chart(fig_abs, width="stretch")
        else:
            bar_src = df_map.copy()
            bar_src = bar_src.sort_values("followers", ascending=False).head(TOP_N_BARS)
            if "percent_on_platform" in bar_src.columns:
                pct = bar_src["percent_on_platform"].astype(float)
                if pct.max() > 1:
                    pct = pct / 100.0
                total = bar_src["followers"].sum()
                computed = bar_src["followers"] / total if total > 0 else 0
                bar_src["pct"] = pct.fillna(computed)
            else:
                total = bar_src["followers"].sum()
                bar_src["pct"] = bar_src["followers"] / total if total > 0 else 0
            fig_abs = px.bar(
                bar_src,
                x="geography_norm",
                y="followers",
                title="Absolute values (top countries)",
                text="pct",
            )
            fig_abs.update_traces(texttemplate="%{text:.1%}", textposition="outside")
            st.plotly_chart(fig_abs, width="stretch")

        if len(non_mappable) > 0:
            with st.expander("Non-mappable rows (regions / unknown country names)"):
                st.dataframe(
                    non_mappable.sort_values(["platform", "geography_norm"]),
                    width="stretch",
                )

    else:
        df_map = agg[agg["level"] == "county"].copy()

        title = f"Romania Map (Counties) – {metric_label}"
        if platform_choice != "All platforms":
            title += f" – {platform_choice}"

        fig, non_mappable = romania_choropleth(df_map, title, metric_label, hover_extra=hover_extra_cols)
        st.plotly_chart(fig, width="stretch")

        # Bar charts
        st.subheader("County Breakdown")
        if platform_choice == "All platforms" and (not highlight_platform or highlight_platform == "(none)"):
            for p in platforms:
                bar_src = df[(df["level"] == "county") & (df["platform"] == p)].copy()
                bar_src = bar_src.sort_values("followers", ascending=False).head(TOP_N_BARS)
                if p == "Fan Youtube (RD1948)" and bar_src["followers"].sum() == 0:
                    continue
                if "percent_on_platform" in bar_src.columns:
                    pct = bar_src["percent_on_platform"].astype(float)
                    if pct.max() > 1:
                        pct = pct / 100.0
                    total = bar_src["followers"].sum()
                    computed = bar_src["followers"] / total if total > 0 else 0
                    bar_src["pct"] = pct.fillna(computed)
                else:
                    total = bar_src["followers"].sum()
                    bar_src["pct"] = bar_src["followers"] / total if total > 0 else 0

                fig_abs = px.bar(
                    bar_src,
                    x="geography_norm",
                    y="followers",
                    title=f"{p} — Absolute values (top counties)",
                    text="pct",
                )
                fig_abs.update_traces(texttemplate="%{text:.1%}", textposition="outside")
                st.plotly_chart(fig_abs, width="stretch")
        else:
            bar_src = df_map.copy()
            bar_src = bar_src.sort_values("followers", ascending=False).head(TOP_N_BARS)
            if "percent_on_platform" in bar_src.columns:
                pct = bar_src["percent_on_platform"].astype(float)
                if pct.max() > 1:
                    pct = pct / 100.0
                total = bar_src["followers"].sum()
                computed = bar_src["followers"] / total if total > 0 else 0
                bar_src["pct"] = pct.fillna(computed)
            else:
                total = bar_src["followers"].sum()
                bar_src["pct"] = bar_src["followers"] / total if total > 0 else 0
            fig_abs = px.bar(
                bar_src,
                x="geography_norm",
                y="followers",
                title="Absolute values (top counties)",
                text="pct",
            )
            fig_abs.update_traces(texttemplate="%{text:.1%}", textposition="outside")
            st.plotly_chart(fig_abs, width="stretch")

        if len(non_mappable) > 0:
            with st.expander("Unmapped county/region labels (missing in geojson)"):
                st.dataframe(
                    non_mappable.sort_values(["platform", "geography_norm"]),
                    width="stretch",
                )

    st.caption("Edit notes: update DATA_PATH, COUNTRY_NORMALIZE, RO_LOCATION_CENTROIDS, and ISO-3 mappings in to_iso3().")


def main():
    st.set_page_config(page_title="Dinamo - Geo Maps", layout="wide")
    st.title("Dinamo Bucuresti fan situation - Interactive Map")
    tab_geo, tab_demo = st.tabs(["Geography", "Demographics"])
    with tab_geo:
        render_geography()
    with tab_demo:
        render_demographics()


if __name__ == "__main__":
    import os
    import sys

    def _is_running_with_streamlit() -> bool:
        if os.environ.get("STREAMLIT_LAUNCHED_BY_SCRIPT") == "1":
            return True
        try:
            from streamlit.runtime.scriptrunner_utils import get_script_run_ctx

            return get_script_run_ctx() is not None
        except Exception:
            return False

    if _is_running_with_streamlit():
        main()
    else:
        from streamlit.web import cli as stcli

        # Run this file through Streamlit so a browser window opens.
        os.environ["STREAMLIT_LAUNCHED_BY_SCRIPT"] = "1"
        sys.argv = ["streamlit", "run", os.path.abspath(__file__)]
        sys.exit(stcli.main())
