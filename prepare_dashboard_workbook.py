from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
SOURCE_PATH = Path(
    r"C:\Users\Razvan\Documents\Scouting\Data analysis department\Idei extrasportive"
    r"\Fan Survey 1 - Profil Suporter & Opinie Club 03.06 - Clean + judet + one word analysis + mediu + sigla analysis + two column categories + message suggestion analysis.xlsx"
)
OUTPUT_PATH = BASE_DIR / "data" / "fan_survey_dashboard.xlsx"
MAIN_SHEET = "responses_with_judet"

SAFE_MAIN_COLUMNS = [
    "Vârstă",
    "Gen",
    "Țară de reședință",
    "Județ atribuit",
    "Regiune atribuită",
    "Mediu atribuit",
    "Educație",
    "Ai copii?",
    "Vii cu copiii la meci?",
    "Dinamo un cuvânt - normalizat",
    "Dinamo un cuvânt - categorie",
    "De cât timp ești suporter Dinamo?",
    "Cum evaluezi clubul Dinamo în afara terenului, în ultima perioadă?",
    "Cât de conectat te simți emoțional cu Dinamo?",
    "Ce face Dinamo BINE - categorie",
    "Dacă ai putea schimba un singur lucru - categorie",
    "Ce te-ar determina să îți faci abonament pentru sezonul viitor?",
    "Cât de mult contează pentru tine dacă un brand pe care îl cumperi se asociază cu un alt club de fotbal?",
    "Dacă un brand sponsorizează Dinamo, cât de mult îți crește intenția de cumpărare?",
    "Mesaj pentru Dinamo - temă",
    "Mesaj pentru Dinamo - ton",
    "Sugestie suplimentară - categorie",
    "Siglă menționată",
    "Siglă sentiment",
    "Coloană mențiune siglă",
]

SAFE_SUMMARY_SHEETS = [
    "judet_summary",
    "match_status_counts",
    "judet_counts",
    "mediu_summary",
    "mediu_county_summary",
    "dinamo_word_counts",
    "dinamo_category_counts",
    "dinamo_taxonomy",
    "sigla_summary_overall",
    "sigla_summary_by_column",
    "bine_category_counts",
    "schimba_category_counts",
    "two_column_taxonomy",
    "mesaj_theme_counts",
    "mesaj_tone_counts",
    "mesaj_theme_tone_counts",
    "sugestie_category_counts",
    "message_suggestion_taxonomy",
]

BLOCKED_MAIN_COLUMNS = {
    "response_id",
    "Submitted",
    "Adresă de email",
    "Nume",
    "Prenume",
    "Oraș de reședință",
    "Oraș de reședință - normalizat",
    "Localitate potrivită",
    "Localitate oficială pentru mediu",
    "Ce înseamnă Dinamo pentru tine, într-un singur cuvânt?",
    "Ce face Dinamo BINE în acest moment?",
    "Dacă ai putea schimba un singur lucru la Dinamo, care ar fi acela?",
    "Dacă ai avea un mesaj pentru Dinamo, care ar fi acela? (opțional)",
    "Orice altă sugestie pe care ai vrea să ne-o transmiți (opțional)",
}


def freeze_and_size(writer: pd.ExcelWriter, sheet_name: str) -> None:
    ws = writer.sheets[sheet_name]
    ws.freeze_panes = "A2"
    for column_cells in ws.columns:
        max_len = 0
        for cell in column_cells:
            if cell.value is not None:
                max_len = max(max_len, len(str(cell.value)))
        ws.column_dimensions[column_cells[0].column_letter].width = min(max(max_len + 2, 12), 80)


def assert_public_safe(df: pd.DataFrame, sheet_names: list[str]) -> None:
    blocked_present = [col for col in BLOCKED_MAIN_COLUMNS if col in df.columns]
    if blocked_present:
        raise ValueError(f"Blocked columns present in public workbook: {blocked_present}")

    blocked_sheet_fragments = ["review", "override", "semantic_map"]
    unsafe_sheets = [
        sheet
        for sheet in sheet_names
        if any(fragment in sheet.lower() for fragment in blocked_sheet_fragments)
    ]
    if unsafe_sheets:
        raise ValueError(f"Unsafe review/override sheets selected: {unsafe_sheets}")


def main() -> None:
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(SOURCE_PATH)

    xl = pd.ExcelFile(SOURCE_PATH)
    source_df = pd.read_excel(SOURCE_PATH, sheet_name=MAIN_SHEET)
    missing = [col for col in SAFE_MAIN_COLUMNS if col not in source_df.columns]
    if missing:
        raise KeyError(f"Missing required dashboard columns: {missing}")

    public_df = source_df[SAFE_MAIN_COLUMNS].copy()
    public_df.insert(0, "anonymous_response_id", range(1, len(public_df) + 1))

    selected_sheets = [sheet for sheet in SAFE_SUMMARY_SHEETS if sheet in xl.sheet_names]
    assert_public_safe(public_df, selected_sheets)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        public_df.to_excel(writer, sheet_name=MAIN_SHEET, index=False)
        freeze_and_size(writer, MAIN_SHEET)

        for sheet in selected_sheets:
            sheet_df = pd.read_excel(SOURCE_PATH, sheet_name=sheet)
            sheet_df.to_excel(writer, sheet_name=sheet, index=False)
            freeze_and_size(writer, sheet)

    print(f"source_rows={len(source_df)}")
    print(f"public_rows={len(public_df)}")
    print(f"public_columns={len(public_df.columns)}")
    print(f"summary_sheets={len(selected_sheets)}")
    print(f"output={OUTPUT_PATH}")


if __name__ == "__main__":
    main()
