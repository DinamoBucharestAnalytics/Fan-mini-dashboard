# Fan Survey Dashboard

Streamlit dashboard for the Dinamo fan survey analysis.

The previous social-media dashboard is preserved on the `social-media` branch.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run "streamlit fans.py"
```

## Data

The dashboard reads:

```text
data/fan_survey_dashboard.xlsx
```

The committed workbook is sanitized for public use:

- no names
- no email addresses
- no exact submission timestamps
- no city/locality-level respondent fields
- no raw long-form open answers
- only derived category/summary fields needed for charts
- includes 10,499 anonymized responses from the 03.06 cleaned survey workbook
- includes county, region, and urban/rural (`Mediu atribuit`) classifications
