import numpy as np
import pandas as pd
import streamlit as st
from snowflake.snowpark import Session
from datetime import datetime

# ============================================================
# Create Snowflake session
# ============================================================
def get_session():
    return Session.builder.configs({
        "account": st.secrets["snowflake"]["account"],
        "user": st.secrets["snowflake"]["user"],
        "password": st.secrets["snowflake"]["password"],
        "role": st.secrets["snowflake"]["role"],
        "warehouse": st.secrets["snowflake"]["warehouse"],
        "database": st.secrets["snowflake"]["database"],
        "schema": st.secrets["snowflake"]["schema"],
    }).create()

session = get_session()

# ============================================================
# Helpers
# ============================================================
def format_url(url):
    if pd.isna(url) or url is None:
        return None
    url = str(url).strip()
    if not url.startswith(("http://", "https://")):
        return "https://" + url
    return url

def parse_funding_amount(series):
    import re

    def _parse(x):
        if pd.isna(x):
            return np.nan
        s = str(x).upper().replace(",", "").strip()

        for sym in ["$", "€", "£"]:
            s = s.replace(sym, "")

        multiplier = 1
        if s.endswith("B"):
            multiplier = 1_000_000_000
            s = s[:-1]
        elif s.endswith("M"):
            multiplier = 1_000_000
            s = s[:-1]
        elif s.endswith("K"):
            multiplier = 1_000

        s = "".join(re.findall(r"[0-9.]", s))
        return float(s) * multiplier if s else np.nan

    return series.apply(_parse)

# ============================================================
# Load data from Snowflake
# ============================================================
@st.cache_data(ttl=300)
def load_data():
    companies = session.sql("""
        SELECT company_id, company_name, website_url, linkedin_url
        FROM companies
    """).to_pandas()

    funding = session.sql("""
        SELECT *
        FROM funding_rounds
    """).to_pandas()

    companies.columns = companies.columns.str.lower()
    funding.columns = funding.columns.str.lower()

    # parse numeric values from amount_raised_total
    funding["amount_num"] = parse_funding_amount(funding["amount_raised_total"])

    # merge
    merged = funding.merge(
        companies,
        on="company_id",
        how="left"
    )

    return companies, funding, merged

def get_last_updated():
    try:
        df = session.sql("SELECT MAX(updated_at) AS last_update FROM funding_rounds").to_pandas()
        if pd.notna(df.loc[0, "LAST_UPDATE"]):
            return str(df.loc[0, "LAST_UPDATE"])
    except:
        return "Unknown"

    return "Unknown"

# ============================================================
# Display table (no downloads, hides IDs)
# ============================================================
def display_table(df, name):
    df = df.drop(columns=["company_id", "round_id"], errors="ignore").copy()

    rename_map = {
        "company_name": "Company",
        "stage_or_funding_round": "Funding Stage",
        "amount_raised_total": "Funding Amount",
        "lead_investor": "Lead Investor",
        "website_url": "Website",
        "linkedin_url": "LinkedIn"
    }
    df.rename(columns=rename_map, inplace=True)

    search = st.text_input(f"Search in {name}:", key=f"search_{name}")

    if search:
        mask = df.astype(str).apply(lambda col: col.str.contains(search, case=False, na=False)).any(axis=1)
        df_show = df[mask]
        st.caption(f"{len(df_show)} matching rows")
    else:
        df_show = df
        st.caption(f"{len(df_show)} rows")

    st.dataframe(df_show, hide_index=True, use_container_width=True)

# ============================================================
# Page Config
# ============================================================
st.set_page_config(page_title="Funding Intelligence Dashboard", layout="wide")

st.title("Funding Intelligence Dashboard")
st.caption("Live VC funding insights powered by Snowflake.")

companies_df, funding_df, merged_df = load_data()

# ============================================================
# Sidebar Filters
# ============================================================
st.sidebar.header("Filters")

stage_options = sorted(merged_df["stage_or_funding_round"].dropna().unique())
investor_options = sorted(merged_df["lead_investor"].dropna().unique())
company_options = sorted(merged_df["company_name"].dropna().unique())

selected_stages = st.sidebar.multiselect("Funding Stage", stage_options)
selected_investors = st.sidebar.multiselect("Lead Investor", investor_options)
selected_companies = st.sidebar.multiselect("Company", company_options)

st.sidebar.markdown("---")
selected_table = st.sidebar.selectbox("Select Table", ["Both", "Funding Rounds", "Companies"])
st.sidebar.caption(f"Last updated: **{get_last_updated()}**")

# ============================================================
# Apply Filters
# ============================================================
filtered = merged_df.copy()

if selected_stages:
    filtered = filtered[filtered["stage_or_funding_round"].isin(selected_stages)]

if selected_investors:
    filtered = filtered[filtered["lead_investor"].isin(selected_investors)]

if selected_companies:
    filtered = filtered[filtered["company_name"].isin(selected_companies)]

# ============================================================
# KPIs
# ============================================================
st.subheader("Key Metrics")

k1, k2, k3 = st.columns(3)
k1.metric("Total Funding", f"${filtered['amount_num'].sum():,.0f}")
k2.metric("Funding Rounds", filtered["round_id"].nunique())
k3.metric("Unique Companies", filtered["company_id"].nunique())

st.markdown("---")

# ============================================================
# Tabs
# ============================================================
tab1, tab2, tab3 = st.tabs(["Rounds", "Investors", "Companies"])

with tab1:
    st.markdown("### Funding by Stage")
    totals = filtered.groupby("stage_or_funding_round")["amount_num"].sum()
    st.bar_chart(totals)

with tab2:
    st.markdown("### Top Investors")
    inv = filtered.groupby("lead_investor")["amount_num"].sum().sort_values(ascending=False)
    st.bar_chart(inv)

with tab3:
    st.markdown("### Top Companies")
    comp = filtered.groupby("company_name")["amount_num"].sum().sort_values(ascending=False)
    st.bar_chart(comp)

# ============================================================
# Tables
# ============================================================
st.markdown("---")
st.subheader("Data Tables")

if selected_table in ["Both", "Funding Rounds"]:
    st.markdown("### Funding Rounds")
    display_table(funding_df, "funding_rounds")

if selected_table in ["Both", "Companies"]:
    st.markdown("### Companies")
    display_table(companies_df, "companies")

st.caption("Dashboard loads live data from Snowflake.")
