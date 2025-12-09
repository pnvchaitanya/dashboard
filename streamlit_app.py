import numpy as np
import pandas as pd
import streamlit as st
from snowflake.snowpark import Session
from datetime import datetime

# ============================================================
# Snowflake session
# ============================================================
def get_session():
    connection_parameters = {
        "account": st.secrets["snowflake"]["account"],
        "user": st.secrets["snowflake"]["user"],
        "password": st.secrets["snowflake"]["password"],
        "role": st.secrets["snowflake"]["role"],
        "warehouse": st.secrets["snowflake"]["warehouse"],
        "database": st.secrets["snowflake"]["database"],
        "schema": st.secrets["snowflake"]["schema"],
    }
    return Session.builder.configs(connection_parameters).create()

session = get_session()


# ============================================================
# Helpers
# ============================================================
def format_url(url):
    """Add protocol to URL if missing."""
    if url is None or pd.isna(url):
        return None
    url = str(url).strip()
    if url == "":
        return None
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url


def parse_funding_amount(series: pd.Series) -> pd.Series:
    """Convert textual amount to numeric."""
    import re

    def _parse(x):
        if pd.isna(x):
            return np.nan

        s = str(x).strip().upper()
        if s in ("", "", "NONE", "N/A", "NA", "-"):
            return np.nan

        s = s.replace(",", "")
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
            s = s[:-1]

        s = "".join(re.findall(r"[0-9.]", s))
        return float(s) * multiplier if s else np.nan

    return series.apply(_parse)


# ============================================================
# Load data from Snowflake
# ============================================================
@st.cache_data(ttl=300)
def load_data():
    # COMPANIES table
    companies = session.sql("""
        SELECT
            company_id,
            company_name,
            website      AS website_url,   -- <== IMPORTANT FIX
            linkedin_url,
            category_group,
            status
        FROM RISKINSIGHTSMEDIA_DB.ANALYTICS.COMPANIES
    """).to_pandas()

    # FUNDING_ROUNDS table
    funding = session.sql("""
        SELECT
            round_id,
            company_id,
            company_name,
            stage_or_funding_round,
            amount_raised_total,
            lead_investor,
            website_url,
            linkedin_url,
            created_at,
            updated_at
        FROM RISKINSIGHTSMEDIA_DB.ANALYTICS.FUNDING_ROUNDS
    """).to_pandas()

    # Lowercase columns
    companies.columns = companies.columns.str.lower()
    funding.columns = funding.columns.str.lower()

    # Parse numeric funding
    if "amount_raised_total" in funding.columns:
        funding["amount_num"] = parse_funding_amount(funding["amount_raised_total"])
    else:
        funding["amount_num"] = np.nan

    # Parse created_at
    if "created_at" in funding.columns:
        funding["created_at_dt"] = pd.to_datetime(
            funding["created_at"], errors="coerce"
        )
    else:
        funding["created_at_dt"] = pd.NaT

    # Columns for merge
    merge_cols = [
        "company_id", "company_name", "website_url", "linkedin_url",
        "category_group", "status"
    ]
    merge_cols = [c for c in merge_cols if c in companies.columns]

    merged = funding.merge(
        companies[merge_cols],
        on="company_id",
        how="left",
        suffixes=("", "_company"),
    )

    return companies, funding, merged


def get_last_updated():
    try:
        df = session.sql("""
            SELECT MAX(updated_at) AS last_update
            FROM RISKINSIGHTSMEDIA_DB.ANALYTICS.FUNDING_ROUNDS
        """).to_pandas()

        if not df.empty and pd.notna(df.loc[0, "LAST_UPDATE"]):
            return str(df.loc[0, "LAST_UPDATE"])
    except Exception:
        pass
    return "Unknown"


# ============================================================
# Table viewer (Hides IDs + Renames Columns + No Download)
# ============================================================
def display_table(df: pd.DataFrame, table_name: str):

    # Hide ID columns + technical columns
    df = df.drop(
        columns=["company_id", "round_id", "created_at", "updated_at", "created_at_dt"],
        errors="ignore"
    ).copy()

    # Rename columns for display only
    rename_map = {
        "company_name": "Company",
        "stage_or_funding_round": "Funding",
        "amount_raised_total": "Amount Raised Total",
        "lead_investor": "Lead Investor",
        "website_url": "Website",
        "linkedin_url": "LinkedIn"
    }

    df.rename(
        columns={k: v for k, v in rename_map.items() if k in df.columns},
        inplace=True
    )

    # Search field
    search_term = st.text_input(
        f"Search in {table_name}:",
        placeholder="Type to search...",
        key=f"search_{table_name}"
    )

    if search_term:
        string_cols = df.select_dtypes(include="object").columns
        mask = (
            df[string_cols].astype(str)
            .apply(lambda col: col.str.contains(search_term, case=False, na=False))
            .any(axis=1)
        )
        view_df = df[mask]
        st.caption(f"{len(view_df)} matching rows")
    else:
        view_df = df
        st.caption(f"{len(view_df)} rows")

    # Show table (no download button)
    st.dataframe(view_df, use_container_width=True, hide_index=True)


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Funding Intelligence Dashboard", layout="wide")

st.title("Funding Intelligence Dashboard")
st.caption("Explore funding rounds, investors, categories, and companies from Snowflake.")

companies_df, funding_df, merged_df = load_data()

if funding_df.empty or companies_df.empty:
    st.error("No data available in Snowflake.")
    st.stop()


# ============================================================
# Sidebar Filters
# ============================================================
st.sidebar.header("Filters")

stage_options = sorted(merged_df["stage_or_funding_round"].dropna().astype(str).unique())
selected_stages = st.sidebar.multiselect("Funding Stage / Round", stage_options)

investor_options = sorted(merged_df["lead_investor"].dropna().astype(str).unique())
selected_investors = st.sidebar.multiselect("Lead Investor", investor_options)

category_options = sorted(merged_df["category_group"].dropna().astype(str).unique()) \
    if "category_group" in merged_df.columns else []
selected_categories = st.sidebar.multiselect("Company Category", category_options)

status_options = sorted(merged_df["status"].dropna().astype(str).unique()) \
    if "status" in merged_df.columns else []
selected_status = st.sidebar.multiselect("Company Status", status_options)

st.sidebar.markdown("---")
selected_table = st.sidebar.selectbox(
    "Select table to view:",
    ["Both Tables", "funding_rounds", "companies"],
    index=2
)

st.sidebar.markdown("---")
st.sidebar.caption(f"Last updated: **{get_last_updated()}**")


# ============================================================
# Apply Filters
# ============================================================
filtered = merged_df.copy()

if selected_stages:
    filtered = filtered[filtered["stage_or_funding_round"].isin(selected_stages)]
if selected_investors:
    filtered = filtered[filtered["lead_investor"].isin(selected_investors)]
if selected_categories and "category_group" in filtered.columns:
    filtered = filtered[filtered["category_group"].isin(selected_categories)]
if selected_status and "status" in filtered.columns:
    filtered = filtered[filtered["status"].isin(selected_status)]


# ============================================================
# KPIs
# ============================================================
st.subheader("Key KPIs (Filtered)")

total_funding = filtered["amount_num"].sum(skipna=True)
total_rounds = filtered["round_id"].nunique() if "round_id" in filtered.columns else 0
unique_companies = filtered["company_id"].nunique() if "company_id" in filtered.columns else 0

top_investor = (
    filtered.groupby("lead_investor")["amount_num"]
    .sum()
    .sort_values(ascending=False)
    .index[0]
    if not filtered.empty and filtered["lead_investor"].notna().any()
    else "N/A"
)

top_company = (
    filtered.groupby("company_name")["amount_num"]
    .sum()
    .sort_values(ascending=False)
    .index[0]
    if not filtered.empty and filtered["company_name"].notna().any()
    else "N/A"
)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Funding", f"${total_funding:,.0f}" if total_funding else "N/A")
k2.metric("Funding Rounds", int(total_rounds))
k3.metric("Unique Companies", int(unique_companies))
k4.metric("Top Investor (by $)", top_investor)
k5.metric("Top Funded Company", top_company)

st.markdown("---")


# ============================================================
# Tabs
# ============================================================
tab1, tab2, tab3 = st.tabs(["Rounds Analysis", "Investors", "Company Explorer"])

# -------------------------
# Tab 1: Rounds Analysis
# -------------------------
with tab1:
    st.markdown("### Rounds Analysis")

    c1, c2 = st.columns(2)

    if "stage_or_funding_round" in filtered.columns:
        with c1:
            st.markdown("**Funding totals by round**")
            totals = (
                filtered.groupby("stage_or_funding_round")["amount_num"]
                .sum()
                .sort_values(ascending=False)
            )
            st.bar_chart(totals)

        with c2:
            st.markdown("**Company count per round**")
            counts = (
                filtered.groupby("stage_or_funding_round")["company_id"]
                .nunique()
                .sort_values(ascending=False)
            )
            st.bar_chart(counts)


# -------------------------
# Tab 2: Investors
# -------------------------
with tab2:
    st.markdown("### Investors")

    c3, c4 = st.columns(2)

    with c3:
        st.markdown("**Top investors by funding**")
        inv_fund = (
            filtered.groupby("lead_investor")["amount_num"]
            .sum()
            .sort_values(ascending=False)
            .head(20)
        )
        st.bar_chart(inv_fund)

    with c4:
        st.markdown("**Top investors by deal count**")
        inv_count = (
            filtered.groupby("lead_investor")["round_id"]
            .nunique()
            .sort_values(ascending=False)
            .head(20)
        )
        st.bar_chart(inv_count)

    st.markdown("---")

    st.markdown("**Top funded companies**")
    comp_fund = (
        filtered.groupby("company_name")["amount_num"]
        .sum()
        .sort_values(ascending=False)
        .head(20)
    )
    st.bar_chart(comp_fund)


# -------------------------
# Tab 3: Company Explorer
# -------------------------
with tab3:
    st.markdown("### Company Explorer")

    group_cols = ["company_id", "company_name", "website_url", "linkedin_url"]
    if "category_group" in filtered.columns:
        group_cols.append("category_group")
    if "status" in filtered.columns:
        group_cols.append("status")

    company_metrics = (
        filtered.groupby(group_cols, dropna=False)
        .agg(total_funding=("amount_num", "sum"))
        .reset_index()
    )

    company_metrics = company_metrics.sort_values(by="total_funding", ascending=False)
    top_companies = company_metrics.head(25)

    for _, row in top_companies.iterrows():
        with st.container(border=True):

            st.markdown(f"#### {row['company_name']}")

            detail_list = []
            if "category_group" in row and pd.notna(row["category_group"]):
                detail_list.append(f"Category: {row['category_group']}")
            if "status" in row and pd.notna(row["status"]):
                detail_list.append(f"Status: {row['status']}")

            if detail_list:
                st.markdown(" • ".join(detail_list))

            cols = st.columns([1, 3])

            with cols[0]:
                website = format_url(row.get("website_url"))
                linkedin = format_url(row.get("linkedin_url"))

                if website:
                    st.link_button("Website", website, use_container_width=True)
                if linkedin:
                    st.link_button("LinkedIn", linkedin, use_container_width=True)

            with cols[1]:
                tf = (
                    f"${row['total_funding']:,.0f}"
                    if pd.notna(row["total_funding"])
                    else "N/A"
                )
                st.markdown("**Funding Summary**")
                st.write(f"- Total funding: {tf}")


# ============================================================
# Data Tables
# ============================================================
st.markdown("---")
st.subheader("Data Tables")

allowed_round_ids = filtered["round_id"].dropna().unique() if "round_id" in filtered.columns else []
allowed_company_ids = filtered["company_id"].dropna().unique() if "company_id" in filtered.columns else []

funding_filtered = funding_df[funding_df["round_id"].isin(allowed_round_ids)]
companies_filtered = companies_df[companies_df["company_id"].isin(allowed_company_ids)]

if selected_table == "Both Tables":
    st.markdown("### Funding Rounds Table")
    display_table(funding_filtered, "funding_rounds")

    st.markdown("### Companies Table")
    display_table(companies_filtered, "companies")

elif selected_table == "funding_rounds":
    display_table(funding_filtered, "funding_rounds")

elif selected_table == "companies":
    display_table(companies_filtered, "companies")

st.caption("Dashboard loads live data from Snowflake.")
