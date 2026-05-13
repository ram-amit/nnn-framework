"""
etl_processor.py — ETL pipeline for the NNN marketing measurement framework.

Two modes:
  1. Snowflake (default): Queries bigbrain warehouse directly
  2. CSV fallback: Loads raw ad platform CSVs

Output: real_marketing_data.csv with schema:
  Date, Geography, Channel, Spend, Enterprise_Trials, Closed_Won, Campaign_Metadata

Usage:
    python3 etl_processor.py                          # Snowflake mode
    python3 etl_processor.py --start-date 2024-06-01  # custom date range
    python3 etl_processor.py --mode csv               # CSV fallback
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Geography mapping
# ─────────────────────────────────────────────
COUNTRY_TO_REGION: Dict[str, str] = {
    # North America
    "US": "NA", "CA": "NA",
    # EMEA
    "GB": "EMEA", "UK": "EMEA", "DE": "EMEA", "FR": "EMEA",
    "NL": "EMEA", "IL": "EMEA", "SE": "EMEA", "ES": "EMEA",
    "IT": "EMEA", "CH": "EMEA", "AT": "EMEA", "BE": "EMEA",
    "DK": "EMEA", "NO": "EMEA", "FI": "EMEA", "PL": "EMEA",
    "IE": "EMEA",
    # Snowflake-specific region codes
    "DACH": "EMEA", "EU1": "EMEA", "INT1_2": "EMEA",
    # APAC
    "AU": "APAC", "NZ": "APAC", "JP": "APAC", "SG": "APAC",
    "IN": "APAC", "KR": "APAC", "HK": "APAC",
    # LATAM (folded into NA for NNN geo simplicity)
    "BR": "LATAM", "MX": "LATAM", "LATAM": "LATAM",
    # Global buckets
    "WW": "GLOBAL", "WW2": "GLOBAL", "ROW": "GLOBAL",
}

# Ad platform SOURCE field → standardized channel name
SOURCE_TO_CHANNEL: Dict[str, str] = {
    "adwordsbrand": "Google Brand",
    "adwordssearch": "Google Non-Brand",
    "adwordslocals": "Google Non-Brand",
    "adwordsyoutube": "YouTube",
    "adwordsmobile": "Google Non-Brand",
    "adwordsproducts": "Google Shopping",
    "linkedin_acq": "LinkedIn",
    "facebook": "Facebook",
    "bing": "Bing",
    "dbm_display": "Display",
    "reddit_ads": "Reddit",
    "ctv": "CTV",
    "capterra": "Review Sites",
    "podcast": "Podcast",
    "natural_intelligence": "Affiliates",
    "partnerstack": "Affiliates",
    "mvf": "Affiliates",
    "dpm": "Affiliates",
    "traffic_point": "Affiliates",
    "usbrands": "Google Brand",
}

# Lead attribution source → channel (for FACT_LEADS_ATTRIBUTION_V2)
LEAD_SOURCE_TO_CHANNEL: Dict[str, str] = {
    "Linkedin": "LinkedIn",
    "Content": "Review Sites",
    "Software Review Vendors": "Review Sites",
    "Vendors": "Affiliates",
    "Partner Link": "Affiliates",
    "Partner Outbound": "Affiliates",
}

# Opportunity attribution source → channel (for DIM_ACCOUNT_MARKETING_ATTRIBUTION)
MARKETING_SOURCE_TO_CHANNEL: Dict[str, str] = {
    "google": "Google Non-Brand",
    "google_brand": "Google Brand",
    "adwords": "Google Non-Brand",
    "youtube": "YouTube",
    "linkedin": "LinkedIn",
    "facebook": "Facebook",
    "bing": "Bing",
    "display": "Display",
    "reddit": "Reddit",
    "capterra": "Review Sites",
}


def map_geography(code: str) -> str:
    if pd.isna(code) or str(code).strip() == "" or str(code).lower() == "none":
        return "UNKNOWN"
    return COUNTRY_TO_REGION.get(str(code).upper().strip(), "OTHER")


def parse_geo_from_campaign_name(name: str) -> str:
    """Extract country code from structured campaign names like 'gb-en-prm-...'"""
    if pd.isna(name) or not isinstance(name, str):
        return "UNKNOWN"
    prefix = name.split("-")[0].upper().strip()
    return COUNTRY_TO_REGION.get(prefix, "UNKNOWN")


def to_week_start(date_series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(date_series)
    return dt - pd.to_timedelta(dt.dt.weekday, unit="D")


# =========================================================================
# Snowflake connection
# =========================================================================

def get_snowflake_connection():
    import snowflake.connector
    return snowflake.connector.connect(
        account="monday-prod.monday1.us-east-1.a.p1.satoricyber.net",
        user="amitram@monday.com",
        host="monday-prod.monday1.us-east-1.a.p1.satoricyber.net",
        authenticator="externalbrowser",
        database="bigbrain",
        warehouse="analytics_wh_01",
    )


# =========================================================================
# Snowflake ETL queries
# =========================================================================

SPEND_QUERY = """
SELECT
    DATE_TRUNC('WEEK', DAY)                     AS WEEK_START,
    COALESCE(COUNTRY_FIELD, SPLIT_PART(CAMPAIGN_NAME, '-', 1)) AS GEO_RAW,
    SOURCE,
    CAMPAIGN_NAME,
    SUM(COST)                                    AS SPEND,
    SUM(IMPRESSIONS)                             AS IMPRESSIONS,
    SUM(CLICKS)                                  AS CLICKS,
    SUM(TOTAL_CONVERSIONS)                       AS PLATFORM_CONVERSIONS,
    ANY_VALUE(CREATIVE_NAME)                     AS CREATIVE_SAMPLE,
    ANY_VALUE(KEYWORD)                           AS KEYWORD_SAMPLE,
    ANY_VALUE(BUSINESS_GOAL)                     AS BUSINESS_GOAL,
    ANY_VALUE(BUDGET_TYPE)                       AS BUDGET_TYPE,
    ANY_VALUE(GROUPED_CHANNEL_DTR)               AS CHANNEL_DTR
FROM bigbrain.L3.FACT_MARKETING_ADN_SPEND_DAILY
WHERE DAY >= %(start_date)s
  AND DAY < %(end_date)s
  AND COST > 0
GROUP BY 1, 2, 3, 4
"""

LEADS_QUERY = """
SELECT
    DATE_TRUNC('WEEK', LEAD_CREATED_AT)         AS WEEK_START,
    COUNTRY                                      AS GEO_RAW,
    ATTRIBUTED_FIRST_SOURCE                      AS LEAD_SOURCE,
    COUNT(*)                                     AS LEAD_COUNT
FROM bigbrain.L3.FACT_LEADS_ATTRIBUTION_V2
WHERE LEAD_CREATED_AT >= %(start_date)s
  AND LEAD_CREATED_AT < %(end_date)s
  AND ATTRIBUTED_FIRST_SOURCE IS NOT NULL
GROUP BY 1, 2, 3
"""

CLOSED_WON_QUERY = """
SELECT
    DATE_TRUNC('WEEK', o.CLOSE_DATE)            AS WEEK_START,
    a.MARKETING_SOURCE,
    a.MARKETING_MEDIUM,
    o.OWNER_OFFICE_REGION_WHEN_CLOSED            AS GEO_RAW,
    COUNT(DISTINCT o.OPPORTUNITY_ID)             AS WON_COUNT,
    SUM(o.ARR)                                   AS WON_ARR
FROM bigbrain.L3.FACT_DAILY_SALESFORCE_OPPORTUNITIES o
JOIN bigbrain.FINAL.DIM_ACCOUNT_MARKETING_ATTRIBUTION a
    ON o.PULSE_ACCOUNT_ID = a.PULSE_ACCOUNT_ID
WHERE o.IS_WON = 1
  AND o.DAY = o.CLOSE_DATE
  AND o.CLOSE_DATE >= %(start_date)s
  AND o.CLOSE_DATE < %(end_date)s
  AND a.MARKETING_SOURCE IS NOT NULL
GROUP BY 1, 2, 3, 4
"""


def extract_spend(conn, start_date: str, end_date: str) -> pd.DataFrame:
    """Pull ad spend data from Snowflake and standardize channels/geos."""
    logger.info("Querying FACT_MARKETING_ADN_SPEND_DAILY...")
    df = pd.read_sql(
        SPEND_QUERY, conn,
        params={"start_date": start_date, "end_date": end_date},
    )
    logger.info("Spend raw: %d rows", len(df))

    # Map source → channel
    df["Channel"] = df["SOURCE"].map(SOURCE_TO_CHANNEL).fillna("Other")

    # Map geo: use GEO_RAW, which is COALESCE(COUNTRY_FIELD, campaign_name prefix)
    df["Geography"] = df["GEO_RAW"].apply(map_geography)
    # For remaining UNKNOWNs, try parsing campaign name
    unknown_mask = df["Geography"] == "UNKNOWN"
    if unknown_mask.any():
        df.loc[unknown_mask, "Geography"] = df.loc[unknown_mask, "CAMPAIGN_NAME"].apply(
            parse_geo_from_campaign_name
        )

    # Build Campaign_Metadata
    df["Campaign_Metadata"] = df.apply(
        lambda r: (
            f"[Campaign: {_safe(r['CAMPAIGN_NAME'])}] "
            f"[Copy: {_safe(r['CREATIVE_SAMPLE'])}] "
            f"[Keywords: {_safe(r['KEYWORD_SAMPLE'])}] "
            f"[Objective: {_safe(r['BUSINESS_GOAL'])}]"
        ),
        axis=1,
    )

    # Aggregate to week / geo / channel
    agg = df.groupby(["WEEK_START", "Geography", "Channel"]).agg(
        Spend=("SPEND", "sum"),
        Impressions=("IMPRESSIONS", "sum"),
        Clicks=("CLICKS", "sum"),
        Enterprise_Trials=("PLATFORM_CONVERSIONS", "sum"),
        Campaign_Metadata=("Campaign_Metadata", "first"),
    ).reset_index()

    agg.rename(columns={"WEEK_START": "week_start"}, inplace=True)
    return agg


def extract_leads(conn, start_date: str, end_date: str) -> pd.DataFrame:
    """Pull attributed leads from Snowflake as supplementary conversion signal."""
    logger.info("Querying FACT_LEADS_ATTRIBUTION_V2...")
    try:
        df = pd.read_sql(
            LEADS_QUERY, conn,
            params={"start_date": start_date, "end_date": end_date},
        )
        logger.info("Leads raw: %d rows", len(df))

        df["Channel"] = df["LEAD_SOURCE"].map(LEAD_SOURCE_TO_CHANNEL)
        df = df.dropna(subset=["Channel"])
        df["Geography"] = df["GEO_RAW"].apply(map_geography)
        df.rename(columns={"WEEK_START": "week_start", "LEAD_COUNT": "Attributed_Leads"}, inplace=True)

        agg = df.groupby(["week_start", "Geography", "Channel"]).agg(
            Attributed_Leads=("Attributed_Leads", "sum"),
        ).reset_index()
        return agg
    except Exception as e:
        logger.warning("Leads query failed: %s", e)
        return pd.DataFrame(columns=["week_start", "Geography", "Channel", "Attributed_Leads"])


def extract_closed_won(conn, start_date: str, end_date: str) -> pd.DataFrame:
    """Pull marketing-attributed closed-won deals from Snowflake."""
    logger.info("Querying FACT_DAILY_SALESFORCE_OPPORTUNITIES + DIM_ACCOUNT_MARKETING_ATTRIBUTION...")
    try:
        df = pd.read_sql(
            CLOSED_WON_QUERY, conn,
            params={"start_date": start_date, "end_date": end_date},
        )
        logger.info("Closed-won raw: %d rows", len(df))

        # Map marketing_source → channel
        df["Channel"] = df["MARKETING_SOURCE"].str.lower().map(
            lambda s: next(
                (ch for prefix, ch in MARKETING_SOURCE_TO_CHANNEL.items() if prefix in str(s)),
                None,
            )
        )
        df = df.dropna(subset=["Channel"])

        # Map geo from sales region
        region_map = {
            "US": "NA", "NA": "NA", "North America": "NA",
            "EMEA": "EMEA", "Europe": "EMEA",
            "APAC": "APAC", "Asia": "APAC",
            "LATAM": "LATAM",
        }
        df["Geography"] = df["GEO_RAW"].map(
            lambda r: next(
                (v for k, v in region_map.items() if str(k).lower() in str(r).lower()),
                "GLOBAL",
            )
        )

        df.rename(columns={"WEEK_START": "week_start"}, inplace=True)
        agg = df.groupby(["week_start", "Geography", "Channel"]).agg(
            Closed_Won=("WON_COUNT", "sum"),
            Won_ARR=("WON_ARR", "sum"),
        ).reset_index()
        return agg
    except Exception as e:
        logger.warning("Closed-won query failed: %s", e)
        return pd.DataFrame(columns=["week_start", "Geography", "Channel", "Closed_Won"])


# =========================================================================
# Grid fill + merge logic (shared by both modes)
# =========================================================================

def build_dense_grid(
    spend_df: pd.DataFrame,
    crm_df: Optional[pd.DataFrame] = None,
    leads_df: Optional[pd.DataFrame] = None,
    drop_channels: Optional[List[str]] = None,
    drop_geos: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Build a dense geo × week × channel grid and merge all data sources.
    Ensures every combination has a row (required for the Rank-4 tensor).
    """
    if drop_channels is None:
        drop_channels = ["Other"]
    if drop_geos is None:
        drop_geos = ["UNKNOWN", "OTHER"]

    # Filter out unwanted channels/geos from spend
    spend_df = spend_df[~spend_df["Channel"].isin(drop_channels)].copy()
    spend_df = spend_df[~spend_df["Geography"].isin(drop_geos)].copy()

    all_weeks = sorted(spend_df["week_start"].unique())
    all_geos = sorted(spend_df["Geography"].unique())
    all_channels = sorted(spend_df["Channel"].unique())

    if crm_df is not None and len(crm_df) > 0:
        crm_df = crm_df[~crm_df["Channel"].isin(drop_channels)].copy()
        crm_df = crm_df[~crm_df["Geography"].isin(drop_geos)].copy()
        all_channels = sorted(set(all_channels) | set(crm_df["Channel"].unique()))

    grid = pd.MultiIndex.from_product(
        [all_weeks, all_geos, all_channels],
        names=["week_start", "Geography", "Channel"],
    )
    filled = pd.DataFrame(index=grid).reset_index()

    # Merge spend
    filled = filled.merge(
        spend_df[["week_start", "Geography", "Channel", "Spend", "Enterprise_Trials", "Campaign_Metadata"]],
        on=["week_start", "Geography", "Channel"],
        how="left",
    )
    filled["Spend"] = filled["Spend"].fillna(0).round(2)
    filled["Enterprise_Trials"] = filled["Enterprise_Trials"].fillna(0).astype(int)
    filled["Campaign_Metadata"] = filled["Campaign_Metadata"].fillna(
        "[Campaign: N/A] [Copy: No active campaign this period] [Keywords: N/A] [Objective: N/A]"
    )

    # Merge leads
    if leads_df is not None and len(leads_df) > 0:
        leads_df = leads_df[~leads_df["Channel"].isin(drop_channels)].copy()
        leads_df = leads_df[~leads_df["Geography"].isin(drop_geos)].copy()
        filled = filled.merge(
            leads_df[["week_start", "Geography", "Channel", "Attributed_Leads"]],
            on=["week_start", "Geography", "Channel"],
            how="left",
        )
        filled["Attributed_Leads"] = filled["Attributed_Leads"].fillna(0).astype(int)
    else:
        filled["Attributed_Leads"] = 0

    # Merge CRM
    if crm_df is not None and len(crm_df) > 0:
        filled = filled.merge(
            crm_df[["week_start", "Geography", "Channel", "Closed_Won"]],
            on=["week_start", "Geography", "Channel"],
            how="left",
        )
    else:
        filled["Closed_Won"] = 0
    filled["Closed_Won"] = filled["Closed_Won"].fillna(0).astype(int)

    # Format dates
    filled["Date"] = pd.to_datetime(filled["week_start"]).dt.strftime("%Y-%m-%d")
    result = filled[[
        "Date", "Geography", "Channel", "Spend",
        "Enterprise_Trials", "Closed_Won", "Campaign_Metadata",
    ]].sort_values(["Date", "Geography", "Channel"]).reset_index(drop=True)

    return result


# =========================================================================
# Main ETL: Snowflake mode
# =========================================================================

def run_snowflake_etl(
    start_date: str = "2024-01-01",
    end_date: str = "2025-12-31",
    output_path: Optional[str] = None,
    include_leads: bool = True,
    include_crm: bool = True,
) -> pd.DataFrame:
    """Full ETL pipeline pulling data directly from Snowflake."""
    out_dir = Path(__file__).parent / "data"
    if output_path is None:
        output_path = str(out_dir / "real_marketing_data.csv")

    conn = get_snowflake_connection()
    try:
        spend_df = extract_spend(conn, start_date, end_date)

        leads_df = None
        if include_leads:
            leads_df = extract_leads(conn, start_date, end_date)

        crm_df = None
        if include_crm:
            crm_df = extract_closed_won(conn, start_date, end_date)
    finally:
        conn.close()

    # Normalize week_start types to datetime across all dataframes
    spend_df["week_start"] = pd.to_datetime(spend_df["week_start"])
    if leads_df is not None and len(leads_df) > 0:
        leads_df["week_start"] = pd.to_datetime(leads_df["week_start"])
    if crm_df is not None and len(crm_df) > 0:
        crm_df["week_start"] = pd.to_datetime(crm_df["week_start"])

    # Print source summaries
    print(f"\n  Spend: {len(spend_df):,} rows, "
          f"{spend_df['Geography'].nunique()} geos, "
          f"{spend_df['Channel'].nunique()} channels")
    if leads_df is not None and len(leads_df) > 0:
        print(f"  Leads: {len(leads_df):,} rows, "
              f"{leads_df['Attributed_Leads'].sum():,} total attributed")
    if crm_df is not None and len(crm_df) > 0:
        print(f"  CRM:   {len(crm_df):,} rows, "
              f"{crm_df['Closed_Won'].sum():,} total won deals")

    result = build_dense_grid(spend_df, crm_df, leads_df)
    result.to_csv(output_path, index=False)
    logger.info("Wrote %d rows to %s", len(result), output_path)

    return result


# =========================================================================
# CSV fallback mode (preserved from previous version)
# =========================================================================

def _safe(val) -> str:
    if pd.isna(val) or str(val).lower() == "none":
        return ""
    return str(val)


def classify_google_channel(campaign_type: str, campaign_name: str) -> str:
    ctype = str(campaign_type).upper().strip()
    cname = str(campaign_name).lower()
    if ctype == "SEARCH":
        if "brand" in cname and "nonbrand" not in cname and "non-brand" not in cname:
            return "Google Brand"
        return "Google Non-Brand"
    if ctype == "DISPLAY":
        return "Display"
    if ctype == "VIDEO":
        return "YouTube"
    return f"Google {ctype.title()}"


def attribute_crm_channel(campaign_source: str) -> str:
    src = str(campaign_source).lower()
    if "linkedin" in src:
        return "LinkedIn"
    if "brand" in src and ("google" in src or "search" in src) and "nonbrand" not in src:
        return "Google Brand"
    if any(k in src for k in ("nonbrand", "non-brand", "sem", "generic")):
        return "Google Non-Brand"
    if "display" in src or "gdn" in src:
        return "Display"
    if "youtube" in src or "video" in src:
        return "YouTube"
    if "content" in src or "syndication" in src:
        return "Review Sites"
    return "Other"


def run_csv_etl(
    linkedin_path: Optional[str] = None,
    google_path: Optional[str] = None,
    crm_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """CSV-based ETL fallback (reads raw ad platform exports)."""
    raw_dir = Path(__file__).parent / "data" / "raw"
    out_dir = Path(__file__).parent / "data"

    linkedin_path = linkedin_path or str(raw_dir / "linkedin_ads_raw.csv")
    google_path = google_path or str(raw_dir / "google_ads_raw.csv")
    crm_path = crm_path or str(raw_dir / "crm_opportunities.csv")
    output_path = output_path or str(out_dir / "real_marketing_data.csv")

    frames = []

    if Path(linkedin_path).exists():
        df = pd.read_csv(linkedin_path)
        logger.info("LinkedIn raw: %d rows", len(df))
        df["week_start"] = to_week_start(df["date"])
        df["Geography"] = df["country_code"].apply(map_geography)
        df["Channel"] = "LinkedIn"
        agg = df.groupby(["week_start", "Geography", "Channel"]).agg(
            Spend=("spend", "sum"),
            Enterprise_Trials=("conversions", "sum"),
            _cn=("campaign_name", "first"), _ct=("creative_text", "first"),
            _au=("audience_targeting", "first"), _ob=("objective", "first"),
        ).reset_index()
        agg["Campaign_Metadata"] = agg.apply(
            lambda r: f"[Campaign: {r['_cn']}] [Copy: {r['_ct']}] [Audience: {r['_au']}] [Objective: {r['_ob']}]",
            axis=1,
        )
        frames.append(agg[["week_start", "Geography", "Channel", "Spend", "Enterprise_Trials", "Campaign_Metadata"]])

    if Path(google_path).exists():
        df = pd.read_csv(google_path)
        logger.info("Google Ads raw: %d rows", len(df))
        df["week_start"] = to_week_start(df["date"])
        df["Geography"] = df["country_code"].apply(map_geography)
        df["Channel"] = df.apply(
            lambda r: classify_google_channel(r["campaign_type"], r["campaign_name"]), axis=1,
        )
        df["spend_dollars"] = df["cost_micros"] / 1e6
        agg = df.groupby(["week_start", "Geography", "Channel"]).agg(
            Spend=("spend_dollars", "sum"),
            Enterprise_Trials=("conversions", "sum"),
            _cn=("campaign_name", "first"), _hl=("headline", "first"),
            _desc=("description", "first"), _au=("audience_segment", "first"),
            _bid=("bidding_strategy", "first"),
        ).reset_index()
        agg["Campaign_Metadata"] = agg.apply(
            lambda r: f"[Campaign: {r['_cn']}] [Copy: {_safe(r['_hl'])} | {_safe(r['_desc'])}] [Audience: {_safe(r['_au'])}] [Objective: {_safe(r['_bid'])}]",
            axis=1,
        )
        frames.append(agg[["week_start", "Geography", "Channel", "Spend", "Enterprise_Trials", "Campaign_Metadata"]])

    if not frames:
        raise FileNotFoundError("No ad platform CSVs found.")

    spend_df = pd.concat(frames, ignore_index=True)

    crm_df = None
    if Path(crm_path).exists():
        df = pd.read_csv(crm_path)
        logger.info("CRM raw: %d rows", len(df))
        df = df[df["stage_name"] == "Closed Won"].copy()
        df["week_start"] = to_week_start(df["close_date"])
        df["Geography"] = df["account_country_code"].apply(map_geography)
        df["Channel"] = df["campaign_source"].apply(attribute_crm_channel)
        crm_df = df.groupby(["week_start", "Geography", "Channel"]).agg(
            Closed_Won=("opportunity_id", "nunique"),
        ).reset_index()

    result = build_dense_grid(spend_df, crm_df)
    result.to_csv(output_path, index=False)
    logger.info("Wrote %d rows to %s", len(result), output_path)
    return result


# =========================================================================
# Validation
# =========================================================================

def validate_output(df: pd.DataFrame) -> None:
    required_cols = {"Date", "Geography", "Channel", "Spend", "Enterprise_Trials", "Closed_Won", "Campaign_Metadata"}
    missing = required_cols - set(df.columns)
    assert not missing, f"Missing columns: {missing}"

    assert df["Spend"].ge(0).all(), "Negative spend values found"
    assert df["Enterprise_Trials"].ge(0).all(), "Negative trial counts found"
    assert df["Closed_Won"].ge(0).all(), "Negative closed-won counts found"
    assert not df["Campaign_Metadata"].isna().any(), "Null Campaign_Metadata found"
    assert not df["Date"].isna().any(), "Null dates found"
    assert not df["Geography"].isna().any(), "Null geographies found"

    n_geos = df["Geography"].nunique()
    n_weeks = df["Date"].nunique()
    n_channels = df["Channel"].nunique()
    expected_rows = n_geos * n_weeks * n_channels

    assert len(df) == expected_rows, (
        f"Grid not dense: {len(df)} rows but expected "
        f"{n_geos}x{n_weeks}x{n_channels}={expected_rows}"
    )

    print(f"  Validation passed:")
    print(f"    {len(df)} rows = {n_geos} geos x {n_weeks} weeks x {n_channels} channels")
    print(f"    Spend range: ${df['Spend'].min():,.0f} - ${df['Spend'].max():,.0f}")
    print(f"    Total Enterprise_Trials: {df['Enterprise_Trials'].sum():,}")
    print(f"    Total Closed_Won: {df['Closed_Won'].sum():,}")


# =========================================================================
# CLI
# =========================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="NNN Framework ETL Processor")
    parser.add_argument("--mode", choices=["snowflake", "csv"], default="snowflake",
                        help="Data source mode (default: snowflake)")
    parser.add_argument("--start-date", default="2024-01-01",
                        help="Start date for Snowflake queries (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2025-12-31",
                        help="End date for Snowflake queries (YYYY-MM-DD)")
    parser.add_argument("--no-leads", action="store_true",
                        help="Skip leads attribution query")
    parser.add_argument("--no-crm", action="store_true",
                        help="Skip CRM closed-won query")
    parser.add_argument("--output", help="Output CSV path")
    # CSV-mode args
    parser.add_argument("--linkedin", help="LinkedIn Ads raw CSV (csv mode)")
    parser.add_argument("--google", help="Google Ads raw CSV (csv mode)")
    parser.add_argument("--crm", help="CRM CSV (csv mode)")
    args = parser.parse_args()

    print("=" * 60)
    print("NNN Framework - ETL Processor")
    print("=" * 60)

    if args.mode == "snowflake":
        print(f"  Mode: Snowflake (bigbrain)")
        print(f"  Date range: {args.start_date} to {args.end_date}")
        result = run_snowflake_etl(
            start_date=args.start_date,
            end_date=args.end_date,
            output_path=args.output,
            include_leads=not args.no_leads,
            include_crm=not args.no_crm,
        )
    else:
        print(f"  Mode: CSV files")
        result = run_csv_etl(
            linkedin_path=args.linkedin,
            google_path=args.google,
            crm_path=args.crm,
            output_path=args.output,
        )

    print()
    print("--- Output Preview (first 20 rows) ---")
    preview_cols = ["Date", "Geography", "Channel", "Spend", "Enterprise_Trials", "Closed_Won"]
    print(result[preview_cols].head(20).to_string(index=False))
    print()

    print("--- Validation ---")
    validate_output(result)

    print()
    print("--- Integration Check ---")
    try:
        from data_prep import RealDataConfig, prepare_real_data
        out_path = args.output or str(Path(__file__).parent / "data" / "real_marketing_data.csv")
        config = RealDataConfig(csv_path=out_path)
        data = prepare_real_data(config)
        m = data["metadata"]
        print(f"  Tensor shape: {data['tensor'].shape}")
        print(f"  input_dim={m['input_dim']} ({m['n_numeric']} numeric + {m['embed_dim']} embedding)")
        print(f"  Ready for NNNModel(n_channels={len(m['channels'])}, input_dim={m['input_dim']})")
    except Exception as e:
        print(f"  Integration check failed: {e}")

    print("\nDone.")
