import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Procurement Cartel Forensic Dashboard",
    page_icon="🔎",
    layout="wide"
)


BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = (
    BASE_DIR
    / "dashboard_data"
    / "full_tender_risk_scores.parquet"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    if not DATA_FILE.exists():

        st.error(
            f"""
            Dashboard data not found.

            Expected:

            {DATA_FILE}

            Run create_full_dashboard_scores.py first.
            """
        )

        st.stop()

    return pd.read_parquet(DATA_FILE)


df = load_data()


# ============================================================
# BASIC PREPARATION
# ============================================================

df["final_risk_score"] = pd.to_numeric(
    df["final_risk_score"],
    errors="coerce"
)

df["investigation_rank"] = pd.to_numeric(
    df["investigation_rank"],
    errors="coerce"
)


valid = df[
    df["final_risk_score"].notna()
].copy()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🔎 Forensic Dashboard")

st.sidebar.markdown(
    """
    ### Procurement Cartel Detection

    **Post-event forensic screening**

    The system ranks completed procurement
    tenders according to evidence of possible
    collusive behaviour.
    """
)

st.sidebar.markdown("---")

screen = st.sidebar.radio(
    "Dashboard Screen",
    [
        "1. Overview",
        "2. Investigation Ranking",
        "3. Tender Forensic Detail"
    ]
)

st.sidebar.markdown("---")

st.sidebar.warning(
    """
    **Important**

    A high risk score does not prove fraud
    or cartel activity.

    It indicates that the tender deserves
    further forensic investigation.
    """
)


# ============================================================
# SCREEN 1 — OVERVIEW
# ============================================================

if screen == "1. Overview":

    st.title(
        "Procurement Cartel Forensic Dashboard"
    )

    st.caption(
        "Post-event forensic screening and investigation prioritization"
    )


    # --------------------------------------------------------
    # MAIN METRICS
    # --------------------------------------------------------

    total_tenders = len(df)

    usable_scores = len(valid)

    high_risk = int(
        (valid["risk_level"] == "High").sum()
    )

    medium_risk = int(
        (valid["risk_level"] == "Medium").sum()
    )

    low_risk = int(
        (valid["risk_level"] == "Low").sum()
    )


    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.metric(
            "Total Completed Tenders",
            f"{total_tenders:,}"
        )


    with c2:

        st.metric(
            "Usable Risk Scores",
            f"{usable_scores:,}"
        )


    with c3:

        st.metric(
            "High-Risk Tenders",
            f"{high_risk:,}"
        )


    with c4:

        st.metric(
            "Modules Available",
            "M1–M4"
        )


    st.markdown("---")


    # --------------------------------------------------------
    # FINAL EVALUATION RESULT
    # --------------------------------------------------------

    st.subheader(
        "Validated Model Performance"
    )


    st.caption(
        "These figures come from the separate labelled chronological evaluation."
    )


    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.metric(
            "ROC-AUC",
            "0.7559"
        )


    with c2:

        st.metric(
            "Average Precision",
            "0.0937"
        )


    with c3:

        st.metric(
            "Top-5% Precision",
            "13.50%"
        )


    with c4:

        st.metric(
            "Top-5% Lift",
            "3.73×"
        )


    st.info(
        """
        Evaluation and investigation are separated:
        labelled historical records are used to validate
        the method, while the dashboard can rank completed
        tenders without requiring a cartel label.
        """
    )


    # --------------------------------------------------------
    # RISK DISTRIBUTION
    # --------------------------------------------------------

    st.subheader(
        "Current Investigation Risk Distribution"
    )


    risk_df = pd.DataFrame({
        "Risk Level": [
            "High",
            "Medium",
            "Low"
        ],
        "Tenders": [
            high_risk,
            medium_risk,
            low_risk
        ]
    })


    fig = px.bar(
        risk_df,
        x="Risk Level",
        y="Tenders",
        title="Risk Categories Across Completed Tenders"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # --------------------------------------------------------
    # MODULE AVAILABILITY
    # --------------------------------------------------------

    st.subheader(
        "Forensic Evidence Availability"
    )


    availability = []


    for module in ["m1", "m2", "m3", "m4"]:

        col = f"{module}_rank"

        count = int(
            df[col].notna().sum()
        )

        availability.append({
            "Module": module.upper(),
            "Available Tenders": count
        })


    availability_df = pd.DataFrame(
        availability
    )


    fig = px.bar(
        availability_df,
        x="Module",
        y="Available Tenders",
        title="Evidence Availability by Module"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # --------------------------------------------------------
    # SYSTEM EXPLANATION
    # --------------------------------------------------------

    st.subheader(
        "How the System Works"
    )


    st.markdown(
        """
        **Completed Tender Data**

        ↓

        **M1 — Price Behaviour**

        **M2 — Competition / Participation**

        **M3 — Winning / Buyer–Bidder Relationships**

        **M4 — Statistical Forensics**

        ↓

        **Evidence Reliability**

        ↓

        **Rank Normalization**

        ↓

        **Reliability-Aware Fusion**

        ↓

        **Investigation Priority Ranking**
        """
    )


# ============================================================
# SCREEN 2 — INVESTIGATION RANKING
# ============================================================

elif screen == "2. Investigation Ranking":

    st.title(
        "Investigation Ranking"
    )

    st.caption(
        "Ranked list of completed tenders requiring forensic attention"
    )


    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

    st.sidebar.subheader(
        "Investigation Filters"
    )


    filtered = valid.copy()


    # Country

    if "country" in filtered.columns:

        countries = sorted(
            filtered["country"]
            .dropna()
            .astype(str)
            .unique()
        )

        selected_country = st.sidebar.multiselect(
            "Country",
            countries
        )

        if selected_country:

            filtered = filtered[
                filtered["country"]
                .astype(str)
                .isin(selected_country)
            ]


    # Year

    if "tender_year" in filtered.columns:

        years = sorted(
            pd.to_numeric(
                filtered["tender_year"],
                errors="coerce"
            )
            .dropna()
            .astype(int)
            .unique()
        )

        selected_year = st.sidebar.multiselect(
            "Tender Year",
            years
        )

        if selected_year:

            filtered = filtered[
                pd.to_numeric(
                    filtered["tender_year"],
                    errors="coerce"
                )
                .isin(selected_year)
            ]


    # Risk

    selected_risk = st.sidebar.multiselect(
        "Risk Level",
        ["High", "Medium", "Low"]
    )


    if selected_risk:

        filtered = filtered[
            filtered["risk_level"]
            .isin(selected_risk)
        ]


    # --------------------------------------------------------
    # TOP-K
    # --------------------------------------------------------

    top_percent = st.slider(
        "Investigation selection (%)",
        min_value=1,
        max_value=20,
        value=5
    )


    number_to_display = max(
        1,
        int(
            np.ceil(
                len(filtered)
                * top_percent
                / 100
            )
        )
    )


    filtered = (
        filtered
        .sort_values(
            "final_risk_score",
            ascending=False
        )
        .head(number_to_display)
        .copy()
    )


    # --------------------------------------------------------
    # TABLE
    # --------------------------------------------------------

    st.subheader(
        f"Top {top_percent}% — {len(filtered):,} Tenders"
    )


    display_columns = [
        "investigation_rank",
        "row_id",
        "country",
        "tender_year",
        "buyer_id",
        "bidder_id",
        "main_cpv_2",
        "lot_bidscount",
        "singleb",
        "available_module_count",
        "final_risk_score",
        "risk_level"
    ]


    display_columns = [
        c for c in display_columns
        if c in filtered.columns
    ]


    table = filtered[
        display_columns
    ].copy()


    table = table.rename(
        columns={
            "investigation_rank": "Rank",
            "row_id": "Tender ID",
            "country": "Country",
            "tender_year": "Year",
            "buyer_id": "Buyer",
            "bidder_id": "Bidder",
            "main_cpv_2": "CPV",
            "lot_bidscount": "Bid Count",
            "singleb": "Single Bid",
            "available_module_count": "Modules",
            "final_risk_score": "Risk Score",
            "risk_level": "Risk"
        }
    )


    table["Risk Score"] = (
        table["Risk Score"]
        .round(4)
    )


    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True
    )


    st.markdown("---")


    # --------------------------------------------------------
    # SCORE DISTRIBUTION
    # --------------------------------------------------------

    st.subheader(
        "Risk Score Distribution"
    )


    fig = px.histogram(
        valid,
        x="final_risk_score",
        nbins=40,
        title="Final Investigation Risk Scores"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    st.caption(
        """
        The ranking is intended to help investigators prioritize
        completed tenders for further examination.
        """
    )


# ============================================================
# SCREEN 3 — FORENSIC DETAIL
# ============================================================

elif screen == "3. Tender Forensic Detail":

    st.title(
        "Tender Forensic Detail"
    )

    st.caption(
        "Evidence breakdown for an individual completed tender"
    )


    # --------------------------------------------------------
    # SELECT TENDER
    # --------------------------------------------------------

    tender_ids = (
        valid["row_id"]
        .dropna()
        .tolist()
    )


    selected_id = st.selectbox(
        "Select Tender ID",
        tender_ids
    )


    tender = valid[
        valid["row_id"] == selected_id
    ].iloc[0]


    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.metric(
            "Tender ID",
            str(tender["row_id"])
        )


    with c2:

        st.metric(
            "Investigation Rank",
            f"#{int(tender['investigation_rank'])}"
        )


    with c3:

        st.metric(
            "Risk Score",
            f"{tender['final_risk_score']:.4f}"
        )


    with c4:

        st.metric(
            "Risk Level",
            tender["risk_level"]
        )


    st.markdown("---")


    # --------------------------------------------------------
    # BASIC INFORMATION
    # --------------------------------------------------------

    st.subheader(
        "Tender Information"
    )


    basic_fields = [
        ("Country", "country"),
        ("Tender Year", "tender_year"),
        ("Buyer", "buyer_id"),
        ("Bidder", "bidder_id"),
        ("CPV Level 2", "main_cpv_2"),
        ("CPV Level 3", "main_cpv_3"),
        ("CPV Level 4", "main_cpv_4"),
        ("Bid Count", "lot_bidscount"),
        ("Single Bid", "singleb"),
        ("Consortium", "bid_isconsortium"),
        ("Subcontracted", "bid_issubcontracted")
    ]


    basic_data = []


    for label, column in basic_fields:

        if column not in tender.index:
            continue

        value = tender[column]

        if pd.isna(value):
            value = "Not available"

        basic_data.append({
            "Field": label,
            "Value": value
        })


    st.dataframe(
        pd.DataFrame(basic_data),
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # PRICE INFORMATION
    # --------------------------------------------------------

    st.subheader(
        "Price Information"
    )


    price_fields = [
        ("Bid Price", "bid_price"),
        ("Lot Estimated Price", "lot_estimatedprice"),
        ("Tender Estimated Price", "tender_estimatedprice"),
        ("Tender Final Price", "tender_finalprice")
    ]


    price_data = []


    for label, column in price_fields:

        if column not in tender.index:
            continue

        value = tender[column]

        if pd.isna(value):
            value = "Not available"

        price_data.append({
            "Field": label,
            "Value": value
        })


    st.dataframe(
        pd.DataFrame(price_data),
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # FORENSIC EVIDENCE
    # --------------------------------------------------------

    st.subheader(
        "Forensic Evidence"
    )


    module_definitions = [
        (
            "M1",
            "Price Behaviour",
            "m1_evidence_score_v3",
            "m1_reliability_v2",
            "m1_rank"
        ),
        (
            "M2",
            "Competition / Participation",
            "m2_evidence_score_v4",
            "m2_reliability_v2",
            "m2_rank"
        ),
        (
            "M3",
            "Winning / Relationship",
            "m3_evidence_score_v3",
            "m3_reliability_v2",
            "m3_rank"
        ),
        (
            "M4",
            "Statistical Forensics",
            "m4_evidence_score_v3",
            "m4_reliability_v2",
            "m4_rank"
        )
    ]


    evidence_data = []


    for module, description, evidence_col, reliability_col, rank_col in module_definitions:

        evidence = tender.get(
            evidence_col,
            np.nan
        )

        reliability = tender.get(
            reliability_col,
            np.nan
        )

        rank = tender.get(
            rank_col,
            np.nan
        )


        evidence_data.append({
            "Module": f"{module} — {description}",
            "Evidence Score": (
                "Unavailable"
                if pd.isna(evidence)
                else round(float(evidence), 4)
            ),
            "Reliability": (
                "Unavailable"
                if pd.isna(reliability)
                else round(float(reliability), 4)
            ),
            "Rank Score": (
                "Unavailable"
                if pd.isna(rank)
                else round(float(rank), 4)
            )
        })


    evidence_df = pd.DataFrame(
        evidence_data
    )


    st.dataframe(
        evidence_df,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # EVIDENCE CHART
    # --------------------------------------------------------

    chart_data = []


    for module, description, evidence_col, _, _ in module_definitions:

        if evidence_col not in tender.index:
            continue

        value = tender[evidence_col]

        if not pd.isna(value):

            chart_data.append({
                "Module": f"{module} — {description}",
                "Evidence Score": float(value)
            })


    if chart_data:

        chart_df = pd.DataFrame(
            chart_data
        )


        fig = px.bar(
            chart_df,
            x="Module",
            y="Evidence Score",
            range_y=[0, 1],
            title="Evidence Score by Module"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # --------------------------------------------------------
    # INVESTIGATION MESSAGE
    # --------------------------------------------------------

    st.subheader(
        "Investigation Interpretation"
    )


    risk = tender["risk_level"]


    if risk == "High":

        st.error(
            """
            **High-priority tender**

            This tender ranks among the higher-risk completed
            tenders according to the forensic evidence fusion.

            It should receive closer investigation.
            """
        )


    elif risk == "Medium":

        st.warning(
            """
            **Medium-priority tender**

            The tender exhibits some evidence patterns that
            may justify additional review.
            """
        )


    else:

        st.success(
            """
            **Lower-priority tender**

            This tender currently has a lower forensic risk
            ranking relative to the other completed tenders.
            """
        )


    st.info(
        """
        **Important:** A forensic risk score is not a finding
        of fraud or cartel activity. It is an investigation
        prioritization signal. Human investigators must examine
        the underlying procurement evidence before drawing conclusions.
        """
    )


    # --------------------------------------------------------
    # LABEL HIDDEN
    # --------------------------------------------------------

    st.caption(
        "Ground-truth cartel labels are intentionally hidden from the investigation interface."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    """
    Procurement Cartel Forensic Screening System |
    Reliability-Aware Rank Fusion |
    Research Prototype
    """
)