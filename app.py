
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="CLV Return Analysis — Priyanshu Gardharia",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>
.headline-box {
    background: linear-gradient(90deg, #1D9E75, #378ADD);
    padding: 20px 30px;
    border-radius: 10px;
    color: white;
    margin-bottom: 20px;
}
.headline-box h2 {
    color: white;
    margin: 0;
    font-size: 22px;
}
.headline-box p {
    color: rgba(255,255,255,0.9);
    margin: 5px 0 0 0;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# COLORS
# =========================================================
PERSONA_COLORS = {
    "Champion Buyers":  "#1D9E75",
    "Regular Buyers":   "#378ADD",
    "Whale Buyer":      "#7F77DD",
    "Extreme Returner": "#D85A30"
}

PERSONA_ORDER = [
    "Champion Buyers",
    "Regular Buyers",
    "Whale Buyer",
    "Extreme Returner"
]

# =========================================================
# LOAD DATA — SAFE VERSION WITH ERROR HANDLING
# =========================================================
@st.cache_data
def load_data():
    try:
        cust       = pd.read_csv("clv_master_final.csv")
        persona_df = pd.read_csv("clv_persona_summary.csv")
        segment_df = pd.read_csv("clv_segment_summary.csv")
        bucket_df  = pd.read_csv("clv_bucket_summary.csv")
        return cust, persona_df, segment_df, bucket_df
    except Exception as e:
        st.error(f"Error loading data files: {e}")
        st.stop()

cust, persona_df, segment_df, bucket_df = load_data()

# =========================================================
# CALCULATE LIVE HEADLINE NUMBERS FROM DATA
# =========================================================
total_loss_all    = cust["Return_Loss"].sum()
whale_loss_all    = cust[cust["Persona"]=="Whale Buyer"]["Return_Loss"].sum()
whale_count_all   = cust[cust["Persona"]=="Whale Buyer"].shape[0]
whale_pct_all     = whale_count_all / len(cust) * 100
whale_loss_pct    = whale_loss_all / max(total_loss_all, 1) * 100
extreme_count_all = cust[cust["Persona"]=="Extreme Returner"].shape[0]
extreme_clv_all   = cust[cust["Persona"]=="Extreme Returner"]["CLV_True"].sum()

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("Dashboard Filters")
st.sidebar.markdown("---")

selected_personas = st.sidebar.multiselect(
    "Filter by Persona",
    options=PERSONA_ORDER,
    default=PERSONA_ORDER
)

selected_buckets = st.sidebar.multiselect(
    "Filter by Price Bucket",
    options=["Low", "Mid", "High"],
    default=["Low", "Mid", "High"]
)

selected_segments = st.sidebar.multiselect(
    "Filter by RFM Segment",
    options=cust["Segment"].unique().tolist(),
    default=cust["Segment"].unique().tolist()
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About This Project")
st.sidebar.markdown(
    "10-day data science project analysing "
    f"{len(cust):,} e-commerce customers to reveal "
    f"£{total_loss_all:,.0f} in hidden return losses."
)
st.sidebar.markdown("**Tools:** Python · XGBoost · KMeans · Streamlit")
st.sidebar.markdown("**Model:** XGBoost R²=0.999")
st.sidebar.markdown("**Author:** Priyanshu Gardharia")
st.sidebar.markdown(
    "[GitHub Repo](https://github.com/Priyanshu-Gardharia/"
    "clv-pricing-return-analysis)"
)

# =========================================================
# FILTER DATA
# =========================================================
filtered = cust[
    (cust["Persona"].isin(selected_personas)) &
    (cust["Price_Bucket"].isin(selected_buckets)) &
    (cust["Segment"].isin(selected_segments))
].copy()

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="headline-box">
<h2>Customer Lifetime Value with Pricing & Return Behavior Analysis</h2>
<p>
541,909 transactions · 4,338 customers · UK Retail Dataset (2010-2011) ·
XGBoost R²=0.999
</p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# HEADLINE ALERT — CALCULATED FROM DATA
# =========================================================
st.error(
    f"🚨 **Headline Finding:** {whale_count_all} Whale Buyers "
    f"({whale_pct_all:.1f}% of customers) are responsible for "
    f"**£{whale_loss_all:,.0f} — {whale_loss_pct:.1f}% of all "
    f"return losses (£{total_loss_all:,.0f} total).** "
    f"Standard CLV models completely miss this."
)

st.markdown("---")

# =========================================================
# KPI CARDS — ALL CALCULATED FROM FILTERED DATA
# =========================================================
st.subheader("Key Metrics")

k1, k2, k3, k4, k5 = st.columns(5)

total_customers = len(filtered)
gross_revenue   = filtered["Monetary"].sum()
net_revenue     = filtered["Adjusted_CLV"].sum()
return_loss     = filtered["Return_Loss"].sum()
avg_return_rate = filtered["Return_Value_Rate"].mean()

with k1:
    st.metric(
        "Total Customers",
        f"{total_customers:,}",
        f"{total_customers/max(len(cust),1)*100:.1f}% of all"
    )

with k2:
    st.metric(
        "Gross Revenue",
        f"£{gross_revenue:,.0f}"
    )

with k3:
    st.metric(
        "Net Revenue",
        f"£{net_revenue:,.0f}",
        f"-£{max(gross_revenue - net_revenue, 0):,.0f}",
        delta_color="inverse"
    )

with k4:
    loss_pct = (
        return_loss /
        max(filtered["CLV_Gross"].sum(), 1) * 100
    )
    st.metric(
        "Total Return Loss",
        f"£{return_loss:,.0f}",
        f"{loss_pct:.1f}% of CLV",
        delta_color="inverse"
    )

with k5:
    st.metric(
        "Avg Return Rate",
        f"{avg_return_rate:.2%}"
    )

st.markdown("---")

# =========================================================
# ROW 1 — PERSONA OVERVIEW
# =========================================================
st.subheader("Persona Overview")
col1, col2 = st.columns(2)

with col1:
    persona_counts = (
        filtered["Persona"]
        .value_counts()
        .reindex(PERSONA_ORDER)
        .dropna()
    )

    fig1, ax1 = plt.subplots(figsize=(7, 4))
    bars = ax1.bar(
        persona_counts.index,
        persona_counts.values,
        color=[PERSONA_COLORS.get(p, "#888780")
               for p in persona_counts.index],
        edgecolor="white", linewidth=0.8
    )
    ax1.set_title("Customers per Persona",
                  fontsize=13, fontweight="bold")
    ax1.set_xlabel("Persona")
    ax1.set_ylabel("Number of Customers")
    ax1.tick_params(axis="x", rotation=15)
    for bar, val in zip(bars, persona_counts.values):
        ax1.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 10,
            f"{val:,}", ha="center",
            fontsize=10, fontweight="bold"
        )
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close()

with col2:
    rev_by_persona = (
        filtered.groupby("Persona")["Monetary"]
        .sum()
        .reindex(PERSONA_ORDER)
        .dropna()
    )

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    bars2 = ax2.bar(
        rev_by_persona.index,
        rev_by_persona.values,
        color=[PERSONA_COLORS.get(p, "#888780")
               for p in rev_by_persona.index],
        edgecolor="white", linewidth=0.8
    )
    ax2.set_title("Total Gross Revenue by Persona",
                  fontsize=13, fontweight="bold")
    ax2.set_xlabel("Persona")
    ax2.set_ylabel("Revenue (£)")
    ax2.tick_params(axis="x", rotation=15)
    for bar, val in zip(bars2, rev_by_persona.values):
        ax2.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 5000,
            f"£{val:,.0f}", ha="center", fontsize=9
        )
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

st.markdown("---")

# =========================================================
# ROW 2 — CLV ANALYSIS
# =========================================================
st.subheader("CLV Analysis — Gross vs True Value")
col3, col4 = st.columns(2)

with col3:
    clv_compare = (
        filtered.groupby("Persona")[["CLV_Gross","CLV_True"]]
        .mean()
        .reindex(PERSONA_ORDER)
        .dropna()
    )

    fig3, ax3 = plt.subplots(figsize=(7, 4))
    x     = range(len(clv_compare))
    width = 0.35
    ax3.bar(
        [i - width/2 for i in x],
        clv_compare["CLV_Gross"],
        width, label="Gross CLV",
        color="#D85A30", edgecolor="white", alpha=0.85
    )
    ax3.bar(
        [i + width/2 for i in x],
        clv_compare["CLV_True"],
        width, label="True CLV (after returns)",
        color="#1D9E75", edgecolor="white", alpha=0.85
    )
    ax3.set_title(
        "Gross CLV vs True CLV by Persona
(gap = revenue lost to returns)",
        fontsize=12, fontweight="bold"
    )
    ax3.set_xlabel("Persona")
    ax3.set_ylabel("Avg CLV per Customer (£)")
    ax3.set_xticks(list(x))
    ax3.set_xticklabels(clv_compare.index, rotation=15)
    ax3.legend()
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

with col4:
    fig4, ax4 = plt.subplots(figsize=(7, 4))
    for persona, group in filtered.groupby("Persona"):
        ax4.scatter(
            group["CLV_Gross"],
            group["CLV_True"],
            c=PERSONA_COLORS.get(persona, "#888780"),
            label=persona, alpha=0.5, s=20
        )
    max_val = filtered["CLV_Gross"].quantile(0.95) if len(filtered) > 0 else 1
    ax4.plot(
        [0, max_val], [0, max_val],
        "k--", lw=1, alpha=0.4,
        label="Perfect line (zero returns)"
    )
    ax4.set_xlim(0, max_val)
    ax4.set_ylim(0, max_val)
    ax4.set_title(
        "Deceptive Customers — CLV Gross vs True
"
        "(points below diagonal = value lost to returns)",
        fontsize=12, fontweight="bold"
    )
    ax4.set_xlabel("CLV Gross (£) — apparent value")
    ax4.set_ylabel("CLV True (£) — real value")
    ax4.legend(fontsize=7)
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()

st.markdown("---")

# =========================================================
# ROW 3 — RETURN ANALYSIS
# =========================================================
st.subheader("Return Behavior Analysis")
col5, col6 = st.columns(2)

with col5:
    loss_by_persona = (
        filtered.groupby("Persona")["Return_Loss"]
        .sum()
        .reindex(PERSONA_ORDER)
        .dropna()
    )

    fig5, ax5 = plt.subplots(figsize=(7, 4))
    bars5 = ax5.bar(
        loss_by_persona.index,
        loss_by_persona.values,
        color=[PERSONA_COLORS.get(p, "#888780")
               for p in loss_by_persona.index],
        edgecolor="white"
    )
    ax5.set_title(
        "Total Return Loss by Persona
(revenue destroyed by returns)",
        fontsize=12, fontweight="bold"
    )
    ax5.set_xlabel("Persona")
    ax5.set_ylabel("Return Loss (£)")
    ax5.tick_params(axis="x", rotation=15)
    for bar, val in zip(bars5, loss_by_persona.values):
        ax5.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 1000,
            f"£{val:,.0f}", ha="center", fontsize=9
        )
    plt.tight_layout()
    st.pyplot(fig5)
    plt.close()

with col6:
    try:
        heatmap_data = (
            filtered.groupby(["Segment","Price_Bucket"])
            ["Return_Damage"]
            .mean()
            .unstack()
        )
        fig6, ax6 = plt.subplots(figsize=(7, 4))
        sns.heatmap(
            heatmap_data,
            ax=ax6,
            cmap="RdYlGn_r",
            annot=True,
            fmt=".3f",
            linewidths=0.5,
            cbar_kws={"label": "Return Damage Score"}
        )
        ax6.set_title(
            "Return Damage by Segment & Price Bucket
(higher = more damaging)",
            fontsize=12, fontweight="bold"
        )
        ax6.set_xlabel("Price Bucket")
        ax6.set_ylabel("Segment")
        plt.tight_layout()
        st.pyplot(fig6)
        plt.close()
    except Exception as e:
        st.warning(f"Heatmap needs more filter combinations: {e}")

st.markdown("---")

# =========================================================
# ROW 4 — ML MODEL
# =========================================================
st.subheader("Machine Learning — XGBoost Model")
col7, col8 = st.columns(2)

with col7:
    feature_importance = {
        "M_Score":           0.4330,
        "Frequency":         0.2219,
        "Monetary":          0.2133,
        "F_Score":           0.0700,
        "Return_Value_Rate": 0.0320,
        "Return_Damage":     0.0180,
        "Avg_Order_Value":   0.0070,
        "Num_Products":      0.0010,
        "Num_Returns":       0.0008,
        "Return_Freq_Rate":  0.0007,
        "Recency":           0.0006,
        "Avg_Unit_Price":    0.0005,
        "R_Score":           0.0004,
        "High_Value_Buyer":  0.0003
    }

    fi_series  = pd.Series(feature_importance).sort_values()
    threshold  = pd.Series(feature_importance).quantile(0.75)
    colors_fi  = [
        "#D85A30" if v >= threshold else "#378ADD"
        for v in fi_series.values
    ]

    fig7, ax7 = plt.subplots(figsize=(7, 5))
    fi_series.plot(
        kind="barh", ax=ax7,
        color=colors_fi, edgecolor="white"
    )
    ax7.set_title(
        "Feature Importance — What Drives Adjusted CLV?
"
        "(red = top 25% most important)",
        fontsize=12, fontweight="bold"
    )
    ax7.set_xlabel("Importance Score")
    plt.tight_layout()
    st.pyplot(fig7)
    plt.close()

with col8:
    st.markdown("### Model Performance Comparison")

    model_data = pd.DataFrame({
        "Model": ["Linear Regression", "XGBoost"],
        "R²":    [0.9390, 0.9990],
        "RMSE":  [0.5218, 0.0660]
    })

    fig8, axes8 = plt.subplots(1, 2, figsize=(7, 4))
    colors_m = ["#378ADD", "#1D9E75"]

    axes8[0].bar(
        model_data["Model"],
        model_data["R²"],
        color=colors_m, edgecolor="white"
    )
    axes8[0].set_title("R² Score
(higher = better)",
                       fontsize=11, fontweight="bold")
    axes8[0].set_ylim(0.9, 1.01)
    axes8[0].tick_params(axis="x", rotation=15)
    for i, v in enumerate(model_data["R²"]):
        axes8[0].text(i, v + 0.001, f"{v:.4f}",
                      ha="center", fontsize=10)

    axes8[1].bar(
        model_data["Model"],
        model_data["RMSE"],
        color=colors_m, edgecolor="white"
    )
    axes8[1].set_title("RMSE
(lower = better)",
                       fontsize=11, fontweight="bold")
    axes8[1].tick_params(axis="x", rotation=15)
    for i, v in enumerate(model_data["RMSE"]):
        axes8[1].text(i, v + 0.005, f"{v:.4f}",
                      ha="center", fontsize=10)

    plt.tight_layout()
    st.pyplot(fig8)
    plt.close()

    st.success(
        "XGBoost improved R² by **+6.4%** and reduced "
        "prediction error by **8x** over Linear Regression"
    )

st.markdown("---")

# =========================================================
# KEY FINDINGS — ALL CALCULATED FROM DATA
# =========================================================
st.subheader("Key Findings")

f1, f2, f3 = st.columns(3)

with f1:
    st.info(
        f"**Finding 1 — The Hidden Gap**\n\n"
        f"£{total_loss_all:,.0f} separates gross CLV "
        f"from true CLV. Standard CLV models that delete "
        f"cancelled orders completely miss this."
    )

with f2:
    st.warning(
        f"**Finding 2 — The {whale_pct_all:.1f}% Problem**\n\n"
        f"{whale_count_all} Whale Buyers cause "
        f"£{whale_loss_all:,.0f} — {whale_loss_pct:.1f}% of "
        f"all return losses. They appear most valuable "
        f"but are most damaging."
    )

with f3:
    st.error(
        f"**Finding 3 — Extreme Returners**\n\n"
        f"{extreme_count_all} customers have true CLV of "
        f"only £{extreme_clv_all:,.0f} total. "
        f"89.5% of their apparent value wiped by returns."
    )

st.markdown("---")

# =========================================================
# DATA EXPLORER
# =========================================================
st.subheader("Customer Data Explorer")

col_s1, col_s2, col_s3 = st.columns(3)

with col_s1:
    sort_by = st.selectbox(
        "Sort by",
        ["CLV_True","CLV_Gross","Return_Loss",
         "Monetary","Return_Value_Rate",
         "Recency","Return_Damage"]
    )
with col_s2:
    sort_order = st.radio(
        "Order",
        ["Descending","Ascending"],
        horizontal=True
    )
with col_s3:
    show_rows = st.slider(
        "Rows to show",
        min_value=10,
        max_value=100,
        value=25,
        step=5
    )

display_cols = [
    "CustomerID","Persona","Segment",
    "Recency","Frequency","Monetary",
    "Adjusted_CLV","CLV_True","Return_Loss",
    "Return_Value_Rate","Price_Bucket","Return_Damage"
]

ascending  = sort_order == "Ascending"
table_data = (
    filtered[display_cols]
    .sort_values(sort_by, ascending=ascending)
    .head(show_rows)
    .reset_index(drop=True)
)

table_display = table_data.copy()
for col in ["Monetary","Adjusted_CLV","CLV_True","Return_Loss"]:
    table_display[col] = table_display[col].apply(
        lambda x: f"£{x:,.0f}"
    )
table_display["Return_Value_Rate"] = table_display[
    "Return_Value_Rate"
].apply(lambda x: f"{x:.2%}")
table_display["Return_Damage"] = table_display[
    "Return_Damage"
].apply(lambda x: f"{x:.3f}")

st.dataframe(table_display, use_container_width=True, height=400)

st.markdown("---")

# =========================================================
# FOOTER
# =========================================================
st.markdown(
    "<center>"
    "Built by <b>Priyanshu Gardharia</b> · "
    "UK Retail Dataset (Kaggle) · "
    "Python · XGBoost · KMeans · Streamlit · "
    "<a href='https://github.com/Priyanshu-Gardharia/"
    "clv-pricing-return-analysis'>GitHub Repository</a>"
    "</center>",
    unsafe_allow_html=True
)
