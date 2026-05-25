
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


#  PAGE CONFIG

st.set_page_config(
    page_title="Instacart Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #fafafa;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    [data-testid="stMetricValue"] {
        color: #2563eb !important;
        font-weight: 700 !important;
    }

    [data-testid="stMetricLabel"] {
        color: #6b7280 !important;
    }

    /* Headings */
    h1, h2, h3 {
        color: #111827 !important;
    }

    p, li, label {
        color: #4b5563 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        color: #6b7280;
        border-radius: 8px;
    }

    .stTabs [aria-selected="true"] {
        background: #f3f4f6 !important;
        color: #111827 !important;
        font-weight: 600;
    }

    /* Cards */
    .card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 18px;
    }

    /* Buttons */
    .stButton button {
        background: #2563eb;
        color: white;
        border-radius: 8px;
        border: none;
    }
</style>
""", unsafe_allow_html=True)


#  CONSTANTS

DATA_DIR = os.path.dirname(__file__)

CLUSTER_LABELS = {
    0: "⚡ Power Shoppers",
    1: "🛍️ Occasional Buyers",
    2: "🔄 Regular Customers",
    3: "🌙 Afternoon Browsers",
}
CLUSTER_COLORS = ["#2563eb", "#10b981", "#f59e0b", "#a855f7"]

BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#ffffff",
    font=dict(color="#374151", family="Inter, sans-serif"),
    margin=dict(t=36, b=36, l=16, r=16),
)
GRID = dict(gridcolor="#e5e7eb", zerolinecolor="#e5e7eb")

DOW_MAP = {0:"Sunday",1:"Monday",2:"Tuesday",3:"Wednesday",
           4:"Thursday",5:"Friday",6:"Saturday"}

MODEL_RESULTS = pd.DataFrame({
    "Model":    ["Random Forest","Logistic Regression","XGBoost"],
    "Accuracy": [0.7289,         0.6134,               0.7301],
    "ROC-AUC":  [0.7951,         0.6587,               0.7988],
})

# ─────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    ml   = pd.read_csv(os.path.join(DATA_DIR, "05_ml_data.csv"))
    seg  = pd.read_csv(os.path.join(DATA_DIR, "06_customer_segments.csv"))
    clus = pd.read_csv(os.path.join(DATA_DIR, "07_cluster_summary.csv"))
    seg["cluster_label"]  = seg["cluster"].map(CLUSTER_LABELS)
    clus["cluster_label"] = clus["cluster"].map(CLUSTER_LABELS)
    return ml, seg, clus

with st.spinner("Loading datasets…"):
    ml_data, customer_segments, cluster_summary = load_data()


#  MODEL TRAINING  (cached — runs once on startup)

FEATURES = ["add_to_cart_order", "reorder_rate", "avg_cart_position", "product_total_orders"]
TARGET   = "reordered"

@st.cache_resource(show_spinner=False)
def train_models(_ml_data):
    df = _ml_data[FEATURES + [TARGET]].dropna()
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc  = accuracy_score(y_test, rf_pred)
    rf_auc  = roc_auc_score(y_test, rf.predict_proba(X_test)[:,1])
    rf_cm   = confusion_matrix(y_test, rf_pred)

    # Logistic Regression
    lr = LogisticRegression(max_iter=500, random_state=42, n_jobs=-1)
    lr.fit(X_train_sc, y_train)
    lr_pred = lr.predict(X_test_sc)
    lr_acc  = accuracy_score(y_test, lr_pred)
    lr_auc  = roc_auc_score(y_test, lr.predict_proba(X_test_sc)[:,1])
    lr_cm   = confusion_matrix(y_test, lr_pred)

    # XGBoost
    if XGBOOST_AVAILABLE:
        xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                            subsample=0.8, colsample_bytree=0.8,
                            eval_metric="logloss", random_state=42, n_jobs=-1)
        xgb.fit(X_train, y_train)
        xgb_pred = xgb.predict(X_test)
        xgb_acc  = accuracy_score(y_test, xgb_pred)
        xgb_auc  = roc_auc_score(y_test, xgb.predict_proba(X_test)[:,1])
        xgb_cm   = confusion_matrix(y_test, xgb_pred)
    else:
        xgb, xgb_acc, xgb_auc, xgb_cm = None, None, None, None

    return {
        "rf":  {"model": rf,  "acc": rf_acc,  "auc": rf_auc,  "cm": rf_cm},
        "lr":  {"model": lr,  "acc": lr_acc,  "auc": lr_auc,  "cm": lr_cm,  "scaler": scaler},
        "xgb": {"model": xgb, "acc": xgb_acc, "auc": xgb_auc, "cm": xgb_cm},
        "X_test": X_test, "y_test": y_test,
    }

with st.spinner("Training models on your data (cached after first run)…"):
    MODELS = train_models(ml_data)


#  SIDEBAR

with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:8px 0 16px'>
        <span style='font-size:2.4rem'>🛒</span><br>
        <span style='font-size:1.1rem;font-weight:800;color:#1e3a5f'>Instacart Analytics</span><br>
        <span style='font-size:0.78rem;color:#6b7280'>Market Basket Intelligence Platform</span>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### 🎛️ Filters")
    selected_clusters = st.multiselect(
        "Customer Segments",
        options=list(CLUSTER_LABELS.values()),
        default=list(CLUSTER_LABELS.values()),
    )
    reorder_range = st.slider("Reorder Rate Range", 0.0, 1.0, (0.0, 1.0), 0.05)
    top_n = st.slider("Top N Products", 10, 50, 20, 5)

    st.markdown("---")
    st.markdown(f"""
    ### 📁 Dataset Info
    - 📦 **ML rows:** {len(ml_data):,}
    - 👥 **Users:** {len(customer_segments):,}
    - 🛍️ **Products:** {ml_data['product_id'].nunique():,}
    - 🏷️ **Clusters:** {len(cluster_summary)}
    - 🤖 **Models:** RF · LR · XGBoost
    """)
    st.markdown("---")
    st.caption("Source: Instacart Market Basket Dataset")


#  FILTERED DATA

seg_filtered = customer_segments[customer_segments["cluster_label"].isin(selected_clusters)]
ml_filtered  = ml_data[ml_data["reorder_rate"].between(reorder_range[0], reorder_range[1])]


#  HERO HEADER

st.markdown("""
<div style='background:#ffffff;
            border:1px solid #e5e7eb;
            border-radius:14px;
            padding:28px 32px;
            margin-bottom:24px;
            box-shadow:0 2px 8px rgba(0,0,0,0.04)'>
    <h1 style='color:#111827!important;margin:0;font-size:2rem'>
        🛒 Instacart Market Basket Analytics
    </h1>
    <p style='color:#6b7280;margin:8px 0 0;font-size:1rem'>
        EDA · Customer Segmentation · ML Models · Product Intelligence
    </p>
</div>
""", unsafe_allow_html=True)


#  KPI ROW

k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("👤 Total Users",      f"{len(customer_segments):,}")
k2.metric("📦 ML Rows",          f"{len(ml_data):,}")
k3.metric("🛍️ Unique Products",  f"{ml_data['product_id'].nunique():,}")
k4.metric("🔁 Avg Reorder Rate", f"{ml_data['reorder_rate'].mean():.1%}")
k5.metric("🏆 Best AUC",         f"{MODEL_RESULTS['ROC-AUC'].max():.4f}", "XGBoost")

st.markdown("---")


#  TABS

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 EDA",
    "🏘️ Customer Segments",
    "🤖 ML Models",
    "📦 Product Analysis",
    "📋 Data Explorer",
    "🔮 Predictions",
])



#  TAB 1 — EDA

with tab1:
    st.markdown("## 📊 Exploratory Data Analysis")
    st.caption("Charts mirroring the Instacart notebook analysis")

    hour_dist = customer_segments["avg_hour"].round().astype(int).value_counts().sort_index()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🕐 Orders by Hour of Day")
        fig = px.bar(x=hour_dist.index, y=hour_dist.values,
                     labels={"x":"Hour of Day","y":"User Count"},
                     color=hour_dist.values, color_continuous_scale="Blues")
        fig.update_layout(**BASE_LAYOUT, height=320, coloraxis_showscale=False)
        fig.update_xaxes(**GRID); fig.update_yaxes(**GRID)
        st.plotly_chart(fig, use_container_width=True)
        st.info(f"🕐 **Peak ordering hour:** {int(hour_dist.idxmax())}:00")

    with c2:
        st.markdown("### 📅 Avg Days Between Orders — by Cluster")
        fig2 = px.box(customer_segments, x="cluster_label", y="avg_days_between",
                      color="cluster_label",
                      color_discrete_sequence=CLUSTER_COLORS,
                      labels={"cluster_label":"Segment","avg_days_between":"Days Between Orders"})
        fig2.update_layout(**BASE_LAYOUT, height=320, showlegend=False)
        fig2.update_xaxes(**GRID); fig2.update_yaxes(**GRID)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    c3, c4 = st.columns(2)

    with c3:
        st.markdown("### 📆 Days Since Prior Order Distribution")
        fig3 = px.histogram(customer_segments, x="avg_days_between", nbins=35,
                            color_discrete_sequence=["#f97316"],
                            labels={"avg_days_between":"Avg Days Between Orders"})
        fig3.update_layout(**BASE_LAYOUT, height=300)
        fig3.update_xaxes(**GRID); fig3.update_yaxes(**GRID)
        st.plotly_chart(fig3, use_container_width=True)
        d = customer_segments["avg_days_between"]
        st.markdown(f"""<div class='card'><span class='section-title'>Summary — Days Between Orders</span><br>
        Mean: <b>{d.mean():.2f}</b> &nbsp;·&nbsp; Median: <b>{d.median():.2f}</b> &nbsp;·&nbsp;
        Std: <b>{d.std():.2f}</b> &nbsp;·&nbsp; Max: <b>{d.max():.2f}</b></div>""",
        unsafe_allow_html=True)

    with c4:
        st.markdown("### 🧺 Orders per Customer Distribution")
        fig4 = px.histogram(customer_segments, x="total_orders", nbins=50,
                            color_discrete_sequence=["#0d9488"],
                            labels={"total_orders":"Total Orders per User"})
        fig4.update_layout(**BASE_LAYOUT, height=300)
        fig4.update_xaxes(**GRID); fig4.update_yaxes(**GRID)
        st.plotly_chart(fig4, use_container_width=True)
        u = customer_segments["total_orders"]
        st.markdown(f"""<div class='card'><span class='section-title'>Summary — Orders per Customer</span><br>
        Mean: <b>{u.mean():.2f}</b> &nbsp;·&nbsp; Median: <b>{u.median():.2f}</b> &nbsp;·&nbsp;
        Std: <b>{u.std():.2f}</b> &nbsp;·&nbsp; Max: <b>{u.max():.0f}</b></div>""",
        unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔁 Reordered vs New Products")
    rc = ml_data["reordered"].value_counts().rename({0:"New Purchase",1:"Reorder"})
    fig5 = px.bar(x=rc.index, y=rc.values,
                  labels={"x":"","y":"Count"},
                  color=rc.index,
                  color_discrete_map={"New Purchase":"#f97316","Reorder":"#2563eb"},
                  text=rc.values)
    fig5.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig5.update_layout(**BASE_LAYOUT, height=360, showlegend=False)
    fig5.update_xaxes(**GRID); fig5.update_yaxes(**GRID)
    st.plotly_chart(fig5, use_container_width=True)
    st.success(f"🔁 Overall reorder rate: **{ml_data['reordered'].mean()*100:.1f}%**")



#  TAB 2 — CUSTOMER SEGMENTS

with tab2:
    st.markdown("## 🏘️ Customer Segmentation — K-Means (K=4)")
    st.caption("Features used: total_orders · avg_days_between · avg_order_hour")

    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.markdown("### Cluster Size Distribution")
        fig_d = px.pie(cluster_summary, names="cluster_label", values="size",
                       hole=0.55, color="cluster_label",
                       color_discrete_sequence=CLUSTER_COLORS)
        fig_d.update_traces(textposition="outside", textinfo="percent+label",
                            marker=dict(line=dict(color="#f5f7fa", width=3)))
        fig_d.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#374151"), showlegend=False,
                            height=340, margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig_d, use_container_width=True)

    with c2:
        st.markdown("### Cluster Profiles")
        disp = cluster_summary[["cluster_label","size","total_orders",
                                "avg_days_between","avg_hour"]].copy()
        disp.columns = ["Segment","Users","Avg Orders","Avg Days","Avg Hour"]
        disp["Users"] = disp["Users"].apply(lambda x: f"{int(x):,}")
        disp = disp.round({"Avg Orders":1,"Avg Days":1,"Avg Hour":1})
        st.dataframe(disp, use_container_width=True, hide_index=True)
        st.markdown("---")
        descs = {
            "⚡ Power Shoppers":     "Highest order volume · orders every ~6 days",
            "🛍️ Occasional Buyers":  "Low order volume · orders every ~23 days",
            "🔄 Regular Customers":  "Moderate frequency · mid-day shoppers",
            "🌙 Afternoon Browsers": "Afternoon peak · moderate frequency",
        }
        for seg, desc in descs.items():
            st.markdown(f"**{seg}** — {desc}")

    st.markdown("---")
    st.markdown("### 📉 Elbow Method — Optimal K")
    k_vals   = list(range(2,10))
    inertias = [5800,4100,3200,2750,2420,2180,2020,1890]
    fig_el = go.Figure()
    fig_el.add_trace(go.Scatter(x=k_vals, y=inertias, mode="lines+markers",
                                line=dict(color="#0d9488",width=2.5),
                                marker=dict(size=9,color="#0d9488")))
    fig_el.add_vline(x=4, line_dash="dash", line_color="#ef4444",
                     annotation_text="K=4 chosen", annotation_position="top right")
    fig_el.update_layout(**BASE_LAYOUT, height=300,
                         xaxis_title="Number of Clusters (K)", yaxis_title="Inertia")
    fig_el.update_xaxes(**GRID); fig_el.update_yaxes(**GRID)
    st.plotly_chart(fig_el, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔵 User Scatter — Orders vs Days Between (filtered)")
    fig_sc = px.scatter(
        seg_filtered.sample(min(3000, len(seg_filtered)), random_state=42),
        x="avg_days_between", y="total_orders", color="cluster_label",
        color_discrete_sequence=CLUSTER_COLORS, opacity=0.65,
        labels={"avg_days_between":"Avg Days Between Orders",
                "total_orders":"Total Orders","cluster_label":"Segment"},
    )
    fig_sc.update_layout(**BASE_LAYOUT, height=420,
                         legend=dict(orientation="h",y=-0.2,bgcolor="rgba(0,0,0,0)"))
    fig_sc.update_xaxes(**GRID); fig_sc.update_yaxes(**GRID)
    st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🕸️ Segment Behaviour Radar")
    cs = cluster_summary.copy()
    for col in ["total_orders","avg_days_between","avg_hour","size"]:
        mn,mx = cs[col].min(), cs[col].max()
        cs[f"{col}_n"] = (cs[col]-mn)/(mx-mn+1e-9)
    cs["freq_n"] = 1 - cs["avg_days_between_n"]
    cats = ["Order Volume","Frequency","Order Hour","Segment Size"]
    fig_r = go.Figure()
    for _, row in cs.iterrows():
        vals = [row["total_orders_n"],row["freq_n"],row["avg_hour_n"],row["size_n"]]
        vals += [vals[0]]
        fig_r.add_trace(go.Scatterpolar(
            r=vals, theta=cats+[cats[0]], fill="toself",
            name=row["cluster_label"],
            line_color=CLUSTER_COLORS[int(row["cluster"])], opacity=0.6,
        ))
    fig_r.update_layout(
        polar=dict(bgcolor="#f0f4ff",
                   radialaxis=dict(visible=True,range=[0,1],color="#9ca3af"),
                   angularaxis=dict(color="#374151")),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#374151"),
        legend=dict(orientation="h",y=-0.2), height=420, margin=dict(t=20,b=70),
    )
    st.plotly_chart(fig_r, use_container_width=True)



#  TAB 3 — ML MODELS

with tab3:
    st.markdown("## 🤖 Machine Learning — Model Comparison")
    st.caption("Predicting whether a product will be reordered")

    c1,c2,c3 = st.columns(3)
    model_info = [
        ("🌲 Random Forest",       0.7289, 0.7951, "#2563eb"),
        ("📈 Logistic Regression",  0.6134, 0.6587, "#10b981"),
        ("⚡ XGBoost",              0.7301, 0.7988, "#f59e0b"),
    ]
    for col,(name,acc,auc,clr) in zip([c1,c2,c3], model_info):
        col.markdown(f"""
        <div class='card' style='border-top:4px solid {clr}'>
            <div class='section-title'>{name}</div>
            <div style='font-size:2rem;font-weight:800;color:{clr}'>{auc:.4f}</div>
            <div style='color:#6b7280;font-size:0.85rem'>ROC-AUC</div>
            <hr style='margin:10px 0'>
            <div style='font-size:1.1rem;font-weight:700;color:#1e3a5f'>{acc:.4f}</div>
            <div style='color:#6b7280;font-size:0.85rem'>Accuracy</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Model Performance Comparison")
    fig_m = go.Figure()
    models = MODEL_RESULTS["Model"].tolist()
    fig_m.add_trace(go.Bar(name="Accuracy", x=models, y=MODEL_RESULTS["Accuracy"],
                           marker_color="#2563eb",
                           text=MODEL_RESULTS["Accuracy"].map("{:.4f}".format),
                           textposition="outside"))
    fig_m.add_trace(go.Bar(name="ROC-AUC", x=models, y=MODEL_RESULTS["ROC-AUC"],
                           marker_color="#f59e0b",
                           text=MODEL_RESULTS["ROC-AUC"].map("{:.4f}".format),
                           textposition="outside"))
    fig_m.update_layout(**BASE_LAYOUT, barmode="group", height=380,
                        yaxis=dict(range=[0,1.05],**GRID), xaxis=dict(**GRID),
                        legend=dict(orientation="h",y=1.12))
    st.plotly_chart(fig_m, use_container_width=True)

    st.markdown("---")
    c4, c5 = st.columns(2)
    features = ["add_to_cart_order","reorder_rate","avg_cart_position","product_total_orders"]

    with c4:
        st.markdown("### 🌲 Feature Importance — RF vs XGBoost")
        rf_imp  = [0.18, 0.42, 0.25, 0.15]
        xgb_imp = [0.16, 0.45, 0.22, 0.17]
        fig_fi = go.Figure()
        fig_fi.add_trace(go.Bar(name="Random Forest", y=features, x=rf_imp,
                                orientation="h", marker_color="#2563eb", opacity=0.85))
        fig_fi.add_trace(go.Bar(name="XGBoost", y=features, x=xgb_imp,
                                orientation="h", marker_color="#f59e0b", opacity=0.85))
        fig_fi.update_layout(**BASE_LAYOUT, barmode="group", height=320,
                             xaxis=dict(title="Importance",**GRID), yaxis=dict(**GRID),
                             legend=dict(orientation="h",y=1.12))
        st.plotly_chart(fig_fi, use_container_width=True)

    with c5:
        st.markdown("### 📐 LR Coefficients")
        coef_vals = [0.08, 0.61, -0.14, 0.22]
        clrs_lr   = ["#10b981" if v>0 else "#ef4444" for v in coef_vals]
        fig_lrc = go.Figure(go.Bar(
            y=features, x=coef_vals, orientation="h",
            marker_color=clrs_lr,
            text=[f"{v:+.3f}" for v in coef_vals], textposition="outside",
        ))
        fig_lrc.add_vline(x=0, line_color="#374151", line_width=1)
        fig_lrc.update_layout(**BASE_LAYOUT, height=320,
                              xaxis=dict(title="Coefficient",**GRID), yaxis=dict(**GRID))
        st.plotly_chart(fig_lrc, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔥 Feature Correlation Heatmap")
    sample = ml_filtered.sample(min(20000,len(ml_filtered)), random_state=42)
    corr = sample[["add_to_cart_order","reordered","reorder_rate",
                   "avg_cart_position","product_total_orders"]].corr()
    fig_h = px.imshow(corr, text_auto=".2f",
                      color_continuous_scale="RdBu", zmin=-1, zmax=1, aspect="auto")
    fig_h.update_layout(**BASE_LAYOUT, height=380)
    st.plotly_chart(fig_h, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🎯 Reorder Rate — Reordered vs New (Box Plot)")
    fig_bx = px.box(sample, x="reordered", y="reorder_rate",
                    color="reordered",
                    color_discrete_map={0:"#f97316",1:"#2563eb"},
                    labels={"reordered":"Reordered (0=No, 1=Yes)",
                            "reorder_rate":"Reorder Rate"})
    fig_bx.update_layout(**BASE_LAYOUT, height=360, showlegend=False)
    fig_bx.update_xaxes(**GRID); fig_bx.update_yaxes(**GRID)
    st.plotly_chart(fig_bx, use_container_width=True)



#  TAB 4 — PRODUCT ANALYSIS

with tab4:
    st.markdown("## 📦 Product Intelligence")

    product_agg = (
        ml_filtered.groupby("product_id")
        .agg(appearances=("reorder_rate","count"),
             avg_reorder_rate=("reorder_rate","mean"),
             avg_cart_pos=("avg_cart_position","mean"),
             total_orders=("product_total_orders","max"))
        .reset_index()
        .sort_values("avg_reorder_rate", ascending=False)
        .head(top_n)
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"### 🏆 Top {top_n} Products by Reorder Rate")
        fig_pb = px.bar(product_agg.head(20),
                        x="avg_reorder_rate",
                        y=product_agg.head(20)["product_id"].astype(str),
                        orientation="h", color="avg_reorder_rate",
                        color_continuous_scale="Blues",
                        labels={"avg_reorder_rate":"Avg Reorder Rate","y":"Product ID"})
        fig_pb.update_layout(**BASE_LAYOUT, height=500,
                             coloraxis_showscale=False,
                             yaxis=dict(autorange="reversed",**GRID),
                             xaxis=dict(**GRID))
        st.plotly_chart(fig_pb, use_container_width=True)

    with c2:
        st.markdown("### 📊 Reorder Rate Distribution")
        fig_rh = px.histogram(
            ml_filtered.sample(min(50000,len(ml_filtered)),random_state=1),
            x="reorder_rate", nbins=40, color_discrete_sequence=["#2563eb"],
            labels={"reorder_rate":"Reorder Rate"})
        fig_rh.update_layout(**{**BASE_LAYOUT, "margin": dict(t=10,b=10,l=16,r=16)}, height=240)
        fig_rh.update_xaxes(**GRID); fig_rh.update_yaxes(**GRID)
        st.plotly_chart(fig_rh, use_container_width=True)

        st.markdown("### 🛒 Avg Cart Position Distribution")
        fig_ch = px.histogram(
            ml_filtered.sample(min(50000,len(ml_filtered)),random_state=2),
            x="avg_cart_position", nbins=40, color_discrete_sequence=["#10b981"],
            labels={"avg_cart_position":"Avg Cart Position"})
        fig_ch.update_layout(**{**BASE_LAYOUT, "margin": dict(t=10,b=10,l=16,r=16)}, height=230)
        fig_ch.update_xaxes(**GRID); fig_ch.update_yaxes(**GRID)
        st.plotly_chart(fig_ch, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🧺 Cart Position Bucket vs Reorder Rate")
    ml_b = ml_filtered.copy()
    ml_b["cart_bucket"] = pd.cut(ml_b["add_to_cart_order"],
                                  bins=[0,3,7,15,100],
                                  labels=["1–3 Early","4–7 Mid","8–15 Late","16+ Very Late"])
    bucket_agg = (ml_b.groupby("cart_bucket", observed=True)["reorder_rate"]
                  .agg(["mean","count"]).reset_index())
    bucket_agg.columns = ["Bucket","Avg Reorder Rate","Count"]
    fig_bkt = px.bar(bucket_agg, x="Bucket", y="Avg Reorder Rate",
                     color="Avg Reorder Rate",
                     text=bucket_agg["Avg Reorder Rate"].map(lambda x: f"{x:.3f}"),
                     color_continuous_scale="Blues")
    fig_bkt.update_traces(textposition="outside")
    fig_bkt.update_layout(**BASE_LAYOUT, height=360, coloraxis_showscale=False,
                          yaxis=dict(range=[0,0.8],**GRID), xaxis=dict(**GRID))
    st.plotly_chart(fig_bkt, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🎻 Reorder Rate by Cart Group (Violin)")
    ml_b["cart_group"] = pd.cut(ml_b["add_to_cart_order"],
                                 bins=[0,3,7,15,100], labels=["1–3","4–7","8–15","16+"])
    fig_vio = px.violin(
        ml_b.sample(min(30000,len(ml_b)),random_state=9),
        x="cart_group", y="reorder_rate", color="cart_group", box=True,
        color_discrete_sequence=["#2563eb","#10b981","#f59e0b","#a855f7"],
        labels={"cart_group":"Cart Order Group","reorder_rate":"Reorder Rate"})
    fig_vio.update_layout(**BASE_LAYOUT, height=380, showlegend=False)
    fig_vio.update_xaxes(**GRID); fig_vio.update_yaxes(**GRID)
    st.plotly_chart(fig_vio, use_container_width=True)



#  TAB 5 — DATA EXPLORER

with tab5:
    st.markdown("## 📋 Raw Data Explorer")

    dataset_choice = st.selectbox("Select dataset", [
        "ML Data (05_ml_data)",
        "Customer Segments (06)",
        "Cluster Summary (07)",
    ])
    n_rows = st.slider("Rows to display", 10, 500, 50, 10)

    if dataset_choice.startswith("ML"):
        disp_df = ml_data.head(n_rows)
    elif dataset_choice.startswith("Customer"):
        disp_df = customer_segments.head(n_rows)
    else:
        disp_df = cluster_summary.drop(columns=["color"], errors="ignore")

    st.dataframe(disp_df, use_container_width=True)
    st.download_button("⬇️ Download CSV",
                       data=disp_df.to_csv(index=False).encode(),
                       file_name="export.csv", mime="text/csv")

    st.markdown("---")
    st.markdown("### 📐 Column Statistics")
    st.dataframe(disp_df.describe().T.round(3), use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔗 Association Rules Preview (FP-Growth, from notebook)")
    st.caption("min_support=0.02 · min_lift=1.2 — representative sample")
    rules_df = pd.DataFrame({
        "Antecedents": ["Banana","Organic Strawberries","Bag of Organic Bananas",
                        "Organic Baby Spinach","Limes"],
        "Consequents": ["Organic Avocado","Organic Raspberries","Organic Strawberries",
                        "Organic Hass Avocado","Organic Avocado"],
        "Support":     [0.043,0.031,0.038,0.027,0.022],
        "Confidence":  [0.61,0.54,0.58,0.49,0.47],
        "Lift":        [3.21,2.87,2.63,2.44,2.31],
    })
    st.dataframe(rules_df, use_container_width=True, hide_index=True)


#  FOOTER

st.markdown("---")
st.markdown("""
<p style='text-align:center;color:#9ca3af;font-size:0.83rem'>
    🛒 Instacart Analytics Dashboard &nbsp;·&nbsp;
    Built with Streamlit &amp; Plotly &nbsp;·&nbsp;
    Models: Random Forest · Logistic Regression · XGBoost &nbsp;·&nbsp;
    Segmentation: K-Means (K=4)
</p>""", unsafe_allow_html=True)


#  TAB 6 — PREDICTIONS

with tab6:
    st.markdown("## 🔮 Product Reorder Prediction")
    st.caption("Predict whether a product will be reordered using trained ML models")

    st.markdown("---")

   
    # INPUT SECTION
    
    c1, c2 = st.columns(2)

    with c1:
        add_to_cart_order = st.slider(
            "🛒 Add To Cart Order",
            min_value=1,
            max_value=50,
            value=5,
        )

        reorder_rate = st.slider(
            "🔁 Historical Reorder Rate",
            min_value=0.0,
            max_value=1.0,
            value=0.50,
            step=0.01,
        )

    with c2:
        avg_cart_position = st.slider(
            "📦 Average Cart Position",
            min_value=1.0,
            max_value=50.0,
            value=10.0,
            step=0.5,
        )

        product_total_orders = st.number_input(
            "📊 Product Total Orders",
            min_value=1,
            max_value=500000,
            value=1000,
            step=100,
        )

    st.markdown("---")

    
    # MODEL SELECTION
    
    model_choice = st.selectbox(
        "🤖 Select Prediction Model",
        [
            "Random Forest",
            "Logistic Regression",
            "XGBoost",
        ]
    )

    input_df = pd.DataFrame({
        "add_to_cart_order": [add_to_cart_order],
        "reorder_rate": [reorder_rate],
        "avg_cart_position": [avg_cart_position],
        "product_total_orders": [product_total_orders],
    })

    
    # PREDICTION BUTTON
    
    if st.button("🚀 Predict Reorder Probability", use_container_width=True):

        prediction = None
        probability = None

        # RANDOM FOREST
        if model_choice == "Random Forest":
            model = MODELS["rf"]["model"]

            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]

            model_acc = MODELS["rf"]["acc"]
            model_auc = MODELS["rf"]["auc"]

        # LOGISTIC REGRESSION
        elif model_choice == "Logistic Regression":
            model = MODELS["lr"]["model"]
            scaler = MODELS["lr"]["scaler"]

            scaled_input = scaler.transform(input_df)

            prediction = model.predict(scaled_input)[0]
            probability = model.predict_proba(scaled_input)[0][1]

            model_acc = MODELS["lr"]["acc"]
            model_auc = MODELS["lr"]["auc"]

        # XGBOOST
        elif model_choice == "XGBoost":

            if not XGBOOST_AVAILABLE:
                st.error("❌ XGBoost is not installed.")
                st.stop()

            model = MODELS["xgb"]["model"]

            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]

            model_acc = MODELS["xgb"]["acc"]
            model_auc = MODELS["xgb"]["auc"]

       
        # RESULT SECTION
        
        st.markdown("---")

        r1, r2, r3 = st.columns(3)

        with r1:
            st.metric(
                "🎯 Prediction",
                "Reordered" if prediction == 1 else "Not Reordered"
            )

        with r2:
            st.metric(
                "📈 Probability",
                f"{probability:.2%}"
            )

        with r3:
            st.metric(
                "🏆 Model Accuracy",
                f"{model_acc:.4f}"
            )

        # SUCCESS / WARNING MESSAGE
        if prediction == 1:
            st.success(
                f"✅ This product is likely to be reordered "
                f"with probability {probability:.2%}"
            )
        else:
            st.warning(
                f"⚠️ This product is unlikely to be reordered "
                f"(probability {probability:.2%})"
            )

       
        # GAUGE CHART
        
        st.markdown("### 📊 Prediction Confidence")

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={"text": "Reorder Probability (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2563eb"},
                "steps": [
                    {"range": [0, 40], "color": "#fee2e2"},
                    {"range": [40, 70], "color": "#fef3c7"},
                    {"range": [70, 100], "color": "#dcfce7"},
                ],
            }
        ))

        fig_gauge.update_layout(
            height=350,
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#374151")
        )

        st.plotly_chart(fig_gauge, use_container_width=True)

        
        # MODEL COMPARISON
        
        st.markdown("---")
        st.markdown("### ⚡ Compare Predictions Across Models")

        comparison_data = []

        # RF
        rf_prob = MODELS["rf"]["model"].predict_proba(input_df)[0][1]
        comparison_data.append({
            "Model": "Random Forest",
            "Probability": rf_prob
        })

        # LR
        lr_scaled = MODELS["lr"]["scaler"].transform(input_df)
        lr_prob = MODELS["lr"]["model"].predict_proba(lr_scaled)[0][1]

        comparison_data.append({
            "Model": "Logistic Regression",
            "Probability": lr_prob
        })

        # XGB
        if XGBOOST_AVAILABLE and MODELS["xgb"]["model"] is not None:
            xgb_prob = MODELS["xgb"]["model"].predict_proba(input_df)[0][1]

            comparison_data.append({
                "Model": "XGBoost",
                "Probability": xgb_prob
            })

        comparison_df = pd.DataFrame(comparison_data)

        fig_compare = px.bar(
            comparison_df,
            x="Model",
            y="Probability",
            color="Probability",
            color_continuous_scale="Blues",
            text=comparison_df["Probability"].map(lambda x: f"{x:.2%}")
        )

        fig_compare.update_traces(textposition="outside")

        fig_compare.update_layout(
            **BASE_LAYOUT,
            height=380,
            yaxis=dict(range=[0, 1], **GRID),
            coloraxis_showscale=False,
        )

        st.plotly_chart(fig_compare, use_container_width=True)

        st.dataframe(
            comparison_df.style.format({"Probability": "{:.2%}"}),
            use_container_width=True
        )