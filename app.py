# app.py
# Talent Match Intelligence - Streamlit (Step 3, dynamic + AI hook)
# Requirements:
#   pip install streamlit pandas numpy seaborn matplotlib plotly openai (openai optional)
# Files expected in same folder:
#   - final_match_results.csv  (required)
#   - merged_dataset.csv       (optional but recommended for accurate dynamic baselines)

import streamlit as st
import pandas as pd
import numpy as np
import uuid
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os
import json

st.set_page_config(page_title="Talent Match Intelligence", layout="wide", initial_sidebar_state="expanded")

# -------------------------
# Helper: load data
# -------------------------
@st.cache_data
def load_final():
    return pd.read_csv("final_match_results.csv")

@st.cache_data
def load_merged_if_exists():
    if os.path.exists("merged_dataset.csv"):
        return pd.read_csv("merged_dataset.csv")
    return None

df_final = load_final()
df_merged = load_merged_if_exists()

# Ensure columns we expect exist (some tolerant naming)
st.header("Talent Match Intelligence — Interactive App")
st.caption("Fill Role Info > pick benchmark employees > Generate profile & ranking")

# -------------------------
# Sidebar: Inputs
# -------------------------
st.sidebar.header("Role & Benchmark Inputs")

role_name = st.sidebar.text_input("Role name (e.g., Data Analyst)")
job_level = st.sidebar.selectbox("Job level", ["Intern", "Junior", "Middle", "Senior", "Lead", "Manager"], index=2)
role_purpose = st.sidebar.text_area("Role purpose (1-2 sentences)")

# select benchmarks from final dataset; use employee_id + name for clarity
if 'employee_id' in df_final.columns and 'fullname' in df_final.columns:
    bench_options = df_final.apply(lambda r: f"{r['employee_id']} — {r['fullname']}", axis=1).tolist()
    selected_bench = st.sidebar.multiselect("Select benchmark employees (max 5)", bench_options, max_selections=5)
else:
    selected_bench = st.sidebar.multiselect("Select benchmark employees", [], max_selections=5)

st.sidebar.write("Tips: choose 1–3 high-performers (rating=5) to define the benchmark profile.")
submit = st.sidebar.button("Generate Job Profile & Compute Matches")

# -------------------------
# Utility functions
# -------------------------
def parse_bench_selected(selected_list):
    # returns list of employee_ids parsed from strings like "EMP100 — Name"
    ids = []
    for s in selected_list:
        if isinstance(s, str) and "—" in s:
            ids.append(s.split("—")[0].strip())
        else:
            ids.append(s)
    return ids

def save_benchmark_entry(job_vacancy_id, role_name, job_level, role_purpose, benchmark_ids, weights, timestamp=None):
    # Append to local CSV (talent_benchmarks.csv)
    fname = "talent_benchmarks.csv"
    row = {
        "job_vacancy_id": job_vacancy_id,
        "role_name": role_name,
        "job_level": job_level,
        "role_purpose": role_purpose,
        "selected_bench_ids": json.dumps(benchmark_ids),
        "weights": json.dumps(weights),
        "timestamp": timestamp or datetime.utcnow().isoformat()
    }
    df_row = pd.DataFrame([row])
    if os.path.exists(fname):
        df_row.to_csv(fname, mode='a', header=False, index=False)
    else:
        df_row.to_csv(fname, index=False)

def compute_baselines_from_merged(merged_df, bench_ids, tv_list):
    bench_rows = merged_df[merged_df['employee_id'].isin(bench_ids)]
    medians = {}
    for tv in tv_list:
        if tv in bench_rows.columns:
            medians[tv] = bench_rows[tv].median()
        else:
            medians[tv] = np.nan
    return medians

def compute_baselines_from_final(final_df, bench_ids, tv_match_cols):
    # fallback: use match% columns present in final_df for bench items, invert to get baseline %
    bench_rows = final_df[final_df['employee_id'].isin(bench_ids)]
    medians = {}
    for col in tv_match_cols:
        if col in bench_rows.columns:
            medians[col] = bench_rows[col].median()
        else:
            medians[col] = np.nan
    return medians

def compute_matches(merged_df, medians, tvs, weights):
    # merged_df contains raw tvs (iq, pauli, gtq, etc.). We'll compute match% per TV using formula:
    # numeric: match = (value / median_baseline) * 100  (if median>0)
    df = merged_df.copy()
    for tv in tvs:
        baseline = medians.get(tv, np.nan)
        out_col = f"{tv}_match"
        if pd.isna(baseline) or baseline == 0:
            df[out_col] = np.nan
        else:
            df[out_col] = (df[tv] / baseline) * 100
    # compute TGV-level aggregation and final weighted score
    # Here we treat each tv as its own TGV for simplicity, or group if user provides mapping
    # Weighted average across tvs using weights dict
    weight_sum = sum(weights.get(tv, 0) for tv in tvs)
    if weight_sum == 0:
        # equal weights fallback
        weight_list = {tv: 1/len(tvs) for tv in tvs}
    else:
        weight_list = {tv: weights.get(tv, 0)/weight_sum for tv in tvs}
    df['final_match_rate'] = 0
    for tv in tvs:
        df['final_match_rate'] += df[f"{tv}_match"].fillna(0) * weight_list.get(tv, 0)
    return df

def generate_job_profile_via_llm(role_name, job_level, role_purpose, success_formula, top_talents, api_key=None):
    # Placeholder: integrate with OpenAI / OpenRouter here.
    # For the case study submission, we provide a local deterministic fallback if api_key is not provided.
    if api_key:
        # Example using openai (user must pip install openai and set key)
        try:
            import openai
            openai.api_key = api_key
            prompt = f"Generate a concise job profile for role '{role_name}' (level: {job_level}). " \
                     f"Purpose: {role_purpose}\n\nSuccess factors: {json.dumps(success_formula)}\n" \
                     f"Top talent characteristics: {top_talents}\n\nProvide: 1-sentence role summary, 6 key responsibilities, 6 competencies."
            resp = openai.ChatCompletion.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}], max_tokens=400)
            text = resp['choices'][0]['message']['content']
            return text
        except Exception as e:
            return f"[LLM ERROR] {e}\nFallback: Role: {role_name}; SuccessFormula: {success_formula}; Top talents: {top_talents}"
    else:
        # deterministic fallback description
        summary = f"{role_name} ({job_level}) — {role_purpose or 'Role purpose not provided.'}"
        bullets = [
            "Analyze data and produce actionable insights",
            "Build and maintain dashboards and reports",
            "Translate business questions into analytic workflows",
            "Collaborate with stakeholders to define KPIs",
            "Ensure data quality and documentation",
            "Coach team/build analytics capability"
        ]
        comp = ["SQL", "Python/Pandas", "Data Visualization", "Problem Solving", "Stakeholder Communication", "Bias-aware analysis"]
        return "SUMMARY:\n" + summary + "\n\nKEY RESPONSIBILITIES:\n- " + "\n- ".join(bullets) + "\n\nKEY COMPETENCIES:\n- " + "\n- ".join(comp)

# -------------------------
# When user hits Submit
# -------------------------
if submit:
    # validate input
    if not role_name or not selected_bench:
        st.error("Please provide a role name and select at least 1 benchmark employee.")
    else:
        # parse benchmark ids
        bench_ids = parse_bench_selected(selected_bench)
        job_vacancy_id = str(uuid.uuid4())[:8]
        st.success(f"Created job_vacancy_id: {job_vacancy_id}")

        # determine TV list and weights — for demo we use IQ, Pauli, GTQ as example TVs
        # In your report you can map more TVs and TGVs
        tvs = ['iq', 'pauli', 'gtq']  # raw measurement names expected in merged_dataset
        # default weights from your Success Formula (adjustable in-app later)
        default_weights = {'iq': 0.4, 'pauli': 0.3, 'gtq': 0.3}

        # Save benchmark entry
        save_benchmark_entry(job_vacancy_id, role_name, job_level, role_purpose, bench_ids, default_weights)

        # Compute baselines (prefer merged dataset raw values)
        if df_merged is not None:
            st.info("Computing medians from merged_dataset.csv (recommended, using raw IQ/Pauli/GTQ).")
            medians = compute_baselines_from_merged(df_merged, bench_ids, tvs)
            st.write("Baselines (medians):", medians)
            # compute matches across whole merged dataset
            result_df = compute_matches(df_merged, medians, tvs, default_weights)
            # add basic profile info (fullname etc) if present
            if 'fullname' not in result_df.columns and 'employee_id' in df_final.columns:
                # try to join with final to get names
                result_df = result_df.merge(df_final[['employee_id','fullname']], on='employee_id', how='left')
        else:
            # fallback: use final_match_results' match% columns as baselines
            st.warning("merged_dataset.csv not found. Using final_match_results.csv fallback (uses *_match columns if present).")
            # try to find match columns in final df (like iq_match, pauli_match, gtq_match)
            tv_match_cols = []
            for tv in tvs:
                col = f"{tv}_match"
                if col in df_final.columns:
                    tv_match_cols.append(col)
            if len(tv_match_cols) >= 1:
                medians = compute_baselines_from_final(df_final, bench_ids, tv_match_cols)
                st.write("Fallback baselines (median of match% in final data):", medians)
                # compute new 'final' by rescaling: we'll interpret final_df's match% as proxies and recompute weighted average
                # start from df_final copy
                result_df = df_final.copy()
                # ensure match cols exist; if not, attempt to compute from raw
                for col in tv_match_cols:
                    # keep as-is
                    pass
                # compute final_score from medians (this is an approximation)
                # Normalize each match column by (col / medians[col]) * 100 -> not strictly necessary; instead compute weighted avg of existing match% using default_weights mapped
                # Map tvs->match cols dict
                mapping = {tv: f"{tv}_match" for tv in tvs if f"{tv}_match" in df_final.columns}
                # compute weighted final
                weight_sum = sum(default_weights[tv] for tv in mapping.keys()) or 1
                result_df['final_match_rate_recomputed'] = 0
                for tv, col in mapping.items():
                    w = default_weights.get(tv, 0)/weight_sum
                    result_df['final_match_rate_recomputed'] += result_df[col].fillna(0) * w
                # set final_match_rate for display
                result_df['final_match_rate_display'] = result_df['final_match_rate_recomputed'].round(2)
                # unify naming
                result_df = result_df.rename(columns={'final_match_rate_display':'final_match_rate'})
            else:
                st.error("Cannot compute baselines: neither merged_dataset.csv nor *_match columns in final_match_results.csv are present.")
                st.stop()

        # -------------------------
        # Rank and show top results
        # -------------------------
        # ensure final_match_rate column exists
        if 'final_match_rate' not in result_df.columns:
            st.error("final_match_rate column missing after computation.")
        else:
            ranked = result_df.sort_values('final_match_rate', ascending=False).reset_index(drop=True)
            st.subheader("✅ Ranked Talent List (Top 50)")
            cols_to_show = ['employee_id','fullname'] + [c for c in result_df.columns if c.endswith('_match')] + ['final_match_rate']
            cols_to_show = [c for c in cols_to_show if c in ranked.columns]
            st.dataframe(ranked[cols_to_show].head(50))

            # Top 5 summary
            top5 = ranked.head(5)
            st.markdown("### Top 5 Candidates")
            for i, row in top5.iterrows():
                st.write(f"**{i+1}. {row.get('fullname', row.get('employee_id'))}** — Final Match: {row['final_match_rate']:.2f}%")
                # show top TV contributions
                match_cols = [c for c in row.index if c.endswith('_match')]
                contrib = {mc: row[mc] for mc in match_cols}
                sorted_contrib = dict(sorted(contrib.items(), key=lambda x: x[1], reverse=True))
                st.write("Top TVs:", sorted_contrib)

            # Visuals: distribution of final_match_rate
            fig = px.histogram(ranked, x='final_match_rate', nbins=30, title='Distribution of Final Match Rate')
            st.plotly_chart(fig, use_container_width=True)

            # Heatmap of top gaps (top N employees vs benchmark medians) - only if raw merged available
            if df_merged is not None:
                st.subheader("Heatmap: Top employees vs benchmark median (raw TVs)")
                topN = ranked.head(10)['employee_id'].tolist()
                raw_top = df_merged[df_merged['employee_id'].isin(topN)]
                # create matrix of tv values vs median
                tv_matrix = pd.DataFrame([raw_top.set_index('employee_id')[tvs].T.mean().to_dict()])
                # but better: create for each employee
                raw_top2 = raw_top.set_index('employee_id')[tvs]
                med = pd.Series(medians)
                # percent gap: (employee - median)
                gap = ((raw_top2 - med)/med*100).T  # rows: tvs, cols: employee_id
                plt.figure(figsize=(10,4))
                sns.heatmap(gap, annot=True, cmap='RdYlGn', center=0)
                st.pyplot(plt.gcf())

            # Radar: allow compare any two employees
            st.subheader("Radar: Compare any two employees (IQ/Pauli/GTQ match%)")
            emp_list = ranked['fullname'].fillna(ranked['employee_id']).tolist()
            c1 = st.selectbox("Candidate A", emp_list, index=0, key="radar1")
            c2 = st.selectbox("Candidate B", emp_list, index=1 if len(emp_list)>1 else 0, key="radar2")
            r1 = ranked[ranked['fullname']==c1].head(1)
            r2 = ranked[ranked['fullname']==c2].head(1)
            if r1.empty or r2.empty:
                st.info("Select candidates available in table.")
            else:
                categories = [tv + " match" for tv in tvs]
                v1 = [r1.iloc[0].get(tv + "_match", 0) for tv in tvs]
                v2 = [r2.iloc[0].get(tv + "_match", 0) for tv in tvs]
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(r=v1, theta=categories, fill='toself', name=c1))
                fig_radar.add_trace(go.Scatterpolar(r=v2, theta=categories, fill='toself', name=c2))
                fig_radar.update_layout(polar=dict(radialaxis=dict(range=[0, max(120, max(v1+v2)+10)])))
                st.plotly_chart(fig_radar, use_container_width=True)

                        # Download results
            st.download_button("Download Ranked Results (CSV)", data=ranked.to_csv(index=False).encode('utf-8'), file_name=f"ranked_{job_vacancy_id}.csv")
