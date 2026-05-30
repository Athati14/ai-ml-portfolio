"""Streamlit dashboard over eval_results.json. Run: streamlit run dashboard.py"""
import json, pandas as pd, streamlit as st

st.set_page_config(page_title="LLM Eval Dashboard", layout="wide")
st.title("LLM Evaluation & Observability")

df = pd.DataFrame(json.load(open("eval_results.json")))
col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{df['correct'].mean():.1%}")
col2.metric("Avg latency", f"{df['latency_s'].mean():.2f}s")
col3.metric("Total est. cost", f"${df['est_cost'].sum():.4f}")

st.subheader("By model x template")
pivot = df.groupby(["model", "template"]).agg(
    accuracy=("correct", "mean"),
    groundedness=("grounded", "mean"),
    avg_latency=("latency_s", "mean"),
    cost=("est_cost", "sum"),
).reset_index()
st.dataframe(pivot, use_container_width=True)
st.bar_chart(pivot.set_index("template")["accuracy"])

with st.expander("Raw outputs"):
    st.dataframe(df, use_container_width=True)
