import streamlit as st
import pandas as pd
import plotly.express as px

def show_brain_activity_insights(row):

    # Extract EEG values from dataset row
    delta = row["delta"]
    theta = row["theta"]
    alpha = row["lowAlpha"] + row["highAlpha"]
    beta = row["lowBeta"] + row["highBeta"]
    gamma = row["lowGamma"] + row["highGamma"]

    total = delta + theta + alpha + beta + gamma

    if total == 0:
        st.warning("No EEG data available")
        return

    percentages = {
        "Delta": (delta/total)*100,
        "Theta": (theta/total)*100,
        "Alpha": (alpha/total)*100,
        "Beta": (beta/total)*100,
        "Gamma": (gamma/total)*100
    }

    df = pd.DataFrame({
        "Brainwave": percentages.keys(),
        "Percentage": percentages.values()
    })

    colors = {
        "Delta":"red",
        "Theta":"orange",
        "Alpha":"yellow",
        "Beta":"green",
        "Gamma":"blue"
    }

    st.subheader("🧠 Brain Activity Insights")

    # BAR GRAPH
    fig = px.bar(
        df,
        x="Brainwave",
        y="Percentage",
        color="Brainwave",
        text=df["Percentage"].round(2).astype(str)+"%",
        color_discrete_map=colors
    )

    fig.update_layout(
        xaxis_title="EEG Brain Waves",
        yaxis_title="Contribution (%)",
        height=220,
        width=540,
        margin=dict(l=30, r=10, t=25, b=35),
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
        font=dict(size=11)
    )

    st.plotly_chart(fig, use_container_width=True, config={"responsive": True})

    st.info(
    """
Interpretation

• Higher **Theta + Delta → Drowsiness**
• Higher **Beta → Alert state**
• Alpha → Relaxed brain
"""
)