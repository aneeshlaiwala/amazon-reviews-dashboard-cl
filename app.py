import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- Utility Functions for Explanations ---
def metric_explanations():
    st.markdown("""
    ### 📖 **Metrics Explained**
    | Metric | What It Means | How It's Calculated / How to Read |
    |--------|---------------|------------------------------------|
    | Trust Score | How reliable and genuine the reviews are. | Based on the percentage of suspicious reviews. Higher is better. |
    | Brand Sentiment | Overall customer feeling (positive/neutral/negative). | Based on review text analysis. |
    | Engagement Rate | How many reviews are detailed and helpful. | % of long, informative reviews. Higher is better. |
    | Business Impact | How much a review can influence others. | Combines rating, sentiment, and review length. |
    | Brand Advocacy | % of customers who strongly recommend the brand. | % of very positive reviews. |
    | Customer Segments | Groups of customers by review behavior. | Derived from review length, sentiment, rating, and risk. |
    """)

def customer_segment_explanations():
    st.markdown("""
    **Customer Segments:**
    - **Engaged Advocate:** Long, detailed, positive reviews.
    - **Satisfied Customer:** High ratings and positive feedback.
    - **Dissatisfied Customer:** Low ratings and negative feedback.
    - **Passive User:** Short or generic reviews.
    - **Suspicious:** Reviews flagged as possibly fake or unreliable.
    """)

def kpi_explanations():
    st.markdown("""
    ### 📖 **Risk KPIs Explained**
    | KPI | What It Means | How It's Calculated / How to Read |
    |-----|---------------|------------------------------------|
    | High Risk Reviews | Reviews likely fake or manipulated. | Patterns like repetition, mismatched sentiment/rating, duplicates. |
    | Dissatisfaction Risk | Chance customers are unhappy. | % of low-rating, negative sentiment reviews. |
    | Inconsistent Reviews | Rating and text don't match. | E.g., 5-star with negative text. |
    | Average Risk Score | Average risk across all reviews. | Score based on suspicious features. |
    """)

def authenticity_explanations():
    st.markdown("""
    ### 📖 **Review Authenticity Labels**
    | Label | What It Means | How It's Derived |
    |-------|---------------|------------------|
    | Legitimate | Review is genuine. | No suspicious patterns. |
    | Low Risk | Slightly suspicious. | Minor issues like short/generic text. |
    | Medium Risk | Could be fake. | Multiple warning signs. |
    | High Risk | Likely fake. | Many red flags (duplicates, mismatched sentiment/rating, etc.). |
    """)

def topic_definitions():
    st.markdown("""
    ### 📖 **Strategic Topics Defined**
    - **Camera & Video Performance:** Camera, video, photo, picture clarity.
    - **Mobile Device Compatibility:** Works with phones/tablets/brands.
    - **Speed & Transfer Performance:** Speed, data transfer, loading.
    - **Storage Capacity & Management:** Memory, storage, capacity.
    - **Price & Value Proposition:** Cost, value for money.
    - **Build Quality & Durability:** Quality, strength, longevity.
    - **Shipping & Delivery Experience:** Delivery speed, packaging.
    - **Ease of Use & Installation:** Setup, user-friendliness.
    - **Overall Customer Satisfaction:** General satisfaction.
    - **Technical Issues & Problems:** Errors, malfunctions.
    """)

# --- Main App ---
st.set_page_config(page_title="Executive Amazon Reviews Intelligence", layout="wide")

st.title("🎯 Executive Intelligence Dashboard")
st.markdown("Strategic insights for data-driven decision making. All metrics and summaries are explained below each section.")

# --- Data Upload ---
uploaded_file = st.file_uploader("Upload your reviews CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    # (Insert your data processing here: sentiment, fraud, segmentation, topics, etc.)
    # For demo, assume columns: rating, reviewText, sentiment, fraudFlag, topic, etc.

    # --- Executive Summary (Layman Language) ---
    avg_rating = df['rating'].mean()
    st.markdown(f"""
    ## 📊 **Business Performance Overview**
    - **Total Reviews:** {len(df)}
    - **Average Rating:** {avg_rating:.2f}/5.0
    - **Customer Engagement:** {np.mean(df['reviewText'].str.len() > 50) * 100:.1f}% detailed reviews
    - **Brand Sentiment:** {df['sentiment'].value_counts().idxmax()}
    """)
    st.info("The average rating shows how happy customers are overall. More detailed reviews mean customers are engaged and sharing useful feedback.")

    # --- Main Charts (Example) ---
    st.header("Customer Segment Performance Matrix")
    # (Insert your segment performance chart here)
    customer_segment_explanations()

    st.header("Strategic Topic Performance")
    # (Insert your topic performance chart here)
    topic_definitions()

    st.header("Rating-Sentiment Business Matrix")
    # Example: Heatmap with improved color coding
    # (Insert your heatmap code with red for misalignment, green for alignment, x-axis ordered negative to positive)
    st.markdown("""
    **How to Read:**  
    - Red boxes = possible fake/inconsistent reviews (e.g., low rating but positive sentiment).
    - Green = rating and sentiment match.
    - The bar on the right shows total reviews for each sentiment.
    """)

    st.header("Enterprise Risk Assessment")
    # (Insert your risk charts here)
    kpi_explanations()
    authenticity_explanations()

    # --- Metrics Explained Section (Bottom of Each Page) ---
    metric_explanations()

else:
    st.warning("Please upload a CSV file to begin.")

# --- General UI/UX Enhancements ---
st.markdown("""
---
**Tips:**
- Hover over info icons for quick explanations.
- Charts use green for positive/good, red for risk/problem.
- All metrics and technical terms are explained below each section.
""")
