import streamlit as st
st.set_page_config(layout="wide", page_title="Customer Reviews Intelligence Platform", page_icon="📊")

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from textblob import TextBlob
import re
from collections import Counter, defaultdict
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import warnings
warnings.filterwarnings('ignore')
import io

# Try to import optional dependencies with fallbacks
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.sentiment import SentimentIntensityAnalyzer
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    NLTK_AVAILABLE = True
except:
    NLTK_AVAILABLE = False

# Enhanced CSS with 3D and sexy effects
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF5733;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        background: linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    @media (max-width: 768px) {
        .main-header { font-size: 1.8rem; }
    }
    
    .subheader {
        font-size: 1.5rem;
        color: #3366FF;
        margin-bottom: 0.5rem;
        border-bottom: 2px solid #3366FF;
        padding-bottom: 0.5rem;
    }
    
    @media (max-width: 768px) {
        .subheader { font-size: 1.2rem; }
    }
    
    .insight-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-left: 5px solid #0066cc;
        padding: 1.2rem;
        margin: 1rem 0;
        border-radius: 0.8rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .insight-box:hover {
        transform: translateY(-2px) rotateX(5deg);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    .insight-title {
        font-weight: bold;
        color: #0066cc;
        font-size: 1.2rem;
        margin-bottom: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .filter-box {
        background: linear-gradient(135deg, #f0f0f0 0%, #e0e0e0 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border: 1px solid #ddd;
    }
    
    .explained-box {
        background: linear-gradient(135deg, #f9f9f9 0%, #e8f5e8 100%);
        border-left: 5px solid #4CAF50;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 0.8rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .explained-title {
        font-weight: bold;
        color: #4CAF50;
        font-size: 1.6rem;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        margin: 0.5rem 0;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: scale(1.05) rotateY(10deg);
    }
    
    .chart-container {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 1rem;
        padding: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
        perspective: 1000px;
    }
    
    .executive-summary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 20px;
        margin: 2rem 0;
        box-shadow: 0 20px 40px rgba(0,0,0,0.2);
        border: 1px solid rgba(255,255,255,0.2);
        position: relative;
        overflow: hidden;
    }
    
    .executive-summary::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.2) 0%, rgba(255,255,255,0) 70%);
        animation: rotate 20s linear infinite;
    }
    
    @keyframes rotate {
        100% { transform: rotate(360deg); }
    }
    
    .recommendation-box {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: #2c3e50;
        font-weight: 500;
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
    }
    
    .verbatim-section {
        background: rgba(255,255,255,0.9);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    
    .positive-verbatim {
        border-left: 4px solid #28a745;
        background: rgba(40, 167, 69, 0.05);
    }
    
    .negative-verbatim {
        border-left: 4px solid #dc3545;
        background: rgba(220, 53, 69, 0.05);
    }
    
    @media (max-width: 768px) {
        .insight-box, .filter-box, .explained-box { padding: 1rem; margin: 0.5rem 0; }
        .metric-card { padding: 1rem; margin: 0.3rem 0; }
    }
    
    @media (max-width: 480px) {
        .insight-title { font-size: 1rem; }
        .explained-title { font-size: 1.3rem; }
    }
    
    .main-content { animation: fadeInUp 0.8s ease-out; }
    
    .glass-effect {
        background: rgba(255, 255, 255, 0.25);
        backdrop-filter: blur(10px);
        border-radius: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.18);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'topics' not in st.session_state:
    st.session_state.topics = []
if 'filters' not in st.session_state:
    st.session_state.filters = {
        'rating': [], 
        'sentiment': [], 
        'trust': 'All Reviews',
        'segment': [],
        'min_impact': -4.5,
        'min_confidence': 0.0
    }
if 'apply_filters' not in st.session_state:
    st.session_state.apply_filters = False

# Load and process data
uploaded_file = st.file_uploader("📁 Upload Amazon Reviews Dataset", type=['csv'])

if uploaded_file:
    with st.spinner('🔄 Loading data...'):
        df = load_and_process_data(uploaded_file)
        if df is not None:
            df, topics = process_data_with_advanced_ml(df)
            st.session_state.processed_data = df
            st.session_state.topics = topics
            st.session_state.filters['rating'] = sorted(df['rating'].unique())
            st.session_state.filters['sentiment'] = df['sentiment'].unique()
            st.session_state.filters['segment'] = df['customerSegment'].unique()
            st.session_state.filters['min_impact'] = float(df['businessImpact'].min())
            st.session_state.apply_filters = True
            st.success(f"✅ Analyzed {len(df):,} customer reviews successfully!")
        else:
            st.error("❌ Something went wrong. Please check your CSV file.")
            return

df = st.session_state.processed_data
topics = st.session_state.topics

if df is None:
    st.info("👆 Upload your Amazon reviews dataset to start the analysis")
    return

# Sidebar filters
with st.sidebar:
    st.markdown('<div class="filter-box glass-effect">', unsafe_allow_html=True)
    st.subheader("🎛️ Dynamic Dashboard Filters")
    st.markdown("💡 **Pro Tip**: Select filters, then click 'Apply Filters' to update.")

    rating_filter_temp = st.multiselect("⭐ Star Ratings", 
                                        sorted(df['rating'].unique()), 
                                        default=st.session_state.filters['rating'])
    
    sentiment_filter_temp = st.multiselect("😊 Customer Feelings", 
                                           df['sentiment'].unique(), 
                                           default=st.session_state.filters['sentiment'])
    
    trust_filter_temp = st.selectbox("🔍 Review Trust", 
                                     ['All Reviews', 'Trusted Only', 'Suspicious Only', 'High Risk Only'],
                                     index=['All Reviews', 'Trusted Only', 'Suspicious Only', 'High Risk Only'].index(st.session_state.filters['trust']))
    
    segment_filter_temp = st.multiselect("👥 Customer Types", 
                                         df['customerSegment'].unique(), 
                                         default=st.session_state.filters['segment'])
    
    min_impact_temp = st.slider("💼 Minimum Business Impact", 
                                float(df['businessImpact'].min()), 
                                float(df['businessImpact'].max()), 
                                st.session_state.filters['min_impact'])
    
    min_confidence_temp = st.slider("🎯 Minimum Analysis Confidence", 0.0, 1.0, st.session_state.filters['min_confidence'], 0.1)
    
    if st.button("Apply Filters"):
        st.session_state.filters = {
            'rating': rating_filter_temp,
            'sentiment': sentiment_filter_temp,
            'trust': trust_filter_temp,
            'segment': segment_filter_temp,
            'min_impact': min_impact_temp,
            'min_confidence': min_confidence_temp
        }
        st.session_state.apply_filters = True
    
    if st.button("Clear Filters"):
        st.session_state.filters = {
            'rating': sorted(df['rating'].unique()),
            'sentiment': df['sentiment'].unique(),
            'trust': 'All Reviews',
            'segment': df['customerSegment'].unique(),
            'min_impact': float(df['businessImpact'].min()),
            'min_confidence': 0.0
        }
        st.session_state.apply_filters = True
    
    st.markdown("</div>", unsafe_allow_html=True)

# Apply filters if button pressed
if st.session_state.apply_filters:
    filtered_df = df[
        (df['rating'].isin(st.session_state.filters['rating'])) & 
        (df['sentiment'].isin(st.session_state.filters['sentiment'])) &
        (df['customerSegment'].isin(st.session_state.filters['segment'])) &
        (df['businessImpact'] >= st.session_state.filters['min_impact']) &
        (df['sentimentConfidence'] >= st.session_state.filters['min_confidence'])
    ]
    
    if st.session_state.filters['trust'] == 'Trusted Only':
        filtered_df = filtered_df[filtered_df['fraudFlag'] == 'Legitimate']
    elif st.session_state.filters['trust'] == 'Suspicious Only':
        filtered_df = filtered_df[filtered_df['fraudFlag'].isin(['Low Risk', 'Medium Risk'])]
    elif st.session_state.filters['trust'] == 'High Risk Only':
        filtered_df = filtered_df[filtered_df['fraudFlag'] == 'High Risk']
else:
    filtered_df = df.copy()

# Show active filters
active_filters = [f"{k}: {v}" for k, v in st.session_state.filters.items() if v]
if active_filters:
    st.markdown(f"""
    <div style='background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                color: white; 
                padding: 1rem; 
                border-radius: 0.8rem; 
                margin: 1rem 0;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);'>
        <strong>🎯 Active Filters:</strong> {', '.join(active_filters)} 
        <br><strong>📈 Filtered Reviews:</strong> {len(filtered_df):,} out of {len(df):,} total reviews
        <br><strong>📊 Coverage:</strong> {len(filtered_df)/len(df)*100:.1f}% of total dataset
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div style='background: linear-gradient(90deg, #28a745 0%, #20c997 100%); 
                color: white; 
                padding: 1rem; 
                border-radius: 0.8rem; 
                margin: 1rem 0;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);'>
        <strong>🌐 Viewing:</strong> Complete Dataset ({len(df):,} reviews)
        <br><strong>📊 Status:</strong> No filters applied - showing all customer data
    </div>
    """, unsafe_allow_html=True)

# Executive Dashboard Tabs with subheader styling
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Executive Summary", "📊 Business Insights", "🔍 Deep Analysis", 
    "💬 Customer Voices", "🚨 Risk Assessment"
])

# TAB 1: Executive Summary - Made sexier with metric cards and insight boxes
with tab1:
    st.markdown('<div class="executive-summary">', unsafe_allow_html=True)
    st.markdown(create_executive_summary(filtered_df))
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Sexy metric cards
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        avg_rating = filtered_df['rating'].mean()
        st.markdown(f"""
        <div class='metric-card'>
            <h3>📊 Customer Happiness</h3>
            <h2>{avg_rating:.2f}/5.0</h2>
            <p>vs neutral { (avg_rating - 3.0):.1f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        fraud_rate = (filtered_df['fraudFlag'].isin(['High Risk', 'Medium Risk'])).sum() / len(filtered_df) * 100
        st.markdown(f"""
        <div class='metric-card'>
            <h3>🔍 Trust Score</h3>
            <h2>{100-fraud_rate:.1f}%</h2>
            <p>-{fraud_rate:.1f}% risk</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        positive_sentiment = (filtered_df['sentiment'].str.contains('Positive')).sum() / len(filtered_df) * 100
        st.markdown(f"""
        <div class='metric-card'>
            <h3>😊 Positive Feelings</h3>
            <h2>{positive_sentiment:.1f}%</h2>
            <p>Brand Sentiment</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        engagement_rate = (filtered_df['wordCount'] > 50).sum() / len(filtered_df) * 100
        st.markdown(f"""
        <div class='metric-card'>
            <h3>💬 Detailed Reviews</h3>
            <h2>{engagement_rate:.1f}%</h2>
            <p>Engagement Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        avg_business_impact = filtered_df['businessImpact'].mean()
        st.markdown(f"""
        <div class='metric-card'>
            <h3>💼 Business Impact</h3>
            <h2>{avg_business_impact:.1f}</h2>
            <p>per review</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Strategic Overview Charts in chart containers
    col1, col2 = st.columns(2)
    
    with col1:
        fig_rating = px.histogram(
            filtered_df, x='rating', 
            title="📊 How Customers Rate Us",
            color_discrete_sequence=['#667eea'],
            text_auto=True,
            category_orders={"rating": [1, 2, 3, 4, 5]}
        )
        fig_rating.update_layout(
            xaxis_title="Star Rating",
            yaxis_title="Number of Reviews",
            showlegend=False, 
            height=400
        )
        
        high_satisfaction = (filtered_df['rating'] >= 4).sum() / len(filtered_df) * 100
        insight_text = f"**{high_satisfaction:.1f}% of customers gave 4 or 5 stars**, showing strong happiness with our product. This suggests we're doing well in the market."
        
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        create_chart_with_insights(fig_rating, insight_text)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        sentiment_counts = filtered_df['sentiment'].value_counts()
        colors = {
            'Extremely Positive': '#0d7377', 'Very Positive': '#14a085', 'Positive': '#2ca02c',
            'Neutral': '#ffbb33', 'Negative': '#ff6b6b', 'Very Negative': '#d62728', 'Extremely Negative': '#8b0000'
        }
        
        fig_sentiment = px.pie(
            values=sentiment_counts.values, 
            names=sentiment_counts.index,
            title="🎭 Customer Feelings Breakdown",
            color=sentiment_counts.index,
            color_discrete_map=colors
        )
        
        brand_advocates = sentiment_counts.get('Extremely Positive', 0) + sentiment_counts.get('Very Positive', 0)
        detractors = sentiment_counts.get('Very Negative', 0) + sentiment_counts.get('Extremely Negative', 0)
        nps_proxy = (brand_advocates - detractors) / len(filtered_df) * 100
        
        insight_text = f"**{brand_advocates:,} customers love our product**, while {detractors:,} are unhappy. This gives us a customer happiness score of {nps_proxy:.1f}%. We should focus on keeping happy customers and helping the unhappy ones."
        
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        create_chart_with_insights(fig_sentiment, insight_text)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Download buttons
    export_df = filtered_df[[
        'reviewId', 'reviewerName', 'reviewText', 'rating', 'reviewDate',
        'sentiment', 'sentimentScore', 'sentimentConfidence', 'emotion',
        'fraudFlag', 'fraudReason', 'fraudScore',
        'topic', 'customerSegment', 'businessImpact', 'reviewValue',
        'wordCount', 'reviewLength'
    ]].copy()
    
    csv_data = export_df.to_csv(index=False)
    st.download_button(
        label="📄 Download Full Data",
        data=csv_data,
        file_name=f"reviews_analysis_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
    
    executive_summary = create_executive_summary(filtered_df)
    
    st.download_button(
        label="📋 Download Summary",
        data=executive_summary,
        file_name=f"executive_summary_{pd.Timestamp.now().strftime('%Y%m%d')}.md",
        mime="text/markdown"
    )
    
    # Metrics Explained
    st.markdown('<div class="explained-box">', unsafe_allow_html=True)
    st.markdown("""
    ### 📘 Understanding the Metrics
    Here's what each number means, how we calculate it, and what it tells you:

    | **Metric** | **What It Means** | **How It's Calculated / How to Read It** |
    |------------|-------------------|-----------------------------------------|
    | **Customer Happiness (Average Rating)** | The average star rating customers give our product. | Add up all star ratings (1-5) and divide by the number of reviews. Higher means customers are happier. |
    | **Trust Score** | How many reviews seem real and trustworthy. | Percentage of reviews not flagged as suspicious. Higher means more genuine feedback. |
    | **Positive Feelings (Brand Sentiment)** | How many customers feel good about our product. | Percentage of reviews with positive words (e.g., "great," "love"). Higher means more happy customers. |
    | **Detailed Reviews (Engagement Rate)** | How many customers write long, detailed reviews. | Percentage of reviews with more than 50 words. Higher means customers are sharing valuable feedback. |
    | **Business Impact** | How much a review influences other customers or our business. | Based on star rating, review positivity, and length. Higher means the review matters more. |
    | **Brand Advocacy** | Customers who love our product and recommend it. | Percentage of reviews that are very positive (Extremely or Very Positive). Higher means more loyal fans. |

    **Customer Types (Segments):**
    - **Engaged Advocate**: Writes long, detailed, positive reviews. They're our biggest fans.
    - **Satisfied Customer**: Gives high ratings (4-5 stars) and positive feedback.
    - **Dissatisfied Customer**: Gives low ratings (1-2 stars) and negative feedback.
    - **Passive User**: Writes short, vague reviews with little detail.
    - **Suspicious**: Reviews that might be fake, based on unusual patterns.
    - **Average Customer**: Doesn't stand out - gives moderate ratings and feedback.

    These groups are based on review length, star rating, positivity, and trustworthiness.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# TAB 2: Business Insights  
with tab2:
    st.markdown("<h2 class='subheader'>📊 Business Insights</h2>", unsafe_allow_html=True)
    st.markdown("*Clear insights to understand customer behavior and key topics*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        # Customer segment analysis
        st.markdown("### 👥 Customer Types Performance")
        st.markdown("This chart shows how different customer groups rate our product and their impact.")
        
        segment_analysis = filtered_df.groupby('customerSegment').agg({
            'rating': 'mean',
            'businessImpact': 'mean',
            'reviewValue': 'mean',
            'reviewId': 'count'
        }).round(2).reset_index()
        segment_analysis.columns = ['Customer Type', 'Average Rating', 'Business Impact', 'Review Value', 'Number of Reviews']
        
        fig_segments = px.scatter_3d(
            segment_analysis, x='Business Impact', y='Average Rating', z='Review Value',
            color='Customer Type', size='Number of Reviews',
            title="3D Customer Types Impact vs Happiness"
        )
        fig_segments.update_layout(scene=dict(
            xaxis_title='Business Impact',
            yaxis_title='Average Rating',
            zaxis_title='Review Value'
        ))
        
        insight_text = f"**{segment_analysis.loc[segment_analysis['Business Impact'].idxmax(), 'Customer Type']}** has the biggest impact on our business. **{segment_analysis.loc[segment_analysis['Number of Reviews'].idxmax(), 'Customer Type']}** is our largest group, so we should focus on keeping them happy."
        
        create_chart_with_insights(fig_segments, insight_text)
        
        st.markdown("""
        **What Are Customer Types?**
        - **Engaged Advocate**: Long, positive, detailed reviews. They're loyal fans.
        - **Satisfied Customer**: High ratings (4-5 stars) with positive comments.
        - **Dissatisfied Customer**: Low ratings (1-2 stars) with negative feedback.
        - **Passive User**: Short, vague reviews with little detail.
        - **Suspicious**: Possibly fake reviews, flagged for unusual patterns.
        - **Average Customer**: Moderate ratings and feedback, not extreme.
        
        **How We Group Them:**
        We look at:
        - How long their reviews are.
        - Whether their comments are positive or negative.
        - The star rating they give.
        - If the review seems trustworthy or suspicious.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Topic performance analysis
        if topics:
            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
            st.markdown("### 🏷️ Key Customer Topics")
            st.markdown("This chart shows what customers talk about most and how they rate those topics.")
            
            topic_performance = filtered_df.groupby('topic').agg({
                'rating': 'mean',
                'businessImpact': 'mean',
                'sentimentScore': 'mean',
                'reviewId': 'count'
            }).round(2).reset_index()
            topic_performance.columns = ['Topic', 'Average Rating', 'Business Impact', 'Sentiment Score', 'Number of Reviews']
            
            top_topics_perf = topic_performance.nlargest(8, 'Business Impact')
            
            fig_topics = px.bar(
                top_topics_perf, 
                x='Business Impact', 
                y='Topic',
                color='Average Rating',
                title="Top Customer Topics by Impact",
                orientation='h',
                color_continuous_scale='RdYlGn'
            )
            fig_topics.update_layout(yaxis={'categoryorder':'total ascending'})
            
            best_topic = top_topics_perf.iloc[0]['Topic']
            worst_topic = topic_performance.nsmallest(1, 'Business Impact').iloc[0]['Topic']
            
            insight_text = f"**'{best_topic}'** is what customers talk about most positively. We should highlight this in marketing. **'{worst_topic}'** needs improvement to make customers happier."
            
            create_chart_with_insights(fig_topics, insight_text)
            
            st.markdown("""
            **What Each Topic Means:**
            - **Camera & Video Performance**: Reviews about camera quality, video, photos, or picture clarity.
            - **Mobile Device Compatibility**: How well the product works with phones or tablets.
            - **Speed & Transfer Performance**: Comments on speed, data transfer, or loading times.
            - **Storage Capacity & Management**: Feedback on memory or storage space.
            - **Price & Value Proposition**: Opinions about cost or value for money.
            - **Build Quality & Durability**: Mentions of product strength or how long it lasts.
            - **Shipping & Delivery Experience**: Comments about delivery speed or packaging.
            - **Ease of Use & Installation**: Feedback on how easy or hard it is to use or set up.
            - **Overall Customer Satisfaction**: General happiness or disappointment.
            - **Technical Issues & Problems**: Reports of errors or product failures.
            - **General Product Discussion**: Other general comments.
            
            **How We Find Topics:**
            We use smart text analysis to group reviews by common words (e.g., "camera" and "photo" go together).
            """)
            st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        # Time-based business trends
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.markdown("### 📈 Trends Over Time")
        monthly_trends = filtered_df.groupby(['year', 'month']).agg({
            'rating': 'mean',
            'businessImpact': 'mean',
            'fraudScore': 'mean',
            'reviewId': 'count'
        }).reset_index()
        monthly_trends['date'] = pd.to_datetime(monthly_trends[['year', 'month']].assign(day=1))
        monthly_trends = monthly_trends.sort_values('date')
        
        fig_trends = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Customer Happiness Trend', 'Business Impact Trend'),
            vertical_spacing=0.15
        )
        
        fig_trends.add_trace(
            go.Scatter(x=monthly_trends['date'], y=monthly_trends['rating'],
                      mode='lines+markers', name='Average Rating', line=dict(color='#667eea')),
            row=1, col=1
        )
        
        fig_trends.add_trace(
            go.Scatter(x=monthly_trends['date'], y=monthly_trends['businessImpact'],
                      mode='lines+markers', name='Business Impact', line=dict(color='#f093fb')),
            row=2, col=1
        )
        
        fig_trends.update_layout(height=500, title_text="Performance Over Time", xaxis_tickangle=45, xaxis2_tickangle=45)
        
        recent_rating = monthly_trends['rating'].tail(3).mean()
        historical_rating = monthly_trends['rating'].head(3).mean()
        trend_direction = "getting better" if recent_rating > historical_rating else "getting worse"
        
        insight_text = f"**Customer happiness is {trend_direction}** with a {abs(recent_rating - historical_rating):.2f} star change. This means we {'should keep up the good work' if trend_direction == 'getting better' else 'need to fix issues quickly'}."
        
        create_chart_with_insights(fig_trends, insight_text)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Business impact heatmap
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.markdown("### 🔥 Rating vs Feelings Matrix")
        impact_heatmap = pd.crosstab(filtered_df['rating'], filtered_df['sentiment'])
        sentiment_order = ['Extremely Negative', 'Very Negative', 'Negative', 'Neutral', 
                          'Positive', 'Very Positive', 'Extremely Positive']
        impact_heatmap = impact_heatmap.reindex(columns=sentiment_order, fill_value=0)
        
        fig_heatmap = px.imshow(
            impact_heatmap.values,
            x=impact_heatmap.columns,
            y=impact_heatmap.index,
            title="Rating vs Customer Feelings",
            color_continuous_scale='RdYlBu_r',
            text_auto=True
        )
        fig_heatmap.update_layout(
            xaxis_title="Customer Feelings",
            yaxis_title="Star Rating",
            xaxis_tickangle=45
        )
        
        concerning_areas = impact_heatmap.loc[4:5, ['Negative', 'Very Negative', 'Extremely Negative']].sum().sum()
        positive_areas = impact_heatmap.loc[4:5, ['Positive', 'Very Positive', 'Extremely Positive']].sum().sum()
        
        insight_text = f"**{concerning_areas} high-rated reviews seem negative**, which could mean fake reviews or mixed feelings. **{positive_areas} reviews show happy customers giving high ratings**, which is great for our brand."
        
        create_chart_with_insights(fig_heatmap, insight_text)
        
        st.markdown("""
        **How to Read This Chart:**
        - Each box shows how many reviews have a certain star rating (1-5) and feeling (e.g., Positive, Negative).
        - **Red boxes**: Reviews where the rating and feeling don't match (e.g., 1 star but positive words). These might be suspicious.
        - **Darker red**: 1-star ratings with positive feelings (very suspicious).
        - **Lighter red**: 2-star ratings with positive feelings (less suspicious).
        - **Green boxes**: Ratings and feelings match (e.g., 5 stars with positive words).
        - **Bar on the right**: Shows the total number of reviews for each star rating.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Metrics Explained
    st.markdown('<div class="explained-box">', unsafe_allow_html=True)
    st.markdown("""
    ### 📘 Understanding the Metrics
    | **Metric** | **What It Means** | **How It's Calculated / How to Read It** |
    |------------|-------------------|-----------------------------------------|
    | **Average Rating** | The average star rating customers give our product. | Add up all star ratings (1-5) and divide by the number of reviews. Higher means customers are happier. |
    | **Business Impact** | How much a review influences other customers or our business. | Based on star rating, review positivity, and length. Higher means the review matters more. |
    | **Review Value** | How useful a review is for understanding customer needs. | Based on length, helpfulness, and trustworthiness. Higher means more valuable feedback. |
    | **Number of Reviews** | How many reviews are in each customer group or topic. | Total count of reviews. More reviews mean the group or topic is more important. |
    | **Sentiment Score** | How positive or negative a review is. | Analyzed from review text. Positive numbers mean happy reviews; negative means unhappy. |

    **Customer Types and Topics are explained above under each chart.**
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# TAB 3: Deep Analysis
with tab3:
    st.markdown("<h2 class='subheader'>🔍 Deep Analysis</h2>", unsafe_allow_html=True)
    st.markdown("*Detailed insights to guide big decisions*")
    
    # Advanced topic analysis with business recommendations
    if topics:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.subheader("🏷️ Key Themes Analysis")
        
        topic_insights = []
        for idx, topic_name in enumerate(sorted(topics), 1):
            topic_reviews = filtered_df[filtered_df['topic'] == topic_name]
            if len(topic_reviews) > 0:
                avg_rating = topic_reviews['rating'].mean()
                avg_sentiment = topic_reviews['sentimentScore'].mean()
                review_count = len(topic_reviews)
                avg_impact = topic_reviews['businessImpact'].mean()
                
                if avg_rating >= 4.5 and avg_sentiment > 0.3:
                    recommendation = "🟢 Highlight in marketing campaigns to attract more customers."
                elif avg_rating >= 4.0 and avg_sentiment > 0.1:
                    recommendation = "🟡 Improve slightly to make this a strength."
                elif avg_rating >= 3.0:
                    recommendation = "🟠 Review processes to reduce complaints by 20%."
                else:
                    recommendation = "🔴 Fix issues urgently to improve customer trust."
                
                topic_insights.append({
                    'ID': idx,
                    'Theme': topic_name,
                    'Volume': review_count,
                    'Avg Rating': f"{avg_rating:.2f}",
                    'Sentiment Score': f"{avg_sentiment:.2f}",
                    'Business Impact': f"{avg_impact:.2f}",
                    'Action': recommendation
                })
        
        if topic_insights:
            insights_df = pd.DataFrame(topic_insights).sort_values('Business Impact', ascending=False)
            st.dataframe(insights_df, use_container_width=True)
            
            st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
            st.markdown("### 💡 **What We Should Do**")
            
            top_opportunity = insights_df.iloc[0]
            biggest_risk = insights_df[insights_df['Action'].str.contains('🔴')].head(1)
            
            st.markdown(f"""
            **🎯 Best Opportunity:** Focus on **{top_opportunity['Theme']}** - it has {top_opportunity['Volume']} reviews and high impact.
            
            **⚠️ Biggest Issue:** {biggest_risk['Theme'].iloc[0] if len(biggest_risk) > 0 else 'No major issues found'} needs urgent fixes.
            
            **📈 Next Steps:** Build on strong themes and fix weak ones to keep customers happy.
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Customer journey analysis
    st.subheader("🛤️ Customer Experience Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        # Review length vs satisfaction analysis
        st.markdown("### 📝 Review Length vs Happiness")
        filtered_df['lengthCategory'] = pd.cut(
            filtered_df['wordCount'],
            bins=[0, 10, 30, 75, 150, float('inf')],
            labels=['Minimal (≤10)', 'Brief (11-30)', 'Standard (31-75)', 'Detailed (76-150)', 'Comprehensive (150+)']
        )
        
        length_analysis = filtered_df.groupby('lengthCategory').agg({
            'rating': 'mean',
            'sentimentScore': 'mean',
            'businessImpact': 'mean',
            'reviewId': 'count'
        }).round(2).reset_index()
        
        fig_length = px.bar(
            length_analysis, 
            x='lengthCategory', 
            y='rating',
            color='businessImpact',
            title="Review Length vs Customer Happiness",
            color_continuous_scale='Viridis'
        )
        
        optimal_length = length_analysis.loc[length_analysis['rating'].idxmax(), 'lengthCategory']
        optimal_rating = length_analysis['rating'].max()
        
        insight_text = f"**{optimal_length} reviews give the highest ratings** ({optimal_rating:.2f}/5.0). These reviews have a {length_analysis.loc[length_analysis['lengthCategory'] == optimal_length, 'businessImpact'].iloc[0]:.1f} impact, showing detailed feedback matters."
        
        create_chart_with_insights(fig_length, insight_text)
        
        st.markdown("""
        **What This Means:**
        - **Review Length**: How many words a customer writes (short, medium, long).
        - **Rating**: The star rating they give (1-5).
        - **Business Detail**: The topics they mention (e.g., shipping, quality).
        - **How to Read**: Longer reviews often give better insights. If long reviews have high ratings, customers are happy and detailed. If they have low ratings, there are serious issues to fix.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        # Customer segment journey
        st.markdown("### 👥 Customer Types Feelings")
        segment_journey = filtered_df.groupby(['customerSegment', 'sentiment']).size().unstack(fill_value=0)
        segment_journey = segment_journey.reindex(columns=sentiment_order, fill_value=0)
        segment_journey_pct = segment_journey.div(segment_journey.sum(axis=1), axis=0) * 100
        
        fig_journey = px.imshow(
            segment_journey_pct.values,
            x=segment_journey_pct.columns,
            y=segment_journey_pct.index,
            title="Customer Types Feelings Breakdown",
            color_continuous_scale='RdYlGn',
            text_auto='.1f'
        )
        
        segment_scores = (segment_journey_pct[['Positive', 'Very Positive', 'Extremely Positive']].sum(axis=1) - 
                          segment_journey_pct[['Negative', 'Very Negative', 'Extremely Negative']].sum(axis=1))
        best_segment = segment_scores.idxmax()
        worst_segment = segment_scores.idxmin()
        
        insight_text = f"**{best_segment}** customers are the happiest ({segment_scores.max():.1f}% positive), so we should reward them. **{worst_segment}** customers are least happy ({segment_scores.min():.1f}%), so we need to help them."
        
        create_chart_with_insights(fig_journey, insight_text)
        
        st.markdown("""
        **How to Read This Chart:**
        - Each box shows the percentage of a customer group (e.g., Satisfied Customers) with a certain feeling (e.g., Positive).
        - **Green boxes**: More customers in that group feel positive.
        - **Red boxes**: More customers feel negative, which needs attention.
        - **Bar on the right**: Shows the total number of reviews for each customer group.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Metrics Explained
    st.markdown('<div class="explained-box">', unsafe_allow_html=True)
    st.markdown("""
    ### 📘 Understanding the Metrics
    | **Metric** | **What It Means** | **How It's Calculated / How to Read It** |
    |------------|-------------------|-----------------------------------------|
    | **Volume** | Number of reviews about a topic. | Total count of reviews for each topic. More reviews mean the topic is important. |
    | **Avg Rating** | Average star rating for a topic or review length. | Add up ratings and divide by number of reviews. Higher means customers like it. |
    | **Sentiment Score** | How positive or negative reviews are. | Analyzed from review text. Positive numbers mean happy reviews; negative means unhappy. |
    | **Business Impact** | How much a topic or review affects the business. | Based on rating, positivity, and length. Higher means bigger influence. |
    | **Action** | What we should do about a topic. | Based on ratings and positivity. Green means promote it; red means fix it. |

    **Customer Types are explained in the Executive Summary tab.**
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# TAB 4: Customer Voices
with tab4:
    st.markdown("<h2 class='subheader'>💬 Customer Voices</h2>", unsafe_allow_html=True)
    st.markdown("*What customers are really saying about our product*")
    
    # Extract and display sample verbatims
    positive_verbatims, negative_verbatims = extract_sample_verbatims(filtered_df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🌟 **Happy Customers**")
        st.markdown("*Customers who love our product and help our brand*")
        
        for idx, (_, review) in enumerate(positive_verbatims.iterrows(), 1):
            st.markdown(f'<div class="verbatim-section positive-verbatim">', unsafe_allow_html=True)
            st.markdown(f"**Happy Customer #{idx}** | ⭐{review['rating']}/5 | Impact: {review['businessImpact']:.1f}")
            review_preview = review['reviewText'][:300]
            if len(review['reviewText']) > 300:
                review_preview += "..."
            st.markdown(f"*\"{review_preview}\"*")
            st.markdown(f"**Topic:** {review['topic']} | **Feeling:** {review['sentiment']}")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### ⚠️ **Customer Concerns**")
        st.markdown("*Issues we need to fix to keep customers happy*")
        
        for idx, (_, review) in enumerate(negative_verbatims.iterrows(), 1):
            st.markdown(f'<div class="verbatim-section negative-verbatim">', unsafe_allow_html=True)
            st.markdown(f"**Concern #{idx}** | ⭐{review['rating']}/5 | Impact: {review['businessImpact']:.1f}")
            review_preview = review['reviewText'][:300]
            if len(review['reviewText']) > 300:
                review_preview += "..."
            st.markdown(f"*\"{review_preview}\"*")
            st.markdown(f"**Topic:** {review['topic']} | **Feeling:** {review['sentiment']}")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Word Cloud for Positive and Negative Reviews
    st.subheader("🌐 Word Cloud Insights")
    col1, col2 = st.columns(2)
    
    with col1:
        positive_text = filtered_df[filtered_df['sentiment'].str.contains('Positive', na=False)]['reviewText'].dropna()
        if not positive_text.empty:
            generate_wordcloud(positive_text, "Word Cloud - Positive Reviews")
        else:
            st.info("No positive reviews available with current filters.")
    
    with col2:
        negative_text = filtered_df[filtered_df['sentiment'].str.contains('Negative', na=False)]['reviewText'].dropna()
        if not negative_text.empty:
            generate_wordcloud(negative_text, "Word Cloud - Negative Reviews")
        else:
            st.info("No negative reviews available with current filters.")
    
    # Impact score explanation
    st.markdown('<div class="explained-box">', unsafe_allow_html=True)
    st.markdown("""
    ### 📘 Understanding the Impact Score
    **What It Means:** The Impact Score shows how much a review could affect your business - positive reviews help sales, negative ones hurt.
    
    **How It's Calculated:** Based on star rating (high = positive), sentiment (positive words = positive), and review length (longer = more impact). Scores range from -4.5 to 4.5.
    
    **How to Read It (3 Buckets):**
    - **Positive Impact (>0):** Good reviews that boost your brand. Focus on sharing these.
    - **Neutral Impact (0):** Balanced reviews with little effect. Monitor for trends.
    - **Negative Impact (<0):** Bad reviews that could harm sales. Address these issues quickly.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Key themes summary (dynamic with filtered_df)
    st.subheader("🔤 Top Customer Topics")
    if topics:
        topic_summary = filtered_df['topic'].value_counts().head(5)
        st.markdown("Here are the top 5 things customers are talking about:")
        for idx, (topic, count) in enumerate(topic_summary.items(), 1):
            st.markdown(f"**{idx}. {topic}**: Mentioned in {count:,} reviews ({count/len(filtered_df)*100:.1f}%)")
        st.markdown("**What to Do:** Use these topics to improve products, marketing, or customer service.")
    
    # Metrics Explained
    st.markdown('<div class="explained-box">', unsafe_allow_html=True)
    st.markdown("""
    ### 📘 Understanding the Metrics
    | **Metric** | **What It Means** | **How It's Calculated / How to Read It** |
    |------------|-------------------|-----------------------------------------|
    | **Rating** | The star rating a customer gave. | From 1 to 5 stars. Higher means they liked the product. |
    | **Business Impact** | How much a review affects our business. | Based on rating, positivity, and length. Higher means bigger influence. |
    | **Topic** | What the review is about (e.g., shipping, quality). | Found by analyzing common words in reviews. Helps us know what matters to customers. |
    | **Feeling (Sentiment)** | Whether the review is positive, negative, or neutral. | Analyzed from review text. Positive means happy; negative means unhappy. |
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# TAB 5: Risk Assessment
with tab5:
    st.markdown("<h2 class='subheader'>🚨 Risk Assessment</h2>", unsafe_allow_html=True)
    st.markdown("*Checking for fake reviews and customer issues*")
    
    # Risk metrics dashboard with metric cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        high_risk_count = (filtered_df['fraudFlag'] == 'High Risk').sum()
        st.markdown(f"""
        <div class='metric-card'>
            <h3>🚨 Very Suspicious Reviews</h3>
            <h2>{high_risk_count}</h2>
            <p>{high_risk_count/len(filtered_df)*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        negative_trend_risk = len(filtered_df[filtered_df['rating'] <= 2]) / len(filtered_df) * 100
        st.markdown(f"""
        <div class='metric-card'>
            <h3>📉 Unhappy Customers</h3>
            <h2>{negative_trend_risk:.1f}%</h2>
            <p>Dissatisfaction Risk</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        inconsistent_reviews = len(filtered_df[
            ((filtered_df['rating'] >= 4) & (filtered_df['sentiment'].str.contains('Negative', na=False))) |
            ((filtered_df['rating'] <= 2) & (filtered_df['sentiment'].str.contains('Positive', na=False)))
        ])
        st.markdown(f"""
        <div class='metric-card'>
            <h3>⚠️ Mismatched Reviews</h3>
            <h2>{inconsistent_reviews}</h2>
            <p>Inconsistent Reviews</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_fraud_score = filtered_df['fraudScore'].mean()
        st.markdown(f"""
        <div class='metric-card'>
            <h3>🎯 Average Risk Score</h3>
            <h2>{avg_fraud_score:.1f}/10</h2>
            <p>Average Risk Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed fraud analysis in chart containers
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        fraud_dist = filtered_df['fraudFlag'].value_counts()
        fig_fraud = px.pie(
            values=fraud_dist.values,
            names=fraud_dist.index,
            title="🔍 Review Trust Breakdown",
            color_discrete_map={
                'Legitimate': '#28a745',
                'Low Risk': '#ffc107', 
                'Medium Risk': '#fd7e14',
                'High Risk': '#dc3545'
            }
        )
        
        legitimate_rate = fraud_dist.get('Legitimate', 0) / len(filtered_df) * 100
        insight_text = f"**{legitimate_rate:.1f}% of reviews are trustworthy**. We should verify the {100-legitimate_rate:.1f}% suspicious reviews to keep our feedback reliable."
        
        create_chart_with_insights(fig_fraud, insight_text)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        risk_trends = filtered_df.groupby(['year', 'month']).agg({
            'fraudScore': 'mean',
            'fraudFlag': lambda x: (x != 'Legitimate').sum(),
            'reviewId': 'count'
        }).reset_index()
        risk_trends['risk_rate'] = risk_trends['fraudFlag'] / risk_trends['reviewId'] * 100
        risk_trends['date'] = pd.to_datetime(risk_trends[['year', 'month']].assign(day=1))
        
        fig_risk_trends = px.line(
            risk_trends, 
            x='date', 
            y='risk_rate',
            title="📈 Suspicious Reviews Over Time",
            color_discrete_sequence=['#dc3545']
        )
        fig_risk_trends.update_layout(xaxis_tickangle=45)
        
        recent_risk = risk_trends['risk_rate'].tail(3).mean()
        historical_risk = risk_trends['risk_rate'].head(3).mean()
        risk_direction = "going up" if recent_risk > historical_risk else "going down"
        
        insight_text = f"**The number of suspicious reviews is {risk_direction}** by {abs(recent_risk - historical_risk):.1f}%. This means our efforts to keep reviews honest are {'working' if risk_direction == 'going down' else 'not enough, and we need stronger checks'}."
        
        create_chart_with_insights(fig_risk_trends, insight_text)
        
        st.markdown("""
        **What is Risk Rate?**
        - **Risk Rate**: The percentage of reviews that seem suspicious (Low, Medium, or High Risk).
        - **How to Read**: A lower risk rate means fewer fake or problematic reviews. If it's going down, our review system is getting more trustworthy.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # New Fraud Detection section
    st.subheader("🕵️ Fraud Detection")
    suspicious_df = filtered_df[filtered_df['fraudFlag'].isin(['High Risk', 'Medium Risk'])]
    if not suspicious_df.empty:
        st.markdown(f"**Found {len(suspicious_df)} suspicious reviews.**")
        st.dataframe(suspicious_df[['reviewId', 'reviewText', 'rating', 'fraudFlag', 'fraudReason']], use_container_width=True)
        st.info("These are added to the downloadable CSV with columns 'fraudFlag' and 'fraudReason'.")
    else:
        st.success("No suspicious reviews found in current filters.")
    
    # Risk mitigation recommendations
    st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
    st.markdown("### 🛡️ **How to Reduce Risks**")
    
    total_risk_rate = (filtered_df['fraudFlag'] != 'Legitimate').sum() / len(filtered_df) * 100
    
    if total_risk_rate > 20:
        st.markdown("🔴 **URGENT:** Too many suspicious reviews - check all reviews carefully.")
    elif total_risk_rate > 10:
        st.markdown("🟠 **WARNING:** Some reviews are suspicious - verify the risky ones.")
    else:
        st.markdown("🟢 **GOOD:** Most reviews are trustworthy - keep monitoring.")
    
    st.markdown("""
    **Quick Actions:**
    1. Flag reviews with unusual patterns (e.g., same text repeated).
    2. Verify reviews with high risk scores (>5).
    3. Watch suspicious customer groups closely.
    
    **Long-Term Plans:**
    1. Use smart tools to spot fake reviews instantly.
    2. Score reviews for trustworthiness.
    3. Reward honest reviewers to encourage real feedback.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Metrics Explained
    st.markdown('<div class="explained-box">', unsafe_allow_html=True)
    st.markdown("""
    ### 📘 Understanding the Metrics
    | **Metric** | **What It Means** | **How It's Calculated / How to Read It** |
    |------------|-------------------|-----------------------------------------|
    | **Very Suspicious Reviews (High Risk Reviews)** | Reviews that are very likely fake. | Flagged for patterns like repeated text, mismatched ratings, or excessive enthusiasm. Higher numbers mean more fake reviews. |
    | **Unhappy Customers (Dissatisfaction Risk)** | Percentage of customers who are not happy. | Percentage of reviews with low ratings (1-2 stars) and negative comments. Higher means more customers are upset. |
    | **Mismatched Reviews (Inconsistent Reviews)** | Reviews where the star rating and comments don't match. | E.g., 5 stars but negative text. Higher numbers mean possible fake reviews or confusion. |
    | **Average Risk Score** | How suspicious reviews are on average. | Each review gets a score based on suspicious features (0-10). Higher means more risk. |

    **Review Trust Levels:**
    - **Legitimate**: Seems genuine with no suspicious signs. No issues found.
    - **Low Risk**: Slightly suspicious, but mostly okay. Might be short or generic.
    - **Medium Risk**: Several warning signs, could be fake. E.g., repetitive words or odd ratings.
    - **High Risk**: Very likely fake. E.g., duplicate text or mismatched rating and comments.
    
    These levels are based on patterns like review length, word repetition, and rating-comment mismatches.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Sidebar with quick stats (no download report)
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Quick Stats")
st.sidebar.metric("📈 Total Reviews", f"{len(filtered_df):,}", help="Number of reviews analyzed.")
st.sidebar.metric("⭐ Average Rating", f"{filtered_df['rating'].mean():.2f}/5.0", help="Average star rating from all reviews.")
st.sidebar.metric("😊 Positive Feelings", f"{(filtered_df['sentiment'].str.contains('Positive', na=False)).sum()/len(filtered_df)*100:.1f}%", help="Percentage of happy reviews.")
st.sidebar.metric("🔍 Trust Score", f"{(filtered_df['fraudFlag'] == 'Legitimate').sum()/len(filtered_df)*100:.1f}%", help="Percentage of genuine reviews.")
st.sidebar.metric("💼 Business Impact", f"{filtered_df['businessImpact'].mean():.2f}", help="Average influence per review.")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center;'>Created by insights3d - email to aneesh@insights3d.com</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
