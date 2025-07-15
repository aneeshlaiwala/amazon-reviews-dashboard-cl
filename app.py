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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import warnings
warnings.filterwarnings('ignore')

# Try to import optional dependencies with fallbacks
try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.sentiment import SentimentIntensityAnalyzer
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    NLTK_AVAILABLE = True
except:
    NLTK_AVAILABLE = False

import datetime
import random

# Ultra-modern CSS for C-level presentation
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #f5576c 75%, #4facfe 100%);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        background-clip: text;
        animation: gradient 3s ease infinite;
        text-shadow: 0 4px 20px rgba(0,0,0,0.3);
        letter-spacing: -2px;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5rem;
            letter-spacing: -1px;
        }
    }
    
    .subheader {
        font-size: 2rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 1rem;
        border-bottom: 3px solid transparent;
        background: linear-gradient(90deg, #667eea, #764ba2);
        background-clip: border-box;
        border-image: linear-gradient(90deg, #667eea, #764ba2) 1;
        padding-bottom: 0.8rem;
    }
    
    .executive-summary {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.95) 0%, rgba(118, 75, 162, 0.95) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 25px;
        padding: 3rem;
        margin: 2rem 0;
        color: white;
        box-shadow: 
            0 25px 50px -12px rgba(0, 0, 0, 0.25),
            0 0 0 1px rgba(255, 255, 255, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .executive-summary::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at 20% 80%, rgba(120, 119, 198, 0.3) 0%, transparent 50%),
                    radial-gradient(circle at 80% 20%, rgba(255, 119, 198, 0.3) 0%, transparent 50%),
                    radial-gradient(circle at 40% 40%, rgba(120, 219, 255, 0.2) 0%, transparent 50%);
        pointer-events: none;
    }
    
    .executive-content {
        position: relative;
        z-index: 1;
    }
    .metric-card {
        background: linear-gradient(135deg, #232526 0%, #414345 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 20px;
        padding: 1rem;
        margin: 1rem 0;
        text-align: center;
        color: #fff !important; /* White text for contrast */
        box-shadow: 
            0 20px 40px -10px rgba(0, 0, 0, 0.3),
            0 0 0 1px rgba(255, 255, 255, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: auto;
        min-height: 180px;
        max-height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        width: 100%;
    }
    .metric-card * {
        color: #fff !important;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.5s;
    }
    
    .metric-card:hover::before {
        left: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 
            0 30px 60px -15px rgba(0, 0, 0, 0.4),
            0 0 0 1px rgba(255, 255, 255, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.3);
    }
    
    .metric-card h3 {
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    .metric-card h2 {
        font-size: 1.8rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
        line-height: 1.3;
    }
    
    .metric-card p {
        font-size: 0.8rem;
        margin: 0.3rem 0 0 0;
        opacity: 0.8;
        font-weight: 500;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    
    .chart-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 25px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 
            0 25px 50px -12px rgba(0, 0, 0, 0.15),
            0 0 0 1px rgba(255, 255, 255, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    
    .chart-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at 10% 20%, rgba(102, 126, 234, 0.05) 0%, transparent 50%),
                    radial-gradient(circle at 90% 80%, rgba(118, 75, 162, 0.05) 0%, transparent 50%);
        pointer-events: none;
    }
    
    .chart-container:hover {
        transform: translateY(-5px);
        box-shadow: 
            0 35px 70px -15px rgba(0, 0, 0, 0.2),
            0 0 0 1px rgba(255, 255, 255, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.3);
    }
    
    .insight-box {
        background: linear-gradient(135deg, rgba(40, 167, 69, 0.1) 0%, rgba(40, 167, 69, 0.05) 100%);
        border-left: 5px solid #28a745;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        backdrop-filter: blur(10px);
        box-shadow: 0 10px 30px rgba(40, 167, 69, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .insight-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 3px;
        height: 100%;
        background: linear-gradient(to bottom, #28a745, #20c997);
        border-radius: 0 3px 3px 0;
    }
    
    .insight-title {
        font-weight: 700;
        color: #28a745;
        font-size: 1.1rem;
        margin-bottom: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .filter-sidebar {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 249, 250, 0.95) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 25px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 
            0 20px 40px -10px rgba(0, 0, 0, 0.1),
            0 0 0 1px rgba(255, 255, 255, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.3);
        position: sticky;
        top: 2rem;
        height: fit-content;
    }
    
    .filter-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 1.5rem;
        text-align: center;
        border-bottom: 2px solid #e9ecef;
        padding-bottom: 1rem;
    }
    
    .filter-section {
        margin-bottom: 1.5rem;
        padding: 1rem;
        background: rgba(255, 255, 255, 0.5);
        border-radius: 15px;
        border: 1px solid rgba(0, 0, 0, 0.05);
    }
    
    .filter-label {
        font-weight: 600;
        color: #495057;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
        display: block;
    }
    
    .active-filters {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.3);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .recommendation-box {
        background: linear-gradient(135deg, rgba(132, 250, 176, 0.2) 0%, rgba(143, 211, 244, 0.2) 100%);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(132, 250, 176, 0.3);
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        color: #2c3e50;
        box-shadow: 0 15px 35px rgba(132, 250, 176, 0.15);
        position: relative;
        overflow: hidden;
    }
    
    .recommendation-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at 20% 20%, rgba(132, 250, 176, 0.1) 0%, transparent 50%);
        pointer-events: none;
    }
    
    .verbatim-section {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(0, 0, 0, 0.1);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
    }
    
    .verbatim-section:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.12);
    }
    
    .positive-verbatim {
        border-left: 4px solid #28a745;
        background: linear-gradient(135deg, rgba(40, 167, 69, 0.05) 0%, rgba(40, 167, 69, 0.02) 100%);
    }
    
    .negative-verbatim {
        border-left: 4px solid #dc3545;
        background: linear-gradient(135deg, rgba(220, 53, 69, 0.05) 0%, rgba(220, 53, 69, 0.02) 100%);
    }
    
    .explained-box {
        background: linear-gradient(135deg, rgba(249, 249, 249, 0.95) 0%, rgba(232, 245, 232, 0.95) 100%);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(76, 175, 80, 0.2);
        border-left: 5px solid #4CAF50;
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 15px 35px rgba(76, 175, 80, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .explained-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at 10% 10%, rgba(76, 175, 80, 0.05) 0%, transparent 50%);
        pointer-events: none;
    }
    
    .explained-title {
        font-weight: 700;
        color: #4CAF50;
        font-size: 1.8rem;
        margin-bottom: 1.5rem;
        text-align: center;
        position: relative;
        z-index: 1;
    }
    
    /* Enhanced tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 249, 250, 0.95) 100%);
        backdrop-filter: blur(20px);
        border-radius: 15px;
        padding: 0.5rem;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 10px;
        color: #6c757d;
        font-weight: 600;
        padding: 1rem 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Enhanced buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 10px;
        color: white;
        font-weight: 600;
        padding: 0.8rem 2rem;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Enhanced selectbox and multiselect */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 0, 0, 0.1);
        border-radius: 10px;
    }
    
    .stMultiSelect > div > div {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 0, 0, 0.1);
        border-radius: 10px;
    }
    
    /* 3D effect for plotly charts */
    .js-plotly-plot {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 
            0 20px 40px rgba(0, 0, 0, 0.1),
            0 0 0 1px rgba(255, 255, 255, 0.1);
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(248, 249, 250, 0.1) 100%);
        backdrop-filter: blur(10px);
        border-top: 1px solid rgba(0, 0, 0, 0.1);
        color: #6c757d;
        font-weight: 500;
    }
    
    /* Mobile responsiveness */
@media (max-width: 768px) {
    .metric-card {
        padding: 0.8rem; /* Further reduced padding for mobile */
        margin: 0.5rem 0;
        min-height: 140px; /* Adjusted minimum height for mobile */
        max-height: 160px; /* Adjusted maximum height for mobile */
    }
    
    .metric-card h2 {
        font-size: 1.4rem; /* Further reduced font size for mobile */
    }
    
    .metric-card h3 {
        font-size: 0.7rem; /* Further reduced font size for mobile */
    }
    
    .metric-card p {
        font-size: 0.6rem; /* Further reduced font size for mobile */
    }
}

    /* Animation for page load */
    .main-content {
        animation: fadeInUp 0.8s ease-out;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Wordcloud container */
    .wordcloud-container {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 249, 250, 0.95) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 25px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.15);
        text-align: center;
    }
    
    .sidebar-content {
        background: linear-gradient(135deg, #f8fafc 0%, #e0e7ef 100%);
        border-radius: 20px;
        box-shadow: 0 4px 24px rgba(102,126,234,0.08);
        padding: 2rem 1.5rem 1.5rem 1.5rem;
        margin-bottom: 2rem;
        border: 1px solid #e3e8f0;
    }
</style>
""", unsafe_allow_html=True)

if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'apply_filters' not in st.session_state:
    st.session_state.apply_filters = False

@st.cache_data
def load_and_process_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        return None
    
    df.columns = ['reviewId', 'reviewerName', 'reviewText', 'rating', 'summary', 
                  'helpful', 'totalVotes', 'reviewDate', 'year']
    
    df['reviewText'] = df['reviewText'].fillna('').astype(str)
    df['summary'] = df['summary'].fillna('').astype(str)
    df['reviewerName'] = df['reviewerName'].fillna('Anonymous').astype(str)
    
    df['reviewDate'] = pd.to_datetime(df['reviewDate'], format='%d-%m-%Y', errors='coerce')
    df['month'] = df['reviewDate'].dt.month
    df['month_name'] = df['reviewDate'].dt.month_name()
    df['quarter'] = df['reviewDate'].dt.quarter
    
    df['reviewLength'] = df['reviewText'].str.len()
    df['wordCount'] = df['reviewText'].str.split().str.len()
    df['sentenceCount'] = df['reviewText'].str.count(r'[.!?]+')
    df['exclamationCount'] = df['reviewText'].str.count('!')
    df['capsCount'] = df['reviewText'].str.count('[A-Z]')
    df['capsRatio'] = df['capsCount'] / (df['reviewLength'] + 1)
    
    return df

def advanced_sentiment_analysis_multilevel(text):
    if not text or len(text.strip()) == 0:
        return {
            'sentiment': 'Neutral',
            'polarity': 0,
            'confidence': 0.5,
            'emotion': 'neutral'
        }
    
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    if NLTK_AVAILABLE:
        try:
            sia = SentimentIntensityAnalyzer()
            vader_scores = sia.polarity_scores(text)
            polarity = vader_scores['compound']
        except:
            pass
    
    if polarity > 0.5:
        sentiment = 'Extremely Positive'
        emotion = 'enthusiastic'
    elif polarity > 0.3:
        sentiment = 'Very Positive'
        emotion = 'satisfied'
    elif polarity > 0.1:
        sentiment = 'Positive'
        emotion = 'pleased'
    elif polarity > -0.1:
        sentiment = 'Neutral'
        emotion = 'neutral'
    elif polarity > -0.3:
        sentiment = 'Negative'
        emotion = 'disappointed'
    elif polarity > -0.5:
        sentiment = 'Very Negative'
        emotion = 'frustrated'
    else:
        sentiment = 'Extremely Negative'
        emotion = 'angry'
    
    confidence = min(abs(polarity) + (1 - subjectivity) * 0.3 + len(text.split()) * 0.01, 1.0)
    
    return {
        'sentiment': sentiment,
        'polarity': polarity,
        'confidence': confidence,
        'emotion': emotion
    }

def sophisticated_fraud_detection(df):
    fraud_flags = []
    fraud_reasons = []
    fraud_scores = []
    
    user_behavior = df.groupby('reviewerName').agg({
        'reviewDate': 'count',
        'rating': ['mean', 'std'],
        'wordCount': 'mean',
        'reviewText': lambda x: len(set(x))
    }).round(2)
    
    user_behavior.columns = ['review_count', 'avg_rating', 'rating_std', 'avg_word_count', 'unique_reviews']
    
    for idx, row in df.iterrows():
        flags = []
        score = 0
        
        words = row['reviewText'].lower().split()
        if len(words) < 3:
            flags.append('Extremely Short Review')
            score += 3
        
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.2:
                flags.append('High Word Repetition')
                score += 2
        
        generic_patterns = [
            r'\b(best|great|excellent|amazing|perfect|awesome)\s+(product|item|purchase|buy|deal)\b',
            r'\b(highly\s+recommend|five\s+stars?|10/10|thumbs\s+up|love\s+it)\b',
            r'\b(fast\s+shipping|quick\s+delivery|arrived\s+quickly|super\s+fast)\b'
        ]
        
        generic_count = sum(1 for pattern in generic_patterns 
                          if re.search(pattern, row['reviewText'].lower()))
        if generic_count >= 2 and len(words) < 25:
            flags.append('Generic Template Language')
            score += 2
        
        user_stats = user_behavior.loc[row['reviewerName']]
        
        if user_stats['review_count'] > 10:
            score += 1
            
        if user_stats['rating_std'] < 0.5 and user_stats['review_count'] > 5:
            flags.append('Consistent Rating Pattern')
            score += 2
        
        if row['capsRatio'] > 0.3:
            flags.append('Excessive Capitalization')
            score += 1
        
        if row['exclamationCount'] > 3 and len(words) < 30:
            flags.append('Excessive Enthusiasm Markers')
            score += 1
        
        try:
            sentiment_result = advanced_sentiment_analysis_multilevel(row['reviewText'])
            sentiment_score = sentiment_result['polarity']
            
            if row['rating'] >= 4 and sentiment_score < -0.4:
                flags.append('High Rating with Negative Sentiment')
                score += 3
            elif row['rating'] <= 2 and sentiment_score > 0.4:
                flags.append('Low Rating with Positive Sentiment')
                score += 3
        except:
            pass
        
        exact_duplicates = df[df['reviewText'] == row['reviewText']]
        if len(exact_duplicates) > 1:
            flags.append('Exact Duplicate Content')
            score += 4
        
        if score >= 6:
            fraud_flag = 'High Risk'
        elif score >= 4:
            fraud_flag = 'Medium Risk'
        elif score >= 2:
            fraud_flag = 'Low Risk'
        else:
            fraud_flag = 'Legitimate'
        
        fraud_flags.append(fraud_flag)
        fraud_reasons.append('; '.join(flags) if flags else 'No Issues Detected')
        fraud_scores.append(score)
    
    return fraud_flags, fraud_reasons, fraud_scores

def executive_topic_modeling(texts, n_topics=8):
    try:
        cleaned_texts = []
        
        for text in texts:
            if text and len(str(text).strip()) > 10:
                clean_text = re.sub(r'[^\w\s]', ' ', str(text).lower())
                clean_text = ' '.join(clean_text.split())
                cleaned_texts.append(clean_text)
        
        if len(cleaned_texts) < n_topics:
            return [], ['General Discussion'] * len(texts)
        
        if NLTK_AVAILABLE:
            try:
                stop_words = set(stopwords.words('english'))
            except:
                stop_words = set()
        else:
            stop_words = set()
        
        business_stops = {
            'product', 'item', 'amazon', 'buy', 'bought', 'purchase', 'purchased',
            'get', 'got', 'use', 'used', 'using', 'work', 'works', 'working',
            'one', 'two', 'would', 'could', 'really', 'very', 'much', 'well',
            'time', 'first', 'last', 'way', 'make', 'made', 'take', 'took'
        }
        stop_words.update(business_stops)
        
        vectorizer = TfidfVectorizer(
            max_features=300,
            stop_words=list(stop_words),
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2),
            token_pattern=r'\b[a-zA-Z]{2,}\b'
        )
        
        doc_term_matrix = vectorizer.fit_transform(cleaned_texts)
        
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=20,
            learning_method='batch'
        )
        lda.fit(doc_term_matrix)
        
        feature_names = vectorizer.get_feature_names_out()
        
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[-15:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            key_words = top_words[:8]
            topic_label = create_business_topic_label(key_words)
            if topic_label not in topics:  # Avoid duplicates
                topics.append(topic_label)
        
        doc_topic_matrix = lda.transform(doc_term_matrix)
        topic_assignments = []
        
        for doc_topics in doc_topic_matrix:
            max_prob = np.max(doc_topics)
            if max_prob > 0.25:
                topic_idx = np.argmax(doc_topics)
                topic_assignments.append(topics[topic_idx] if topic_idx < len(topics) else "Mixed Themes")
            else:
                topic_assignments.append("Mixed Themes")
        
        full_topic_assignments = ['General Discussion'] * len(texts)
        j = 0
        for i, text in enumerate(texts):
            if text and len(str(text).strip()) > 10:
                if j < len(topic_assignments):
                    full_topic_assignments[i] = topic_assignments[j]
                j += 1
        
        return topics, full_topic_assignments
        
    except Exception as e:
        return [], ['General Discussion'] * len(texts)

def create_business_topic_label(key_words):
    if any(word in key_words for word in ['camera', 'video', 'photo', 'picture']):
        return "Camera & Video Performance"
    elif any(word in key_words for word in ['phone', 'samsung', 'galaxy', 'android']):
        return "Mobile Device Compatibility"
    elif any(word in key_words for word in ['speed', 'fast', 'slow', 'transfer']):
        return "Speed & Transfer Performance"
    elif any(word in key_words for word in ['storage', 'capacity', 'space', 'memory', 'gb']):
        return "Storage Capacity & Management"
    elif any(word in key_words for word in ['price', 'value', 'money', 'cost', 'cheap']):
        return "Price & Value Proposition"
    elif any(word in key_words for word in ['quality', 'build', 'durable', 'solid']):
        return "Build Quality & Durability"
    elif any(word in key_words for word in ['shipping', 'delivery', 'arrived', 'package']):
        return "Shipping & Delivery Experience"
    elif any(word in key_words for word in ['easy', 'simple', 'difficult', 'install']):
        return "Ease of Use & Installation"
    elif any(word in key_words for word in ['recommend', 'satisfied', 'happy', 'disappointed']):
        return "Overall Customer Satisfaction"
    elif any(word in key_words for word in ['problem', 'issue', 'error', 'failed']):
        return "Technical Issues & Problems"
    else:
        return "General Product Discussion"

def create_executive_summary(df):
    total_reviews = len(df)
    avg_rating = df['rating'].mean()
    
    sentiment_dist = df['sentiment'].value_counts(normalize=True) * 100
    
    high_risk_rate = (df['fraudFlag'] == 'High Risk').sum() / total_reviews * 100
    medium_risk_rate = (df['fraudFlag'] == 'Medium Risk').sum() / total_reviews * 100
    total_risk_rate = high_risk_rate + medium_risk_rate
    
    recent_reviews = df[df['reviewDate'] > df['reviewDate'].max() - pd.Timedelta(days=90)]
    recent_avg_rating = recent_reviews['rating'].mean() if len(recent_reviews) > 0 else avg_rating
    rating_trend = "📈 Getting Better" if recent_avg_rating > avg_rating else "📉 Getting Worse" if recent_avg_rating < avg_rating else "➡️ Staying the Same"
    
    detailed_reviews = df[df['wordCount'] > 50]
    engagement_rate = len(detailed_reviews) / total_reviews * 100
    
    # Top topics, excluding 'Mixed Themes' and 'General Discussion' if possible
    top_topics = df['topic'].value_counts()
    top_topics = top_topics[~top_topics.index.isin(['Mixed Themes', 'General Discussion'])].head(3)
    if len(top_topics) < 3:
        top_topics = df['topic'].value_counts().head(3)
    
    summary = f"""
    <div class="executive-content">
    
    ## 🎯 **Executive Summary for Business Leaders**
    *Transform customer feedback into strategic business decisions*

    ### 📊 **Customer Satisfaction Overview**
    - **Total Customer Reviews:** {total_reviews:,} authentic customer voices analyzed
    - **Customer Happiness Score:** {avg_rating:.2f}/5.0 ⭐ ({get_satisfaction_grade(avg_rating)})
    - **Market Trend:** {rating_trend} compared to historical performance
    - **Customer Engagement Level:** {"High" if engagement_rate > 30 else "Moderate" if engagement_rate > 15 else "Low"} ({engagement_rate:.1f}% write detailed feedback)
    
    ### 🎭 **Customer Sentiment Breakdown**
    - **Brand Advocates:** {sentiment_dist.get('Extremely Positive', 0) + sentiment_dist.get('Very Positive', 0) + sentiment_dist.get('Positive', 0):.1f}% actively promote our product
    - **Neutral Experience:** {sentiment_dist.get('Neutral', 0):.1f}% have average expectations met
    - **Customer Concerns:** {sentiment_dist.get('Negative', 0) + sentiment_dist.get('Very Negative', 0) + sentiment_dist.get('Extremely Negative', 0):.1f}% require immediate attention
    
    ### 🔍 **Review Authenticity & Trust**
    - **High-Risk Reviews:** {high_risk_rate:.1f}% flagged for suspicious patterns 🚨
    - **Medium-Risk Reviews:** {medium_risk_rate:.1f}% require verification ⚠️
    - **Verified Authentic Reviews:** {100 - total_risk_rate:.1f}% confirmed genuine feedback ✅
    - **Overall Trust Score:** {get_trust_score(total_risk_rate)}/10 (Market benchmark: 7.5+)
    </div>
    """
    
    return summary

def create_enhanced_executive_summary_with_topics(df):
    """Create a comprehensive executive summary with strategic insights"""
    total_reviews = len(df)
    avg_rating = df['rating'].mean()
    
    sentiment_dist = df['sentiment'].value_counts(normalize=True) * 100
    
    high_risk_rate = (df['fraudFlag'] == 'High Risk').sum() / total_reviews * 100
    medium_risk_rate = (df['fraudFlag'] == 'Medium Risk').sum() / total_reviews * 100
    total_risk_rate = high_risk_rate + medium_risk_rate
    
    recent_reviews = df[df['reviewDate'] > df['reviewDate'].max() - pd.Timedelta(days=90)]
    recent_avg_rating = recent_reviews['rating'].mean() if len(recent_reviews) > 0 else avg_rating
    rating_trend = "📈 Improving" if recent_avg_rating > avg_rating else "📉 Declining" if recent_avg_rating < avg_rating else "➡️ Stable"
    
    detailed_reviews = df[df['wordCount'] > 50]
    engagement_rate = len(detailed_reviews) / total_reviews * 100
    
    # Top topics analysis
    top_topics = df['topic'].value_counts()
    top_topics = top_topics[~top_topics.index.isin(['Mixed Themes', 'General Discussion'])].head(3)
    if len(top_topics) < 3:
        top_topics = df['topic'].value_counts().head(3)
    
    # Strategic recommendations
    strategic_recommendations = get_strategic_recommendations(avg_rating, total_risk_rate, engagement_rate, sentiment_dist)
    
    return f"""
    <div class="executive-content">
    
    ## 🎯 **Strategic Customer Intelligence Report**
    *Comprehensive analysis of {total_reviews:,} customer interactions*

    ### 📈 **Market Position & Performance**
    - **Customer Satisfaction Index:** {avg_rating:.2f}/5.0 ⭐ ({get_satisfaction_grade(avg_rating)})
    - **Market Trajectory:** {rating_trend} ({abs(recent_avg_rating - avg_rating):.2f} point change)
    - **Customer Engagement Quality:** {"Premium" if engagement_rate > 30 else "Standard" if engagement_rate > 15 else "Basic"} ({engagement_rate:.1f}% detailed feedback)
    - **Brand Health Score:** {calculate_brand_health_score(avg_rating, sentiment_dist, total_risk_rate)}/100

    ### 🎭 **Customer Sentiment Intelligence**
    - **🟢 Brand Champions:** {sentiment_dist.get('Extremely Positive', 0) + sentiment_dist.get('Very Positive', 0) + sentiment_dist.get('Positive', 0):.1f}% (Premium customer advocates)
    - **🟡 Neutral Zone:** {sentiment_dist.get('Neutral', 0):.1f}% (Conversion opportunity)
    - **🔴 At-Risk Customers:** {sentiment_dist.get('Negative', 0) + sentiment_dist.get('Very Negative', 0) + sentiment_dist.get('Extremely Negative', 0):.1f}% (Retention priority)

    ### 🛡️ **Data Integrity & Trust Metrics**
    - **Verified Authentic Reviews:** {100 - total_risk_rate:.1f}% ✅
    - **Suspicious Activity:** {total_risk_rate:.1f}% flagged for review
    - **Data Confidence Level:** {get_trust_score(total_risk_rate)}/10 (Industry standard: 7.5+)

    ### 🏆 **Top Customer Priorities**
    {get_formatted_top_topics(top_topics, total_reviews)}

    ### 💡 **Strategic Action Plan**
    {strategic_recommendations}
    
    </div>
    """

def get_formatted_top_topics(top_topics, total_reviews):
    """Format top topics for executive summary"""
    formatted_topics = []
    for i, (topic, count) in enumerate(top_topics.items(), 1):
        impact_level = "🔥 Critical" if count/total_reviews > 0.15 else "⚡ High" if count/total_reviews > 0.10 else "📊 Moderate"
        formatted_topics.append(f"**{i}. {topic}**: {count:,} mentions ({count/total_reviews*100:.1f}%) - {impact_level} Impact")
    
    return "\n".join(formatted_topics)

def get_strategic_recommendations(avg_rating, total_risk_rate, engagement_rate, sentiment_dist):
    """Generate strategic recommendations based on data"""
    recommendations = []
    
    # Rating-based recommendations
    if avg_rating >= 4.5:
        recommendations.append("🎯 **Leverage Excellence**: Use positive reviews in marketing campaigns and case studies")
    elif avg_rating >= 4.0:
        recommendations.append("📈 **Optimize Performance**: Focus on converting 4-star to 5-star experiences")
    elif avg_rating >= 3.0:
        recommendations.append("🔧 **Quality Enhancement**: Address core product/service issues immediately")
    else:
        recommendations.append("🚨 **Crisis Management**: Implement emergency customer satisfaction protocols")
    
    # Trust-based recommendations
    if total_risk_rate > 15:
        recommendations.append("🛡️ **Review Verification**: Implement stricter review authenticity measures")
    elif total_risk_rate > 5:
        recommendations.append("👀 **Monitor Closely**: Enhanced suspicious activity detection required")
    
    # Engagement-based recommendations
    if engagement_rate < 15:
        recommendations.append("📢 **Increase Engagement**: Launch customer feedback incentive programs")
    
    return "\n".join(recommendations)

def calculate_brand_health_score(avg_rating, sentiment_dist, total_risk_rate):
    """Calculate overall brand health score out of 100"""
    rating_score = (avg_rating / 5.0) * 40  # 40% weight
    sentiment_score = (sentiment_dist.get('Extremely Positive', 0) + sentiment_dist.get('Very Positive', 0) + sentiment_dist.get('Positive', 0)) * 0.4  # 40% weight
    trust_score = (100 - total_risk_rate) * 0.2  # 20% weight
    
    return int(rating_score + sentiment_score + trust_score)

def get_satisfaction_grade(rating):
    if rating >= 4.5:
        return "Excellent (A+)"
    elif rating >= 4.0:
        return "Good (A-)"
    elif rating >= 3.5:
        return "Fair (B)"
    elif rating >= 3.0:
        return "Poor (C)"
    else:
        return "Needs Work (D)"

def get_trust_score(risk_rate):
    if risk_rate < 5:
        return 9
    elif risk_rate < 10:
        return 8
    elif risk_rate < 20:
        return 6
    else:
        return 4

def process_data_with_advanced_ml(df):
    st.info("🤖 Analyzing reviews with advanced AI algorithms...")
    
    sentiment_results = []
    progress_bar = st.progress(0)
    
    for i, text in enumerate(df['reviewText']):
        result = advanced_sentiment_analysis_multilevel(text)
        sentiment_results.append(result)
        progress_bar.progress((i + 1) / len(df) * 0.4)
    
    df['sentiment'] = [r['sentiment'] for r in sentiment_results]
    df['sentimentScore'] = [r['polarity'] for r in sentiment_results]
    df['sentimentConfidence'] = [r['confidence'] for r in sentiment_results]
    df['emotion'] = [r['emotion'] for r in sentiment_results]
    
    fraud_flags, fraud_reasons, fraud_scores = sophisticated_fraud_detection(df)
    df['fraudFlag'] = fraud_flags
    df['fraudReason'] = fraud_reasons
    df['fraudScore'] = fraud_scores
    progress_bar.progress(0.7)
    
    topics, topic_assignments = executive_topic_modeling(df['reviewText'].tolist())
    df['topic'] = topic_assignments
    progress_bar.progress(0.9)
    
    df['reviewValue'] = calculate_review_value(df)
    df['customerSegment'] = segment_customers(df)
    df['businessImpact'] = calculate_business_impact(df)
    
    progress_bar.progress(1.0)
    progress_bar.empty()
    
    return df, topics

def calculate_review_value(df):
    value_scores = []
    for _, row in df.iterrows():
        score = 0
        if row['wordCount'] > 50:
            score += 2
        elif row['wordCount'] > 20:
            score += 1
        
        if row['helpful'] > 0:
            score += 2
        
        if row['sentimentConfidence'] > 0.8:
            score += 1
        
        if row['fraudFlag'] == 'Legitimate':
            score += 2
        elif row['fraudFlag'] == 'Low Risk':
            score += 1
        
        value_scores.append(min(score, 8))
    
    return value_scores

def segment_customers(df):
    segments = []
    for _, row in df.iterrows():
        if row['fraudFlag'] in ['High Risk', 'Medium Risk']:
            segment = 'Suspicious'
        elif row['wordCount'] > 100 and row['sentimentConfidence'] > 0.7:
            segment = 'Engaged Advocate'
        elif row['rating'] >= 4 and row['wordCount'] > 30:
            segment = 'Satisfied Customer'
        elif row['rating'] <= 2:
            segment = 'Dissatisfied Customer'
        elif row['wordCount'] < 10:
            segment = 'Passive User'
        else:
            segment = 'Average Customer'
        segments.append(segment)
    return segments

def calculate_business_impact(df):
    """
    Calculate business impact score ranging from -5.0 to +5.0
    Positive scores indicate positive business impact, negative scores indicate negative impact
    """
    impact_scores = []
    for _, row in df.iterrows():
        impact = 0
        
        # Rating contribution (scale: -3 to +3)
        if row['rating'] == 5:
            impact += 3
        elif row['rating'] == 4:
            impact += 1
        elif row['rating'] == 3:
            impact += 0
        elif row['rating'] == 2:
            impact -= 2
        elif row['rating'] == 1:
            impact -= 3
        
        # Sentiment contribution (scale: -2 to +2)
        if 'Extremely Positive' in row['sentiment']:
            impact += 2
        elif 'Very Positive' in row['sentiment']:
            impact += 1
        elif 'Positive' in row['sentiment']:
            impact += 0.5
        elif 'Negative' in row['sentiment']:
            impact -= 1
        elif 'Very Negative' in row['sentiment'] or 'Extremely Negative' in row['sentiment']:
            impact -= 2
        
        # Word count multiplier (detailed reviews have more impact)
        if row['wordCount'] > 100:
            impact = impact * 1.5
        elif row['wordCount'] > 50:
            impact = impact * 1.2
        elif row['wordCount'] < 10:
            impact = impact * 0.5
        
        # Trust factor (fraudulent reviews reduce impact)
        if row['fraudFlag'] == 'High Risk':
            impact = impact * 0.3
        elif row['fraudFlag'] == 'Medium Risk':
            impact = impact * 0.7
        
        # Cap the score between -5.0 and +5.0
        impact = max(-5.0, min(5.0, impact))
        impact_scores.append(round(impact, 1))
    
    return impact_scores

def create_chart_with_insights(fig, insight_text):
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f'<div class="insight-box"><div class="insight-title">📊 Key Insight</div> {insight_text}</div>', 
                unsafe_allow_html=True)

def extract_sample_verbatims(df):
    positive_reviews = df[
        (df['sentiment'].str.contains('Positive')) & 
        (df['fraudFlag'] == 'Legitimate') &
        (df['wordCount'] > 20) &
        (df['sentimentConfidence'] > 0.7)
    ].nlargest(5, 'businessImpact')
    
    negative_reviews = df[
        (df['sentiment'].str.contains('Negative')) & 
        (df['fraudFlag'] == 'Legitimate') &
        (df['wordCount'] > 20) &
        (df['sentimentConfidence'] > 0.7)
    ].nsmallest(5, 'businessImpact')
    
    return positive_reviews, negative_reviews

def get_word_frequencies(text_series):
    text = ' '.join(text_series.dropna()).lower()
    
    if NLTK_AVAILABLE:
        try:
            words = word_tokenize(text)
            stop_words = set(stopwords.words('english'))
        except:
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
            stop_words = set()
    else:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
        stop_words = set()
    
    business_stops = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'card', 'memory', 'product', 'item', 'amazon', 'one', 'get', 'use', 'work'
    }
    stop_words.update(business_stops)
    
    filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
    return Counter(filtered_words).most_common(20)

def create_3d_enhanced_charts():
    """Create enhanced 3D-style charts with modern styling"""
    # Enhanced color palettes for 3D effect
    colors_3d = [
        '#667eea', '#764ba2', '#f093fb', '#f5576c',
        '#4facfe', '#00f2fe', '#43e97b', '#38f9d7'
    ]
    
    return colors_3d

def generate_wordcloud(df):
    """Generate word cloud from review text"""
    if not WORDCLOUD_AVAILABLE:
        return None
    
    # Combine all review text
    text = ' '.join(df['reviewText'].dropna())
    
    # Remove common stop words
    if NLTK_AVAILABLE:
        try:
            stop_words = set(stopwords.words('english'))
        except:
            stop_words = set()
    else:
        stop_words = set()
    
    # Add business-specific stop words
    business_stops = {
        'product', 'item', 'amazon', 'buy', 'bought', 'purchase', 'purchased',
        'get', 'got', 'use', 'used', 'using', 'work', 'works', 'working',
        'one', 'two', 'would', 'could', 'really', 'very', 'much', 'well',
        'time', 'first', 'last', 'way', 'make', 'made', 'take', 'took'
    }
    stop_words.update(business_stops)
    
    try:
        wordcloud = WordCloud(
            width=800, 
            height=400,
            background_color='white',
            stopwords=stop_words,
            max_words=100,
            colormap='viridis',
            relative_scaling=0.5,
            random_state=42
        ).generate(text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        plt.tight_layout(pad=0)
        
        return fig
    except:
        return None

def generate_executive_summary_card(df):
    total_reviews = len(df)
    avg_rating = df['rating'].mean()
    sentiment_dist = df['sentiment'].value_counts(normalize=True) * 100
    high_risk_rate = (df['fraudFlag'] == 'High Risk').sum() / total_reviews * 100
    medium_risk_rate = (df['fraudFlag'] == 'Medium Risk').sum() / total_reviews * 100
    total_risk_rate = high_risk_rate + medium_risk_rate
    recent_reviews = df[df['reviewDate'] > df['reviewDate'].max() - pd.Timedelta(days=90)]
    recent_avg_rating = recent_reviews['rating'].mean() if len(recent_reviews) > 0 else avg_rating
    rating_trend = "📈 Improving" if recent_avg_rating > avg_rating else "📉 Declining" if recent_avg_rating < avg_rating else "➡️ Stable"
    detailed_reviews = df[df['wordCount'] > 50]
    engagement_rate = len(detailed_reviews) / total_reviews * 100
    brand_health_score = int((avg_rating / 5.0) * 40 + (sentiment_dist.get('Extremely Positive', 0) + sentiment_dist.get('Very Positive', 0) + sentiment_dist.get('Positive', 0)) * 0.4 + (100 - total_risk_rate) * 0.2)
    top_topics = df['topic'].value_counts().head(3).index.tolist()
    actionable_insight = (
        f"Focus on <b>{top_topics[0]}</b> and <b>{top_topics[1]}</b> to maximize customer satisfaction. "
        f"Monitor <b>{top_topics[2]}</b> for emerging issues. "
        f"Engagement is {'excellent' if engagement_rate > 30 else 'good' if engagement_rate > 15 else 'needs improvement'}."
    )
    return f"""
    <div style='
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 20px;
        padding: 2.5rem 2.5rem 2rem 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    '>
        <h2 style='font-size:2.2rem; font-weight:800; margin-bottom:0.5rem;'>🎯 Strategic Customer Intelligence Report</h2>
        <div style='font-size:1.1rem; line-height:1.7;'>
            <b>🗓️ Date:</b> {datetime.datetime.now().strftime('%B %d, %Y')}<br>
            <b>🧮 Total Reviews:</b> {total_reviews:,}<br>
            <b>⭐ Customer Satisfaction:</b> <span style='color:#ffeb3b;'>{avg_rating:.2f}/5.0</span> ({rating_trend})<br>
            <b>💬 Engagement Quality:</b> <span style='color:#00bcd4;'>{engagement_rate:.1f}% detailed feedback</span><br>
            <b>🏆 Brand Health Score:</b> <span style='color:#ff4081;'>{brand_health_score}/100</span><br>
            <b>🟢 Brand Champions:</b> {sentiment_dist.get('Extremely Positive', 0) + sentiment_dist.get('Very Positive', 0) + sentiment_dist.get('Positive', 0):.1f}%<br>
            <b>🔴 At-Risk Customers:</b> {sentiment_dist.get('Negative', 0) + sentiment_dist.get('Very Negative', 0) + sentiment_dist.get('Extremely Negative', 0):.1f}%<br>
            <b>🛡️ Verified Reviews:</b> {100 - total_risk_rate:.1f}%<br>
            <b>🏅 Top Priorities:</b> <span style='color:#ffd700;'>{', '.join(top_topics)}</span><br>
            <b>💡 Actionable Insight:</b> {actionable_insight}
        </div>
    </div>
    """

def main():
    st.markdown("""
<div style='text-align:center; margin-bottom:2rem;'>
    <h1 class="main-header" style="margin-bottom:0.5rem;">📊 Customer Reviews Intelligence Platform</h1>
    <div style="font-style:italic; font-size:1.2rem; color:#444; margin-top:0;">Transform customer feedback into strategic business intelligence for C-level decision making</div>
</div>
""", unsafe_allow_html=True)
    
    # Create layout with sidebar for filters
    col1, col2 = st.columns([1, 4])
    
    with col1:
        # Enhanced filter sidebar
        st.markdown('<div class="filter-sidebar">', unsafe_allow_html=True)
        st.markdown('<div class="filter-header">🎛️ Smart Filters</div>', unsafe_allow_html=True)
        
        # File upload in sidebar
        uploaded_file = st.file_uploader("📁 Upload Dataset", type=['csv'], help="Upload your Amazon reviews CSV file")
        
        if uploaded_file or st.session_state.processed_data is not None:
            if uploaded_file and st.session_state.processed_data is None:
                with st.spinner('🤖 Processing with AI...'):
                    df = load_and_process_data(uploaded_file)
                    if df is not None:
                        df, topics = process_data_with_advanced_ml(df)
                        st.session_state.processed_data = df
                        st.session_state.topics = topics
                        st.success("✅ Analysis Complete!")
            
            if st.session_state.processed_data is not None:
                df = st.session_state.processed_data
                topics = getattr(st.session_state, 'topics', [])
                
                st.markdown('<div class="filter-section">', unsafe_allow_html=True)
                st.markdown('<span class="filter-label">⭐ Rating Filter</span>', unsafe_allow_html=True)
                rating_filter = st.multiselect("", 
                                              sorted(df['rating'].unique()), 
                                              default=sorted(df['rating'].unique()),
                                              key="rating_filter")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="filter-section">', unsafe_allow_html=True)
                st.markdown('<span class="filter-label">😊 Sentiment Filter</span>', unsafe_allow_html=True)
                sentiment_filter = st.multiselect("", 
                                                 df['sentiment'].unique(), 
                                                 default=df['sentiment'].unique(),
                                                 key="sentiment_filter")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="filter-section">', unsafe_allow_html=True)
                st.markdown('<span class="filter-label">🔍 Trust Level</span>', unsafe_allow_html=True)
                trust_filter = st.selectbox("", 
                                           ['All Reviews', 'Trusted Only', 'Suspicious Only', 'High Risk Only'],
                                           key="trust_filter")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="filter-section">', unsafe_allow_html=True)
                st.markdown('<span class="filter-label">👥 Customer Segments</span>', unsafe_allow_html=True)
                segment_filter = st.multiselect("", 
                                               df['customerSegment'].unique(), 
                                               default=df['customerSegment'].unique(),
                                               key="segment_filter")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="filter-section">', unsafe_allow_html=True)
                st.markdown('<span class="filter-label">💼 Business Impact</span>', unsafe_allow_html=True)
                min_impact = st.slider("", 
                                      float(df['businessImpact'].min()), 
                                      float(df['businessImpact'].max()), 
                                      float(df['businessImpact'].min()),
                                      key="impact_filter")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="filter-section">', unsafe_allow_html=True)
                st.markdown('<span class="filter-label">🎯 Confidence Level</span>', unsafe_allow_html=True)
                min_confidence = st.slider("", 0.0, 1.0, 0.0, 0.1, key="confidence_filter")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Filter action buttons
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("Apply Filters", type="primary"):
                        st.session_state.apply_filters = True
                
                with col_btn2:
                    if st.button("Reset All"):
                        st.session_state.apply_filters = False
                        st.rerun()
                
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Main content area
        if st.session_state.processed_data is not None:
            df = st.session_state.processed_data
            topics = getattr(st.session_state, 'topics', [])
            
            # Apply filters only when button is clicked
            if st.session_state.apply_filters:
                filtered_df = df[
                    (df['rating'].isin(rating_filter)) & 
                    (df['sentiment'].isin(sentiment_filter)) &
                    (df['customerSegment'].isin(segment_filter)) &
                    (df['businessImpact'] >= min_impact) &
                    (df['sentimentConfidence'] >= min_confidence)
                ]
                
                if trust_filter == 'Trusted Only':
                    filtered_df = filtered_df[filtered_df['fraudFlag'] == 'Legitimate']
                elif trust_filter == 'Suspicious Only':
                    filtered_df = filtered_df[filtered_df['fraudFlag'].isin(['Low Risk', 'Medium Risk'])]
                elif trust_filter == 'High Risk Only':
                    filtered_df = filtered_df[filtered_df['fraudFlag'] == 'High Risk']
            else:
                filtered_df = df
            
            # Show filter status
            if st.session_state.apply_filters:
                st.markdown(f"""
                <div class="active-filters">
                    <strong>🎯 Filters Applied</strong><br>
                    Showing {len(filtered_df):,} of {len(df):,} reviews ({len(filtered_df)/len(df)*100:.1f}% of dataset)
                </div>
                """, unsafe_allow_html=True)
            
            # Enhanced tabs for C-level presentation
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "🎯 Executive Dashboard", "📊 Strategic Analytics", "🔍 Deep Intelligence", 
                "💬 Customer Voices", "🚨 Risk Management"
            ])
            
            # TAB 1: Executive Dashboard
            with tab1:
                # Enhanced Executive Summary
                st.markdown(generate_executive_summary_card(filtered_df), unsafe_allow_html=True)
                
                # Premium metric cards with 3D effects
                st.markdown("### 📊 Key Performance Indicators")
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    avg_rating = filtered_df['rating'].mean()
                    rating_change = "📈" if avg_rating > 3.5 else "📉" if avg_rating < 3.0 else "➡️"
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h3>⭐ Customer Satisfaction</h3>
                        <h2>{avg_rating:.2f}/5.0</h2>
                        <p>{rating_change} Market Position</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    fraud_rate = (filtered_df['fraudFlag'].isin(['High Risk', 'Medium Risk'])).sum() / len(filtered_df) * 100
                    trust_icon = "🛡️" if fraud_rate < 10 else "⚠️" if fraud_rate < 20 else "🚨"
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h3>🔍 Trust Score</h3>
                        <h2>{100-fraud_rate:.1f}%</h2>
                        <p>{trust_icon} Data Integrity</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    positive_sentiment = (filtered_df['sentiment'].str.contains('Positive')).sum() / len(filtered_df) * 100
                    sentiment_trend = "📈" if positive_sentiment > 70 else "📊" if positive_sentiment > 50 else "📉"
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h3>😊 Brand Sentiment</h3>
                        <h2>{positive_sentiment:.1f}%</h2>
                        <p>{sentiment_trend} Customer Love</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    engagement_rate = (filtered_df['wordCount'] > 50).sum() / len(filtered_df) * 100
                    engagement_icon = "🔥" if engagement_rate > 30 else "📝" if engagement_rate > 15 else "💤"
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h3>💬 Engagement Rate</h3>
                        <h2>{engagement_rate:.1f}%</h2>
                        <p>{engagement_icon} Detail Level</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col5:
                    avg_business_impact = filtered_df['businessImpact'].mean()
                    impact_icon = "🚀" if avg_business_impact > 2 else "📈" if avg_business_impact > 0 else "📉"
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h3>💼 Business Impact</h3>
                        <h2>{avg_business_impact:.1f}/5.0</h2>
                        <p>{impact_icon} Revenue Effect</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Enhanced 3D-style charts
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.markdown("<h3 style='text-align:center; margin-bottom:1rem;'>📊 Customer Rating Distribution</h3>", unsafe_allow_html=True)
                    
                    # 3D-enhanced rating distribution
                    fig_rating = px.histogram(
                        filtered_df, x='rating', 
                        color_discrete_sequence=['#667eea'],
                        text_auto=True,
                        category_orders={"rating": [1, 2, 3, 4, 5]}
                    )
                    
                    # Enhanced 3D styling
                    fig_rating.update_layout(
                        xaxis_title="Star Rating",
                        yaxis_title="Number of Reviews",
                        showlegend=False, 
                        height=450,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Inter, sans-serif", size=12),
                        title_font_size=16,
                        title_x=0.5,
                        xaxis=dict(
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='rgba(128,128,128,0.2)',
                            showline=True,
                            linewidth=2,
                            linecolor='rgb(204, 204, 204)'
                        ),
                        yaxis=dict(
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='rgba(128,128,128,0.2)',
                            showline=True,
                            linewidth=2,
                            linecolor='rgb(204, 204, 204)'
                        )
                    )
                    
                    # Add shadow effect
                    fig_rating.update_traces(
                        marker=dict(
                            line=dict(width=2, color='rgba(102, 126, 234, 0.8)'),
                            opacity=0.8
                        )
                    )
                    
                    high_satisfaction = (filtered_df['rating'] >= 4).sum() / len(filtered_df) * 100
                    insight_text = f"**{high_satisfaction:.1f}% customer satisfaction rate** indicates strong market position. Focus on converting 4-star experiences to 5-star loyalty."
                    
                    create_chart_with_insights(fig_rating, insight_text)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.markdown("<h3 style='text-align:center; margin-bottom:1rem;'>🛡️ Customer Sentiment Analysis</h3>", unsafe_allow_html=True)
                    
                    # 3D-enhanced sentiment pie chart
                    sentiment_counts = filtered_df['sentiment'].value_counts()
                    colors = {
                        'Extremely Positive': '#0d7377', 'Very Positive': '#14a085', 'Positive': '#2ca02c',
                        'Neutral': '#ffbb33', 'Negative': '#ff6b6b', 'Very Negative': '#d62728', 'Extremely Negative': '#8b0000'
                    }
                    
                    fig_sentiment = px.pie(
                        values=sentiment_counts.values, 
                        names=sentiment_counts.index,
                        color=sentiment_counts.index,
                        color_discrete_map=colors
                    )
                    
                    # Enhanced 3D styling for pie chart
                    fig_sentiment.update_layout(
                        height=450,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Inter, sans-serif", size=12),
                        title_font_size=16,
                        title_x=0.5,
                        showlegend=True,
                        legend=dict(
                            orientation="v",
                            yanchor="middle",
                            y=0.5,
                            xanchor="left",
                            x=1.05
                        )
                    )
                    
                    # Add 3D effect to pie slices
                    fig_sentiment.update_traces(
                        textposition='inside',
                        textinfo='percent+label',
                        marker=dict(
                            line=dict(color='#FFFFFF', width=3)
                        ),
                        pull=[0.1 if 'Positive' in name else 0 for name in sentiment_counts.index]
                    )
                    
                    brand_advocates = sentiment_counts.get('Extremely Positive', 0) + sentiment_counts.get('Very Positive', 0)
                    detractors = sentiment_counts.get('Very Negative', 0) + sentiment_counts.get('Extremely Negative', 0)
                    nps_proxy = (brand_advocates - detractors) / len(filtered_df) * 100
                    
                    insight_text = f"**Net Promoter Score: {nps_proxy:.1f}%** - {brand_advocates:,} brand advocates vs {detractors:,} detractors. Excellent customer loyalty foundation."
                    
                    create_chart_with_insights(fig_sentiment, insight_text)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Word Cloud Section
                st.markdown("### 🔤 Customer Voice Cloud")
                if WORDCLOUD_AVAILABLE:
                    st.markdown('<div class="wordcloud-container">', unsafe_allow_html=True)
                    wordcloud_fig = generate_wordcloud(filtered_df)
                    if wordcloud_fig:
                        st.pyplot(wordcloud_fig)
                        word_freq = get_word_frequencies(filtered_df['reviewText'])
                        if word_freq:
                            top_words = [word for word, count in word_freq[:5]]
                            # Executive-style summary
                            summary_sentence = (
                                f"The most discussed topics are <span style='color:#667eea;font-weight:bold'>{', '.join(top_words[:3])}</span>, "
                                f"indicating that customers are primarily focused on these aspects. "
                                f"Prioritize these areas for maximum impact on satisfaction and loyalty."
                            )
                            st.markdown(
                                f"<div class='insight-box'><div class='insight-title'>🔑 Executive Insight</div> {summary_sentence}</div>",
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown("<div class='insight-box'><div class='insight-title'>🔑 Executive Insight</div> No significant keywords found.</div>", unsafe_allow_html=True)
                    else:
                        st.info("Word cloud generation failed - using alternative word frequency analysis")
                        word_freq = get_word_frequencies(filtered_df['reviewText'])
                        if word_freq:
                            freq_df = pd.DataFrame(word_freq[:15], columns=['Word', 'Frequency'])
                            fig_words = px.bar(freq_df, x='Frequency', y='Word', orientation='h',
                                             title="Top 15 Most Mentioned Words",
                                             color='Frequency', color_continuous_scale='viridis')
                            fig_words.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
                            st.plotly_chart(fig_words, use_container_width=True)
                            top_words = [word for word, count in word_freq[:5]]
                            summary_sentence = (
                                f"The most discussed topics are <span style='color:#667eea;font-weight:bold'>{', '.join(top_words[:3])}</span>, "
                                f"indicating that customers are primarily focused on these aspects. "
                                f"Prioritize these areas for maximum impact on satisfaction and loyalty."
                            )
                            st.markdown(
                                f"<div class='insight-box'><div class='insight-title'>🔑 Executive Insight</div> {summary_sentence}</div>",
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown("<div class='insight-box'><div class='insight-title'>🔑 Executive Insight</div> No significant keywords found.</div>", unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="wordcloud-container">', unsafe_allow_html=True)
                    word_freq = get_word_frequencies(filtered_df['reviewText'])
                    if word_freq:
                        freq_df = pd.DataFrame(word_freq[:15], columns=['Word', 'Frequency'])
                        fig_words = px.bar(freq_df, x='Frequency', y='Word', orientation='h',
                                         title="🔤 Top Customer Keywords",
                                         color='Frequency', color_continuous_scale='viridis')
                        fig_words.update_layout(
                            height=400, 
                            yaxis={'categoryorder':'total ascending'},
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig_words, use_container_width=True)
                        top_words = [word for word, count in word_freq[:5]]
                        summary_sentence = (
                            f"The most discussed topics are <span style='color:#667eea;font-weight:bold'>{', '.join(top_words[:3])}</span>, "
                            f"indicating that customers are primarily focused on these aspects. "
                            f"Prioritize these areas for maximum impact on satisfaction and loyalty."
                        )
                        st.markdown(
                            f"<div class='insight-box'><div class='insight-title'>🔑 Executive Insight</div> {summary_sentence}</div>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown("<div class='insight-box'><div class='insight-title'>🔑 Executive Insight</div> No significant keywords found.</div>", unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Business Impact Explanation
                st.markdown('<div class="explained-box">', unsafe_allow_html=True)
                st.markdown("""
                ### 📊 Business Impact Score Explained
                
                **What it measures:** The Business Impact Score quantifies how much each review influences your business success, ranging from **-5.0 to +5.0**.
                
                **How it's calculated:**
                - **Rating Component (-3 to +3):** 5-star = +3, 4-star = +1, 3-star = 0, 2-star = -2, 1-star = -3
                - **Sentiment Component (-2 to +2):** Extremely Positive = +2, Very Positive = +1, Positive = +0.5, Negative = -1, Very/Extremely Negative = -2
                - **Engagement Multiplier:** Detailed reviews (100+ words) get 1.5x impact, medium reviews (50+ words) get 1.2x impact
                - **Trust Factor:** High-risk reviews reduced to 30% impact, medium-risk to 70% impact
                
                **Interpretation:**
                - **+3.0 to +5.0:** Premium advocates driving significant business value
                - **+1.0 to +2.9:** Positive contributors to brand reputation
                - **-1.0 to +0.9:** Neutral impact on business metrics
                - **-2.9 to -1.1:** Negative impact requiring attention
                - **-5.0 to -3.0:** Crisis-level detractors needing immediate intervention
                
                **Strategic Use:** Prioritize high-impact positive reviews for marketing and address high-impact negative reviews immediately.
                """)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # TAB 2: Strategic Analytics
            with tab2:
                st.markdown("<h2 class='subheader'>📊 Strategic Business Analytics</h2>", unsafe_allow_html=True)
                st.markdown("*Advanced insights for strategic decision making*")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                    
                    # Customer segment performance analysis
                    st.markdown("### 👥 Customer Segment Performance Matrix")
                    
                    segment_analysis = filtered_df.groupby('customerSegment').agg({
                        'rating': 'mean',
                        'businessImpact': 'mean',
                        'reviewValue': 'mean',
                        'reviewId': 'count'
                    }).round(2).reset_index()
                    segment_analysis.columns = ['Customer Segment', 'Avg Rating', 'Business Impact', 'Review Value', 'Volume']
                    
                    # 3D-enhanced scatter plot
                    fig_segments = px.scatter(
                        segment_analysis, 
                        x='Business Impact', 
                        y='Avg Rating',
                        size='Volume',
                        color='Customer Segment',
                        title="Customer Segments: Impact vs Satisfaction",
                        hover_data=['Review Value'],
                        size_max=60
                    )
                    
                    # Enhanced 3D styling
                    fig_segments.update_layout(
                        height=500,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Inter, sans-serif"),
                        xaxis=dict(
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='rgba(128,128,128,0.2)',
                            zeroline=True,
                            zerolinecolor='rgba(128,128,128,0.5)',
                            zerolinewidth=2
                        ),
                        yaxis=dict(
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='rgba(128,128,128,0.2)',
                            zeroline=True,
                            zerolinecolor='rgba(128,128,128,0.5)',
                            zerolinewidth=2
                        )
                    )
                    
                    # Add drop shadow effect to markers
                    fig_segments.update_traces(
                        marker=dict(
                            line=dict(width=2, color='rgba(255,255,255,0.8)'),
                            opacity=0.8
                        )
                    )
                    
                    best_segment = segment_analysis.loc[segment_analysis['Business Impact'].idxmax(), 'Customer Segment']
                    largest_segment = segment_analysis.loc[segment_analysis['Volume'].idxmax(), 'Customer Segment']
                    
                    insight_text = f"**Strategic Priority:** {best_segment} delivers highest business value. **Scale Focus:** {largest_segment} represents your largest customer base."
                    
                    create_chart_with_insights(fig_segments, insight_text)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                    
                    # Performance trends over time
                    st.markdown("### 📈 Performance Trends Analysis")
                    
                    monthly_trends = filtered_df.groupby(['year', 'month']).agg({
                        'rating': 'mean',
                        'businessImpact': 'mean',
                        'fraudScore': 'mean',
                        'reviewId': 'count'
                    }).reset_index()
                    monthly_trends['date'] = pd.to_datetime(monthly_trends[['year', 'month']].assign(day=1))
                    monthly_trends = monthly_trends.sort_values('date')
                    
                    # 3D-enhanced multi-line chart
                    fig_trends = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=('Customer Satisfaction Trend', 'Business Impact Evolution'),
                        vertical_spacing=0.15,
                        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
                    )
                    
                    # Add satisfaction trend
                    fig_trends.add_trace(
                        go.Scatter(
                            x=monthly_trends['date'], 
                            y=monthly_trends['rating'],
                            mode='lines+markers', 
                            name='Satisfaction Rating',
                            line=dict(color='#667eea', width=4, shape='spline'),
                            marker=dict(size=8, line=dict(width=2, color='white')),
                            fill='tonexty',
                            fillcolor='rgba(102, 126, 234, 0.1)'
                        ),
                        row=1, col=1
                    )
                    
                    # Add impact trend
                    fig_trends.add_trace(
                        go.Scatter(
                            x=monthly_trends['date'], 
                            y=monthly_trends['businessImpact'],
                            mode='lines+markers', 
                            name='Business Impact',
                            line=dict(color='#f093fb', width=4, shape='spline'),
                            marker=dict(size=8, line=dict(width=2, color='white')),
                            fill='tonexty',
                            fillcolor='rgba(240, 147, 251, 0.1)'
                        ),
                        row=2, col=1
                    )
                    
                    fig_trends.update_layout(
                        height=550, 
                        title_text="Strategic Performance Metrics Over Time",
                        title_x=0.5,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Inter, sans-serif"),
                        showlegend=False,
                        xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
                        xaxis2=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
                        yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
                        yaxis2=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
                    )
                    
                    # Calculate trend direction
                    recent_rating = monthly_trends['rating'].tail(3).mean()
                    historical_rating = monthly_trends['rating'].head(3).mean()
                    trend_direction = "improving" if recent_rating > historical_rating else "declining"
                    trend_magnitude = abs(recent_rating - historical_rating)
                    
                    insight_text = f"**Trend Analysis:** Customer satisfaction is {trend_direction} with {trend_magnitude:.2f} point change. {'Maintain momentum with current strategies.' if trend_direction == 'improving' else 'Implement corrective measures immediately.'}"
                    
                    create_chart_with_insights(fig_trends, insight_text)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Topic performance analysis
                if topics:
                    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                    st.markdown("### 🏷️ Strategic Topic Performance Dashboard")
                    
                    topic_performance = filtered_df.groupby('topic').agg({
                        'rating': 'mean',
                        'businessImpact': 'mean',
                        'sentimentScore': 'mean',
                        'reviewId': 'count'
                    }).round(2).reset_index()
                    topic_performance.columns = ['Topic', 'Avg Rating', 'Business Impact', 'Sentiment Score', 'Volume']
                    
                    # Get top topics by business impact
                    top_topics_perf = topic_performance.nlargest(8, 'Business Impact')
                    
                    # 3D-enhanced horizontal bar chart
                    fig_topics = px.bar(
                        top_topics_perf, 
                        x='Business Impact', 
                        y='Topic',
                        color='Avg Rating',
                        title="Strategic Topic Impact Matrix",
                        orientation='h',
                        color_continuous_scale='RdYlGn',
                        hover_data=['Volume', 'Sentiment Score']
                    )
                    
                    # Enhanced 3D styling
                    fig_topics.update_layout(
                        height=500,
                        yaxis={'categoryorder':'total ascending'},
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Inter, sans-serif"),
                        xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
                       # yaxis=dict(showgrid=False),
                    )
                    
                    # Add shadow effect to bars
                    fig_topics.update_traces(
                        marker=dict(
                            line=dict(width=1, color='rgba(255,255,255,0.6)'),
                            opacity=0.9
                        )
                    )
                    
                    best_topic = top_topics_perf.iloc[0]['Topic']
                    worst_topic = topic_performance.nsmallest(1, 'Business Impact').iloc[0]['Topic']
                    
                    insight_text = f"**Market Opportunity:** '{best_topic}' drives highest customer value - leverage in marketing. **Risk Area:** '{worst_topic}' needs strategic intervention."
                    
                    create_chart_with_insights(fig_topics, insight_text)
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.markdown("<div class='chart-container' style='margin-top:1.5rem;'>", unsafe_allow_html=True)
                    for idx, row in top_topics_perf.iterrows():
                        topic = row['Topic']
                        avg_rating = row['Avg Rating']
                        business_impact = row['Business Impact']
                        sentiment_score = row['Sentiment Score']
                        volume = row['Volume']
                        # Placeholder for topic description (could be replaced with a lookup or ML summary)
                        topic_desc = f"<b>{topic}</b>: This topic covers customer feedback related to {topic.lower()}."
                        # Generate a summary and actionable recommendation
                        summary = f"<b>Summary:</b> {volume} reviews, Avg Rating: {avg_rating:.2f}, Impact: {business_impact:.2f}, Sentiment Score: {sentiment_score:.2f}."
                        if avg_rating >= 4.5:
                            recommendation = "Maintain excellence and leverage this topic in marketing."
                        elif avg_rating >= 4.0:
                            recommendation = "Solid performance, but monitor for emerging issues."
                        elif avg_rating >= 3.5:
                            recommendation = "Opportunity for improvement; address minor pain points."
                        else:
                            recommendation = "Critical area: prioritize improvements and customer outreach."
                        st.markdown(f"""
                        <div style='background:linear-gradient(135deg,#f8fafc 0%,#e0e7ef 100%);border-radius:15px;padding:1.2rem 1.5rem;margin-bottom:1rem;box-shadow:0 2px 8px rgba(102,126,234,0.08);'>
                            <div style='font-size:1.1rem;margin-bottom:0.3rem;'>{topic_desc}</div>
                            <div style='margin-bottom:0.2rem;'>{summary}</div>
                            <div style='color:#764ba2;font-weight:600;'>Action: {recommendation}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
            
            # TAB 3: Deep Intelligence
            with tab3:
                st.markdown("<h2 class='subheader'>🔍 Deep Customer Intelligence</h2>", unsafe_allow_html=True)
                st.markdown("*Advanced analytics for competitive advantage*")
                
                # Advanced topic analysis with business recommendations
                if topics:
                    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                    st.subheader("🎯 Strategic Topic Intelligence Matrix")
                    
                    topic_insights = []
                    for idx, topic_name in enumerate(sorted(topics), 1):
                        topic_reviews = filtered_df[filtered_df['topic'] == topic_name]
                        if len(topic_reviews) > 0:
                            avg_rating = topic_reviews['rating'].mean()
                            avg_sentiment = topic_reviews['sentimentScore'].mean()
                            review_count = len(topic_reviews)
                            avg_impact = topic_reviews['businessImpact'].mean()
                            trust_score = (topic_reviews['fraudFlag'] == 'Legitimate').sum() / len(topic_reviews) * 100
                            
                            # Strategic classification
                            if avg_rating >= 4.5 and avg_sentiment > 0.3:
                                classification = "🟢 Strategic Strength"
                                recommendation = "Leverage in premium marketing campaigns and competitive positioning"
                            elif avg_rating >= 4.0 and avg_sentiment > 0.1:
                                classification = "🟡 Growth Opportunity"
                                recommendation = "Optimize to convert into strategic strength"
                            elif avg_rating >= 3.0:
                                classification = "🟠 Performance Gap"
                                recommendation = "Implement immediate process improvements"
                            else:
                                classification = "🔴 Critical Risk"
                                recommendation = "Crisis management and emergency intervention required"
                            
                            topic_insights.append({
                                'Rank': idx,
                                'Strategic Topic': topic_name,
                                'Customer Volume': f"{review_count:,}",
                                'Satisfaction': f"{avg_rating:.2f}/5.0",
                                'Business Impact': f"{avg_impact:.2f}",
                                'Trust Level': f"{trust_score:.1f}%",
                                'Classification': classification,
                                'Strategic Action': recommendation
                            })
                    
                    if topic_insights:
                        insights_df = pd.DataFrame(topic_insights)
                        
                        # Sort by business impact
                        insights_df['Impact_Numeric'] = insights_df['Business Impact'].str.replace(r'[^\d.-]', '', regex=True).astype(float)
                        insights_df = insights_df.sort_values('Impact_Numeric', ascending=False).drop('Impact_Numeric', axis=1)
                        
                        st.dataframe(
                            insights_df, 
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Strategic Topic": st.column_config.TextColumn("Strategic Topic", width="medium"),
                                "Classification": st.column_config.TextColumn("Classification", width="medium"),
                                "Strategic Action": st.column_config.TextColumn("Strategic Action", width="large")
                            }
                        )
                        
                        # Strategic recommendations
                        st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
                        st.markdown("### 🎯 **C-Level Strategic Recommendations**")
                        
                        top_opportunity = insights_df.iloc[0]
                        biggest_risks = insights_df[insights_df['Classification'].str.contains('🔴')].head(2)
                        growth_opportunities = insights_df[insights_df['Classification'].str.contains('🟡')].head(2)
                        
                        st.markdown(f"""
                        **🚀 Primary Market Opportunity:** 
                        - **Focus Area:** {top_opportunity['Strategic Topic']} 
                        - **Market Size:** {top_opportunity['Customer Volume']} customer touchpoints
                        - **Revenue Impact:** {top_opportunity['Business Impact']} business value score
                        - **Action:** {top_opportunity['Strategic Action']}
                        
                        **⚠️ Critical Risk Management:**
                        {chr(10).join([f"- **{row['Strategic Topic']}**: {row['Strategic Action']}" for _, row in biggest_risks.iterrows()])}
                        
                        **📈 Growth Acceleration Targets:**
                        {chr(10).join([f"- **{row['Strategic Topic']}**: Convert to market leadership position" for _, row in growth_opportunities.iterrows()])}
                        
                        **🎯 Executive Action Items:**
                        1. **Immediate (0-30 days):** Address critical risk topics with crisis management protocols
                        2. **Short-term (1-3 months):** Launch optimization initiatives for growth opportunities  
                        3. **Long-term (3-12 months):** Build competitive moats around strategic strengths
                        """)
                        st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Customer journey intelligence
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                    
                    # Review engagement vs satisfaction analysis
                    st.markdown("### 📝 Customer Engagement Intelligence")
                    
                    filtered_df['engagementTier'] = pd.cut(
                        filtered_df['wordCount'],
                        bins=[0, 10, 30, 75, 150, float('inf')],
                        labels=['Minimal (≤10)', 'Brief (11-30)', 'Standard (31-75)', 'Detailed (76-150)', 'Comprehensive (150+)']
                    )
                    
                    engagement_analysis = filtered_df.groupby('engagementTier').agg({
                        'rating': 'mean',
                        'sentimentScore': 'mean',
                        'businessImpact': 'mean',
                        'reviewId': 'count',
                        'fraudFlag': lambda x: (x == 'Legitimate').sum() / len(x) * 100
                    }).round(2).reset_index()
                    engagement_analysis.columns = ['Engagement Level', 'Avg Rating', 'Sentiment Score', 'Business Impact', 'Volume', 'Trust %']
                    
                    # 3D-enhanced bar chart
                    fig_engagement = px.bar(
                        engagement_analysis, 
                        x='Engagement Level', 
                        y='Avg Rating',
                        color='Business Impact',
                        title="Customer Engagement vs Satisfaction Matrix",
                        color_continuous_scale='Viridis',
                        hover_data=['Volume', 'Trust %']
                    )
                    
                    # Enhanced styling
                    fig_engagement.update_layout(
                        height=450,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Inter, sans-serif"),
                        xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
                        yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
                    )
                    
                    fig_engagement.update_traces(
                        marker=dict(
                            line=dict(width=2, color='rgba(255,255,255,0.8)'),
                            opacity=0.9
                        )
                    )
                    
                    optimal_engagement = engagement_analysis.loc[engagement_analysis['Avg Rating'].idxmax(), 'Engagement Level']
                    optimal_impact = engagement_analysis.loc[engagement_analysis['Business Impact'].idxmax(), 'Business Impact']
                    
                    insight_text = f"**Engagement Sweet Spot:** {optimal_engagement} reviews generate highest satisfaction. **Business Optimization:** Focus on encouraging detailed feedback (impact score: {optimal_impact})."
                    
                    create_chart_with_insights(fig_engagement, insight_text)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                    
                    # Customer sentiment journey heatmap
                    st.markdown("### 🎭 Customer Sentiment Journey Map")
                    
                    # Create sentiment order for consistent display
                    sentiment_order = ['Extremely Negative', 'Very Negative', 'Negative', 'Neutral', 
                                      'Positive', 'Very Positive', 'Extremely Positive']
                    
                    segment_journey = filtered_df.groupby(['customerSegment', 'sentiment']).size().unstack(fill_value=0)
                    segment_journey = segment_journey.reindex(columns=sentiment_order, fill_value=0)
                    segment_journey_pct = segment_journey.div(segment_journey.sum(axis=1), axis=0) * 100
                    
                    # 3D-enhanced heatmap
                    fig_journey = px.imshow(
                        segment_journey_pct.values,
                        x=segment_journey_pct.columns,
                        y=segment_journey_pct.index,
                        title="Customer Segment Sentiment Distribution",
                        color_continuous_scale='RdYlGn',
                        text_auto='.1f',
                        aspect="auto"
                    )
                    
                    # Enhanced styling
                    fig_journey.update_layout(
                        height=450,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Inter, sans-serif"),
                        xaxis_title="Customer Sentiment",
                        yaxis_title="Customer Segment",
                        coloraxis_colorbar=dict(
                            title="Percentage",
                            thickness=15,
                            len=0.7
                        )
                    )
                    
                    # Calculate segment scores
                    segment_scores = (segment_journey_pct[['Positive', 'Very Positive', 'Extremely Positive']].sum(axis=1) - 
                                    segment_journey_pct[['Negative', 'Very Negative', 'Extremely Negative']].sum(axis=1))
                    best_segment = segment_scores.idxmax()
                    worst_segment = segment_scores.idxmin()
                    
                    insight_text = f"**Champion Segment:** {best_segment} shows {segment_scores.max():.1f}% net positive sentiment. **Priority Segment:** {worst_segment} needs immediate attention ({segment_scores.min():.1f}% net sentiment)."
                    
                    create_chart_with_insights(fig_journey, insight_text)
                    st.markdown("</div>", unsafe_allow_html=True)
            
            # TAB 4: Customer Voices
            with tab4:
                st.markdown("<h2 class='subheader'>💬 Customer Voice Intelligence</h2>", unsafe_allow_html=True)
                st.markdown("*Authentic customer feedback driving strategic decisions*")
                
                # Extract enhanced verbatims
                if 'sample_pos' not in st.session_state or 'sample_neg' not in st.session_state:
                    pos_reviews = filtered_df[(filtered_df['rating'] == 5) & (filtered_df['sentiment'].str.contains('Positive'))]
                    neg_reviews = filtered_df[(filtered_df['rating'] == 1) & (filtered_df['sentiment'].str.contains('Negative'))]
                    st.session_state['sample_pos'] = pos_reviews.sample(n=min(5, len(pos_reviews)), random_state=42)
                    st.session_state['sample_neg'] = neg_reviews.sample(n=min(5, len(neg_reviews)), random_state=42)
                
                positive_verbatims = st.session_state['sample_pos']
                negative_verbatims = st.session_state['sample_neg']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🌟 **Brand Champions & Success Stories**")
                    st.markdown("*High-impact positive feedback driving business growth*")
                    
                    for idx, (_, review) in enumerate(positive_verbatims.iterrows(), 1):
                        st.markdown(f'<div class="verbatim-section positive-verbatim">', unsafe_allow_html=True)
                        st.markdown(f"**Champion Voice #{idx}** | ⭐{review['rating']}/5 | Business Impact: **{review['businessImpact']:.1f}/5.0**")
                        
                        # Enhanced review preview with smart truncation
                        review_text = review['reviewText']
                        if len(review_text) > 200:
                            sentences = review_text.split('.')
                            preview = '. '.join(sentences[:2]) + '...' if len(sentences) > 2 else review_text
                        else:
                            preview = review_text
                        
                        st.markdown(f"*\"{preview}\"*")
                        
                        # Enhanced metadata
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown(f"**Focus Area:** {review['topic']}")
                            st.markdown(f"**Customer Type:** {review['customerSegment']}")
                        with col_b:
                            st.markdown(f"**Emotion:** {review['sentiment']}")
                            st.markdown(f"**Confidence:** {review['sentimentConfidence']:.0%}")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown("### ⚠️ **Critical Customer Concerns**")
                    st.markdown("*High-priority issues requiring immediate executive attention*")
                    
                    for idx, (_, review) in enumerate(negative_verbatims.iterrows(), 1):
                        st.markdown(f'<div class="verbatim-section negative-verbatim">', unsafe_allow_html=True)
                        st.markdown(f"**Critical Issue #{idx}** | ⭐{review['rating']}/5 | Business Impact: **{review['businessImpact']:.1f}/5.0**")
                        
                        # Enhanced review preview
                        review_text = review['reviewText']
                        if len(review_text) > 200:
                            sentences = review_text.split('.')
                            preview = '. '.join(sentences[:2]) + '...' if len(sentences) > 2 else review_text
                        else:
                            preview = review_text
                        
                        st.markdown(f"*\"{preview}\"*")
                        
                        # Enhanced metadata with urgency indicators
                        col_a, col_b = st.columns(2)
                        with col_a:
                            urgency = "🚨 Critical" if review['businessImpact'] < -3 else "⚠️ High" if review['businessImpact'] < -1 else "📊 Monitor"
                            st.markdown(f"**Urgency Level:** {urgency}")
                            st.markdown(f"**Issue Category:** {review['topic']}")
                        with col_b:
                            st.markdown(f"**Customer Type:** {review['customerSegment']}")
                            st.markdown(f"**Confidence:** {review['sentimentConfidence']:.0%}")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                
                # Enhanced impact analysis
                st.markdown('<div class="explained-box">', unsafe_allow_html=True)
                st.markdown("""
                ### 📊 Business Impact Score - Executive Guide
                
                **Strategic Framework:** The Business Impact Score is your north star metric for prioritizing customer feedback actions.
                
                **Scoring Methodology:**
                - **Foundation Layer:** Star ratings provide base impact (-3 to +3)
                - **Sentiment Amplification:** Customer emotion multiplies influence (-2 to +2)  
                - **Engagement Weighting:** Detailed reviews carry more market weight (up to 1.5x multiplier)
                - **Trust Validation:** Suspicious reviews are discounted to maintain data integrity
                
                **Executive Decision Matrix:**
                
                | **Score Range** | **Business Meaning** | **Strategic Action** | **Priority Level** |
                |-----------------|----------------------|---------------------|-------------------|
                | **+3.0 to +5.0** | Premium brand advocates | Leverage in marketing campaigns | 🟢 Strategic Asset |
                | **+1.0 to +2.9** | Satisfied customers | Nurture loyalty programs | 🟡 Growth Opportunity |
                | **-1.0 to +0.9** | Neutral market position | Monitor and optimize | 📊 Baseline Management |
                | **-2.9 to -1.1** | Customer satisfaction risk | Immediate process review | 🟠 Action Required |
                | **-5.0 to -3.0** | Brand reputation threat | Crisis management protocol | 🔴 Executive Escalation |
                
                **ROI Optimization:** Focus resources on high-impact positive reviews for marketing leverage and high-impact negative reviews for risk mitigation.
                """)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Strategic topic summary
                st.markdown("### 🏆 **Strategic Customer Priorities Dashboard**")
                if topics:
                    col1, col2, col3 = st.columns(3)
                    
                    topic_summary = filtered_df['topic'].value_counts().head(9)
                    topic_sentiment = filtered_df.groupby('topic')['businessImpact'].mean().round(2)
                    
                    for idx, (topic, count) in enumerate(topic_summary.items()):
                        col_idx = idx % 3
                        impact = topic_sentiment.get(topic, 0)
                        
                        # Determine priority level
                        if impact > 2:
                            priority = "🟢 Strategic Strength"
                            action = "Leverage in marketing"
                        elif impact > 0:
                            priority = "🟡 Growth Area"
                            action = "Optimize for excellence"
                        elif impact > -1:
                            priority = "📊 Monitor"
                            action = "Track performance"
                        else:
                            priority = "🔴 Critical Issue"
                            action = "Immediate intervention"
                        
                        if col_idx == 0:
                            with col1:
                                st.markdown(f"""
                                **{idx+1}. {topic}**
                                - **Volume:** {count:,} mentions ({count/len(filtered_df)*100:.1f}%)
                                - **Impact:** {impact:.1f}/5.0
                                - **Status:** {priority}
                                - **Action:** {action}
                                """)
                        elif col_idx == 1:
                            with col2:
                                st.markdown(f"""
                                **{idx+1}. {topic}**
                                - **Volume:** {count:,} mentions ({count/len(filtered_df)*100:.1f}%)
                                - **Impact:** {impact:.1f}/5.0
                                - **Status:** {priority}
                                - **Action:** {action}
                                """)
                        else:
                            with col3:
                                st.markdown(f"""
                                **{idx+1}. {topic}**
                                - **Volume:** {count:,} mentions ({count/len(filtered_df)*100:.1f}%)
                                - **Impact:** {impact:.1f}/5.0
                                - **Status:** {priority}
                                - **Action:** {action}
                                """)
            
            # TAB 5: Risk Management
            with tab5:
                st.markdown("<h2 class='subheader'>🚨 Enterprise Risk Management</h2>", unsafe_allow_html=True)
                st.markdown("*Advanced threat detection and business continuity analytics*")
                
                # Enhanced risk metrics with executive KPIs
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    high_risk_count = (filtered_df['fraudFlag'] == 'High Risk').sum()
                    risk_trend = "📈" if high_risk_count > len(filtered_df) * 0.1 else "📊" if high_risk_count > len(filtered_df) * 0.05 else "📉"
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h3>🚨 Critical Threats</h3>
                        <h2>{high_risk_count}</h2>
                        <p>{high_risk_count/len(filtered_df)*100:.1f}% {risk_trend}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    reputation_risk = len(filtered_df[filtered_df['rating'] <= 2]) / len(filtered_df) * 100
                    rep_status = "🟢 Stable" if reputation_risk < 10 else "🟡 Monitor" if reputation_risk < 20 else "🔴 Alert"
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h3>📉 Reputation Risk</h3>
                        <h2>{reputation_risk:.1f}%</h2>
                        <p>{rep_status}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    data_integrity = len(filtered_df[filtered_df['fraudFlag'] == 'Legitimate']) / len(filtered_df) * 100
                    integrity_grade = "A+" if data_integrity > 90 else "A" if data_integrity > 80 else "B" if data_integrity > 70 else "C"
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h3>🛡️ Data Integrity</h3>
                        <h2>{data_integrity:.1f}%</h2>
                        <p>Grade: {integrity_grade}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    avg_fraud_score = filtered_df['fraudScore'].mean()
                    risk_level = "Low" if avg_fraud_score < 2 else "Medium" if avg_fraud_score < 4 else "High"
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h3>⚖️ Risk Index</h3>
                        <h2>{avg_fraud_score:.1f}/10</h2>
                        <p>{risk_level} Risk</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Advanced risk analytics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                    
                    # Enhanced fraud detection dashboard
                    fraud_dist = filtered_df['fraudFlag'].value_counts()
                    
                    # 3D-enhanced donut chart
                    fig_fraud = px.pie(
                        values=fraud_dist.values,
                        names=fraud_dist.index,
                        title="🔍 Review Authenticity Intelligence",
                        color_discrete_map={
                            'Legitimate': '#28a745',
                            'Low Risk': '#ffc107', 
                            'Medium Risk': '#fd7e14',
                            'High Risk': '#dc3545'
                        },
                        hole=0.4
                    )
                    
                    # Enhanced styling
                    fig_fraud.update_layout(
                        height=450,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Inter, sans-serif"),
                        showlegend=True,
                        legend=dict(
                            orientation="v",
                            yanchor="middle",
                            y=0.5,
                            xanchor="left",
                            x=1.05
                        )
                    )
                    
                    # Add 3D effect with pull and shadow
                    fig_fraud.update_traces(
                        textposition='inside',
                        textinfo='percent+label',
                        marker=dict(
                            line=dict(color='#FFFFFF', width=3)
                        ),
                        pull=[0.1 if 'High Risk' in name else 0.05 if 'Medium Risk' in name else 0 for name in fraud_dist.index]
                    )
                    
                    legitimate_rate = fraud_dist.get('Legitimate', 0) / len(filtered_df) * 100
                    total_risk = 100 - legitimate_rate
                    
                    insight_text = f"**Data Integrity Status:** {legitimate_rate:.1f}% verified authentic reviews. **Risk Exposure:** {total_risk:.1f}% requires monitoring. {'Implement enhanced verification protocols.' if total_risk > 15 else 'Maintain current security standards.'}"
                    
                    create_chart_with_insights(fig_fraud, insight_text)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                    
                    # Risk trend analysis
                    st.markdown("### 📈 Risk Evolution Timeline")
                    
                    risk_trends = filtered_df.groupby(['year', 'month']).agg({
                        'fraudScore': 'mean',
                        'fraudFlag': lambda x: (x != 'Legitimate').sum(),
                        'reviewId': 'count',
                        'businessImpact': 'mean'
                    }).reset_index()
                    
                    risk_trends['risk_rate'] = risk_trends['fraudFlag'] / risk_trends['reviewId'] * 100
                    risk_trends['date'] = pd.to_datetime(risk_trends[['year', 'month']].assign(day=1))
                    risk_trends = risk_trends.sort_values('date')
                    
                    # 3D-enhanced dual axis chart
                    fig_risk = make_subplots(
                        specs=[[{"secondary_y": True}]],
                        subplot_titles=["Risk Rate vs Business Impact Correlation"]
                    )
                    
                    # Risk rate line
                    fig_risk.add_trace(
                        go.Scatter(
                            x=risk_trends['date'], 
                            y=risk_trends['risk_rate'],
                            mode='lines+markers',
                            name='Suspicious Review Rate (%)',
                            line=dict(color='#dc3545', width=4, shape='spline'),
                            marker=dict(size=8, line=dict(width=2, color='white')),
                            fill='tonexty',
                            fillcolor='rgba(220, 53, 69, 0.1)'
                        ),
                        secondary_y=False
                    )
                    
                    # Business impact line
                    fig_risk.add_trace(
                        go.Scatter(
                            x=risk_trends['date'], 
                            y=risk_trends['businessImpact'],
                            mode='lines+markers',
                            name='Business Impact Score',
                            line=dict(color='#28a745', width=4, shape='spline'),
                            marker=dict(size=8, line=dict(width=2, color='white')),
                            fill='tonexty',
                            fillcolor='rgba(40, 167, 69, 0.1)'
                        ),
                        secondary_y=True
                    )
                    
                    # Enhanced styling
                    fig_risk.update_layout(
                        height=450,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Inter, sans-serif"),
                        title_x=0.5,
                        showlegend=True,
                        legend=dict(x=0, y=1.1, orientation="h"),
                        xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
                    )
                    
                    fig_risk.update_yaxes(
                        title_text="Suspicious Review Rate (%)",
                        secondary_y=False,
                        showgrid=True,
                        gridcolor='rgba(128,128,128,0.2)'
                    )
                    fig_risk.update_yaxes(
                        title_text="Business Impact Score",
                        secondary_y=True,
                        showgrid=False
                    )
                    
                    # Calculate risk trajectory
                    recent_risk = risk_trends['risk_rate'].tail(3).mean()
                    historical_risk = risk_trends['risk_rate'].head(3).mean()
                    risk_direction = "increasing" if recent_risk > historical_risk else "decreasing"
                    risk_change = abs(recent_risk - historical_risk)
                    
                    insight_text = f"**Risk Trajectory:** Suspicious activity is {risk_direction} by {risk_change:.1f}%. **Correlation Impact:** {'Negative correlation detected - higher risk = lower business value' if risk_trends['risk_rate'].corr(risk_trends['businessImpact']) < -0.3 else 'Monitor for emerging patterns'}."
                    
                    create_chart_with_insights(fig_risk, insight_text)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Detailed fraud detection results
                st.markdown("### 🕵️ **Advanced Threat Detection Results**")
                suspicious_df = filtered_df[filtered_df['fraudFlag'].isin(['High Risk', 'Medium Risk'])]
                
                if not suspicious_df.empty:
                    st.markdown(f"**🚨 Security Alert:** {len(suspicious_df)} suspicious reviews detected requiring investigation")
                    
                    # Enhanced suspicious review analysis
                    fraud_analysis = suspicious_df.groupby(['fraudFlag', 'fraudReason']).size().reset_index(name='Count')
                    fraud_analysis = fraud_analysis.sort_values('Count', ascending=False)
                    
                    st.markdown("#### **Threat Pattern Analysis**")
                    for _, row in fraud_analysis.head(10).iterrows():
                        threat_level = "🔴 Critical" if row['fraudFlag'] == 'High Risk' else "🟡 Moderate"
                        st.markdown(f"- **{threat_level}**: {row['fraudReason']} ({row['Count']} instances)")
                    
                    # Sample suspicious reviews table
                    st.markdown("#### **High-Risk Review Samples**")
                    display_cols = ['reviewId', 'reviewText', 'rating', 'fraudFlag', 'fraudReason', 'fraudScore']
                    sample_suspicious = suspicious_df.nlargest(5, 'fraudScore')[display_cols]
                    
                    st.dataframe(
                        sample_suspicious, 
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "reviewText": st.column_config.TextColumn("Review Text", width="large"),
                            "fraudReason": st.column_config.TextColumn("Risk Factors", width="medium")
                        }
                    )
                    
                else:
                    st.success("✅ **All Clear:** No suspicious reviews detected in current filter selection")
                
                # Executive risk management recommendations
                st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
                st.markdown("### 🛡️ **Executive Risk Management Strategy**")
                
                total_risk_rate = (filtered_df['fraudFlag'] != 'Legitimate').sum() / len(filtered_df) * 100
                high_impact_negative = len(filtered_df[(filtered_df['businessImpact'] < -2) & (filtered_df['rating'] <= 2)])
                
                # Dynamic risk assessment
                if total_risk_rate > 20:
                    risk_status = "🔴 **CRISIS LEVEL**"
                    action_urgency = "Immediate C-suite intervention required"
                elif total_risk_rate > 10:
                    risk_status = "🟠 **ELEVATED RISK**"
                    action_urgency = "Deploy enhanced monitoring protocols"
                elif total_risk_rate > 5:
                    risk_status = "🟡 **MODERATE RISK**"
                    action_urgency = "Implement preventive measures"
                else:
                    risk_status = "🟢 **LOW RISK**"
                    action_urgency = "Maintain current security posture"
                
                st.markdown(f"""
                **Current Threat Level:** {risk_status}
                **Immediate Action Required:** {action_urgency}
                **High-Impact Negative Reviews:** {high_impact_negative} requiring crisis management
                
                **Strategic Risk Mitigation Framework:**
                
                **🚨 Immediate Actions (0-7 days):**
                - Investigate all high-risk flagged reviews for authenticity
                - Implement emergency response for reviews with business impact < -3.0
                - Deploy advanced AI detection for similar pattern recognition
                
                **🛡️ Short-term Fortification (1-4 weeks):**
                - Establish review verification protocols with multiple validation layers
                - Create customer feedback authentication system
                - Launch proactive customer satisfaction intervention programs
                
                **📊 Long-term Strategic Defense (1-6 months):**
                - Build predictive analytics for early threat detection
                - Implement blockchain-based review authenticity verification
                - Develop customer advocacy programs to increase authentic positive reviews
                - Create competitor analysis to identify market manipulation attempts
                
                **💼 Business Continuity Measures:**
                - Monitor brand reputation metrics across all channels
                - Establish crisis communication protocols for reputation management
                - Create legal framework for fraudulent review prosecution
                - Develop customer trust restoration programs
                """)
                st.markdown('</div>', unsafe_allow_html=True)
        
        else:
            # Enhanced onboarding experience
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("""
            ### 🚀 **Welcome to Your C-Level Customer Intelligence Platform**
            
            **Transform Raw Reviews into Strategic Business Intelligence**
            
            This enterprise-grade dashboard converts customer feedback into actionable insights for executive decision making.
            
            #### **🎯 What You'll Discover:**
            - **📊 Executive KPIs:** Customer satisfaction, business impact, and market position metrics
            - **🧠 AI-Powered Analytics:** Advanced sentiment analysis and topic modeling
            - **🔍 Fraud Detection:** Sophisticated algorithms to identify suspicious reviews
            - **💬 Customer Voices:** Strategic verbatims driving business decisions
            - **🚨 Risk Management:** Enterprise-level threat detection and mitigation
            - **📈 Competitive Intelligence:** Market positioning and performance benchmarks
            
            #### **📋 Required Data Format:**
            Upload a CSV file containing Amazon reviews with these columns:
            - Review ID, Reviewer Name, Review Text, Rating (1-5)
            - Summary, Helpful Votes, Total Votes, Review Date, Year
            
            #### **🎨 Advanced Features:**
            - **3D-Enhanced Visualizations** for premium presentation quality
            - **Interactive Filtering** with smart controls in the left sidebar
            - **Word Cloud Analysis** showing customer priority keywords
            - **Strategic Recommendations** with C-level action items
            - **Export Capabilities** for board presentations and reports
            
            **Ready to unlock your customer intelligence? Upload your dataset to begin the analysis.**
            """)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Enhanced sidebar with quick analytics
    if st.session_state.processed_data is not None:
        with st.sidebar:
            st.markdown("---")
            st.subheader("📊 **Executive Summary**")
            
            df = st.session_state.processed_data
            filtered_for_sidebar = df if not st.session_state.apply_filters else filtered_df
            
            # Key metrics with trend indicators
            avg_rating = filtered_for_sidebar['rating'].mean()
            rating_grade = "🟢" if avg_rating >= 4.0 else "🟡" if avg_rating >= 3.0 else "🔴"
            st.metric("⭐ Customer Satisfaction", f"{avg_rating:.2f}/5.0", 
                     delta=f"{rating_grade} {get_satisfaction_grade(avg_rating)}")
            
            positive_pct = (filtered_for_sidebar['sentiment'].str.contains('Positive', na=False)).sum()/len(filtered_for_sidebar)*100
            sentiment_trend = "🟢" if positive_pct >= 70 else "🟡" if positive_pct >= 50 else "🔴"
            st.metric("😊 Brand Sentiment", f"{positive_pct:.1f}%", 
                     delta=f"{sentiment_trend} Customer Love")
            
            trust_pct = (filtered_for_sidebar['fraudFlag'] == 'Legitimate').sum()/len(filtered_for_sidebar)*100
            trust_grade = "🟢" if trust_pct >= 85 else "🟡" if trust_pct >= 70 else "🔴"
            st.metric("🔍 Data Integrity", f"{trust_pct:.1f}%", 
                     delta=f"{trust_grade} Trust Level")
            
            business_impact = filtered_for_sidebar['businessImpact'].mean()
            impact_trend = "🟢" if business_impact >= 1.0 else "🟡" if business_impact >= 0 else "🔴"
            st.metric("💼 Business Impact", f"{business_impact:.2f}/5.0", 
                     delta=f"{impact_trend} Revenue Effect")
            
            # Data export options
            st.markdown("---")
            st.subheader("📥 **Export Options**")
            
            # Prepare enhanced export data
            export_df = filtered_for_sidebar[[
                'reviewId', 'reviewerName', 'reviewText', 'rating', 'reviewDate',
                'sentiment', 'sentimentScore', 'sentimentConfidence', 'emotion',
                'fraudFlag', 'fraudReason', 'fraudScore',
                'topic', 'customerSegment', 'businessImpact', 'reviewValue',
                'wordCount', 'reviewLength'
            ]].copy()
            
            csv_data = export_df.to_csv(index=False)
            st.download_button(
                label="📄 Complete Dataset",
                data=csv_data,
                file_name=f"customer_intelligence_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                help="Download complete analyzed dataset with all AI insights"
            )
            
            # Executive summary export
            executive_summary = create_enhanced_executive_summary_with_topics(filtered_for_sidebar)
            st.download_button(
                label="📋 Executive Report",
                data=executive_summary,
                file_name=f"executive_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown",
                help="Download C-level summary report for board presentations"
            )
    
    # Enhanced footer
    st.markdown("---")
    st.markdown(
        '<div class="footer">Customer Reviews Intelligence Platform | Created by insights3d - email to aneesh@insights3d.com</div>', 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
