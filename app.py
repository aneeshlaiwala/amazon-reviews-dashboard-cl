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

# Enhanced CSS with fixed styling issues
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
    
    .subheader {
        font-size: 2rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 1.5rem;
        border-bottom: 3px solid transparent;
        background: linear-gradient(90deg, #667eea, #764ba2);
        background-clip: border-box;
        border-image: linear-gradient(90deg, #667eea, #764ba2) 1;
        padding-bottom: 0.8rem;
    }
    
    /* FIXED EXECUTIVE SUMMARY */
    .executive-summary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 20px;
        padding: 2.5rem;
        margin: 2rem 0;
        color: white;
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
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
        background: radial-gradient(circle at 20% 80%, rgba(255, 255, 255, 0.1) 0%, transparent 50%);
        pointer-events: none;
    }
    
    .executive-content {
        position: relative;
        z-index: 1;
        color: white;
    }
    
    .executive-content h2, .executive-content h3 {
        color: white !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .executive-content strong {
        color: #f8f9fa;
        font-weight: 700;
    }
    
    /* FIXED METRIC CARDS - SLEEK AND SMALLER */
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 249, 250, 0.95) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(0, 0, 0, 0.1);
        border-radius: 15px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.15);
    }
    
    .metric-card h3 {
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        margin-bottom: 0.5rem !important;
        color: #6c757d !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-card h2 {
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        margin: 0.3rem 0 !important;
        color: #2c3e50 !important;
        line-height: 1.2;
    }
    
    .metric-card p {
        font-size: 0.75rem !important;
        margin: 0 !important;
        color: #495057 !important;
        font-weight: 500;
    }
    
    /* FIXED FILTER SIDEBAR */
    .filter-sidebar {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.98) 0%, rgba(248, 249, 250, 0.98) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(0, 0, 0, 0.1);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        position: sticky;
        top: 2rem;
        height: fit-content;
    }
    
    .filter-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 1rem;
        text-align: center;
        border-bottom: 2px solid #e9ecef;
        padding-bottom: 0.8rem;
    }
    
    .filter-section {
        margin-bottom: 1rem;
        padding: 0.8rem;
        background: rgba(255, 255, 255, 0.7);
        border-radius: 12px;
        border: 1px solid rgba(0, 0, 0, 0.05);
    }
    
    .filter-label {
        font-weight: 600;
        color: #495057;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.4rem;
        display: block;
    }
    
    /* CHART CONTAINERS */
    .chart-container {
        background: rgba(255, 255, 255, 0.98);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(0, 0, 0, 0.08);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
    }
    
    .chart-container:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.12);
    }
    
    /* INSIGHT BOXES */
    .insight-box {
        background: linear-gradient(135deg, rgba(40, 167, 69, 0.1) 0%, rgba(40, 167, 69, 0.05) 100%);
        border-left: 4px solid #28a745;
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        box-shadow: 0 5px 15px rgba(40, 167, 69, 0.1);
    }
    
    .insight-title {
        font-weight: 700;
        color: #28a745;
        font-size: 1rem;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* RECOMMENDATION BOXES - FIXED STYLING */
    .recommendation-box {
        background: linear-gradient(135deg, rgba(132, 250, 176, 0.15) 0%, rgba(143, 211, 244, 0.15) 100%);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(132, 250, 176, 0.3);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        color: #2c3e50;
        box-shadow: 0 10px 25px rgba(132, 250, 176, 0.1);
    }
    
    .recommendation-box h3 {
        color: #28a745 !important;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    /* VERBATIM SECTIONS */
    .verbatim-section {
        background: rgba(255, 255, 255, 0.98);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(0, 0, 0, 0.08);
        border-radius: 12px;
        padding: 1.2rem;
        margin: 1rem 0;
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    
    .verbatim-section:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
    }
    
    .positive-verbatim {
        border-left: 4px solid #28a745;
        background: linear-gradient(135deg, rgba(40, 167, 69, 0.05) 0%, rgba(40, 167, 69, 0.02) 100%);
    }
    
    .negative-verbatim {
        border-left: 4px solid #dc3545;
        background: linear-gradient(135deg, rgba(220, 53, 69, 0.05) 0%, rgba(220, 53, 69, 0.02) 100%);
    }
    
    /* ENHANCED TABS */
    .stTabs [data-baseweb="tab-list"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 249, 250, 0.95) 100%);
        backdrop-filter: blur(20px);
        border-radius: 12px;
        padding: 0.4rem;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #6c757d;
        font-weight: 600;
        padding: 0.8rem 1.2rem;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 3px 10px rgba(102, 126, 234, 0.3);
    }
    
    /* ENHANCED BUTTONS */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 8px;
        color: white;
        font-weight: 600;
        padding: 0.6rem 1.5rem;
        box-shadow: 0 3px 10px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* FORM ELEMENTS */
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 0, 0, 0.15);
        border-radius: 8px;
    }
    
    /* SIDEBAR STYLING */
    .css-1d391kg {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 249, 250, 0.95) 100%);
        backdrop-filter: blur(20px);
    }
    
    /* WORDCLOUD CONTAINER */
    .wordcloud-container {
        background: rgba(255, 255, 255, 0.98);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(0, 0, 0, 0.08);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
        text-align: center;
    }
    
    /* EXPLAINED BOX */
    .explained-box {
        background: linear-gradient(135deg, rgba(249, 249, 249, 0.98) 0%, rgba(232, 245, 232, 0.98) 100%);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(76, 175, 80, 0.2);
        border-left: 5px solid #4CAF50;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 10px 25px rgba(76, 175, 80, 0.1);
    }
    
    .explained-title {
        font-weight: 700;
        color: #4CAF50;
        font-size: 1.5rem;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    /* ACTIVE FILTERS */
    .active-filters {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
        text-align: center;
    }
    
    /* DOWNLOAD BUTTONS */
    .download-section {
        background: rgba(255, 255, 255, 0.98);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(0, 0, 0, 0.08);
        border-radius: 15px;
        padding: 1.2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
    }
    
    /* MOBILE RESPONSIVENESS */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5rem;
            letter-spacing: -1px;
        }
        
        .metric-card {
            padding: 1rem;
            height: 100px;
        }
        
        .metric-card h2 {
            font-size: 1.5rem !important;
        }
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
            if topic_label not in topics:
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
## 🎯 Strategic Customer Intelligence Report
*Comprehensive analysis of {total_reviews:,} customer interactions*

### 📈 Market Position & Performance
- **Customer Satisfaction Index:** {avg_rating:.2f}/5.0 ⭐ ({get_satisfaction_grade(avg_rating)})
- **Market Trajectory:** {rating_trend} ({abs(recent_avg_rating - avg_rating):.2f} point change)
- **Customer Engagement Quality:** {"Premium" if engagement_rate > 30 else "Standard" if engagement_rate > 15 else "Basic"} ({engagement_rate:.1f}% detailed feedback)
- **Brand Health Score:** {calculate_brand_health_score(avg_rating, sentiment_dist, total_risk_rate)}/100

### 🎭 Customer Sentiment Intelligence
- **🟢 Brand Champions:** {sentiment_dist.get('Extremely Positive', 0) + sentiment_dist.get('Very Positive', 0) + sentiment_dist.get('Positive', 0):.1f}% (Premium customer advocates)
- **🟡 Neutral Zone:** {sentiment_dist.get('Neutral', 0):.1f}% (Conversion opportunity)
- **🔴 At-Risk Customers:** {sentiment_dist.get('Negative', 0) + sentiment_dist.get('Very Negative', 0) + sentiment_dist.get('Extremely Negative', 0):.1f}% (Retention priority)

### 🛡️ Data Integrity & Trust Metrics
- **Verified Authentic Reviews:** {100 - total_risk_rate:.1f}% ✅
- **Suspicious Activity:** {total_risk_rate:.1f}% flagged for review
- **Data Confidence Level:** {get_trust_score(total_risk_rate)}/10 (Industry standard: 7.5+)

### 🏆 Top Customer Priorities
{get_formatted_top_topics(top_topics, total_reviews)}

### 💡 Strategic Action Plan
{strategic_recommendations}
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
    """Calculate business impact score ranging from -5.0 to +5.0"""
    impact_scores = []
    for _, row in df.iterrows():
        impact = 0
        
        # Rating contribution
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
        
        # Sentiment contribution
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
        
        # Word count multiplier
        if row['wordCount'] > 100:
            impact = impact * 1.5
        elif row['wordCount'] > 50:
            impact = impact * 1.2
        elif row['wordCount'] < 10:
            impact = impact * 0.5
        
        # Trust factor
        if row['fraudFlag'] == 'High Risk':
            impact = impact * 0.3
        elif row['fraudFlag'] == 'Medium Risk':
            impact = impact * 0.7
        
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

def main():
    st.markdown('<h1 class="main-header">📊 Customer Reviews Intelligence Platform</h1>', unsafe_allow_html=True)
    st.markdown("*Transform customer feedback into strategic business intelligence for C-level decision making*")
    
    # Create layout with sidebar for filters
    col1, col2 = st.columns([1, 4])
    
    with col1:
        # File upload section
        st.markdown("### 📁 Data Upload")
        uploaded_file = st.file_uploader("Upload Dataset", type=['csv'], help="Upload your Amazon reviews CSV file")
        
        # Enhanced filter sidebar
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
                
                st.markdown('<div class="filter-sidebar">', unsafe_allow_html=True)
                st.markdown('<div class="filter-header">🎛️ Smart Filters</div>', unsafe_allow_html=True)
                
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
                        st.experimental_rerun()
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Download section in sidebar
                st.markdown("---")
                st.markdown('<div class="download-section">', unsafe_allow_html=True)
                st.markdown("### 📥 Export Data")
                
                # Prepare filtered data for export
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
                
                export_df = filtered_df[[
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
                executive_summary = create_enhanced_executive_summary_with_topics(filtered_df)
                st.download_button(
                    label="📋 Executive Report",
                    data=executive_summary,
                    file_name=f"executive_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.md",
                    mime="text/markdown",
                    help="Download C-level summary report for board presentations"
                )
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
                st.markdown('<div class="executive-summary">', unsafe_allow_html=True)
                st.markdown('<div class="executive-content">', unsafe_allow_html=True)
                executive_summary = create_enhanced_executive_summary_with_topics(filtered_df)
                st.markdown(executive_summary, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Enhanced metric cards with better spacing
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
                
                # Enhanced charts section
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    
                    # Enhanced rating distribution
                    fig_rating = px.histogram(
                        filtered_df, x='rating', 
                        title="📊 Customer Rating Distribution",
                        color_discrete_sequence=['#667eea'],
                        text_auto=True,
                        category_orders={"rating": [1, 2, 3, 4, 5]}
                    )
                    
                    fig_rating.update_layout(
                        xaxis_title="Star Rating",
                        yaxis_title="Number of Reviews",
                        showlegend=False, 
                        height=450,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Inter, sans-serif", size=12),
                        title_font_size=16,
                        title_x=0.5
                    )
                    
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
                    
                    # Enhanced sentiment pie chart
                    sentiment_counts = filtered_df['sentiment'].value_counts()
                    colors = {
                        'Extremely Positive': '#0d7377', 'Very Positive': '#14a085', 'Positive': '#2ca02c',
                        'Neutral': '#ffbb33', 'Negative': '#ff6b6b', 'Very Negative': '#d62728', 'Extremely Negative': '#8b0000'
                    }
                    
                    fig_sentiment = px.pie(
                        values=sentiment_counts.values, 
                        names=sentiment_counts.index,
                        title="🎭 Customer Sentiment Analysis",
                        color=sentiment_counts.index,
                        color_discrete_map=colors
                    )
                    
                    fig_sentiment.update_layout(
                        height=450,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Inter, sans-serif", size=12),
                        title_font_size=16,
                        title_x=0.5,
                        showlegend=True
                    )
                    
                    fig_sentiment.update_traces(
                        textposition='inside',
                        textinfo='percent+label',
                        marker=dict(
                            line=dict(color='#FFFFFF', width=3)
                        )
                    )
                    
                    brand_advocates = sentiment_counts.get('Extremely Positive', 0) + sentiment_counts.get('Very Positive', 0)
                    detractors = sentiment_counts.get('Very Negative', 0) + sentiment_counts.get('Extremely Negative', 0)
                    nps_proxy = (brand_advocates - detractors) / len(filtered_df) * 100
                    
                    insight_text = f"**Net Promoter Score: {nps_proxy:.1f}%** - {brand_advocates:,} brand advocates vs {detractors:,} detractors. Excellent customer loyalty foundation."
                    
                    create_chart_with_insights(fig_sentiment, insight_text)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Word Cloud Section
                st.markdown("### 🔤 Customer Voice Analysis")
                st.markdown('<div class="wordcloud-container">', unsafe_allow_html=True)
                
                if WORDCLOUD_AVAILABLE:
                    wordcloud_fig = generate_wordcloud(filtered_df)
                    if wordcloud_fig:
                        st.pyplot(wordcloud_fig)
                    else:
                        # Fallback to word frequency analysis
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
                else:
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
                
                # Word analysis insights
                if word_freq:
                    top_words = [word for word, _ in word_freq[:5]]
                    st.markdown(f"""
                    <div class="insight-box">
                        <div class="insight-title">📊 Word Analysis Insights</div>
                        <strong>Top customer priorities:</strong> {', '.join(top_words)}. These keywords represent your brand's core value propositions and customer focus areas. Consider leveraging these terms in marketing messaging and product development.
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Business Impact Explanation
                st.markdown('<div class="explained-box">', unsafe_allow_html=True)
                st.markdown('<div class="explained-title">📊 Business Impact Score Explained</div>', unsafe_allow_html=True)
                st.markdown("""
                **What it measures:** The Business Impact Score quantifies how much each review influences your business success, ranging from **-5.0 to +5.0**.
                
                **How it's calculated:**
                - **Rating Component (-3 to +3):** 5-star = +3, 4-star = +1, 3-star = 0, 2-star = -2, 1-star = -3
                - **Sentiment Component (-2 to +2):** Extremely Positive = +2, Very Positive = +1, Positive = +0.5, Negative = -1, Very/Extremely Negative = -2
                - **Engagement Multiplier:** Detailed reviews (100+ words) get 1.5x impact, medium reviews (50+ words) get 1.2x impact
                - **Trust Factor:** High-risk reviews reduced to 30% impact, medium-risk to 70% impact
                
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
                    st.markdown("### 👥 Customer Segment Performance Matrix")
                    
                    segment_analysis = filtered_df.groupby('customerSegment').agg({
                        'rating': 'mean',
                        'businessImpact': 'mean',
                        'reviewValue': 'mean',
                        'reviewId': 'count'
                    }).round(2).reset_index()
                    segment_analysis.columns = ['Customer Segment', 'Avg Rating', 'Business Impact', 'Review Value', 'Volume']
                    
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
                    
                    fig_segments.update_layout(
                        height=500,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Inter, sans-serif")
                    )
                    
                    best_segment = segment_analysis.loc[segment_analysis['Business Impact'].idxmax(), 'Customer Segment']
                    largest_segment = segment_analysis.loc[segment_analysis['Volume'].idxmax(), 'Customer Segment']
                    
                    insight_text = f"**Strategic Priority:** {best_segment} delivers highest business value. **Scale Focus:** {largest_segment} represents your largest customer base."
                    
                    create_chart_with_insights(fig_segments, insight_text)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                    st.markdown("### 📈 Performance Trends Analysis")
                    
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
                        subplot_titles=('Customer Satisfaction Trend', 'Business Impact Evolution'),
                        vertical_spacing=0.15
                    )
                    
                    fig_trends.add_trace(
                        go.Scatter(
                            x=monthly_trends['date'], 
                            y=monthly_trends['rating'],
                            mode='lines+markers', 
                            name='Satisfaction Rating',
                            line=dict(color='#667eea', width=3)
                        ),
                        row=1, col=1
                    )
                    
                    fig_trends.add_trace(
                        go.Scatter(
                            x=monthly_trends['date'], 
                            y=monthly_trends['businessImpact'],
                            mode='lines+markers', 
                            name='Business Impact',
                            line=dict(color='#f093fb', width=3)
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
                        showlegend=False
                    )
                    
                    recent_rating = monthly_trends['rating'].tail(3).mean()
                    historical_rating = monthly_trends['rating'].head(3).mean()
                    trend_direction = "improving" if recent_rating > historical_rating else "declining"
                    trend_magnitude = abs(recent_rating - historical_rating)
                    
                    insight_text = f"**Trend Analysis:** Customer satisfaction is {trend_direction} with {trend_magnitude:.2f} point change."
                    
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
                    
                    top_topics_perf = topic_performance.nlargest(8, 'Business Impact')
                    
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
                    
                    fig_topics.update_layout(
                        height=500,
                        yaxis={'categoryorder':'total ascending'},
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Inter, sans-serif")
                    )
                    
                    best_topic = top_topics_perf.iloc[0]['Topic']
                    worst_topic = topic_performance.nsmallest(1, 'Business Impact').iloc[0]['Topic']
                    
                    insight_text = f"**Market Opportunity:** '{best_topic}' drives highest customer value. **Risk Area:** '{worst_topic}' needs strategic intervention."
                    
                    create_chart_with_insights(fig_topics, insight_text)
                    st.markdown("</div>", unsafe_allow_html=True)
            
            # TAB 3: Deep Intelligence
            with tab3:
                st.markdown("<h2 class='subheader'>🔍 Deep Customer Intelligence</h2>", unsafe_allow_html=True)
                
                # Strategic recommendations
                if topics:
                    st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
                    st.markdown("### 🎯 C-Level Strategic Recommendations")
                    
                    top_opportunity = filtered_df['topic'].value_counts().index[0]
                    top_sentiment = filtered_df.groupby('topic')['businessImpact'].mean().nlargest(1).index[0]
                    risk_topics = filtered_df[filtered_df['businessImpact'] < 0]['topic'].value_counts().head(2)
                    
                    st.markdown(f"""
                    **🚀 Primary Market Opportunity:** 
                    - **Focus Area:** {top_opportunity} (highest customer interest)
                    - **Performance Leader:** {top_sentiment} (highest business impact)
                    
                    **⚠️ Risk Management Areas:**
                    {chr(10).join([f"- **{topic}**: {count} negative mentions requiring attention" for topic, count in risk_topics.items()])}
                    
                    **📈 Executive Action Items:**
                    1. **Immediate (0-30 days):** Leverage {top_sentiment} in marketing campaigns
                    2. **Short-term (1-3 months):** Address risk areas with targeted improvements  
                    3. **Long-term (3-12 months):** Build competitive advantages around top opportunities
                    """)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Customer journey analysis
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
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
                        'reviewId': 'count'
                    }).round(2).reset_index()
                    engagement_analysis.columns = ['Engagement Level', 'Avg Rating', 'Sentiment Score', 'Business Impact', 'Volume']
                    
                    fig_engagement = px.bar(
                        engagement_analysis, 
                        x='Engagement Level', 
                        y='Avg Rating',
                        color='Business Impact',
                        title="Customer Engagement vs Satisfaction Matrix",
                        color_continuous_scale='Viridis',
                        hover_data=['Volume']
                    )
                    
                    fig_engagement.update_layout(
                        height=450,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Inter, sans-serif")
                    )
                    
                    optimal_engagement = engagement_analysis.loc[engagement_analysis['Avg Rating'].idxmax(), 'Engagement Level']
                    
                    insight_text = f"**Engagement Sweet Spot:** {optimal_engagement} reviews generate highest satisfaction."
                    
                    create_chart_with_insights(fig_engagement, insight_text)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                    st.markdown("### 🎭 Customer Sentiment Journey Map")
                    
                    sentiment_order = ['Extremely Negative', 'Very Negative', 'Negative', 'Neutral', 
                                      'Positive', 'Very Positive', 'Extremely Positive']
                    
                    segment_journey = filtered_df.groupby(['customerSegment', 'sentiment']).size().unstack(fill_value=0)
                    segment_journey = segment_journey.reindex(columns=sentiment_order, fill_value=0)
                    segment_journey_pct = segment_journey.div(segment_journey.sum(axis=1), axis=0) * 100
                    
                    fig_journey = px.imshow(
                        segment_journey_pct.values,
                        x=segment_journey_pct.columns,
                        y=segment_journey_pct.index,
                        title="Customer Segment Sentiment Distribution",
                        color_continuous_scale='RdYlGn',
                        text_auto='.1f',
                        aspect="auto"
                    )
                    
                    fig_journey.update_layout(
                        height=450,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Inter, sans-serif")
                    )
                    
                    segment_scores = (segment_journey_pct[['Positive', 'Very Positive', 'Extremely Positive']].sum(axis=1) - 
                                    segment_journey_pct[['Negative', 'Very Negative', 'Extremely Negative']].sum(axis=1))
                    best_segment = segment_scores.idxmax()
                    worst_segment = segment_scores.idxmin()
                    
                    insight_text = f"**Champion Segment:** {best_segment} shows highest positive sentiment. **Priority Segment:** {worst_segment} needs attention."
                    
                    create_chart_with_insights(fig_journey, insight_text)
                    st.markdown("</div>", unsafe_allow_html=True)
            
            # TAB 4: Customer Voices
            with tab4:
                st.markdown("<h2 class='subheader'>💬 Customer Voice Intelligence</h2>", unsafe_allow_html=True)
                
                positive_verbatims, negative_verbatims = extract_sample_verbatims(filtered_df)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🌟 Brand Champions & Success Stories")
                    
                    for idx, (_, review) in enumerate(positive_verbatims.iterrows(), 1):
                        st.markdown(f'<div class="verbatim-section positive-verbatim">', unsafe_allow_html=True)
                        st.markdown(f"**Champion Voice #{idx}** | ⭐{review['rating']}/5 | Impact: **{review['businessImpact']:.1f}/5.0**")
                        
                        review_text = review['reviewText']
                        if len(review_text) > 200:
                            sentences = review_text.split('.')
                            preview = '. '.join(sentences[:2]) + '...' if len(sentences) > 2 else review_text
                        else:
                            preview = review_text
                        
                        st.markdown(f"*\"{preview}\"*")
                        st.markdown(f"**Topic:** {review['topic']} | **Segment:** {review['customerSegment']}")
                        st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown("### ⚠️ Critical Customer Concerns")
                    
                    for idx, (_, review) in enumerate(negative_verbatims.iterrows(), 1):
                        st.markdown(f'<div class="verbatim-section negative-verbatim">', unsafe_allow_html=True)
                        st.markdown(f"**Critical Issue #{idx}** | ⭐{review['rating']}/5 | Impact: **{review['businessImpact']:.1f}/5.0**")
                        
                        review_text = review['reviewText']
                        if len(review_text) > 200:
                            sentences = review_text.split('.')
                            preview = '. '.join(sentences[:2]) + '...' if len(sentences) > 2 else review_text
                        else:
                            preview = review_text
                        
                        st.markdown(f"*\"{preview}\"*")
                        urgency = "🚨 Critical" if review['businessImpact'] < -3 else "⚠️ High" if review['businessImpact'] < -1 else "📊 Monitor"
                        st.markdown(f"**Urgency:** {urgency} | **Topic:** {review['topic']}")
                        st.markdown('</div>', unsafe_allow_html=True)
            
            # TAB 5: Risk Management
            with tab5:
                st.markdown("<h2 class='subheader'>🚨 Enterprise Risk Management</h2>", unsafe_allow_html=True)
                
                # Risk metrics with fixed styling
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    high_risk_count = (filtered_df['fraudFlag'] == 'High Risk').sum()
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h3>🚨 Critical Threats</h3>
                        <h2>{high_risk_count}</h2>
                        <p>{high_risk_count/len(filtered_df)*100:.1f}% of reviews</p>
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
                    integrity_grade = "A+" if data_integrity > 90 else "A" if data_integrity > 80 else "B"
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
                
                # Fraud detection analysis
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                    st.markdown("### 🔍 Review Authenticity Intelligence")
                    
                    fraud_dist = filtered_df['fraudFlag'].value_counts()
                    
                    fig_fraud = px.pie(
                        values=fraud_dist.values,
                        names=fraud_dist.index,
                        title="Review Trust Analysis",
                        color_discrete_map={
                            'Legitimate': '#28a745',
                            'Low Risk': '#ffc107', 
                            'Medium Risk': '#fd7e14',
                            'High Risk': '#dc3545'
                        }
                    )
                    
                    fig_fraud.update_layout(
                        height=450,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Inter, sans-serif")
                    )
                    
                    legitimate_rate = fraud_dist.get('Legitimate', 0) / len(filtered_df) * 100
                    total_risk = 100 - legitimate_rate
                    
                    insight_text = f"**Data Integrity Status:** {legitimate_rate:.1f}% verified authentic reviews. **Risk Exposure:** {total_risk:.1f}% requires monitoring."
                    
                    create_chart_with_insights(fig_fraud, insight_text)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                    st.markdown("### 📈 Risk vs Business Impact")
                    
                    # Risk impact scatter plot
                    fig_risk_impact = px.scatter(
                        filtered_df,
                        x='fraudScore',
                        y='businessImpact',
                        color='fraudFlag',
                        title="Fraud Risk vs Business Impact Analysis",
                        color_discrete_map={
                            'Legitimate': '#28a745',
                            'Low Risk': '#ffc107', 
                            'Medium Risk': '#fd7e14',
                            'High Risk': '#dc3545'
                        }
                    )
                    
                    fig_risk_impact.update_layout(
                        height=450,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Inter, sans-serif")
                    )
                    
                    high_risk_negative_impact = len(filtered_df[(filtered_df['fraudScore'] > 5) & (filtered_df['businessImpact'] < -2)])
                    
                    insight_text = f"**Critical Alert:** {high_risk_negative_impact} reviews show both high fraud risk and negative business impact - priority investigation required."
                    
                    create_chart_with_insights(fig_risk_impact, insight_text)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Suspicious reviews analysis
                suspicious_df = filtered_df[filtered_df['fraudFlag'].isin(['High Risk', 'Medium Risk'])]
                
                if not suspicious_df.empty:
                    st.markdown("### 🕵️ Suspicious Review Detection Results")
                    st.markdown(f"**🚨 Security Alert:** {len(suspicious_df)} suspicious reviews detected")
                    
                    fraud_analysis = suspicious_df.groupby(['fraudFlag', 'fraudReason']).size().reset_index(name='Count')
                    fraud_analysis = fraud_analysis.sort_values('Count', ascending=False)
                    
                    st.markdown("#### **Threat Pattern Analysis**")
                    for _, row in fraud_analysis.head(5).iterrows():
                        threat_level = "🔴 Critical" if row['fraudFlag'] == 'High Risk' else "🟡 Moderate"
                        st.markdown(f"- **{threat_level}**: {row['fraudReason']} ({row['Count']} instances)")
                    
                    # Sample suspicious reviews
                    st.markdown("#### **High-Risk Review Samples**")
                    sample_suspicious = suspicious_df.nlargest(3, 'fraudScore')[['reviewText', 'rating', 'fraudFlag', 'fraudReason']]
                    
                    for idx, (_, review) in enumerate(sample_suspicious.iterrows(), 1):
                        st.markdown(f'<div class="verbatim-section negative-verbatim">', unsafe_allow_html=True)
                        st.markdown(f"**Suspicious Review #{idx}** | Rating: {review['rating']} | Risk: {review['fraudFlag']}")
                        st.markdown(f"*\"{review['reviewText'][:200]}...\"*")
                        st.markdown(f"**Risk Factors:** {review['fraudReason']}")
                        st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.success("✅ **All Clear:** No suspicious reviews detected in current selection")
                
                # Risk management recommendations
                st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
                st.markdown("### 🛡️ Executive Risk Management Strategy")
                
                total_risk_rate = (filtered_df['fraudFlag'] != 'Legitimate').sum() / len(filtered_df) * 100
                
                if total_risk_rate > 20:
                    risk_status = "🔴 CRISIS LEVEL"
                    action_urgency = "Immediate C-suite intervention required"
                elif total_risk_rate > 10:
                    risk_status = "🟠 ELEVATED RISK"
                    action_urgency = "Deploy enhanced monitoring protocols"
                else:
                    risk_status = "🟢 MANAGEABLE RISK"
                    action_urgency = "Maintain current security posture"
                
                st.markdown(f"""
                **Current Threat Level:** {risk_status}  
                **Recommended Action:** {action_urgency}
                
                **Strategic Risk Mitigation:**
                - **Immediate (0-7 days):** Investigate all high-risk flagged reviews
                - **Short-term (1-4 weeks):** Implement review verification protocols
                - **Long-term (1-6 months):** Build predictive fraud detection systems
                """)
                st.markdown('</div>', unsafe_allow_html=True)
        
        else:
            # Enhanced onboarding experience
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("""
            ### 🚀 Welcome to Your C-Level Customer Intelligence Platform
            
            **Transform Raw Reviews into Strategic Business Intelligence**
            
            This enterprise-grade dashboard converts customer feedback into actionable insights for executive decision making.
            
            #### 🎯 What You'll Discover:
            - **📊 Executive KPIs:** Customer satisfaction, business impact, and market position metrics
            - **🧠 AI-Powered Analytics:** Advanced sentiment analysis and topic modeling
            - **🔍 Fraud Detection:** Sophisticated algorithms to identify suspicious reviews
            - **💬 Customer Voices:** Strategic verbatims driving business decisions
            - **🚨 Risk Management:** Enterprise-level threat detection and mitigation
            - **📈 Competitive Intelligence:** Market positioning and performance benchmarks
            
            #### 📋 Required Data Format:
            Upload a CSV file containing Amazon reviews with these columns:
            - Review ID, Reviewer Name, Review Text, Rating (1-5)
            - Summary, Helpful Votes, Total Votes, Review Date, Year
            
            #### 🎨 Advanced Features:
            - **Interactive Filtering** with smart controls in the left sidebar
            - **Word Cloud Analysis** showing customer priority keywords
            - **Strategic Recommendations** with C-level action items
            - **Export Capabilities** for board presentations and reports
            
            **Ready to unlock your customer intelligence? Upload your dataset to begin the analysis.**
            """)
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
