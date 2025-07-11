import streamlit as st
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

st.set_page_config(page_title="Executive Amazon Reviews Intelligence", page_icon="📊", layout="wide")

# Enhanced CSS styling for C-level presentation
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .executive-summary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 20px;
        margin: 2rem 0;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
    }
    .chart-insight {
        background: rgba(102, 126, 234, 0.1);
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .recommendation-box {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: #2c3e50;
        font-weight: 500;
    }
    .verbatim-section {
        background: rgba(255,255,255,0.9);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
    }
    .positive-verbatim {
        border-left: 4px solid #28a745;
        background: rgba(40, 167, 69, 0.05);
    }
    .negative-verbatim {
        border-left: 4px solid #dc3545;
        background: rgba(220, 53, 69, 0.05);
    }
</style>
""", unsafe_allow_html=True)

if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

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
            topics.append(topic_label)
        
        doc_topic_matrix = lda.transform(doc_term_matrix)
        topic_assignments = []
        
        for doc_topics in doc_topic_matrix:
            max_prob = np.max(doc_topics)
            if max_prob > 0.25:
                topic_idx = np.argmax(doc_topics)
                topic_assignments.append(topics[topic_idx])
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
    rating_trend = "📈 Improving" if recent_avg_rating > avg_rating else "📉 Declining" if recent_avg_rating < avg_rating else "➡️ Stable"
    
    detailed_reviews = df[df['wordCount'] > 50]
    engagement_rate = len(detailed_reviews) / total_reviews * 100
    
    top_topics = df['topic'].value_counts().head(3)
    
    summary = f"""
    # 🎯 Executive Intelligence Dashboard
    *Strategic insights for data-driven decision making*
    
    ## 📊 **Business Performance Overview**
    - **Total Customer Voices Analyzed:** {total_reviews:,}
    - **Overall Customer Satisfaction:** {avg_rating:.2f}/5.0 ⭐ ({get_satisfaction_grade(avg_rating)})
    - **Satisfaction Trajectory:** {rating_trend}
    - **Customer Engagement Level:** {"High" if engagement_rate > 30 else "Moderate" if engagement_rate > 15 else "Low"} ({engagement_rate:.1f}% detailed reviews)
    
    ## 🎭 **Customer Sentiment Intelligence**
    - **Brand Advocates:** {sentiment_dist.get('Extremely Positive', 0) + sentiment_dist.get('Very Positive', 0) + sentiment_dist.get('Positive', 0):.1f}%
    - **Neutral/Undecided:** {sentiment_dist.get('Neutral', 0):.1f}%
    - **Detractors & Critics:** {sentiment_dist.get('Negative', 0) + sentiment_dist.get('Very Negative', 0) + sentiment_dist.get('Extremely Negative', 0):.1f}%
    
    ## 🔍 **Trust & Authenticity Assessment**
    - **High-Risk Reviews:** {high_risk_rate:.1f}% 🚨
    - **Medium-Risk Reviews:** {medium_risk_rate:.1f}% ⚠️
    - **Trusted Customer Feedback:** {100 - total_risk_rate:.1f}% ✅
    - **Overall Trust Score:** {get_trust_score(total_risk_rate)}/10
    
    ## 🏆 **Strategic Business Themes**
    {chr(10).join([f"**{i+1}. {topic}:** {count:,} mentions ({count/total_reviews*100:.1f}%)" for i, (topic, count) in enumerate(top_topics.items())])}
    
    ## 💡 **Executive Action Items**
    
    ### 🎯 **Immediate Priorities**
    - **Customer Satisfaction:** {get_satisfaction_recommendation(avg_rating)}
    - **Review Authenticity:** {get_trust_recommendation(total_risk_rate)}
    - **Engagement Strategy:** {get_engagement_recommendation(engagement_rate)}
    
    ### 📈 **Performance Indicators**
    - **Market Position:** {get_market_position(avg_rating, sentiment_dist)}
    - **Trust Level:** {'High' if total_risk_rate < 10 else 'Moderate' if total_risk_rate < 20 else 'Low'}
    """
    
    return summary

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
        return "Critical (D)"

def get_trust_score(risk_rate):
    if risk_rate < 5:
        return 9
    elif risk_rate < 10:
        return 8
    elif risk_rate < 20:
        return 6
    else:
        return 4

def get_satisfaction_recommendation(rating):
    if rating >= 4.5:
        return "🟢 Maintain excellence - leverage positive feedback"
    elif rating >= 4.0:
        return "🟡 Good performance - identify improvement opportunities"
    elif rating >= 3.0:
        return "🟠 Below expectations - immediate quality review needed"
    else:
        return "🔴 Critical - urgent intervention required"

def get_trust_recommendation(risk_rate):
    if risk_rate < 5:
        return "🟢 High trust - reviews are largely authentic"
    elif risk_rate < 15:
        return "🟡 Monitor closely - implement review verification"
    else:
        return "🔴 Trust deficit - immediate fraud prevention needed"

def get_engagement_recommendation(engagement_rate):
    if engagement_rate > 30:
        return "🟢 High-quality engagement - customers provide valuable feedback"
    elif engagement_rate > 15:
        return "🟡 Moderate engagement - encourage more detailed reviews"
    else:
        return "🔴 Low engagement - implement review incentive programs"

def get_market_position(rating, sentiment_dist):
    positive_rate = sentiment_dist.get('Extremely Positive', 0) + sentiment_dist.get('Very Positive', 0) + sentiment_dist.get('Positive', 0)
    if rating >= 4.5 and positive_rate > 80:
        return "🟢 Market Leader - Strong competitive advantage"
    elif rating >= 4.0 and positive_rate > 65:
        return "🟡 Strong Performer - Above market average"
    elif rating >= 3.5:
        return "🟠 Market Follower - Room for differentiation"
    else:
        return "🔴 At Risk - Competitive disadvantage"

def process_data_with_advanced_ml(df):
    st.info("🤖 Running enterprise-grade analytics algorithms...")
    
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
    impact_scores = []
    for _, row in df.iterrows():
        impact = 0
        
        if row['rating'] == 5:
            impact += 3
        elif row['rating'] == 4:
            impact += 1
        elif row['rating'] == 2:
            impact -= 2
        elif row['rating'] == 1:
            impact -= 3
        
        if 'Extremely Positive' in row['sentiment']:
            impact += 2
        elif 'Very Positive' in row['sentiment']:
            impact += 1
        elif 'Negative' in row['sentiment']:
            impact -= 1
        elif 'Very Negative' in row['sentiment'] or 'Extremely Negative' in row['sentiment']:
            impact -= 2
        
        if row['wordCount'] > 50:
            impact = impact * 1.5
        
        impact_scores.append(round(impact, 1))
    
    return impact_scores

def create_chart_with_insights(fig, insight_text):
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f'<div class="chart-insight"><strong>📊 Executive Insight:</strong> {insight_text}</div>', 
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

def main():
    st.markdown('<div class="main-header">📊 Executive Amazon Reviews Intelligence</div>', unsafe_allow_html=True)
    st.markdown("*Transforming customer feedback into strategic business intelligence*")
    
    uploaded_file = st.file_uploader("📁 Upload Amazon Reviews Dataset", type=['csv'])
    
    if uploaded_file:
        with st.spinner('🔄 Processing data with enterprise-grade AI algorithms...'):
            df = load_and_process_data(uploaded_file)
            if df is not None:
                df, topics = process_data_with_advanced_ml(df)
                st.session_state.processed_data = df
                st.session_state.topics = topics
                st.success(f"✅ Successfully analyzed {len(df):,} customer reviews with advanced AI")
            else:
                st.error("❌ Failed to process data. Please verify CSV format.")
                return
    elif st.session_state.processed_data is not None:
        df = st.session_state.processed_data
        topics = getattr(st.session_state, 'topics', [])
    else:
        st.info("👆 Upload your Amazon reviews dataset to begin executive analysis")
        st.markdown("""
        ### 🎯 **Executive-Grade Analytics Platform**
        
        **Advanced Capabilities:**
        - 🧠 **Multi-Algorithm Sentiment Analysis** with confidence scoring
        - 🔍 **6-Layer Fraud Detection** with behavioral analysis  
        - 🏷️ **Business-Context Topic Modeling** with strategic themes
        - 📈 **Predictive Trend Analysis** for proactive decision-making
        - 💼 **ROI Impact Assessment** for each customer review
        - 🎭 **Customer Segmentation** based on engagement patterns
        
        **Expected Data Format:** Standard Amazon reviews CSV with review text, ratings, dates, and reviewer information.
        """)
        return
    
    # Advanced Filtering System
    st.sidebar.header("🎛️ Executive Filters")
    
    rating_filter = st.sidebar.multiselect("⭐ Customer Satisfaction Level", 
                                          sorted(df['rating'].unique()), 
                                          default=sorted(df['rating'].unique()))
    
    sentiment_filter = st.sidebar.multiselect("😊 Brand Sentiment", 
                                             df['sentiment'].unique(), 
                                             default=df['sentiment'].unique())
    
    trust_filter = st.sidebar.selectbox("🔍 Trust Level", 
                                       ['All Reviews', 'Trusted Only', 'Suspicious Only', 'High Risk Only'])
    
    segment_filter = st.sidebar.multiselect("👥 Customer Segment", 
                                           df['customerSegment'].unique(), 
                                           default=df['customerSegment'].unique())
    
    min_impact = st.sidebar.slider("💼 Minimum Business Impact", 
                                  float(df['businessImpact'].min()), 
                                  float(df['businessImpact'].max()), 
                                  float(df['businessImpact'].min()))
    
    min_confidence = st.sidebar.slider("🎯 Minimum Analysis Confidence", 0.0, 1.0, 0.0, 0.1)
    
    # Apply filters
    filtered_df = df[
        (df['rating'].isin(rating_filter)) & 
        (df['sentiment'].isin(sentiment_filter)) &
        (df['customerSegment'].isin(segment_filter)) &
        (df['businessImpact'] >= min_impact) &
        (df['sentimentConfidence'] >= min_confidence)
    ]
    
    # Trust filter application
    if trust_filter == 'Trusted Only':
        filtered_df = filtered_df[filtered_df['fraudFlag'] == 'Legitimate']
    elif trust_filter == 'Suspicious Only':
        filtered_df = filtered_df[filtered_df['fraudFlag'].isin(['Low Risk', 'Medium Risk'])]
    elif trust_filter == 'High Risk Only':
        filtered_df = filtered_df[filtered_df['fraudFlag'] == 'High Risk']
    
    # Executive Dashboard Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Executive Summary", "📊 Business Intelligence", "🔍 Strategic Insights", 
        "💬 Voice of Customer", "🚨 Risk Assessment"
    ])
    
    # TAB 1: Executive Summary
    with tab1:
        st.markdown('<div class="executive-summary">', unsafe_allow_html=True)
        st.markdown(create_executive_summary(filtered_df))
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Key Performance Indicators
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            avg_rating = filtered_df['rating'].mean()
            st.metric("📊 Customer Satisfaction", f"{avg_rating:.2f}/5.0", 
                     delta=f"{(avg_rating - 3.0):.1f} vs neutral")
        
        with col2:
            fraud_rate = (filtered_df['fraudFlag'].isin(['High Risk', 'Medium Risk'])).sum() / len(filtered_df) * 100
            st.metric("🔍 Trust Score", f"{100-fraud_rate:.1f}%", 
                     delta=f"-{fraud_rate:.1f}% risk")
        
        with col3:
            positive_sentiment = (filtered_df['sentiment'].str.contains('Positive')).sum() / len(filtered_df) * 100
            st.metric("😊 Brand Sentiment", f"{positive_sentiment:.1f}%")
        
        with col4:
            engagement_rate = (filtered_df['wordCount'] > 50).sum() / len(filtered_df) * 100
            st.metric("💬 Engagement Rate", f"{engagement_rate:.1f}%")
        
        with col5:
            avg_business_impact = filtered_df['businessImpact'].mean()
            st.metric("💼 Business Impact", f"{avg_business_impact:.1f}", 
                     delta="per review")
        
        # Strategic Overview Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Enhanced rating distribution
            fig_rating = px.histogram(
                filtered_df, x='rating', 
                title="📊 Customer Satisfaction Distribution",
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
            insight_text = f"**{high_satisfaction:.1f}% of customers are highly satisfied** (4-5 stars). This indicates {get_satisfaction_grade(filtered_df['rating'].mean()).split('(')[0]} market performance with strong customer advocacy potential."
            
            create_chart_with_insights(fig_rating, insight_text)
        
        with col2:
            # Enhanced sentiment distribution
            sentiment_counts = filtered_df['sentiment'].value_counts()
            colors = {
                'Extremely Positive': '#0d7377', 'Very Positive': '#14a085', 'Positive': '#2ca02c',
                'Neutral': '#ffbb33', 'Negative': '#ff6b6b', 'Very Negative': '#d62728', 'Extremely Negative': '#8b0000'
            }
            
            fig_sentiment = px.pie(
                values=sentiment_counts.values, 
                names=sentiment_counts.index,
                title="🎭 Brand Sentiment Landscape",
                color=sentiment_counts.index,
                color_discrete_map=colors
            )
            
            brand_advocates = sentiment_counts.get('Extremely Positive', 0) + sentiment_counts.get('Very Positive', 0)
            detractors = sentiment_counts.get('Very Negative', 0) + sentiment_counts.get('Extremely Negative', 0)
            nps_proxy = (brand_advocates - detractors) / len(filtered_df) * 100
            
            insight_text = f"**Brand health score: {nps_proxy:.1f}%** (Advocates minus Detractors). {brand_advocates:,} customers are strong advocates, while {detractors:,} are potential detractors requiring immediate attention."
            
            create_chart_with_insights(fig_sentiment, insight_text)
    
    # TAB 2: Business Intelligence  
    with tab2:
        st.markdown("## 📊 Strategic Business Intelligence")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Customer segment analysis
            segment_analysis = filtered_df.groupby('customerSegment').agg({
                'rating': 'mean',
                'businessImpact': 'mean',
                'reviewValue': 'mean',
                'reviewId': 'count'
            }).round(2).reset_index()
            segment_analysis.columns = ['Segment', 'Avg Rating', 'Business Impact', 'Review Value', 'Count']
            
            fig_segments = px.scatter(
                segment_analysis, 
                x='Business Impact', 
                y='Avg Rating',
                size='Count',
                color='Segment',
                title="💼 Customer Segment Performance Matrix",
                hover_data=['Review Value']
            )
            
            insight_text = f"**{segment_analysis.loc[segment_analysis['Business Impact'].idxmax(), 'Segment']}** segment drives highest business impact. **{segment_analysis.loc[segment_analysis['Count'].idxmax(), 'Segment']}** represents the largest customer group requiring strategic focus."
            
            create_chart_with_insights(fig_segments, insight_text)
            
            # Topic performance analysis
            if topics:
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
                    title="🏷️ Strategic Topic Performance",
                    orientation='h',
                    color_continuous_scale='RdYlGn'
                )
                fig_topics.update_layout(yaxis={'categoryorder':'total ascending'})
                
                best_topic = top_topics_perf.iloc[0]['Topic']
                worst_topic = topic_performance.nsmallest(1, 'Business Impact').iloc[0]['Topic']
                
                insight_text = f"**'{best_topic}'** generates highest business value, while **'{worst_topic}'** requires strategic intervention. Focus marketing and product development on high-impact themes."
                
                create_chart_with_insights(fig_topics, insight_text)
        
        with col2:
            # Time-based business trends
            monthly_trends = filtered_df.groupby(['year', 'month']).agg({
                'rating': 'mean',
                'businessImpact': 'mean',
                'fraudScore': 'mean',
                'reviewId': 'count'
            }).reset_index()
            monthly_trends['date'] = pd.to_datetime(monthly_trends[['year', 'month']].assign(day=1))
            monthly_trends = monthly_trends.sort_values('date')
            
            # Create subplots for multiple metrics
            fig_trends = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Customer Satisfaction Trend', 'Business Impact Trend'),
                vertical_spacing=0.1
            )
            
            fig_trends.add_trace(
                go.Scatter(x=monthly_trends['date'], y=monthly_trends['rating'],
                          mode='lines+markers', name='Avg Rating', line=dict(color='#667eea')),
                row=1, col=1
            )
            
            fig_trends.add_trace(
                go.Scatter(x=monthly_trends['date'], y=monthly_trends['businessImpact'],
                          mode='lines+markers', name='Business Impact', line=dict(color='#f093fb')),
                row=2, col=1
            )
            
            fig_trends.update_layout(height=500, title_text="📈 Strategic Performance Trends")
            
            recent_rating = monthly_trends['rating'].tail(3).mean()
            historical_rating = monthly_trends['rating'].head(3).mean()
            trend_direction = "improving" if recent_rating > historical_rating else "declining"
            
            insight_text = f"**Customer satisfaction is {trend_direction}** with {abs(recent_rating - historical_rating):.2f} point change. Recent performance indicates {'strong momentum' if trend_direction == 'improving' else 'need for strategic intervention'}."
            
            create_chart_with_insights(fig_trends, insight_text)
            
            # Business impact heatmap
            impact_heatmap = pd.crosstab(filtered_df['rating'], filtered_df['sentiment'])
            
            fig_heatmap = px.imshow(
                impact_heatmap.values,
                x=impact_heatmap.columns,
                y=impact_heatmap.index,
                title="🔥 Rating-Sentiment Business Matrix",
                color_continuous_scale='RdYlBu_r',
                text_auto=True
            )
            fig_heatmap.update_layout(
                xaxis_title="Customer Sentiment",
                yaxis_title="Star Rating"
            )
            
            concerning_areas = impact_heatmap.loc[4:5, ['Negative', 'Very Negative', 'Extremely Negative']].sum().sum()
            positive_areas = impact_heatmap.loc[4:5, ['Positive', 'Very Positive', 'Extremely Positive']].sum().sum()
            
            insight_text = f"**{concerning_areas} high-rated reviews contain negative sentiment** - potential service recovery opportunities. **{positive_areas} reviews show strong rating-sentiment alignment** indicating authentic customer satisfaction."
            
            create_chart_with_insights(fig_heatmap, insight_text)
    
    # TAB 3: Strategic Insights
    with tab3:
        st.markdown("## 🔍 Deep Strategic Analysis")
        
        # Advanced topic analysis with business recommendations
        if topics:
            st.subheader("🏷️ Strategic Theme Intelligence")
            
            topic_insights = []
            for topic_name in topics:
                topic_reviews = filtered_df[filtered_df['topic'] == topic_name]
                if len(topic_reviews) > 0:
                    avg_rating = topic_reviews['rating'].mean()
                    avg_sentiment = topic_reviews['sentimentScore'].mean()
                    review_count = len(topic_reviews)
                    avg_impact = topic_reviews['businessImpact'].mean()
                    
                    if avg_rating >= 4.5 and avg_sentiment > 0.3:
                        recommendation = "🟢 Leverage as competitive advantage"
                    elif avg_rating >= 4.0 and avg_sentiment > 0.1:
                        recommendation = "🟡 Maintain and optimize"
                    elif avg_rating >= 3.0:
                        recommendation = "🟠 Requires improvement focus"
                    else:
                        recommendation = "🔴 Critical intervention needed"
                    
                    topic_insights.append({
                        'Theme': topic_name,
                        'Volume': review_count,
                        'Avg Rating': f"{avg_rating:.2f}",
                        'Sentiment Score': f"{avg_sentiment:.2f}",
                        'Business Impact': f"{avg_impact:.2f}",
                        'Strategic Action': recommendation
                    })
            
            if topic_insights:
                insights_df = pd.DataFrame(topic_insights).sort_values('Business Impact', ascending=False)
                st.dataframe(insights_df, use_container_width=True)
                
                st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
                st.markdown("### 💡 **Executive Recommendations**")
                
                top_opportunity = insights_df.iloc[0]
                biggest_risk = insights_df[insights_df['Strategic Action'].str.contains('🔴')].head(1)
                
                st.markdown(f"""
                **🎯 Primary Growth Opportunity:** Focus on **{top_opportunity['Theme']}** theme - highest business impact with {top_opportunity['Volume']} customer mentions.
                
                **⚠️ Immediate Risk:** {biggest_risk['Theme'].iloc[0] if len(biggest_risk) > 0 else 'No critical risks identified'} requires urgent attention.
                
                **📈 Strategic Priority:** Invest in themes showing positive sentiment trends while addressing negative feedback patterns systematically.
                """)
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Customer journey analysis
        st.subheader("🛤️ Customer Journey Intelligence")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Review length vs satisfaction analysis
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
                title="📝 Review Depth vs Customer Satisfaction",
                color_continuous_scale='Viridis'
            )
            
            optimal_length = length_analysis.loc[length_analysis['rating'].idxmax(), 'lengthCategory']
            optimal_rating = length_analysis['rating'].max()
            
            insight_text = f"**{optimal_length} reviews achieve highest satisfaction** ({optimal_rating:.2f}/5.0). Customers providing detailed feedback show {length_analysis.loc[length_analysis['lengthCategory'] == optimal_length, 'businessImpact'].iloc[0]:.1f} average business impact."
            
            create_chart_with_insights(fig_length, insight_text)
        
        with col2:
            # Customer segment journey
            segment_journey = filtered_df.groupby(['customerSegment', 'sentiment']).size().unstack(fill_value=0)
            segment_journey_pct = segment_journey.div(segment_journey.sum(axis=1), axis=0) * 100
            
            fig_journey = px.imshow(
                segment_journey_pct.values,
                x=segment_journey_pct.columns,
                y=segment_journey_pct.index,
                title="👥 Customer Segment Sentiment Journey",
                color_continuous_scale='RdYlGn',
                text_auto='.1f'
            )
            
            segment_scores = (segment_journey_pct[['Positive', 'Very Positive', 'Extremely Positive']].sum(axis=1) - 
                            segment_journey_pct[['Negative', 'Very Negative', 'Extremely Negative']].sum(axis=1))
            best_segment = segment_scores.idxmax()
            worst_segment = segment_scores.idxmin()
            
            insight_text = f"**{best_segment}** segment shows strongest positive sentiment ({segment_scores.max():.1f}% net positive), while **{worst_segment}** needs retention strategies ({segment_scores.min():.1f}% net sentiment)."
            
            create_chart_with_insights(fig_journey, insight_text)
    
    # TAB 4: Voice of Customer
    with tab4:
        st.markdown("## 💬 Voice of Customer Intelligence")
        
        # Extract and display sample verbatims
        positive_verbatims, negative_verbatims = extract_sample_verbatims(filtered_df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🌟 **Top Customer Advocates**")
            st.markdown("*High-impact positive feedback driving business value*")
            
            for idx, (_, review) in enumerate(positive_verbatims.iterrows(), 1):
                st.markdown(f'<div class="verbatim-section positive-verbatim">', unsafe_allow_html=True)
                st.markdown(f"**Advocate #{idx}** | ⭐{review['rating']}/5 | Impact: {review['businessImpact']:.1f}")
                review_preview = review['reviewText'][:300]
                if len(review['reviewText']) > 300:
                    review_preview += "..."
                st.markdown(f"*\"{review_preview}\"*")
                st.markdown(f"**Theme:** {review['topic']} | **Sentiment:** {review['sentiment']} ({review['sentimentConfidence']:.2f} confidence)")
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("### ⚠️ **Critical Customer Feedback**")
            st.markdown("*High-priority concerns requiring immediate attention*")
            
            for idx, (_, review) in enumerate(negative_verbatims.iterrows(), 1):
                st.markdown(f'<div class="verbatim-section negative-verbatim">', unsafe_allow_html=True)
                st.markdown(f"**Critical Issue #{idx}** | ⭐{review['rating']}/5 | Impact: {review['businessImpact']:.1f}")
                review_preview = review['reviewText'][:300]
                if len(review['reviewText']) > 300:
                    review_preview += "..."
                st.markdown(f"*\"{review_preview}\"*")
                st.markdown(f"**Theme:** {review['topic']} | **Sentiment:** {review['sentiment']} ({review['sentimentConfidence']:.2f} confidence)")
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Word frequency analysis
        st.subheader("🔤 Strategic Keyword Intelligence")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Positive keywords
            positive_reviews_text = filtered_df[filtered_df['sentiment'].str.contains('Positive', na=False)]['reviewText']
            positive_words = get_word_frequencies(positive_reviews_text)
            
            if positive_words:
                pos_df = pd.DataFrame(positive_words, columns=['Keyword', 'Frequency'])
                fig_pos_words = px.bar(
                    pos_df.head(15), 
                    x='Frequency', 
                    y='Keyword',
                    title="🌟 Positive Brand Drivers",
                    orientation='h',
                    color='Frequency',
                    color_continuous_scale='Greens'
                )
                fig_pos_words.update_layout(yaxis={'categoryorder':'total ascending'})
                
                top_positive_word = pos_df.iloc[0]['Keyword']
                insight_text = f"**'{top_positive_word}'** is the strongest positive brand driver. Leverage these keywords in marketing campaigns and product positioning to amplify customer satisfaction themes."
                
                create_chart_with_insights(fig_pos_words, insight_text)
        
        with col2:
            # Negative keywords
            negative_reviews_text = filtered_df[filtered_df['sentiment'].str.contains('Negative', na=False)]['reviewText']
            negative_words = get_word_frequencies(negative_reviews_text)
            
            if negative_words:
                neg_df = pd.DataFrame(negative_words, columns=['Keyword', 'Frequency'])
                fig_neg_words = px.bar(
                    neg_df.head(15), 
                    x='Frequency', 
                    y='Keyword',
                    title="⚠️ Critical Improvement Areas",
                    orientation='h',
                    color='Frequency',
                    color_continuous_scale='Reds'
                )
                fig_neg_words.update_layout(yaxis={'categoryorder':'total ascending'})
                
                top_negative_word = neg_df.iloc[0]['Keyword']
                insight_text = f"**'{top_negative_word}'** is the primary concern driver. Address this systematically through product development, quality assurance, and customer service improvements."
                
                create_chart_with_insights(fig_neg_words, insight_text)
    
    # TAB 5: Risk Assessment
    with tab5:
        st.markdown("## 🚨 Enterprise Risk Assessment")
        
        # Risk metrics dashboard
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            high_risk_count = (filtered_df['fraudFlag'] == 'High Risk').sum()
            st.metric("🚨 High Risk Reviews", high_risk_count, delta=f"{high_risk_count/len(filtered_df)*100:.1f}%")
        
        with col2:
            negative_trend_risk = len(filtered_df[filtered_df['rating'] <= 2]) / len(filtered_df) * 100
            st.metric("📉 Dissatisfaction Risk", f"{negative_trend_risk:.1f}%")
        
        with col3:
            inconsistent_reviews = len(filtered_df[
                ((filtered_df['rating'] >= 4) & (filtered_df['sentiment'].str.contains('Negative', na=False))) |
                ((filtered_df['rating'] <= 2) & (filtered_df['sentiment'].str.contains('Positive', na=False)))
            ])
            st.metric("⚠️ Inconsistent Reviews", inconsistent_reviews)
        
        with col4:
            avg_fraud_score = filtered_df['fraudScore'].mean()
            st.metric("🎯 Average Risk Score", f"{avg_fraud_score:.1f}/10")
        
        # Detailed fraud analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Fraud distribution analysis
            fraud_dist = filtered_df['fraudFlag'].value_counts()
            fig_fraud = px.pie(
                values=fraud_dist.values,
                names=fraud_dist.index,
                title="🔍 Review Authenticity Distribution",
                color_discrete_map={
                    'Legitimate': '#28a745',
                    'Low Risk': '#ffc107', 
                    'Medium Risk': '#fd7e14',
                    'High Risk': '#dc3545'
                }
            )
            
            legitimate_rate = fraud_dist.get('Legitimate', 0) / len(filtered_df) * 100
            insight_text = f"**{legitimate_rate:.1f}% of reviews are verified authentic**. Implement review verification processes for the {100-legitimate_rate:.1f}% flagged as potentially suspicious to maintain platform trust."
            
            create_chart_with_insights(fig_fraud, insight_text)
        
        with col2:
            # Risk trends over time
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
                title="📈 Risk Trend Analysis",
                color_discrete_sequence=['#dc3545']
            )
            
            recent_risk = risk_trends['risk_rate'].tail(3).mean()
            historical_risk = risk_trends['risk_rate'].head(3).mean()
            risk_direction = "increasing" if recent_risk > historical_risk else "decreasing"
            
            insight_text = f"**Platform risk is {risk_direction}** with {abs(recent_risk - historical_risk):.1f}% change. {'Immediate intervention required' if risk_direction == 'increasing' else 'Current risk management is effective'}."
            
            create_chart_with_insights(fig_risk_trends, insight_text)
        
        # Risk mitigation recommendations
        st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
        st.markdown("### 🛡️ **Risk Mitigation Strategy**")
        
        total_risk_rate = (filtered_df['fraudFlag'] != 'Legitimate').sum() / len(filtered_df) * 100
        
        if total_risk_rate > 20:
            st.markdown("🔴 **CRITICAL:** Immediate platform-wide review verification needed")
        elif total_risk_rate > 10:
            st.markdown("🟠 **HIGH:** Enhanced monitoring and selective verification required")
        else:
            st.markdown("🟢 **MODERATE:** Current risk levels manageable with standard protocols")
        
        st.markdown(f"""
        **Immediate Actions:**
        1. Implement automated flagging for top 3 risk factors
        2. Enhanced verification for reviews with >5 risk score
        3. Monitor suspicious customer segments closely
        
        **Strategic Initiatives:**
        1. Develop ML models for real-time risk detection
        2. Implement review authenticity scoring
        3. Create reviewer reputation systems
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Enhanced sidebar with executive summary
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Executive Dashboard")
    st.sidebar.metric("📈 Total Reviews Analyzed", f"{len(filtered_df):,}")
    st.sidebar.metric("⭐ Customer Satisfaction", f"{filtered_df['rating'].mean():.2f}/5.0")
    st.sidebar.metric("🎭 Brand Sentiment", f"{(filtered_df['sentiment'].str.contains('Positive', na=False)).sum()/len(filtered_df)*100:.1f}%")
    st.sidebar.metric("🔍 Trust Score", f"{(filtered_df['fraudFlag'] == 'Legitimate').sum()/len(filtered_df)*100:.1f}%")
    st.sidebar.metric("💼 Avg Business Impact", f"{filtered_df['businessImpact'].mean():.2f}")
    
    # Export functionality
    st.sidebar.markdown("---")
    st.sidebar.subheader("📥 Executive Reports")
    
    if st.sidebar.button("📊 Generate Executive Summary"):
        # Create comprehensive export dataset
        export_df = filtered_df[[
            'reviewId', 'reviewerName', 'reviewText', 'rating', 'reviewDate',
            'sentiment', 'sentimentScore', 'sentimentConfidence', 'emotion',
            'fraudFlag', 'fraudReason', 'fraudScore',
            'topic', 'customerSegment', 'businessImpact', 'reviewValue',
            'wordCount', 'reviewLength'
        ]].copy()
        
        csv_data = export_df.to_csv(index=False)
        
        st.sidebar.download_button(
            label="📄 Download Executive Dataset",
            data=csv_data,
            file_name=f"executive_amazon_reviews_analysis_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        # Generate executive summary report
        executive_summary = create_executive_summary(filtered_df)
        
        st.sidebar.download_button(
            label="📋 Download Executive Summary",
            data=executive_summary,
            file_name=f"executive_summary_{pd.Timestamp.now().strftime('%Y%m%d')}.md",
            mime="text/markdown"
        )

if __name__ == "__main__":
    main()
