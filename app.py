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
    .metrics-explained {
        background: rgba(255,255,255,0.95);
        padding: 2rem;
        border-radius: 15px;
        margin-top: 2rem;
        border: 1px solid #e0e0e0;
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
    
    top_topics = df['topic'].value_counts().head(3)
    
    summary = f"""
    # 🎯 Executive Summary for Business Leaders
    *Simple insights to help you make smart decisions*

    ## 📊 **How Customers Feel About Our Product**
    - **Total Reviews:** {total_reviews:,} customers shared their thoughts.
    - **Average Star Rating:** {avg_rating:.2f}/5.0 ⭐ ({get_satisfaction_grade(avg_rating)}).
    - **Recent Trend:** {rating_trend} compared to past reviews.
    - **Detailed Feedback:** {"Many" if engagement_rate > 30 else "Some" if engagement_rate > 15 else "Few"} customers ({engagement_rate:.1f}%) wrote detailed reviews.
    
    ## 🎭 **What Customers Are Saying**
    - **Happy Customers:** {sentiment_dist.get('Extremely Positive', 0) + sentiment_dist.get('Very Positive', 0) + sentiment_dist.get('Positive', 0):.1f}% love our product.
    - **Neutral Customers:** {sentiment_dist.get('Neutral', 0):.1f}% are okay with it.
    - **Unhappy Customers:** {sentiment_dist.get('Negative', 0) + sentiment_dist.get('Very Negative', 0) + sentiment_dist.get('Extremely Negative', 0):.1f}% are not satisfied.
    
    ## 🔍 **Can We Trust These Reviews?**
    - **Very Suspicious Reviews:** {high_risk_rate:.1f}% might be fake 🚨.
    - **Somewhat Suspicious Reviews:** {medium_risk_rate:.1f}% need checking ⚠️.
    - **Trustworthy Reviews:** {100 - total_risk_rate:.1f}% seem genuine ✅.
    - **Trust Level:** {get_trust_score(total_risk_rate)}/10 (higher is better).
    
    ## 🏆 **Top Customer Topics**
    {chr(10).join([f"**{i+1}. {topic}:** {count:,} reviews ({count/total_reviews*100:.1f}%)" for i, (topic, count) in enumerate(top_topics.items())])}
    
    ## 💡 **What We Should Do Next**
    
    ### 🎯 **Top Priorities**
    - **Customer Happiness:** {get_satisfaction_recommendation(avg_rating)}
    - **Review Trust:** {get_trust_recommendation(total_risk_rate)}
    - **Customer Engagement:** {get_engagement_recommendation(engagement_rate)}
    
    ### 📈 **How We're Doing**
    - **Market Strength:** {get_market_position(avg_rating, sentiment_dist)}
    - **Trustworthiness:** {'High' if total_risk_rate < 10 else 'Medium' if total_risk_rate < 20 else 'Low'}
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

def get_satisfaction_recommendation(rating):
    if rating >= 4.5:
        return "🟢 Keep up the great work - use positive reviews in marketing."
    elif rating >= 4.0:
        return "🟡 Doing well - find ways to make customers even happier."
    elif rating >= 3.0:
        return "🟠 Customers aren't thrilled - check product or service quality."
    else:
        return "🔴 Urgent action needed - fix major customer issues."

def get_trust_recommendation(risk_rate):
    if risk_rate < 5:
        return "🟢 Reviews are mostly genuine - keep monitoring."
    elif risk_rate < 15:
        return "🟡 Some reviews might be fake - verify suspicious ones."
    else:
        return "🔴 Many reviews could be fake - take action to ensure trust."

def get_engagement_recommendation(engagement_rate):
    if engagement_rate > 30:
        return "🟢 Customers are sharing great feedback - keep encouraging it."
    elif engagement_rate > 15:
        return "🟡 Some customers write detailed reviews - ask for more."
    else:
        return "🔴 Few detailed reviews - offer incentives for feedback."

def get_market_position(rating, sentiment_dist):
    positive_rate = sentiment_dist.get('Extremely Positive', 0) + sentiment_dist.get('Very Positive', 0) + sentiment_dist.get('Positive', 0)
    if rating >= 4.5 and positive_rate > 80:
        return "🟢 Top of the Market - Customers love us."
    elif rating >= 4.0 and positive_rate > 65:
        return "🟡 Strong Player - Better than average."
    elif rating >= 3.5:
        return "🟠 Average - Need to stand out more."
    else:
        return "🔴 Falling Behind - Need major improvements."

def process_data_with_advanced_ml(df):
    st.info("🤖 Analyzing reviews with smart algorithms...")
    
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
    st.markdown(f'<div class="chart-insight"><strong>📊 Key Insight:</strong> {insight_text}</div>', 
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
    st.markdown("*Turning customer feedback into easy-to-understand business insights*")
    
    uploaded_file = st.file_uploader("📁 Upload Amazon Reviews Dataset", type=['csv'])
    
    if uploaded_file:
        with st.spinner('🔄 Analyzing data with smart algorithms...'):
            df = load_and_process_data(uploaded_file)
            if df is not None:
                df, topics = process_data_with_advanced_ml(df)
                st.session_state.processed_data = df
                st.session_state.topics = topics
                st.success(f"✅ Analyzed {len(df):,} customer reviews successfully!")
            else:
                st.error("❌ Something went wrong. Please check your CSV file.")
                return
    elif st.session_state.processed_data is not None:
        df = st.session_state.processed_data
        topics = getattr(st.session_state, 'topics', [])
    else:
        st.info("👆 Upload your Amazon reviews dataset to start the analysis")
        st.markdown("""
        ### 🎯 **Smart Analytics Dashboard**
        
        **What This Dashboard Does:**
        - 🧠 Understands if customers are happy or upset.
        - 🔍 Checks if reviews are real or possibly fake.
        - 🏷️ Finds key topics customers talk about.
        - 📈 Tracks trends to help you make better decisions.
        - 💼 Measures how reviews impact your business.
        - 👥 Groups customers by their behavior.
        
        **Data Needed:** A CSV file with Amazon reviews, including review text, star ratings, dates, and reviewer names.
        """)
        return
    
    # Advanced Filtering System
    st.sidebar.header("🎛️ Control Panel")
    
    rating_filter = st.sidebar.multiselect("⭐ Star Ratings", 
                                          sorted(df['rating'].unique()), 
                                          default=sorted(df['rating'].unique()),
                                          help="Choose which star ratings (1-5) to include in the analysis.")
    
    sentiment_filter = st.sidebar.multiselect("😊 Customer Feelings", 
                                             df['sentiment'].unique(), 
                                             default=df['sentiment'].unique(),
                                             help="Select the emotions shown in reviews (e.g., Positive, Negative).")
    
    trust_filter = st.sidebar.selectbox("🔍 Review Trust", 
                                       ['All Reviews', 'Trusted Only', 'Suspicious Only', 'High Risk Only'],
                                       help="Filter reviews by how trustworthy they seem.")
    
    segment_filter = st.sidebar.multiselect("👥 Customer Types", 
                                           df['customerSegment'].unique(), 
                                           default=df['customerSegment'].unique(),
                                           help="Choose which customer groups to analyze (e.g., Satisfied Customers).")
    
    min_impact = st.sidebar.slider("💼 Minimum Business Impact", 
                                  float(df['businessImpact'].min()), 
                                  float(df['businessImpact'].max()), 
                                  float(df['businessImpact'].min()),
                                  help="Set the minimum impact a review has on your business.")
    
    min_confidence = st.sidebar.slider("🎯 Minimum Analysis Confidence", 0.0, 1.0, 0.0, 0.1,
                                      help="Set how certain the analysis should be (0 to 1).")
    
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
        "🎯 Executive Summary", "📊 Business Insights", "🔍 Deep Analysis", 
        "💬 Customer Voices", "🚨 Risk Assessment"
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
            st.metric("📊 Customer Happiness", f"{avg_rating:.2f}/5.0", 
                     delta=f"{(avg_rating - 3.0):.1f} vs neutral",
                     help="Average star rating from all reviews.")
        
        with col2:
            fraud_rate = (filtered_df['fraudFlag'].isin(['High Risk', 'Medium Risk'])).sum() / len(filtered_df) * 100
            st.metric("🔍 Trust Score", f"{100-fraud_rate:.1f}%", 
                     delta=f"-{fraud_rate:.1f}% risk",
                     help="Percentage of reviews that seem genuine.")
        
        with col3:
            positive_sentiment = (filtered_df['sentiment'].str.contains('Positive')).sum() / len(filtered_df) * 100
            st.metric("😊 Positive Feelings", f"{positive_sentiment:.1f}%",
                     help="Percentage of reviews with happy or positive comments.")
        
        with col4:
            engagement_rate = (filtered_df['wordCount'] > 50).sum() / len(filtered_df) * 100
            st.metric("💬 Detailed Reviews", f"{engagement_rate:.1f}%",
                     help="Percentage of reviews that are long and detailed.")
        
        with col5:
            avg_business_impact = filtered_df['businessImpact'].mean()
            st.metric("💼 Business Impact", f"{avg_business_impact:.1f}", 
                     delta="per review",
                     help="Average influence of each review on the business.")
        
        # Strategic Overview Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Enhanced rating distribution
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
                title="🎭 Customer Feelings Breakdown",
                color=sentiment_counts.index,
                color_discrete_map=colors
            )
            
            brand_advocates = sentiment_counts.get('Extremely Positive', 0) + sentiment_counts.get('Very Positive', 0)
            detractors = sentiment_counts.get('Very Negative', 0) + sentiment_counts.get('Extremely Negative', 0)
            nps_proxy = (brand_advocates - detractors) / len(filtered_df) * 100
            
            insight_text = f"**{brand_advocates:,} customers love our product**, while {detractors:,} are unhappy. This gives us a customer happiness score of {nps_proxy:.1f}%. We should focus on keeping happy customers and helping the unhappy ones."
            
            create_chart_with_insights(fig_sentiment, insight_text)
        
        # Metrics Explained
        st.markdown('<div class="metrics-explained">', unsafe_allow_html=True)
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
    
    # TAB 2: Business Intelligence  
    with tab2:
        st.markdown("## 📊 Business Insights")
        st.markdown("*Clear insights to understand customer behavior and key topics*")
        
        col1, col2 = st.columns(2)
        
        with col1:
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
            
            fig_segments = px.scatter(
                segment_analysis, 
                x='Business Impact', 
                y='Average Rating',
                size='Number of Reviews',
                color='Customer Type',
                title="Customer Types Impact vs Happiness",
                hover_data=['Review Value']
            )
            
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
            
            # Topic performance analysis
            if topics:
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
        
        with col2:
            # Time-based business trends
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
                vertical_spacing=0.1
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
            
            fig_trends.update_layout(height=500, title_text="Performance Over Time")
            
            recent_rating = monthly_trends['rating'].tail(3).mean()
            historical_rating = monthly_trends['rating'].head(3).mean()
            trend_direction = "getting better" if recent_rating > historical_rating else "getting worse"
            
            insight_text = f"**Customer happiness is {trend_direction}** with a {abs(recent_rating - historical_rating):.2f} star change. This means we {'should keep up the good work' if trend_direction == 'getting better' else 'need to fix issues quickly'}."
            
            create_chart_with_insights(fig_trends, insight_text)
            
            # Business impact heatmap
            st.markdown("### 🔥 Rating vs Feelings Matrix")
            impact_heatmap = pd.crosstab(filtered_df['rating'], filtered_df['sentiment'])
            sentiment_order = ['Extremely Negative', 'Very Negative', 'Negative', 'Neutral', 
                              'Positive', 'Very Positive', 'Extremely Positive']
            impact_heatmap = impact_heatmap.reindex(columns=sentiment_order, fill_value=0)
            
            # Custom color scale: red for misalignments, green for alignments
            colorscale = []
            for i in range(5):  # Rows (ratings 1 to 5)
                row_colors = []
                for j in range(7):  # Columns (sentiments)
                    rating = i + 1
                    if rating <= 2 and j >= 4:  # Low rating (1,2) with positive sentiment
                        red_intensity = 255 if rating == 1 else 200  # Darker red for rating 1
                        row_colors.append(f'rgb({red_intensity}, 50, 50)')
                    elif rating >= 4 and j <= 2:  # High rating (4,5) with negative sentiment
                        row_colors.append('rgb(200, 50, 50)')  # Red for misalignment
                    else:
                        green_intensity = 100 + (rating * 30) if j >= 4 else 50
                        row_colors.append(f'rgb(50, {green_intensity}, 50)')  # Green for alignment
                colorscale.append(row_colors)
            
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
                yaxis_title="Star Rating"
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
        
        # Metrics Explained
        st.markdown('<div class="metrics-explained">', unsafe_allow_html=True)
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
    
    # TAB 3: Strategic Insights
    with tab3:
        st.markdown("## 🔍 Deep Analysis")
        st.markdown("*Detailed insights to guide big decisions*")
        
        # Advanced topic analysis with business recommendations
        if topics:
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
        
        # Customer journey analysis
        st.subheader("🛤️ Customer Experience Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
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
        
        with col2:
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
        
        # Metrics Explained
        st.markdown('<div class="metrics-explained">', unsafe_allow_html=True)
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
    
    # TAB 4: Voice of Customer
    with tab4:
        st.markdown("## 💬 Customer Voices")
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
        
        # Key themes summary
        st.subheader("🔤 Top Customer Topics")
        if topics:
            topic_summary = filtered_df['topic'].value_counts().head(5)
            st.markdown("Here are the top 5 things customers are talking about:")
            for idx, (topic, count) in enumerate(topic_summary.items(), 1):
                st.markdown(f"**{idx}. {topic}**: Mentioned in {count:,} reviews ({count/len(filtered_df)*100:.1f}%)")
            st.markdown("**What to Do:** Use these topics to improve products, marketing, or customer service.")
        
        # Metrics Explained
        st.markdown('<div class="metrics-explained">', unsafe_allow_html=True)
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
        st.markdown("## 🚨 Risk Assessment")
        st.markdown("*Checking for fake reviews and customer issues*")
        
        # Risk metrics dashboard
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            high_risk_count = (filtered_df['fraudFlag'] == 'High Risk').sum()
            st.metric("🚨 Very Suspicious Reviews", high_risk_count, 
                     delta=f"{high_risk_count/len(filtered_df)*100:.1f}%",
                     help="Number of reviews that seem very likely to be fake.")
        
        with col2:
            negative_trend_risk = len(filtered_df[filtered_df['rating'] <= 2]) / len(filtered_df) * 100
            st.metric("📉 Unhappy Customers", f"{negative_trend_risk:.1f}%",
                     help="Percentage of reviews with low ratings (1-2 stars).")
        
        with col3:
            inconsistent_reviews = len(filtered_df[
                ((filtered_df['rating'] >= 4) & (filtered_df['sentiment'].str.contains('Negative', na=False))) |
                ((filtered_df['rating'] <= 2) & (filtered_df['sentiment'].str.contains('Positive', na=False)))
            ])
            st.metric("⚠️ Mismatched Reviews", inconsistent_reviews,
                     help="Reviews where rating and comments don't match.")
        
        with col4:
            avg_fraud_score = filtered_df['fraudScore'].mean()
            st.metric("🎯 Average Risk Score", f"{avg_fraud_score:.1f}/10",
                     help="Average suspicion level across all reviews.")
        
        # Detailed fraud analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Fraud distribution analysis
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
                title="📈 Suspicious Reviews Over Time",
                color_discrete_sequence=['#dc3545']
            )
            
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
        st.markdown('<div class="metrics-explained">', unsafe_allow_html=True)
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
    
    # Enhanced sidebar with executive summary
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Quick Stats")
    st.sidebar.metric("📈 Total Reviews", f"{len(filtered_df):,}",
                     help="Number of reviews analyzed.")
    st.sidebar.metric("⭐ Average Rating", f"{filtered_df['rating'].mean():.2f}/5.0",
                     help="Average star rating from all reviews.")
    st.sidebar.metric("😊 Positive Feelings", f"{(filtered_df['sentiment'].str.contains('Positive', na=False)).sum()/len(filtered_df)*100:.1f}%",
                     help="Percentage of happy reviews.")
    st.sidebar.metric("🔍 Trust Score", f"{(filtered_df['fraudFlag'] == 'Legitimate').sum()/len(filtered_df)*100:.1f}%",
                     help="Percentage of genuine reviews.")
    st.sidebar.metric("💼 Business Impact", f"{filtered_df['businessImpact'].mean():.2f}",
                     help="Average influence per review.")
    
    # Export functionality
    st.sidebar.markdown("---")
    st.sidebar.subheader("📥 Download Reports")
    
    if st.sidebar.button("📊 Create Reports"):
        export_df = filtered_df[[
            'reviewId', 'reviewerName', 'reviewText', 'rating', 'reviewDate',
            'sentiment', 'sentimentScore', 'sentimentConfidence', 'emotion',
            'fraudFlag', 'fraudReason', 'fraudScore',
            'topic', 'customerSegment', 'businessImpact', 'reviewValue',
            'wordCount', 'reviewLength'
        ]].copy()
        
        csv_data = export_df.to_csv(index=False)
        
        st.sidebar.download_button(
            label="📄 Download Full Data",
            data=csv_data,
            file_name=f"amazon_reviews_analysis_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        executive_summary = create_executive_summary(filtered_df)
        
        st.sidebar.download_button(
            label="📋 Download Summary",
            data=executive_summary,
            file_name=f"executive_summary_{pd.Timestamp.now().strftime('%Y%m%d')}.md",
            mime="text/markdown"
        )

if __name__ == "__main__":
    main()
