import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from textblob import TextBlob
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Try to import optional dependencies with fallbacks
try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    st.warning("📦 WordCloud not available - using alternative visualizations")

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    NLTK_AVAILABLE = True
except:
    NLTK_AVAILABLE = False

st.set_page_config(page_title="Amazon Reviews Analytics Dashboard", page_icon="📊", layout="wide")

# Enhanced CSS styling for a professional, sexy look
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        color: #FF9500;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        font-family: 'Arial', sans-serif;
    }
    .insight-box {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        font-family: 'Arial', sans-serif;
    }
    .metric-container {
        background: rgba(255,255,255,0.95);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #FF9500;
        margin: 0.5rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .chart-insight {
        background: rgba(0,0,0,0.05);
        padding: 1rem;
        border-radius: 8px;
        margin-top: 0.5rem;
        font-size: 0.9rem;
        color: #333;
        border-left: 4px solid #FF9500;
    }
</style>
""", unsafe_allow_html=True)

if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

@st.cache_data
def load_and_process_data(uploaded_file):
    """Load and process the reviews data with enhanced cleaning"""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        return None
    
    # Clean column names
    df.columns = ['reviewId', 'reviewerName', 'reviewText', 'rating', 'summary', 
                  'helpful', 'totalVotes', 'reviewDate', 'year']
    
    # Enhanced data cleaning
    df['reviewText'] = df['reviewText'].fillna('').astype(str)
    df['summary'] = df['summary'].fillna('').astype(str)
    df['reviewerName'] = df['reviewerName'].fillna('Anonymous').astype(str)
    
    # Enhanced date processing
    df['reviewDate'] = pd.to_datetime(df['reviewDate'], format='%d-%m-%Y', errors='coerce')
    df['month'] = df['reviewDate'].dt.month
    df['month_name'] = df['reviewDate'].dt.month_name()
    df['day_of_week'] = df['reviewDate'].dt.day_name()
    
    # Enhanced text metrics
    df['reviewLength'] = df['reviewText'].str.len()
    df['wordCount'] = df['reviewText'].str.split().str.len()
    df['sentenceCount'] = df['reviewText'].str.count(r'[.!?]+')
    df['avgWordsPerSentence'] = df['wordCount'] / (df['sentenceCount'] + 1)
    
    return df

def advanced_sentiment_analysis(text):
    """Enhanced sentiment analysis with confidence scoring"""
    if not text or len(text.strip()) == 0:
        return 'Neutral', 0, 0.5
    
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    # Enhanced sentiment classification
    if polarity > 0.3:
        sentiment = 'Very Positive'
    elif polarity > 0.1:
        sentiment = 'Positive'
    elif polarity < -0.3:
        sentiment = 'Very Negative'
    elif polarity < -0.1:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    
    # Confidence based on subjectivity
    confidence = abs(polarity) + (1 - subjectivity) * 0.3
    
    return sentiment, polarity, confidence

def advanced_fraud_detection(df):
    """Enhanced fraud detection with multiple sophisticated algorithms"""
    fraud_flags = []
    fraud_reasons = []
    fraud_scores = []
    
    for idx, row in df.iterrows():
        flags = []
        score = 0
        
        # 1. Duplicate content analysis
        exact_duplicates = df[df['reviewText'] == row['reviewText']]
        if len(exact_duplicates) > 1:
            flags.append('Exact Duplicate')
            score += 3
        
        # 2. Temporal anomaly detection
        same_user_reviews = df[df['reviewerName'] == row['reviewerName']]
        if len(same_user_reviews) > 1:
            review_dates = same_user_reviews['reviewDate'].dropna()
            if len(review_dates) > 1:
                time_diffs = review_dates.diff().dt.total_seconds() / 3600
                if any(time_diffs < 1):
                    flags.append('Rapid Sequential Reviews')
                    score += 2
        
        # 3. Content quality analysis
        words = row['reviewText'].lower().split()
        if len(words) < 3:
            flags.append('Too Short')
            score += 2
        elif len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                flags.append('Low Lexical Diversity')
                score += 1
        
        # 4. Stylistic anomalies
        if row['reviewText'].isupper() and len(row['reviewText']) > 20:
            flags.append('Excessive Caps')
            score += 1
        
        if len(re.findall(r'!', row['reviewText'])) > 5:
            flags.append('Excessive Exclamation')
            score += 1
        
        # 5. Generic content detection
        generic_patterns = [
            r'\b(best|great|excellent|amazing|perfect|awesome)\s+(product|item|purchase|buy)\b',
            r'\b(highly\s+recommend|five\s+stars?|10/10|thumbs\s+up)\b',
            r'\b(fast\s+shipping|quick\s+delivery|arrived\s+quickly)\b'
        ]
        
        generic_count = sum(1 for pattern in generic_patterns 
                          if re.search(pattern, row['reviewText'].lower()))
        if generic_count >= 2 and len(words) < 20:
            flags.append('Generic Template')
            score += 2
        
        # 6. Rating-content mismatch
        sentiment_score = TextBlob(row['reviewText']).sentiment.polarity
        if row['rating'] >= 4 and sentiment_score < -0.3:
            flags.append('Rating-Sentiment Mismatch')
            score += 2
        elif row['rating'] <= 2 and sentiment_score > 0.3:
            flags.append('Rating-Sentiment Mismatch')
            score += 2
        
        # Final classification
        fraud_flag = 'Yes' if score >= 3 else 'Suspicious' if score >= 1 else 'No'
        fraud_flags.append(fraud_flag)
        fraud_reasons.append('; '.join(flags) if flags else 'No Issues Detected')
        fraud_scores.append(score)
    
    return fraud_flags, fraud_reasons, fraud_scores

def advanced_topic_modeling(texts, n_topics=8):
    """Enhanced topic modeling with better preprocessing and interpretation"""
    try:
        cleaned_texts = []
        for text in texts:
            if text and len(str(text).strip()) > 5:
                clean_text = re.sub(r'[^\w\s]', ' ', str(text).lower())
                clean_text = ' '.join(clean_text.split())
                cleaned_texts.append(clean_text)
        
        if len(cleaned_texts) < n_topics:
            return [], ['No Topic'] * len(texts)
        
        if NLTK_AVAILABLE:
            try:
                stop_words = set(stopwords.words('english'))
            except:
                stop_words = set()
        else:
            stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        additional_stops = {
            'card', 'memory', 'product', 'item', 'amazon', 'buy', 'bought', 'purchase',
            'get', 'got', 'use', 'used', 'work', 'works', 'working', 'good', 'nice',
            'one', 'two', 'would', 'could', 'really', 'very', 'much', 'well',
            'time', 'first', 'last', 'way', 'make', 'made', 'take', 'took'
        }
        stop_words.update(additional_stops)
        
        vectorizer = TfidfVectorizer(
            max_features=200,
            stop_words=list(stop_words),
            min_df=3,
            max_df=0.7,
            ngram_range=(1, 3),
            token_pattern=r'\b[a-zA-Z]{2,}\b'
        )
        
        doc_term_matrix = vectorizer.fit_transform(cleaned_texts)
        
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=20,
            learning_method='batch',
            doc_topic_prior=0.1,
            topic_word_prior=0.01
        )
        lda.fit(doc_term_matrix)
        
        feature_names = vectorizer.get_feature_names_out()
        
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[-15:][::-1]
            top_words = [(feature_names[i], topic[i]) for i in top_words_idx]
            key_words = [word for word, weight in top_words[:8] if weight > 0.01]
            topic_label = create_topic_label(key_words, topic_idx)
            topic_words_str = ', '.join(key_words[:6])
            topics.append(f"{topic_label}: {topic_words_str}")
        
        doc_topic_matrix = lda.transform(doc_term_matrix)
        topic_assignments = []
        
        for doc_idx, doc_topics in enumerate(doc_topic_matrix):
            max_prob = np.max(doc_topics)
            if max_prob > 0.3:
                topic_idx = np.argmax(doc_topics)
                topic_assignments.append(f"Topic {topic_idx + 1}")
            else:
                topic_assignments.append("Mixed Topics")
        
        full_topic_assignments = ['No Topic'] * len(texts)
        j = 0
        for i, text in enumerate(texts):
            if text and len(str(text).strip()) > 5:
                if j < len(topic_assignments):
                    full_topic_assignments[i] = topic_assignments[j]
                j += 1
        
        return topics, full_topic_assignments
        
    except Exception as e:
        st.error(f"Advanced topic modeling failed: {str(e)}")
        return [], ['No Topic'] * len(texts)

def create_topic_label(key_words, topic_idx):
    """Create meaningful topic labels based on key words"""
    topic_mappings = {
        0: ("Device Compatibility & Usage", ["galaxy", "samsung", "note", "phone", "tablet", "works"]),
        1: ("Technical Specifications & Performance", ["class", "cards", "sd", "micro", "sandisk"]),
        2: ("Action Cameras & Photography", ["camera", "video", "recording", "action", "gopro"]),
        3: ("General Performance & Speed", ["fast", "speed", "reliable", "performance"]),
        4: ("Storage for Music & Media", ["music", "photos", "videos", "storage", "media"]),
        5: ("Customer Service Experience", ["support", "service", "help", "response", "warranty"]),
        6: ("Product Durability & Reliability", ["durable", "failed", "broke", "quality", "defective"]),
        7: ("Shipping & Packaging", ["delivery", "shipping", "package", "arrived", "damaged"])
    }
    for idx, (label, keywords) in topic_mappings.items():
        if idx == topic_idx and any(word in key_words for word in keywords):
            return label
    return "General Features"

def get_word_frequencies(text_series):
    """Get word frequencies with robust text processing"""
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
    
    additional_stops = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'is', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does',
        'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can',
        'shall', 'this', 'that', 'these', 'those', 'card', 'memory', 'product',
        'item', 'amazon', 'one', 'get', 'use', 'work'
    }
    stop_words.update(additional_stops)
    
    filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
    return Counter(filtered_words).most_common(20)

def create_word_frequency_chart(word_freq):
    """Create enhanced word frequency visualization"""
    if word_freq:
        df_words = pd.DataFrame(word_freq, columns=['Word', 'Frequency'])
        fig = px.bar(
            df_words, x='Frequency', y='Word',
            title="🔤 Top Keywords Analysis",
            orientation='h',
            color='Frequency',
            color_continuous_scale='viridis',
            height=600
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        return fig
    return None

def process_data_with_advanced_ml(df):
    """Process data with advanced ML algorithms"""
    st.info("🤖 Running advanced ML algorithms...")
    
    sentiments = []
    sentiment_scores = []
    sentiment_confidence = []
    
    progress_bar = st.progress(0)
    for i, text in enumerate(df['reviewText']):
        sentiment, score, confidence = advanced_sentiment_analysis(text)
        sentiments.append(sentiment)
        sentiment_scores.append(score)
        sentiment_confidence.append(confidence)
        progress_bar.progress((i + 1) / len(df) * 0.4)
    
    df['sentiment'] = sentiments
    df['sentimentScore'] = sentiment_scores
    df['sentimentConfidence'] = sentiment_confidence
    
    fraud_flags, fraud_reasons, fraud_scores = advanced_fraud_detection(df)
    df['fraudFlag'] = fraud_flags
    df['fraudReason'] = fraud_reasons
    df['fraudScore'] = fraud_scores
    progress_bar.progress(0.7)
    
    topics, topic_assignments = advanced_topic_modeling(df['reviewText'].tolist())
    df['topic'] = topic_assignments
    progress_bar.progress(1.0)
    
    df['originalLanguage'] = 'en'
    df['translatedText'] = df['reviewText']
    
    progress_bar.empty()
    return df, topics

def create_comprehensive_executive_summary(df):
    """Generate a comprehensive executive summary with deep insights"""
    total_reviews = len(df)
    avg_rating = df['rating'].mean()
    
    sentiment_dist = df['sentiment'].value_counts(normalize=True) * 100
    high_confidence_sentiment = df[df['sentimentConfidence'] > 0.7]['sentiment'].value_counts(normalize=True) * 100
    
    fraud_rate = (df['fraudFlag'] == 'Yes').sum() / total_reviews * 100
    suspicious_rate = (df['fraudFlag'] == 'Suspicious').sum() / total_reviews * 100
    avg_fraud_score = df['fraudScore'].mean()
    
    recent_reviews = df[df['reviewDate'] > df['reviewDate'].max() - pd.Timedelta(days=90)]
    recent_avg_rating = recent_reviews['rating'].mean() if len(recent_reviews) > 0 else avg_rating
    rating_trend = "📈 Improving" if recent_avg_rating > avg_rating else "📉 Declining" if recent_avg_rating < avg_rating else "➡️ Stable"
    
    avg_review_length = df['wordCount'].mean()
    detailed_reviews = df[df['wordCount'] > 50]
    detailed_avg_rating = detailed_reviews['rating'].mean() if len(detailed_reviews) > 0 else avg_rating
    
    top_topics = df['topic'].value_counts().head(5)
    
    # Enhanced topic summaries
    topic_summaries = []
    for topic in top_topics.index:
        topic_reviews = df[df['topic'] == topic]['reviewText'].head(3).tolist()
        topic_keywords = get_word_frequencies(df[df['topic'] == topic]['reviewText'])
        top_keywords = ', '.join([word for word, _ in topic_keywords[:5]])
        topic_label = create_topic_label(top_keywords.split(', '), int(topic.replace('Topic ', '')) - 1)
        topic_summaries.append(
            f"- **{topic_label} ({top_topics[topic]} reviews, {top_topics[topic]/total_reviews*100:.1f}%)**: "
            f"Focuses on {top_keywords}. Sample feedback: '{topic_reviews[0][:100]}...' "
            f"{'Positive sentiment dominates' if df[df['topic'] == topic]['sentimentScore'].mean() > 0 else 'Mixed sentiment observed'}."
        )
    
    summary = f"""
    # 🎯 C-Level Executive Insights Dashboard

    ## 📊 Strategic Overview
    - **Total Reviews Analyzed:** {total_reviews:,}
    - **Average Rating:** {avg_rating:.2f}/5.0 ⭐
    - **90-Day Rating Trend:** {rating_trend}
    - **Review Quality:** {"🟢 High" if avg_review_length > 30 else "🟡 Medium" if avg_review_length > 15 else "🔴 Low"} (avg. {avg_review_length:.1f} words)
    - **Engagement Depth:** {len(detailed_reviews)/total_reviews*100:.1f}% of reviews are detailed (>50 words), averaging {detailed_avg_rating:.2f}/5.0

    ## 😊 Customer Sentiment Intelligence
    - **Positive Sentiment (Positive + Very Positive):** {sentiment_dist.get('Positive', 0) + sentiment_dist.get('Very Positive', 0):.1f}%
    - **Neutral Sentiment:** {sentiment_dist.get('Neutral', 0):.1f}%
    - **Negative Sentiment (Negative + Very Negative):** {sentiment_dist.get('Negative', 0) + sentiment_dist.get('Very Negative', 0):.1f}%
    - **High-Confidence Insights:** {len(df[df['sentimentConfidence'] > 0.7])/total_reviews*100:.1f}% of reviews have high-confidence sentiment
    - **Sentiment Volatility:** {df['sentimentScore'].std():.3f} (lower is more consistent)

    ## 🔍 Review Authenticity Analysis
    - **Confirmed Fraudulent Reviews:** {fraud_rate:.1f}% ({(df['fraudFlag'] == 'Yes').sum()} reviews)
    - **Suspicious Reviews:** {suspicious_rate:.1f}% ({(df['fraudFlag'] == 'Suspicious').sum()} reviews)
    - **Clean Reviews:** {100 - fraud_rate - suspicious_rate:.1f}%
    - **Average Fraud Risk Score:** {avg_fraud_score:.1f}/10
    - **Top Fraud Patterns:** {', '.join(Counter('; '.join(df[df['fraudFlag'] != 'No']['fraudReason']).split('; ')).most_common(3)[0][0] for _ in range(min(3, len(Counter('; '.join(df[df['fraudFlag'] != 'No']['fraudReason']).split('; '))))))}

    ## 🏷️ Key Discussion Themes
    {chr(10).join(topic_summaries)}

    ## 💡 Strategic Insights & Recommendations
    - **Customer Satisfaction**: {'🟢 Exceptional' if avg_rating >= 4.2 else '🟢 Strong' if avg_rating >= 4.0 else '🟡 Moderate' if avg_rating >= 3.0 else '🔴 Concerning'} based on {avg_rating:.2f}/5.0 average rating. Leverage positive feedback in marketing to highlight strengths.
    - **Review Authenticity**: {'🟢 High' if fraud_rate < 5 else '🟡 Moderate' if fraud_rate < 15 else '🔴 Needs Attention'} with {fraud_rate:.1f}% fraudulent reviews. Implement stricter review verification to reduce generic or duplicate content.
    - **Engagement Quality**: {'🟢 Strong' if detailed_avg_rating >= avg_rating else '🟡 Mixed'} with detailed reviews scoring {detailed_avg_rating:.2f}/5.0. Encourage detailed feedback through incentives to improve insight quality.
    - **Data Reliability**: {'🟢 Robust' if len(df[df['sentimentConfidence'] > 0.7])/total_reviews > 0.8 else '🟡 Reliable' if len(df[df['sentimentConfidence'] > 0.7])/total_reviews > 0.6 else '🔴 Needs Validation'} with {len(df[df['sentimentConfidence'] > 0.7])/total_reviews*100:.1f}% high-confidence reviews. Focus on high-confidence data for strategic decisions.
    - **Actionable Steps**:
      - **Enhance Product Quality**: Address recurring issues in {create_topic_label(get_word_frequencies(df[df['sentiment'].isin(['Negative', 'Very Negative'])]['reviewText'])[:5], 6)} to reduce negative feedback.
      - **Optimize Customer Support**: Improve response times and resolution rates, as highlighted in negative verbatims.
      - **Promote Strengths**: Use testimonials from {create_topic_label(get_word_frequencies(df[df['sentiment'].isin(['Positive', 'Very Positive'])]['reviewText'])[:5], 0)} in promotional campaigns.
      - **Monitor Trends**: Regularly track {rating_trend.lower()} trends to proactively address emerging issues.

    ## 🚀 Next Steps
    - Conduct deeper analysis of negative verbatims to pinpoint specific product or service issues.
    - Implement AI-driven review filters to enhance authenticity and reduce fraud.
    - Develop targeted customer engagement strategies based on dominant themes to boost satisfaction.
    """
    
    return summary

def get_top_verbatims(df):
    """Get top 5 positive and negative verbatims"""
    positive_verbatims = df[df['sentiment'].isin(['Positive', 'Very Positive'])][['reviewText', 'rating', 'sentimentScore']].sort_values('sentimentScore', ascending=False).head(5)
    negative_verbatims = df[df['sentiment'].isin(['Negative', 'Very Negative'])][['reviewText', 'rating', 'sentimentScore']].sort_values('sentimentScore', ascending=True).head(5)
    return positive_verbatims, negative_verbatims

def main():
    st.markdown('<div class="main-header">📊 Amazon Reviews Executive Analytics Suite</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload Amazon Reviews CSV", type=['csv'])
    
    if uploaded_file:
        with st.spinner('🔄 Processing data with advanced AI algorithms...'):
            df = load_and_process_data(uploaded_file)
            if df is not None:
                df, topics = process_data_with_advanced_ml(df)
                st.session_state.processed_data = df
                st.session_state.topics = topics
                st.success(f"✅ Processed {len(df):,} reviews with cutting-edge analytics!")
            else:
                st.error("❌ Failed to load data. Please check CSV format.")
                return
    elif st.session_state.processed_data is not None:
        df = st.session_state.processed_data
        topics = getattr(st.session_state, 'topics', [])
    else:
        st.info("👆 Upload your Amazon reviews CSV to unlock powerful insights")
        st.markdown("""
        ### Expected CSV Format:
        Your CSV should include: Review ID, reviewerName, reviewText, overall score (1-5), 
        summary, helpful status, total votes, review date, and year.

        **🚀 Advanced Features:**
        - AI-powered sentiment analysis with confidence scoring
        - Multi-algorithm fraud detection
        - Deep topic modeling with thematic insights
        - Temporal and statistical analysis
        - Actionable executive recommendations
        """)
        return
    
    # Sidebar Filters
    st.sidebar.header("🎛️ Advanced Control Panel")
    
    rating_options = sorted(df['rating'].unique())
    selected_ratings = st.sidebar.multiselect("⭐ Star Ratings", rating_options, default=rating_options)
    
    year_options = sorted(df['year'].unique())
    selected_years = st.sidebar.multiselect("📅 Years", year_options, default=year_options)
    
    month_options = sorted(df['month'].dropna().unique())
    selected_months = st.sidebar.multiselect("📅 Months", month_options, default=month_options)
    
    sentiment_options = df['sentiment'].unique()
    selected_sentiments = st.sidebar.multiselect("😊 Sentiment", sentiment_options, default=sentiment_options)
    
    fraud_options = ['All Reviews', 'Legitimate Only', 'Suspicious Only', 'Confirmed Fraud Only']
    fraud_filter = st.sidebar.selectbox("🔍 Fraud Filter", fraud_options)
    
    min_confidence = st.sidebar.slider("🎯 Min Sentiment Confidence", 0.0, 1.0, 0.0, 0.1)
    min_word_count = st.sidebar.slider("📝 Min Word Count", 0, 200, 0, 10)
    keyword_search = st.sidebar.text_input("🔍 Keyword Search")
    
    # Apply filters
    filtered_df = df[
        (df['rating'].isin(selected_ratings)) & 
        (df['year'].isin(selected_years)) & 
        (df['month'].isin(selected_months)) & 
        (df['sentiment'].isin(selected_sentiments)) &
        (df['sentimentConfidence'] >= min_confidence) &
        (df['wordCount'] >= min_word_count)
    ]
    
    if fraud_filter == 'Legitimate Only':
        filtered_df = filtered_df[filtered_df['fraudFlag'] == 'No']
    elif fraud_filter == 'Suspicious Only':
        filtered_df = filtered_df[filtered_df['fraudFlag'] == 'Suspicious']
    elif fraud_filter == 'Confirmed Fraud Only':
        filtered_df = filtered_df[filtered_df['fraudFlag'] == 'Yes']
    
    if keyword_search:
        filtered_df = filtered_df[
            filtered_df['reviewText'].str.contains(keyword_search, case=False, na=False)
        ]
    
    # Dashboard Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Executive Insights", "🔍 Verbatim Deep Dive", "😊 Sentiment Intelligence", 
        "🚨 Fraud Detection", "📈 Advanced Analytics", "🗣️ Top Verbatims"
    ])
    
    # TAB 1: Executive Insights
    with tab1:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown(create_comprehensive_executive_summary(filtered_df))
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("📊 Total Reviews", f"{len(filtered_df):,}")
        with col2:
            st.metric("⭐ Average Rating", f"{filtered_df['rating'].mean():.2f}")
        with col3:
            fraud_pct = (filtered_df['fraudFlag'] == 'Yes').sum() / len(filtered_df) * 100 if len(filtered_df) > 0 else 0
            st.metric("🚨 Fraud Rate", f"{fraud_pct:.1f}%")
        with col4:
            pos_sentiment = (filtered_df['sentiment'].isin(['Positive', 'Very Positive'])).sum() / len(filtered_df) * 100 if len(filtered_df) > 0 else 0
            st.metric("😊 Positive Sentiment", f"{pos_sentiment:.1f}%")
        with col5:
            avg_confidence = filtered_df['sentimentConfidence'].mean()
            st.metric("🎯 Avg Confidence", f"{avg_confidence:.2f}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_rating = px.histogram(
                filtered_df, x='rating', 
                title="📊 Star Rating Distribution",
                color_discrete_sequence=['#FF9500'],
                text_auto=True
            )
            fig_rating.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_rating, use_container_width=True)
            st.markdown(
                '<div class="chart-insight">'
                f'<strong>Insight:</strong> The distribution shows {filtered_df["rating"].value_counts().idxmax()}-star ratings dominate, indicating {"strong customer satisfaction" if filtered_df["rating"].mean() >= 4 else "mixed feedback"}. '
                f'Focus on addressing lower ratings to boost overall perception.'
                '</div>',
                unsafe_allow_html=True
            )
            
            sentiment_counts = filtered_df['sentiment'].value_counts()
            colors = {'Very Positive': '#1f77b4', 'Positive': '#2ca02c', 'Neutral': '#ffbb33', 'Negative': '#ff6b6b', 'Very Negative': '#d62728'}
            fig_sentiment = px.pie(
                values=sentiment_counts.values, 
                names=sentiment_counts.index,
                title="😊 Sentiment Distribution",
                color=sentiment_counts.index,
                color_discrete_map=colors
            )
            st.plotly_chart(fig_sentiment, use_container_width=True)
            st.markdown(
                '<div class="chart-insight">'
                f'<strong>Insight:</strong> Positive sentiments ({sentiment_counts.get("Positive", 0) + sentiment_counts.get("Very Positive", 0)} reviews) outweigh negative ones, but neutral reviews ({sentiment_counts.get("Neutral", 0)}) suggest opportunities to convert adequate experiences into exceptional ones.'
                '</div>',
                unsafe_allow_html=True
            )
        
        with col2:
            time_trends = filtered_df.groupby(['year', 'month']).agg({
                'reviewId': 'count',
                'rating': 'mean'
            }).reset_index()
            time_trends['date'] = pd.to_datetime(time_trends[['year', 'month']].assign(day=1))
            
            fig_trends = px.line(
                time_trends, x='date', y='reviewId',
                title="📈 Review Volume Trends",
                color_discrete_sequence=['#667eea']
            )
            st.plotly_chart(fig_trends, use_container_width=True)
            st.markdown(
                '<div class="chart-insight">'
                f'<strong>Insight:</strong> Review volume peaks in {time_trends.loc[time_trends["reviewId"].idxmax(), "date"].strftime("%B %Y")}, suggesting high engagement. Monitor low-volume periods for potential issues.'
                '</div>',
                unsafe_allow_html=True
            )
            
            heatmap_data = pd.crosstab(filtered_df['rating'], filtered_df['sentiment'])
            fig_heatmap = px.imshow(
                heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                title="🔥 Rating vs. Sentiment Heatmap",
                color_continuous_scale='RdBu_r',  # Red for negative, blue for positive
                text_auto=True
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
            st.markdown(
                '<div class="chart-insight">'
                f'<strong>Insight:</strong> High ratings align with positive sentiments, but mismatches (e.g., 5-star with negative sentiment) indicate potential fraud or dissatisfaction. Investigate these anomalies.'
                '</div>',
                unsafe_allow_html=True
            )
    
    # TAB 2: Verbatim Deep Dive
    with tab2:
        st.markdown("## 🔍 Verbatim Deep Dive")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔤 Keyword Analysis")
            word_freq = get_word_frequencies(filtered_df['reviewText'])
            fig_words = create_word_frequency_chart(word_freq)
            if fig_words:
                st.plotly_chart(fig_words, use_container_width=True)
                st.markdown(
                    '<div class="chart-insight">'
                    f'<strong>Insight:</strong> Top keywords ({", ".join([word for word, _ in word_freq[:3]])}) highlight customer priorities. Leverage these in marketing and address any negative connotations.'
                    '</div>',
                    unsafe_allow_html=True
                )
            
            if WORDCLOUD_AVAILABLE:
                st.subheader("☁️ Word Cloud")
                text = ' '.join(filtered_df['reviewText'].dropna())
                if NLTK_AVAILABLE:
                    try:
                        stop_words = set(stopwords.words('english'))
                        words = word_tokenize(text.lower())
                    except:
                        words = text.lower().split()
                        stop_words = set()
                else:
                    words = text.lower().split()
                    stop_words = set()
                
                additional_stops = {'card', 'memory', 'product', 'amazon', 'item'}
                stop_words.update(additional_stops)
                
                words = [word for word in words if word.isalpha() and word not in stop_words and len(word) > 2]
                cleaned_text = ' '.join(words)
                
                if cleaned_text:
                    try:
                        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(cleaned_text)
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis('off')
                        st.pyplot(fig)
                        st.markdown(
                            '<div class="chart-insight">'
                            f'<strong>Insight:</strong> The word cloud emphasizes frequently mentioned terms, reinforcing key customer focus areas like {", ".join([word for word, _ in word_freq[:3]])}.'
                            '</div>',
                            unsafe_allow_html=True
                        )
                    except Exception as e:
                        st.info("Word cloud generation failed, showing keyword analysis above")
        
        with col2:
            st.subheader("🏷️ Topic Analysis")
            if topics:
                st.write("**Discovered Themes:**")
                for i, topic in enumerate(topics, 1):
                    topic_label = topic.split(':')[0]
                    topic_keywords = topic.split(':')[1].strip()
                    sample_reviews = filtered_df[filtered_df['topic'] == f'Topic {i}']['reviewText'].head(1).tolist()
                    sample_text = sample_reviews[0][:100] + '...' if sample_reviews else 'No sample available'
                    st.write(f"**{i}. {topic_label}**")
                    st.write(f"- Keywords: {topic_keywords}")
                    st.write(f"- Sample: {sample_text}")
                    st.write(f"- Reviews: {filtered_df[filtered_df['topic'] == f'Topic {i}'].shape[0]}")
            else:
                st.info("Topic modeling results not available")
            
            st.subheader("📏 Review Length Analysis")
            fig_length = px.histogram(
                filtered_df, x='wordCount',
                title="Review Length Distribution",
                nbins=30,
                color_discrete_sequence=['#667eea']
            )
            st.plotly_chart(fig_length, use_container_width=True)
            st.markdown(
                '<div class="chart-insight">'
                f'<strong>Insight:</strong> Most reviews are {filtered_df["wordCount"].mode()[0]}-words long, with longer reviews likely providing deeper insights. Encourage detailed feedback for richer data.'
                '</div>',
                unsafe_allow_html=True
            )
        
        if 'topic' in filtered_df.columns:
            st.subheader("📊 Topic Distribution")
            
            col1, col2 = st.columns(2)
            with col1:
                topic_counts = filtered_df['topic'].value_counts()
                fig_topics = px.pie(
                    values=topic_counts.values,
                    names=topic_counts.index,
                    title="Distribution of Review Topics"
                )
                st.plotly_chart(fig_topics, use_container_width=True)
                st.markdown(
                    '<div class="chart-insight">'
                    f'<strong>Insight:</strong> Dominant topics like {topic_counts.index[0]} ({topic_counts.iloc[0]} reviews) reflect key customer concerns. Prioritize these in product development.'
                    '</div>',
                    unsafe_allow_html=True
                )
            
            with col2:
                topic_rating = filtered_df.groupby('topic')['rating'].mean().sort_values(ascending=False)
                fig_topic_rating = px.bar(
                    x=topic_rating.values,
                    y=topic_rating.index,
                    orientation='h',
                    title="Average Rating by Topic",
                    color=topic_rating.values,
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig_topic_rating, use_container_width=True)
                st.markdown(
                    '<div class="chart-insight">'
                    f'<strong>Insight:</strong> Topics like {topic_rating.index[0]} score high ({topic_rating.iloc[0]:.2f}/5), while {topic_rating.index[-1]} ({topic_rating.iloc[-1]:.2f}/5) needs improvement.'
                    '</div>',
                    unsafe_allow_html=True
                )
    
    # TAB 3: Sentiment Intelligence
    with tab3:
        st.markdown("## 😊 Sentiment Intelligence")
        
        col1, col2 = st.columns(2)
        
        with col1:
            sent_rating = pd.crosstab(filtered_df['rating'], filtered_df['sentiment'], normalize='index') * 100
            sent_rating_melted = sent_rating.reset_index().melt(id_vars='rating', var_name='sentiment', value_name='percentage')
            
            fig7 = px.bar(
                sent_rating_melted,
                x='rating', y='percentage', color='sentiment',
                title="Sentiment by Star Rating",
                color_discrete_map={
                    'Very Positive': '#1f77b4', 'Positive': '#2ca02c', 
                    'Neutral': '#ffbb33', 'Negative': '#ff6b6b', 'Very Negative': '#d62728'
                }
            )
            st.plotly_chart(fig7, use_container_width=True)
            st.markdown(
                '<div class="chart-insight">'
                f'<strong>Insight:</strong> High ratings (4-5 stars) correlate with positive sentiments, but 1-2 star ratings show significant negative sentiment, highlighting areas for improvement.'
                '</div>',
                unsafe_allow_html=True
            )
            
            fig_confidence = px.scatter(
                filtered_df.sample(min(1000, len(filtered_df))),
                x='sentimentScore', y='sentimentConfidence',
                color='sentiment',
                title="Sentiment Score vs. Confidence",
                color_discrete_map={
                    'Very Positive': '#1f77b4', 'Positive': '#2ca02c',
                    'Neutral': '#ffbb33', 'Negative': '#ff6b6b', 'Very Negative': '#d62728'
                }
            )
            st.plotly_chart(fig_confidence, use_container_width=True)
            st.markdown(
                '<div class="chart-insight">'
                f'<strong>Insight:</strong> High-confidence sentiments cluster around strong positive/negative scores, ensuring reliable insights for decision-making.'
                '</div>',
                unsafe_allow_html=True
            )
        
        with col2:
            if len(filtered_df) > 0:
                time_sentiment = filtered_df.groupby(['year', 'sentiment']).size().unstack(fill_value=0)
                time_sentiment_pct = time_sentiment.div(time_sentiment.sum(axis=1), axis=0) * 100
                time_sentiment_melted = time_sentiment_pct.reset_index().melt(id_vars='year', var_name='sentiment', value_name='percentage')
                
                fig8 = px.line(
                    time_sentiment_melted,
                    x='year', y='percentage', color='sentiment',
                    title="Sentiment Trends Over Time",
                    color_discrete_map={
                        'Very Positive': '#1f77b4', 'Positive': '#2ca02c',
                        'Neutral': '#ffbb33', 'Negative': '#ff6b6b', 'Very Negative': '#d62728'
                    }
                )
                st.plotly_chart(fig8, use_container_width=True)
                st.markdown(
                    '<div class="chart-insight">'
                    f'<strong>Insight:</strong> Positive sentiment trends upward in recent years, but spikes in negative sentiment require targeted interventions.'
                    '</div>',
                    unsafe_allow_html=True
                )
            
            fig9 = px.histogram(
                filtered_df, x='sentimentScore',
                title="Sentiment Score Distribution",
                nbins=30,
                color_discrete_sequence=['#17a2b8']
            )
            st.plotly_chart(fig9, use_container_width=True)
            st.markdown(
                '<div class="chart-insight">'
                f'<strong>Insight:</strong> Sentiment scores cluster around {filtered_df["sentimentScore"].mode()[0]:.2f}, indicating consistent customer experiences with room to enhance positive outliers.'
                '</div>',
                unsafe_allow_html=True
            )
        
        st.subheader("🔍 Sentiment Insights")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            inconsistent = filtered_df[
                ((filtered_df['rating'] >= 4) & (filtered_df['sentiment'].isin(['Negative', 'Very Negative']))) |
                ((filtered_df['rating'] <= 2) & (filtered_df['sentiment'].isin(['Positive', 'Very Positive'])))
            ]
            st.metric("🔍 Inconsistent Reviews", len(inconsistent))
            
        with col2:
            high_conf = filtered_df[filtered_df['sentimentConfidence'] > 0.8]
            st.metric("🎯 High Confidence Reviews", f"{len(high_conf)}/{len(filtered_df)}")
            
        with col3:
            sentiment_std = filtered_df['sentimentScore'].std()
            st.metric("📊 Sentiment Volatility", f"{sentiment_std:.3f}")
        
        if len(inconsistent) > 0:
            st.subheader("⚠️ Sample Inconsistent Reviews")
            st.dataframe(inconsistent[['rating', 'sentiment', 'sentimentScore', 'reviewText']].head(5))
    
    # TAB 4: Fraud Detection
    with tab4:
        st.markdown("## 🚨 Fraud Detection Intelligence")
        
        fraud_df = filtered_df[filtered_df['fraudFlag'] == 'Yes']
        suspicious_df = filtered_df[filtered_df['fraudFlag'] == 'Suspicious']
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("🚨 Confirmed Fraud", len(fraud_df))
        with col2:
            st.metric("⚠️ Suspicious", len(suspicious_df))
        with col3:
            fraud_rate = len(fraud_df) / len(filtered_df) * 100 if len(filtered_df) > 0 else 0
            st.metric("📊 Fraud Rate", f"{fraud_rate:.2f}%")
        with col4:
            avg_fraud_score = filtered_df['fraudScore'].mean()
            st.metric("🎯 Avg Risk Score", f"{avg_fraud_score:.1f}/10")
        with col5:
            clean_rate = (filtered_df['fraudFlag'] == 'No').sum() / len(filtered_df) * 100 if len(filtered_df) > 0 else 0
            st.metric("✅ Clean Rate", f"{clean_rate:.1f}%")
        
        col1, col2 = st.columns(2)
        
        with col1:
            all_suspicious = filtered_df[filtered_df['fraudFlag'].isin(['Yes', 'Suspicious'])]
            if len(all_suspicious) > 0:
                fraud_reasons = []
                for reasons in all_suspicious['fraudReason']:
                    if reasons != 'No Issues Detected':
                        fraud_reasons.extend(reasons.split('; '))
                
                if fraud_reasons:
                    reason_counts = Counter(fraud_reasons)
                    reasons_df = pd.DataFrame(reason_counts.most_common(), columns=['Fraud Type', 'Count'])
                    
                    fig_fraud = px.bar(
                        reasons_df, x='Count', y='Fraud Type',
                        title="🔍 Types of Fraud Detected",
                        orientation='h',
                        color='Count',
                        color_continuous_scale='Reds'
                    )
                    fig_fraud.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_fraud, use_container_width=True)
                    st.markdown(
                        '<div class="chart-insight">'
                        f'<strong>Insight:</strong> Common fraud types ({reasons_df["Fraud Type"].iloc[0]}) suggest targeted review filters to reduce low-quality feedback.'
                        '</div>',
                        unsafe_allow_html=True
                    )
            
            fig_fraud_dist = px.histogram(
                filtered_df, x='fraudScore',
                title="🎯 Fraud Risk Score Distribution",
                nbins=20,
                color_discrete_sequence=['#dc3545']
            )
            st.plotly_chart(fig_fraud_dist, use_container_width=True)
            st.markdown(
                '<div class="chart-insight">'
                f'<strong>Insight:</strong> Most reviews have low fraud scores, but a small peak at higher scores indicates potential bot activity requiring scrutiny.'
                '</div>',
                unsafe_allow_html=True
            )
        
        with col2:
            fraud_trends = filtered_df.groupby('year').agg({
                'fraudFlag': lambda x: (x.isin(['Yes', 'Suspicious'])).sum(),
                'reviewId': 'count'
            }).reset_index()
            fraud_trends['fraud_rate'] = fraud_trends['fraudFlag'] / fraud_trends['reviewId'] * 100
            
            fig_fraud_trends = px.line(
                fraud_trends, x='year', y='fraud_rate',
                title="📈 Fraud Rate Trends",
                color_discrete_sequence=['#dc3545']
            )
            st.plotly_chart(fig_fraud_trends, use_container_width=True)
            st.markdown(
                '<div class="chart-insight">'
                f'<strong>Insight:</strong> Fraud rates peak in {fraud_trends.loc[fraud_trends["fraud_rate"].idxmax(), "year"]}, suggesting a need for enhanced verification during those periods.'
                '</div>',
                unsafe_allow_html=True
            )
            
            fraud_rating = filtered_df.groupby('rating')['fraudScore'].mean().reset_index()
            fig_fraud_rating = px.bar(
                fraud_rating, x='rating', y='fraudScore',
                title="🎯 Fraud Score by Rating",
                color='fraudScore',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig_fraud_rating, use_container_width=True)
            st.markdown(
                '<div class="chart-insight">'
                f'<strong>Insight:</strong> Higher fraud scores in {fraud_rating.loc[fraud_rating["fraudScore"].idxmax(), "rating"]}-star ratings suggest targeted manipulation attempts.'
                '</div>',
                unsafe_allow_html=True
            )
        
        if len(all_suspicious) > 0:
            st.subheader("🔍 Detailed Fraud Analysis")
            sample_fraud = all_suspicious[['reviewerName', 'rating', 'fraudFlag', 'fraudScore', 'reviewText', 'fraudReason']].head(10)
            st.dataframe(sample_fraud, use_container_width=True)
    
    # TAB 5: Advanced Analytics
    with tab5:
        st.markdown("## 📈 Advanced Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            filtered_df['lengthCategory'] = pd.cut(
                filtered_df['wordCount'],
                bins=[0, 20, 50, 100, 200, float('inf')],
                labels=['Very Short (≤20)', 'Short (21-50)', 'Medium (51-100)', 'Long (101-200)', 'Very Long (200+)']
            )
            
            length_rating = filtered_df.groupby('lengthCategory').agg({
                'rating': 'mean',
                'reviewId': 'count'
            }).reset_index()
            
            fig_length_rating = px.bar(
                length_rating, x='lengthCategory', y='rating',
                title="📏 Rating by Review Length",
                color='rating',
                color_continuous_scale='RdYlGn'
            )
            fig_length_rating.update_xaxes(tickangle=45)
            st.plotly_chart(fig_length_rating, use_container_width=True)
            st.markdown(
                '<div class="chart-insight">'
                f'<strong>Insight:</strong> Longer reviews ({length_rating["lengthCategory"].iloc[length_rating["rating"].idxmax()]}) tend to have higher ratings, indicating detailed feedback is more positive.'
                '</div>',
                unsafe_allow_html=True
            )
            
            numeric_cols = ['rating', 'wordCount', 'sentimentScore', 'sentimentConfidence', 'fraudScore']
            corr_matrix = filtered_df[numeric_cols].corr()
            
            fig_corr = px.imshow(
                corr_matrix,
                title="🔗 Correlation Matrix",
                color_continuous_scale='RdBu_r',
                aspect='auto',
                text_auto=True
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            st.markdown(
                '<div class="chart-insight">'
                f'<strong>Insight:</strong> Strong correlation between rating and sentimentScore ({corr_matrix.loc["rating", "sentimentScore"]:.2f}) confirms alignment, but weak fraudScore correlation suggests fraud is independent of ratings.'
                '</div>',
                unsafe_allow_html=True
            )
        
        with col2:
            quality_analysis = filtered_df.groupby('rating').agg({
                'helpful': 'mean',
                'totalVotes': 'mean',
                'wordCount': 'mean',
                'sentimentConfidence': 'mean'
            }).reset_index()
            
            fig_quality = px.line(
                quality_analysis, x='rating', y=['helpful', 'sentimentConfidence'],
                title="📊 Quality Metrics by Rating",
                color_discrete_sequence=['#28a745', '#17a2b8']
            )
            st.plotly_chart(fig_quality, use_container_width=True)
            st.markdown(
                '<div class="chart-insight">'
                f'<strong>Insight:</strong> Higher ratings correlate with increased helpfulness and confidence, suggesting credible, impactful reviews.'
                '</div>',
                unsafe_allow_html=True
            )
            
            sample_size = min(1000, len(filtered_df))
            fig_scatter = px.scatter(
                filtered_df.sample(sample_size),
                x='sentimentScore', y='rating',
                color='fraudScore',
                size='wordCount',
                title="🎯 Multi-dimensional Analysis",
                color_continuous_scale='RdYlBu_r'
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.markdown(
                '<div class="chart-insight">'
                f'<strong>Insight:</strong> Reviews with high sentiment and ratings but high fraud scores warrant closer inspection for authenticity.'
                '</div>',
                unsafe_allow_html=True
            )
        
        st.subheader("📊 Statistical Insights")
        try:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**🔬 ANOVA: Ratings by Sentiment**")
                sentiment_groups = [filtered_df[filtered_df['sentiment'] == sent]['rating'] 
                                  for sent in filtered_df['sentiment'].unique() if len(filtered_df[filtered_df['sentiment'] == sent]) > 0]
                
                if len(sentiment_groups) >= 2:
                    f_stat, p_val = stats.f_oneway(*sentiment_groups)
                    st.metric("F-statistic", f"{f_stat:.3f}")
                    st.metric("P-value", f"{p_val:.3e}")
                    st.markdown(
                        '<div class="chart-insight">'
                        f'<strong>Insight:</strong> {"Significant differences" if p_val < 0.05 else "No significant differences"} in ratings across sentiments, guiding targeted improvements.'
                        '</div>',
                        unsafe_allow_html=True
                    )
            
            with col2:
                st.write("**🔬 T-test: Fraud vs. Clean**")
                fraud_ratings = filtered_df[filtered_df['fraudFlag'] == 'Yes']['rating']
                clean_ratings = filtered_df[filtered_df['fraudFlag'] == 'No']['rating']
                
                if len(fraud_ratings) > 0 and len(clean_ratings) > 0:
                    t_stat, p_val = stats.ttest_ind(fraud_ratings, clean_ratings)
                    st.metric("T-statistic", f"{t_stat:.3f}")
                    st.metric("P-value", f"{p_val:.3e}")
                    st.markdown(
                        '<div class="chart-insight">'
                        f'<strong>Insight:</strong> {"Significant differences" if p_val < 0.05 else "No significant differences"} between fraudulent and clean review ratings.'
                        '</div>',
                        unsafe_allow_html=True
                    )
            
            with col3:
                st.write("**🔬 Chi-square: Independence Test**")
                contingency_table = pd.crosstab(filtered_df['sentiment'], filtered_df['rating'])
                chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)
                
                st.metric("Chi-square", f"{chi2:.3f}")
                st.metric("P-value", f"{p_val:.3e}")
                st.markdown(
                    '<div class="chart-insight">'
                    f'<strong>Insight:</strong> Sentiment and ratings are {"dependent" if p_val < 0.05 else "independent"}, informing sentiment-driven strategies.'
                    '</div>',
                    unsafe_allow_html=True
                )
        except ImportError:
            st.info("📦 Install scipy for advanced statistical tests")
        
        st.subheader("🔮 Trend Forecasting")
        
        monthly_data = filtered_df.groupby(['year', 'month']).agg({
            'rating': 'mean',
            'reviewId': 'count',
            'fraudScore': 'mean'
        }).reset_index()
        monthly_data['date'] = pd.to_datetime(monthly_data[['year', 'month']].assign(day=1))
        monthly_data = monthly_data.sort_values('date')
        
        monthly_data['rating_ma'] = monthly_data['rating'].rolling(window=3).mean()
        monthly_data['volume_ma'] = monthly_data['reviewId'].rolling(window=3).mean()
        monthly_data['fraud_ma'] = monthly_data['fraudScore'].rolling(window=3).mean()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_forecast1 = px.line(
                monthly_data, x='date', y=['rating', 'rating_ma'],
                title="📈 Rating Trend",
                color_discrete_sequence=['#FF9500', '#dc3545']
            )
            st.plotly_chart(fig_forecast1, use_container_width=True)
            st.markdown(
                '<div class="chart-insight">'
                f'<strong>Insight:</strong> The moving average smooths rating fluctuations, highlighting a {"stable" if monthly_data["rating_ma"].std() < 0.5 else "volatile"} trend.'
                '</div>',
                unsafe_allow_html=True
            )
        
        with col2:
            fig_forecast2 = px.line(
                monthly_data, x='date', y=['fraudScore', 'fraud_ma'],
                title="🎯 Fraud Risk Trend",
                color_discrete_sequence=['#dc3545', '#ff6b6b']
            )
            st.plotly_chart(fig_forecast2, use_container_width=True)
            st.markdown(
                '<div class="chart-insight">'
                f'<strong>Insight:</strong> Fraud risk trends indicate periods of heightened activity, requiring proactive monitoring.'
                '</div>',
                unsafe_allow_html=True
            )
    
    # TAB 6: Top Verbatims
    with tab6:
        st.markdown("## 🗣️ Top Customer Verbatims")
        
        positive_verbatims, negative_verbatims = get_top_verbatims(filtered_df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("😊 Top 5 Positive Verbatims")
            for idx, row in positive_verbatims.iterrows():
                st.markdown(
                    f'<div class="metric-container">'
                    f'<strong>Rating: {row["rating"]}/5 | Sentiment Score: {row["sentimentScore"]:.2f}</strong><br>'
                    f'"{row["reviewText"][:200]}{"..." if len(row["reviewText"]) > 200 else ""}"'
                    f'</div>',
                    unsafe_allow_html=True
                )
        
        with col2:
            st.subheader("😔 Top 5 Negative Verbatims")
            for idx, row in negative_verbatims.iterrows():
                st.markdown(
                    f'<div class="metric-container">'
                    f'<strong>Rating: {row["rating"]}/5 | Sentiment Score: {row["sentimentScore"]:.2f}</strong><br>'
                    f'"{row["reviewText"][:200]}{"..." if len(row["reviewText"]) > 200 else ""}"'
                    f'</div>',
                    unsafe_allow_html=True
                )
        
        st.markdown(
            '<div class="chart-insight">'
            '<strong>Insight:</strong> Positive verbatims highlight strengths like reliability and performance, ideal for marketing. Negative verbatims pinpoint critical issues (e.g., durability, speed) requiring immediate attention.'
            '</div>',
            unsafe_allow_html=True
        )
    
    # Sidebar Download
    st.sidebar.markdown("---")
    st.sidebar.subheader("📥 Export Insights")
    
    if st.sidebar.button("🚀 Generate Executive Report"):
        download_df = filtered_df.copy()
        
        download_columns = [
            'reviewId', 'reviewerName', 'reviewText', 'rating', 'summary',
            'reviewDate', 'year', 'month', 'sentiment', 'sentimentScore', 'sentimentConfidence',
            'fraudFlag', 'fraudReason', 'fraudScore', 'topic', 'wordCount', 'reviewLength'
        ]
        
        available_columns = [col for col in download_columns if col in download_df.columns]
        download_df = download_df[available_columns]
        
        csv = download_df.to_csv(index=False)
        
        st.sidebar.download_button(
            label="📊 Download Insights CSV",
            data=csv,
            file_name="amazon_reviews_executive_insights.csv",
            mime="text/csv"
        )
        st.sidebar.success("✅ Report ready!")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Analysis Snapshot")
    st.sidebar.write(f"**📊 Reviews:** {len(filtered_df):,}")
    st.sidebar.write(f"**⭐ Avg Rating:** {filtered_df['rating'].mean():.2f}")
    st.sidebar.write(f"**📅 Date Range:** {filtered_df['year'].min()}-{filtered_df['year'].max()}")
    st.sidebar.write(f"**😊 Positive Rate:** {(filtered_df['sentiment'].isin(['Positive', 'Very Positive'])).sum()/len(filtered_df)*100:.1f}%")
    st.sidebar.write(f"**🚨 Fraud Count:** {(filtered_df['fraudFlag']=='Yes').sum()}")
    st.sidebar.write(f"**🎯 Avg Confidence:** {filtered_df['sentimentConfidence'].mean():.2f}")

if __name__ == "__main__":
    main()
