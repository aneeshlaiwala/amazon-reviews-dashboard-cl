import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import warnings
warnings.filterwarnings('ignore')

# Try to import optional dependencies
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    NLTK_AVAILABLE = True
except:
    NLTK_AVAILABLE = False

st.set_page_config(page_title="Amazon Reviews Analytics", page_icon="📊", layout="wide")

st.markdown("""
<style>
.main-header {font-size: 3rem; font-weight: bold; color: #FF9500; text-align: center; margin-bottom: 2rem;}
.insight-box {background-color: #f0f2f6; padding: 1rem; border-radius: 10px; border-left: 5px solid #FF9500; margin: 1rem 0;}
</style>
""", unsafe_allow_html=True)

if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df.columns = ['reviewId', 'reviewerName', 'reviewText', 'rating', 'summary', 'helpful', 'totalVotes', 'reviewDate', 'year']
    df['reviewText'] = df['reviewText'].fillna('')
    df['reviewerName'] = df['reviewerName'].fillna('Anonymous')
    df['reviewDate'] = pd.to_datetime(df['reviewDate'], format='%d-%m-%Y', errors='coerce')
    df['month'] = df['reviewDate'].dt.month
    df['wordCount'] = df['reviewText'].str.split().str.len()
    return df

def analyze_sentiment(text):
    if not text or len(text.strip()) == 0:
        return 'Neutral', 0
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return 'Positive', polarity
    elif polarity < -0.1:
        return 'Negative', polarity
    return 'Neutral', polarity

def detect_fraud(df):
    flags, reasons = [], []
    for _, row in df.iterrows():
        issues = []
        if len(df[df['reviewText'] == row['reviewText']]) > 1:
            issues.append('Duplicate')
        if len(row['reviewText'].split()) < 3:
            issues.append('Too Short')
        if row['reviewText'].isupper() and len(row['reviewText']) > 10:
            issues.append('All Caps')
        words = row['reviewText'].lower().split()
        if len(words) > 0 and len(set(words)) / len(words) < 0.3:
            issues.append('Low Diversity')
        flags.append('Yes' if issues else 'No')
        reasons.append('; '.join(issues) if issues else 'Clean')
    return flags, reasons

def topic_modeling(texts, n_topics=5):
    try:
        cleaned = [re.sub(r'[^a-zA-Z\s]', '', str(text).lower()) for text in texts if text]
        if len(cleaned) < n_topics:
            return [], ['No Topic'] * len(texts)
        
        stop_words = {'card', 'memory', 'product', 'item', 'amazon', 'the', 'and', 'or', 'but'}
        vectorizer = TfidfVectorizer(max_features=50, stop_words=list(stop_words), min_df=2, max_df=0.8)
        matrix = vectorizer.fit_transform(cleaned)
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=5)
        lda.fit(matrix)
        
        features = vectorizer.get_feature_names_out()
        topics = []
        for i, topic in enumerate(lda.components_):
            words = [features[j] for j in topic.argsort()[-5:][::-1]]
            topics.append(f"Topic {i+1}: {', '.join(words)}")
        
        assignments = [f"Topic {np.argmax(lda.transform(vectorizer.transform([text])))+1}" for text in cleaned]
        return topics, assignments + ['No Topic'] * (len(texts) - len(assignments))
    except:
        return [], ['No Topic'] * len(texts)

def get_word_frequencies(text_series):
    """Get word frequencies without NLTK dependency"""
    text = ' '.join(text_series.dropna()).lower()
    # Simple word extraction
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
    # Basic stop words
    stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'shall', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'a', 'an', 'card', 'memory', 'product', 'item', 'amazon'}
    filtered_words = [word for word in words if word not in stop_words]
    return Counter(filtered_words).most_common(15)

def create_simple_wordcloud_chart(word_freq):
    """Create a bar chart instead of word cloud if wordcloud is not available"""
    if word_freq:
        df_words = pd.DataFrame(word_freq, columns=['Word', 'Frequency'])
        fig = px.bar(df_words, x='Frequency', y='Word', orientation='h', 
                    title="Top Keywords", height=400)
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        return fig
    return None

def process_ml(df):
    sentiments, scores = zip(*[analyze_sentiment(text) for text in df['reviewText']])
    df['sentiment'] = sentiments
    df['sentimentScore'] = scores
    
    flags, reasons = detect_fraud(df)
    df['fraudFlag'] = flags
    df['fraudReason'] = reasons
    
    topics, assignments = topic_modeling(df['reviewText'].tolist())
    df['topic'] = assignments
    return df, topics

def executive_summary(df):
    total = len(df)
    avg_rating = df['rating'].mean()
    sentiment_dist = df['sentiment'].value_counts(normalize=True) * 100
    fraud_rate = (df['fraudFlag'] == 'Yes').sum() / total * 100 if total > 0 else 0
    
    return f"""
    ## 🎯 Executive Summary
    **Total Reviews:** {total:,} | **Avg Rating:** {avg_rating:.2f}/5.0 ⭐
    
    **Sentiment:** Positive {sentiment_dist.get('Positive', 0):.1f}% | Neutral {sentiment_dist.get('Neutral', 0):.1f}% | Negative {sentiment_dist.get('Negative', 0):.1f}%
    
    **Fraud Rate:** {fraud_rate:.1f}% | **Status:** {'High satisfaction' if avg_rating >= 4 else 'Moderate satisfaction'}
    """

def main():
    st.markdown('<div class="main-header">📊 Amazon Reviews Analytics Dashboard</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload Amazon Reviews CSV", type=['csv'])
    
    if uploaded_file:
        with st.spinner('Processing data with advanced ML algorithms...'):
            df = load_data(uploaded_file)
            df, topics = process_ml(df)
            st.session_state.processed_data = df
            st.session_state.topics = topics
            st.success(f"✅ Successfully processed {len(df)} reviews with sentiment analysis, fraud detection, and topic modeling!")
    elif st.session_state.processed_data is not None:
        df = st.session_state.processed_data
        topics = getattr(st.session_state, 'topics', [])
    else:
        st.info("👆 Upload your Amazon reviews CSV file to begin comprehensive analysis")
        st.markdown("""
        ### Expected CSV Format:
        Your CSV should contain columns for: Review #, reviewerName, reviewText, overall score (1-5), 
        summary, helpful status, total votes, review date, and year.
        """)
        return
    
    # Filters
    st.sidebar.header("🎛️ Advanced Filters")
    ratings = st.sidebar.multiselect("Star Ratings", sorted(df['rating'].unique()), default=sorted(df['rating'].unique()))
    years = st.sidebar.multiselect("Years", sorted(df['year'].unique()), default=sorted(df['year'].unique()))
    sentiments = st.sidebar.multiselect("Sentiment", df['sentiment'].unique(), default=df['sentiment'].unique())
    fraud_filter = st.sidebar.selectbox("Fraud Detection Filter", ['All Reviews', 'Legitimate Only', 'Suspicious Only'])
    keyword = st.sidebar.text_input("Keyword Search in Reviews")
    
    # Apply filters
    filtered_df = df[
        (df['rating'].isin(ratings)) & 
        (df['year'].isin(years)) & 
        (df['sentiment'].isin(sentiments))
    ]
    
    if fraud_filter == 'Legitimate Only':
        filtered_df = filtered_df[filtered_df['fraudFlag'] == 'No']
    elif fraud_filter == 'Suspicious Only':
        filtered_df = filtered_df[filtered_df['fraudFlag'] == 'Yes']
    
    if keyword:
        filtered_df = filtered_df[filtered_df['reviewText'].str.contains(keyword, case=False, na=False)]
    
    # Main Dashboard Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview Dashboard", "🔍 Verbatim Analysis", "😊 Sentiment Analysis", "🚨 Fraud Detection", "📈 Advanced Analytics"])
    
    with tab1:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown(executive_summary(filtered_df))
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Reviews", f"{len(filtered_df):,}")
        col2.metric("Average Rating", f"{filtered_df['rating'].mean():.2f}")
        col3.metric("Fraud Rate", f"{(filtered_df['fraudFlag']=='Yes').sum()/len(filtered_df)*100:.1f}%")
        col4.metric("Positive Sentiment", f"{(filtered_df['sentiment']=='Positive').sum()/len(filtered_df)*100:.1f}%")
        
        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.histogram(filtered_df, x='rating', title="Star Rating Distribution", 
                               color_discrete_sequence=['#FF9500'])
            st.plotly_chart(fig1, use_container_width=True)
            
            sentiment_counts = filtered_df['sentiment'].value_counts()
            fig2 = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, 
                         title="Sentiment Distribution",
                         color_discrete_map={'Positive': '#28a745', 'Neutral': '#ffc107', 'Negative': '#dc3545'})
            st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            trends = filtered_df.groupby(['year', 'month']).size().reset_index(name='count')
            trends['date'] = pd.to_datetime(trends[['year', 'month']].assign(day=1))
            fig3 = px.line(trends, x='date', y='count', title="Review Volume Trends Over Time",
                          color_discrete_sequence=['#667eea'])
            st.plotly_chart(fig3, use_container_width=True)
            
            heatmap = pd.crosstab(filtered_df['rating'], filtered_df['sentiment'])
            fig4 = px.imshow(heatmap.values, x=heatmap.columns, y=heatmap.index, 
                           title="Rating vs Sentiment Heatmap", color_continuous_scale='Blues')
            st.plotly_chart(fig4, use_container_width=True)
    
    with tab2:
        st.markdown("## 🔍 Verbatim Analysis & Topic Modeling")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Word Frequency Analysis")
            
            if WORDCLOUD_AVAILABLE and MATPLOTLIB_AVAILABLE:
                # Try to create word cloud
                try:
                    if NLTK_AVAILABLE:
                        text = ' '.join(filtered_df['reviewText'].dropna())
                        stop_words = set(stopwords.words('english'))
                        stop_words.update({'card', 'memory', 'product', 'amazon'})
                        words = word_tokenize(text.lower())
                        words = [w for w in words if w.isalpha() and w not in stop_words and len(w) > 2]
                        cleaned_text = ' '.join(words)
                    else:
                        # Fallback without NLTK
                        text = ' '.join(filtered_df['reviewText'].dropna())
                        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
                        stop_words = {'the', 'and', 'or', 'but', 'card', 'memory', 'product', 'amazon'}
                        words = [w for w in words if w not in stop_words]
                        cleaned_text = ' '.join(words)
                    
                    if cleaned_text:
                        wc = WordCloud(width=600, height=300, background_color='white').generate(cleaned_text)
                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.imshow(wc, interpolation='bilinear')
                        ax.axis('off')
                        st.pyplot(fig)
                    else:
                        st.info("Insufficient text data for word cloud")
                except Exception as e:
                    st.warning("Word cloud unavailable, showing keyword chart instead")
                    word_freq = get_word_frequencies(filtered_df['reviewText'])
                    fig_words = create_simple_wordcloud_chart(word_freq)
                    if fig_words:
                        st.plotly_chart(fig_words, use_container_width=True)
            else:
                # Fallback to bar chart
                word_freq = get_word_frequencies(filtered_df['reviewText'])
                fig_words = create_simple_wordcloud_chart(word_freq)
                if fig_words:
                    st.plotly_chart(fig_words, use_container_width=True)
        
        with col2:
            st.subheader("Topic Modeling Results")
            if topics:
                for i, topic in enumerate(topics, 1):
                    st.write(f"**{topic}**")
            else:
                st.info("Topic modeling results not available")
            
            st.subheader("Review Length Analysis")
            fig6 = px.histogram(filtered_df, x='wordCount', title="Review Length Distribution (Words)", 
                               nbins=20, color_discrete_sequence=['#667eea'])
            st.plotly_chart(fig6, use_container_width=True)
        
        if 'topic' in filtered_df.columns:
            st.subheader("Topic Distribution")
            topic_counts = filtered_df['topic'].value_counts()
            fig_topics = px.pie(values=topic_counts.values, names=topic_counts.index, 
                               title="Distribution of Review Topics")
            st.plotly_chart(fig_topics, use_container_width=True)
    
    with tab3:
        st.markdown("## 😊 Advanced Sentiment Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            sent_rating = pd.crosstab(filtered_df['rating'], filtered_df['sentiment'], normalize='index') * 100
            fig7 = px.bar(sent_rating.reset_index().melt(id_vars='rating'), 
                         x='rating', y='value', color='variable', 
                         title="Sentiment Distribution by Star Rating (%)",
                         color_discrete_map={'Positive': '#28a745', 'Neutral': '#ffc107', 'Negative': '#dc3545'})
            st.plotly_chart(fig7, use_container_width=True)
            
            sent_time = filtered_df.groupby(['year', 'sentiment']).size().unstack(fill_value=0)
            sent_time_pct = sent_time.div(sent_time.sum(axis=1), axis=0) * 100
            fig8 = px.line(sent_time_pct.reset_index().melt(id_vars='year'), 
                          x='year', y='value', color='variable', title="Sentiment Trends Over Time (%)",
                          color_discrete_map={'Positive': '#28a745', 'Neutral': '#ffc107', 'Negative': '#dc3545'})
            st.plotly_chart(fig8, use_container_width=True)
        
        with col2:
            fig9 = px.histogram(filtered_df, x='sentimentScore', title="Sentiment Score Distribution", 
                               nbins=20, color_discrete_sequence=['#17a2b8'])
            st.plotly_chart(fig9, use_container_width=True)
            
            st.subheader("Rating-Sentiment Inconsistencies")
            inconsistent = filtered_df[
                ((filtered_df['rating'] >= 4) & (filtered_df['sentiment'] == 'Negative')) |
                ((filtered_df['rating'] <= 2) & (filtered_df['sentiment'] == 'Positive'))
            ]
            st.metric("Inconsistent Reviews Found", len(inconsistent))
            if len(inconsistent) > 0:
                st.subheader("Sample Inconsistent Reviews")
                st.dataframe(inconsistent[['rating', 'sentiment', 'reviewText']].head(5), use_container_width=True)
        
        if 'topic' in filtered_df.columns:
            st.subheader("Sentiment Analysis by Topic")
            topic_sentiment = pd.crosstab(filtered_df['topic'], filtered_df['sentiment'], normalize='index') * 100
            fig_topic_sent = px.bar(
                topic_sentiment.reset_index().melt(id_vars='topic'),
                x='topic', y='value', color='variable',
                title="Sentiment Distribution by Topic (%)",
                color_discrete_map={'Positive': '#28a745', 'Neutral': '#ffc107', 'Negative': '#dc3545'}
            )
            fig_topic_sent.update_xaxes(tickangle=45)
            st.plotly_chart(fig_topic_sent, use_container_width=True)
    
    with tab4:
        st.markdown("## 🚨 Advanced Fraud Detection Analysis")
        
        fraud_df = filtered_df[filtered_df['fraudFlag'] == 'Yes']
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Suspicious Reviews", len(fraud_df))
        col2.metric("Fraud Rate", f"{len(fraud_df)/len(filtered_df)*100:.2f}%")
        col3.metric("Avg Rating (Suspicious)", f"{fraud_df['rating'].mean():.2f}" if len(fraud_df) > 0 else "N/A")
        col4.metric("Avg Rating (Legitimate)", f"{filtered_df[filtered_df['fraudFlag']=='No']['rating'].mean():.2f}")
        
        col1, col2 = st.columns(2)
        with col1:
            if len(fraud_df) > 0:
                reasons = []
                for reason in fraud_df['fraudReason']:
                    if reason != 'Clean':
                        reasons.extend(reason.split('; '))
                
                if reasons:
                    reason_counts = Counter(reasons)
                    df_reasons = pd.DataFrame(reason_counts.most_common(), columns=['Fraud Type', 'Count'])
                    fig10 = px.bar(df_reasons, x='Count', y='Fraud Type', orientation='h', 
                                  title="Types of Fraud Detected", color_discrete_sequence=['#dc3545'])
                    fig10.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig10, use_container_width=True)
            
            fraud_by_rating = filtered_df.groupby('rating')['fraudFlag'].apply(lambda x: (x == 'Yes').sum()).reset_index()
            fraud_by_rating.columns = ['Rating', 'Fraud Count']
            fig11 = px.bar(fraud_by_rating, x='Rating', y='Fraud Count', 
                          title="Suspicious Reviews by Star Rating", color_discrete_sequence=['#dc3545'])
            st.plotly_chart(fig11, use_container_width=True)
        
        with col2:
            fraud_trends = filtered_df.groupby('year').agg({
                'fraudFlag': lambda x: (x == 'Yes').sum(),
                'reviewId': 'count'
            }).reset_index()
            fraud_trends['fraud_rate'] = fraud_trends['fraudFlag'] / fraud_trends['reviewId'] * 100
            
            fig12 = px.line(fraud_trends, x='year', y='fraud_rate', 
                           title="Fraud Rate Trends Over Time (%)", color_discrete_sequence=['#dc3545'])
            st.plotly_chart(fig12, use_container_width=True)
            
            fig13 = px.box(filtered_df, x='fraudFlag', y='wordCount', 
                          title="Review Length vs Fraud Status", color='fraudFlag',
                          color_discrete_map={'Yes': '#dc3545', 'No': '#28a745'})
            st.plotly_chart(fig13, use_container_width=True)
        
        if len(fraud_df) > 0:
            st.subheader("Sample Suspicious Reviews")
            sample_fraud = fraud_df[['reviewerName', 'rating', 'reviewText', 'fraudReason']].head(10)
            st.dataframe(sample_fraud, use_container_width=True)
        else:
            st.info("No suspicious reviews found in current filter selection.")
        
        st.subheader("Reviewer Behavior Analysis")
        reviewer_stats = filtered_df.groupby('reviewerName').agg({
            'reviewId': 'count',
            'rating': 'mean',
            'fraudFlag': lambda x: (x == 'Yes').sum()
        }).reset_index()
        reviewer_stats.columns = ['Reviewer', 'Total Reviews', 'Avg Rating', 'Suspicious Count']
        reviewer_stats = reviewer_stats[reviewer_stats['Total Reviews'] > 1].sort_values('Suspicious Count', ascending=False)
        
        if len(reviewer_stats) > 0:
            st.dataframe(reviewer_stats.head(10), use_container_width=True)
        else:
            st.info("No reviewers with multiple reviews found.")
    
    with tab5:
        st.markdown("## 📈 Advanced Analytics & Statistical Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Review Length vs Rating Analysis")
            filtered_df['lengthCat'] = pd.cut(filtered_df['wordCount'], 
                                            bins=[0, 20, 50, 100, 200, float('inf')],
                                            labels=['Very Short (≤20)', 'Short (21-50)', 'Medium (51-100)', 'Long (101-200)', 'Very Long (200+)'])
            
            length_rating = filtered_df.groupby('lengthCat')['rating'].mean().reset_index()
            fig14 = px.bar(length_rating, x='lengthCat', y='rating', 
                          title="Average Rating by Review Length", color_discrete_sequence=['#17a2b8'])
            fig14.update_xaxes(tickangle=45)
            st.plotly_chart(fig14, use_container_width=True)
            
            st.subheader("Correlation Analysis")
            numeric_cols = ['rating', 'wordCount', 'sentimentScore', 'helpful', 'totalVotes']
            corr_matrix = filtered_df[numeric_cols].corr()
            fig15 = px.imshow(corr_matrix, title="Correlation Matrix", 
                             color_continuous_scale='RdBu_r', aspect='auto')
            st.plotly_chart(fig15, use_container_width=True)
        
        with col2:
            st.subheader("Review Helpfulness Analysis")
            helpful_analysis = filtered_df.groupby('rating').agg({
                'helpful': 'mean',
                'totalVotes': 'mean'
            }).reset_index()
            fig16 = px.bar(helpful_analysis, x='rating', y='helpful', 
                          title="Average Helpfulness by Star Rating", color_discrete_sequence=['#28a745'])
            st.plotly_chart(fig16, use_container_width=True)
            
            st.subheader("Sentiment Score vs Star Rating")
            sample_size = min(1000, len(filtered_df))
            fig17 = px.scatter(filtered_df.sample(sample_size), x='sentimentScore', y='rating', 
                              color='sentiment', title="Sentiment Score vs Star Rating",
                              color_discrete_map={'Positive': '#28a745', 'Neutral': '#ffc107', 'Negative': '#dc3545'})
            st.plotly_chart(fig17, use_container_width=True)
        
        # Statistical Analysis
        st.subheader("Statistical Significance Testing")
        try:
            from scipy import stats
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**ANOVA Test - Ratings by Sentiment**")
                pos = filtered_df[filtered_df['sentiment'] == 'Positive']['rating']
                neu = filtered_df[filtered_df['sentiment'] == 'Neutral']['rating']
                neg = filtered_df[filtered_df['sentiment'] == 'Negative']['rating']
                
                if len(pos) > 0 and len(neu) > 0 and len(neg) > 0:
                    f_stat, p_val = stats.f_oneway(pos, neu, neg)
                    st.metric("F-statistic", f"{f_stat:.3f}")
                    st.metric("P-value", f"{p_val:.3e}")
                    if p_val < 0.05:
                        st.success("Significant difference found")
                    else:
                        st.info("No significant difference")
            
            with col2:
                st.write("**T-test - Fraud vs Legitimate**")
                fraud_ratings = filtered_df[filtered_df['fraudFlag'] == 'Yes']['rating']
                clean_ratings = filtered_df[filtered_df['fraudFlag'] == 'No']['rating']
                
                if len(fraud_ratings) > 0 and len(clean_ratings) > 0:
                    t_stat, p_val = stats.ttest_ind(fraud_ratings, clean_ratings)
                    st.metric("T-statistic", f"{t_stat:.3f}")
                    st.metric("P-value", f"{p_val:.3e}")
                    if p_val < 0.05:
                        st.success("Significant difference found")
                    else:
                        st.info("No significant difference")
            
            with col3:
                st.write("**Chi-square - Sentiment Independence**")
                contingency_table = pd.crosstab(filtered_df['sentiment'], filtered_df['rating'])
                chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)
                st.metric("Chi-square", f"{chi2:.3f}")
                st.metric("P-value", f"{p_val:.3e}")
                if p_val < 0.05:
                    st.success("Variables are dependent")
                else:
                    st.info("Variables are independent")
        except ImportError:
            st.info("Install scipy for advanced statistical tests: `pip install scipy`")
        
        # Trend Forecasting
        st.subheader("Trend Forecasting with Moving Averages")
        monthly_data = filtered_df.groupby(['year', 'month']).agg({
            'rating': 'mean',
            'reviewId': 'count'
        }).reset_index()
        monthly_data['date'] = pd.to_datetime(monthly_data[['year', 'month']].assign(day=1))
        monthly_data = monthly_data.sort_values('date')
        
        # Simple moving average forecast
        window = 3
        monthly_data['rating_ma'] = monthly_data['rating'].rolling(window=window).mean()
        monthly_data['volume_ma'] = monthly_data['reviewId'].rolling(window=window).mean()
        
        col1, col2 = st.columns(2)
        with col1:
            fig18 = px.line(monthly_data, x='date', y=['rating', 'rating_ma'],
                           title="Rating Trend with Moving Average",
                           color_discrete_sequence=['#FF9500', '#dc3545'])
            st.plotly_chart(fig18, use_container_width=True)
        
        with col2:
            fig19 = px.line(monthly_data, x='date', y=['reviewId', 'volume_ma'],
                           title="Review Volume Trend with Moving Average",
                           color_discrete_sequence=['#667eea', '#28a745'])
            st.plotly_chart(fig19, use_container_width=True)
    
    # Enhanced Sidebar Features
    st.sidebar.markdown("---")
    st.sidebar.subheader("📥 Download Enhanced Data")
    
    if st.sidebar.button("Generate Enhanced CSV with ML Results"):
        # Prepare enhanced dataset
        download_df = filtered_df.copy()
        
        # Select columns for download
        download_columns = [
            'reviewId', 'reviewerName', 'reviewText', 'rating', 'summary',
            'reviewDate', 'year', 'month', 'sentiment', 'sentimentScore',
            'fraudFlag', 'fraudReason', 'topic', 'wordCount'
        ]
        
        # Only include columns that exist
        available_columns = [col for col in download_columns if col in download_df.columns]
        download_df = download_df[available_columns]
        
        # Convert to CSV
        csv = download_df.to_csv(index=False)
        
        st.sidebar.download_button(
            label="📊 Download Enhanced Dataset",
            data=csv,
            file_name="amazon_reviews_enhanced_analysis.csv",
            mime="text/csv"
        )
        st.sidebar.success("✅ Enhanced dataset ready for download!")
    
    # Current Filter Summary
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Current Analysis Summary")
    st.sidebar.write(f"**Total Reviews:** {len(filtered_df):,}")
    st.sidebar.write(f"**Average Rating:** {filtered_df['rating'].mean():.2f}/5.0")
    st.sidebar.write(f"**Date Range:** {filtered_df['year'].min()}-{filtered_df['year'].max()}")
    st.sidebar.write(f"**Positive Sentiment:** {(filtered_df['sentiment']=='Positive').sum()/len(filtered_df)*100:.1f}%")
    st.sidebar.write(f"**Fraud Detection:** {(filtered_df['fraudFlag']=='Yes').sum()} suspicious reviews")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p><strong>🚀 Amazon Reviews Analytics Dashboard</strong></p>
        <p>Powered by Advanced AI & Machine Learning</p>
        <p>Features: Sentiment Analysis • Fraud Detection • Topic Modeling • Statistical Testing • Multi-language Support</p>
        <p><em>Built with Streamlit, Plotly, TextBlob, and Scikit-learn</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
