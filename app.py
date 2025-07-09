import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
import warnings
warnings.filterwarnings('ignore')

try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
except:
    pass

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
        with st.spinner('Processing...'):
            df = load_data(uploaded_file)
            df, topics = process_ml(df)
            st.session_state.processed_data = df
            st.session_state.topics = topics
            st.success(f"✅ Processed {len(df)} reviews!")
    elif st.session_state.processed_data is not None:
        df = st.session_state.processed_data
        topics = getattr(st.session_state, 'topics', [])
    else:
        st.info("👆 Upload your CSV file to start")
        return
    
    # Filters
    st.sidebar.header("🎛️ Filters")
    ratings = st.sidebar.multiselect("Ratings", sorted(df['rating'].unique()), default=sorted(df['rating'].unique()))
    years = st.sidebar.multiselect("Years", sorted(df['year'].unique()), default=sorted(df['year'].unique()))
    sentiments = st.sidebar.multiselect("Sentiment", df['sentiment'].unique(), default=df['sentiment'].unique())
    fraud_filter = st.sidebar.selectbox("Fraud", ['All', 'Clean Only', 'Suspicious Only'])
    keyword = st.sidebar.text_input("Keyword Search")
    
    # Apply filters
    filtered_df = df[
        (df['rating'].isin(ratings)) & 
        (df['year'].isin(years)) & 
        (df['sentiment'].isin(sentiments))
    ]
    
    if fraud_filter == 'Clean Only':
        filtered_df = filtered_df[filtered_df['fraudFlag'] == 'No']
    elif fraud_filter == 'Suspicious Only':
        filtered_df = filtered_df[filtered_df['fraudFlag'] == 'Yes']
    
    if keyword:
        filtered_df = filtered_df[filtered_df['reviewText'].str.contains(keyword, case=False, na=False)]
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🔍 Analysis", "😊 Sentiment", "🚨 Fraud", "📈 Advanced"])
    
    with tab1:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown(executive_summary(filtered_df))
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Reviews", f"{len(filtered_df):,}")
        col2.metric("Avg Rating", f"{filtered_df['rating'].mean():.2f}")
        col3.metric("Fraud %", f"{(filtered_df['fraudFlag']=='Yes').sum()/len(filtered_df)*100:.1f}")
        col4.metric("Positive %", f"{(filtered_df['sentiment']=='Positive').sum()/len(filtered_df)*100:.1f}")
        
        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.histogram(filtered_df, x='rating', title="Rating Distribution")
            st.plotly_chart(fig1, use_container_width=True)
            
            sentiment_counts = filtered_df['sentiment'].value_counts()
            fig2 = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, title="Sentiment")
            st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            trends = filtered_df.groupby(['year', 'month']).size().reset_index(name='count')
            trends['date'] = pd.to_datetime(trends[['year', 'month']].assign(day=1))
            fig3 = px.line(trends, x='date', y='count', title="Volume Trends")
            st.plotly_chart(fig3, use_container_width=True)
            
            heatmap = pd.crosstab(filtered_df['rating'], filtered_df['sentiment'])
            fig4 = px.imshow(heatmap.values, x=heatmap.columns, y=heatmap.index, title="Rating vs Sentiment")
            st.plotly_chart(fig4, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Word Cloud")
            text = ' '.join(filtered_df['reviewText'].dropna())
            try:
                stop_words = set(stopwords.words('english'))
            except:
                stop_words = set()
            stop_words.update({'card', 'memory', 'product', 'amazon'})
            
            try:
                words = word_tokenize(text.lower())
            except:
                words = text.lower().split()
            
            words = [w for w in words if w.isalpha() and w not in stop_words and len(w) > 2]
            if words:
                wc = WordCloud(width=600, height=300, background_color='white').generate(' '.join(words))
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            
            # Top keywords
            word_freq = Counter(words).most_common(10)
            if word_freq:
                df_words = pd.DataFrame(word_freq, columns=['Word', 'Freq'])
                fig5 = px.bar(df_words, x='Freq', y='Word', orientation='h', title="Top Keywords")
                st.plotly_chart(fig5, use_container_width=True)
        
        with col2:
            st.subheader("Topics")
            for topic in topics:
                st.write(f"**{topic}**")
            
            st.subheader("Review Length")
            fig6 = px.histogram(filtered_df, x='wordCount', title="Word Count Distribution", nbins=20)
            st.plotly_chart(fig6, use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            sent_rating = pd.crosstab(filtered_df['rating'], filtered_df['sentiment'], normalize='index') * 100
            fig7 = px.bar(sent_rating.reset_index().melt(id_vars='rating'), 
                         x='rating', y='value', color='variable', title="Sentiment by Rating %")
            st.plotly_chart(fig7, use_container_width=True)
            
            sent_time = filtered_df.groupby(['year', 'sentiment']).size().unstack(fill_value=0)
            sent_time_pct = sent_time.div(sent_time.sum(axis=1), axis=0) * 100
            fig8 = px.line(sent_time_pct.reset_index().melt(id_vars='year'), 
                          x='year', y='value', color='variable', title="Sentiment Trends")
            st.plotly_chart(fig8, use_container_width=True)
        
        with col2:
            fig9 = px.histogram(filtered_df, x='sentimentScore', title="Sentiment Scores", nbins=20)
            st.plotly_chart(fig9, use_container_width=True)
            
            inconsistent = filtered_df[
                ((filtered_df['rating'] >= 4) & (filtered_df['sentiment'] == 'Negative')) |
                ((filtered_df['rating'] <= 2) & (filtered_df['sentiment'] == 'Positive'))
            ]
            st.metric("Inconsistent Reviews", len(inconsistent))
            if len(inconsistent) > 0:
                st.dataframe(inconsistent[['rating', 'sentiment', 'reviewText']].head(3))
    
    with tab4:
        fraud_df = filtered_df[filtered_df['fraudFlag'] == 'Yes']
        col1, col2, col3 = st.columns(3)
        col1.metric("Suspicious", len(fraud_df))
        col2.metric("Rate %", f"{len(fraud_df)/len(filtered_df)*100:.1f}")
        col3.metric("Avg Rating", f"{fraud_df['rating'].mean():.2f}" if len(fraud_df) > 0 else "N/A")
        
        if len(fraud_df) > 0:
            reasons = []
            for reason in fraud_df['fraudReason']:
                if reason != 'Clean':
                    reasons.extend(reason.split('; '))
            
            if reasons:
                reason_counts = Counter(reasons)
                df_reasons = pd.DataFrame(reason_counts.most_common(), columns=['Type', 'Count'])
                fig10 = px.bar(df_reasons, x='Count', y='Type', orientation='h', title="Fraud Types")
                st.plotly_chart(fig10, use_container_width=True)
            
            st.subheader("Sample Suspicious Reviews")
            st.dataframe(fraud_df[['reviewerName', 'rating', 'reviewText', 'fraudReason']].head(5))
    
    with tab5:
        col1, col2 = st.columns(2)
        with col1:
            filtered_df['lengthCat'] = pd.cut(filtered_df['wordCount'], 
                                            bins=[0, 20, 50, 100, 200, float('inf')],
                                            labels=['≤20', '21-50', '51-100', '101-200', '200+'])
            
            length_rating = filtered_df.groupby('lengthCat')['rating'].mean().reset_index()
            fig11 = px.bar(length_rating, x='lengthCat', y='rating', title="Rating by Length")
            st.plotly_chart(fig11, use_container_width=True)
            
            numeric_cols = ['rating', 'wordCount', 'sentimentScore']
            corr_matrix = filtered_df[numeric_cols].corr()
            fig12 = px.imshow(corr_matrix, title="Correlations")
            st.plotly_chart(fig12, use_container_width=True)
        
        with col2:
            helpful_analysis = filtered_df.groupby('rating')['helpful'].mean().reset_index()
            fig13 = px.bar(helpful_analysis, x='rating', y='helpful', title="Helpfulness by Rating")
            st.plotly_chart(fig13, use_container_width=True)
            
            sample_size = min(500, len(filtered_df))
            fig14 = px.scatter(filtered_df.sample(sample_size), x='sentimentScore', y='rating', 
                              color='sentiment', title="Sentiment vs Rating")
            st.plotly_chart(fig14, use_container_width=True)
        
        # Statistical tests
        try:
            from scipy import stats
            st.subheader("Statistical Tests")
            col1, col2 = st.columns(2)
            
            with col1:
                pos = filtered_df[filtered_df['sentiment'] == 'Positive']['rating']
                neu = filtered_df[filtered_df['sentiment'] == 'Neutral']['rating']
                neg = filtered_df[filtered_df['sentiment'] == 'Negative']['rating']
                
                if len(pos) > 0 and len(neu) > 0 and len(neg) > 0:
                    f_stat, p_val = stats.f_oneway(pos, neu, neg)
                    st.metric("ANOVA F-stat", f"{f_stat:.3f}")
                    st.metric("P-value", f"{p_val:.3e}")
            
            with col2:
                fraud_ratings = filtered_df[filtered_df['fraudFlag'] == 'Yes']['rating']
                clean_ratings = filtered_df[filtered_df['fraudFlag'] == 'No']['rating']
                
                if len(fraud_ratings) > 0 and len(clean_ratings) > 0:
                    t_stat, p_val = stats.ttest_ind(fraud_ratings, clean_ratings)
                    st.metric("T-test stat", f"{t_stat:.3f}")
                    st.metric("P-value", f"{p_val:.3e}")
        except:
            st.info("Install scipy for statistical tests")
    
    # Sidebar download
    st.sidebar.markdown("---")
    if st.sidebar.button("Download Enhanced CSV"):
        csv = filtered_df.to_csv(index=False)
        st.sidebar.download_button("Download", csv, "enhanced_reviews.csv", "text/csv")
        st.sidebar.success("Ready!")
    
    st.sidebar.markdown("---")
    st.sidebar.write(f"**Reviews:** {len(filtered_df):,}")
    st.sidebar.write(f"**Avg Rating:** {filtered_df['rating'].mean():.2f}")

if __name__ == "__main__":
    main()