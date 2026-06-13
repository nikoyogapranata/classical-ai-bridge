# ─── Imports ──────────────────────────────────────────────────────────────────
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')
import os

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Classical AI Bridge",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #5b4ef0, #e03050);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #666;
        margin-bottom: 0.5rem;
    }
    .hero-author {
        font-size: 0.85rem;
        color: #999;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f8fc;
        border: 1px solid #e8e8f0;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #5b4ef0;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #999;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .result-card {
        background: #f8f8fc;
        border: 1px solid #e8e8f0;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
        border-left: 3px solid #5b4ef0;
    }
    .result-card-neural {
        border-left: 3px solid #e03050;
    }
    .result-score { font-size: 0.8rem; color: #999; }
    .result-text {
        font-size: 0.9rem;
        color: #333;
        margin-top: 0.3rem;
        line-height: 1.4;
    }
    .result-category {
        font-size: 0.75rem;
        color: #5b4ef0;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.3rem;
        margin-top: 1rem;
    }
    .section-desc {
        font-size: 0.9rem;
        color: #888;
        margin-bottom: 1.5rem;
    }
    .divider {
        border: none;
        border-top: 1px solid #e8e8f0;
        margin: 1.5rem 0;
    }
    .badge {
        display: inline-block;
        background: #f0f0f8;
        border: 1px solid #e0e0ee;
        border-radius: 20px;
        padding: 0.2rem 0.8rem;
        font-size: 0.75rem;
        color: #666;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    .pred-bar-container {
        background: #f8f8fc;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
        border: 1px solid #e8e8f0;
    }
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
CATEGORIES = [
    'sci.space',
    'rec.sport.hockey',
    'comp.graphics',
    'talk.politics.guns'
]

CATEGORY_LABELS = {
    'sci.space': '🚀 Space',
    'rec.sport.hockey': '🏒 Hockey',
    'comp.graphics': '💻 Graphics',
    'talk.politics.guns': '🗳️ Politics'
}

CATEGORY_COLORS = {
    'sci.space': '#7c6af7',
    'rec.sport.hockey': '#f7516a',
    'comp.graphics': '#36d399',
    'talk.politics.guns': '#f4a62a'
}

RANDOM_STATE = 42

# ─── Data Loading (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="📥 Loading dataset...")
def load_data():
    newsgroups = fetch_20newsgroups(
        subset='all',
        categories=CATEGORIES,
        remove=('headers', 'footers', 'quotes'),
        random_state=RANDOM_STATE
    )

    np.random.seed(RANDOM_STATE)
    indices = np.random.choice(len(newsgroups.data), size=500, replace=False)

    texts  = [newsgroups.data[i] for i in indices]
    labels = [newsgroups.target[i] for i in indices]
    label_names = newsgroups.target_names

    df = pd.DataFrame({
        'text'      : texts,
        'label'     : labels,
        'category'  : [label_names[l] for l in labels],
        'emoji_label': [CATEGORY_LABELS[label_names[l]] for l in labels]
    })

    return df, label_names

# ─── TF-IDF (cached) ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔢 Building TF-IDF vectors...")
def build_tfidf(_df):
    tfidf = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2)
    )
    matrix = tfidf.fit_transform(_df['text'])
    return tfidf, matrix

# ─── Sentence Transformer (cached) ───────────────────────────────────────────
@st.cache_resource(show_spinner="🤖 Loading neural embedding model...")
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource(show_spinner="⚙️ Generating neural embeddings...")
def build_embeddings(_df, _model):
    texts_truncated = [t[:512] for t in _df['text'].tolist()]
    embeddings = _model.encode(
        texts_truncated,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True
    )
    return embeddings

# ─── K-Means (cached) ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🗂️ Running K-Means clustering...")
def build_kmeans(_tfidf_matrix, _embeddings):
    # K-Means on TF-IDF
    kmeans_tfidf = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=10)
    tfidf_clusters = kmeans_tfidf.fit_predict(_tfidf_matrix)

    # PCA for TF-IDF
    pca_tfidf = PCA(n_components=2, random_state=RANDOM_STATE)
    tfidf_2d  = pca_tfidf.fit_transform(_tfidf_matrix.toarray())

    # PCA for embeddings
    pca_emb = PCA(n_components=2, random_state=RANDOM_STATE)
    emb_2d  = pca_emb.fit_transform(_embeddings)

    return kmeans_tfidf, tfidf_clusters, tfidf_2d, emb_2d

# ─── Naïve Bayes (cached) ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🎲 Training Naïve Bayes...")
def build_naive_bayes(_tfidf, _tfidf_matrix, _df):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix

    X_train, X_test, y_train, y_test = train_test_split(
        _tfidf_matrix, _df['label'].values,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=_df['label'].values
    )

    nb = MultinomialNB(alpha=0.1)
    nb.fit(X_train, y_train)
    y_pred   = nb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm       = confusion_matrix(y_test, y_pred)

    return nb, accuracy, cm, y_test, y_pred

# ─── Load everything ──────────────────────────────────────────────────────────
df, label_names          = load_data()
tfidf, tfidf_matrix      = build_tfidf(df)
emb_model                = load_embedding_model()
embeddings               = build_embeddings(df, emb_model)
kmeans, tfidf_clusters, tfidf_2d, emb_2d = build_kmeans(tfidf_matrix, embeddings)
nb_model, nb_accuracy, nb_cm, y_test, y_pred = build_naive_bayes(tfidf, tfidf_matrix, df)

feature_names = tfidf.get_feature_names_out()

# ─── Hero Header ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-title">Classical AI Bridge 🧠</div>
<div class="hero-subtitle">How Classical Data Mining Powers Modern AI</div>
<div class="hero-author">林宇辰 (Niko Yoga Pranata) · L25020019 · Data Mining Final Project</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ─── Sidebar Navigation ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 Classical AI Bridge")
    st.markdown('<div style="font-size:0.8rem; color:#888; margin-bottom:1.5rem">林宇辰 · L25020019</div>', unsafe_allow_html=True)
    
    page = st.radio(
        label="Navigation",
        options=[
            "🔍 Semantic Search",
            "🗂️ Cluster Explorer",
            "🎲 Naïve Bayes Classifier",
            "📖 How It Works"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown('<div style="font-size:0.75rem; color:#888">Data Mining Final Project<br>20 Newsgroups Dataset<br>500 articles · 4 categories</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — SEMANTIC SEARCH
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔍 Semantic Search":

    st.markdown('<div class="section-header">🔍 Semantic Search</div>', 
                unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Type any query and compare how Classical TF-IDF vs Neural Embeddings retrieve results. Same cosine similarity math — different vectors.</div>', 
                unsafe_allow_html=True)

    # ─── Search input ─────────────────────────────────────────────────────────
    query = st.text_input(
        label="Search query",
        placeholder="e.g. space exploration and rockets, hockey game playoffs, computer image rendering...",
        label_visibility="collapsed"
    )

    col_k, col_btn = st.columns([1, 4])
    with col_k:
        top_k = st.slider("Results", min_value=3, max_value=10, value=5)

    if query and query.strip():
        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # ── TF-IDF search ──────────────────────────────────────────────────
        tfidf_norm     = normalize(tfidf_matrix, norm='l2')
        query_vec      = tfidf.transform([query])
        query_vec_norm = normalize(query_vec, norm='l2')
        tfidf_scores   = cosine_similarity(query_vec_norm, tfidf_norm).flatten()
        tfidf_top      = tfidf_scores.argsort()[-top_k:][::-1]

        # ── Neural embedding search ────────────────────────────────────────
        emb_norm       = normalize(embeddings, norm='l2')
        query_emb      = emb_model.encode([query[:512]], convert_to_numpy=True)
        query_emb_norm = normalize(query_emb, norm='l2')
        emb_scores     = cosine_similarity(query_emb_norm, emb_norm).flatten()
        emb_top        = emb_scores.argsort()[-top_k:][::-1]

        # ── Results columns ────────────────────────────────────────────────
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### 📊 TF-IDF + Cosine Similarity")
            st.markdown('<div class="section-desc">Classical — matches by word frequency</div>',
                        unsafe_allow_html=True)
            for rank, idx in enumerate(tfidf_top, 1):
                score    = tfidf_scores[idx]
                category = df.iloc[idx]['emoji_label']
                preview  = df.iloc[idx]['text'][:150].replace('\n', ' ').strip()
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-category">{category} &nbsp;·&nbsp; #{rank}</div>
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div class="result-text">{preview}...</div>
                    </div>
                    <div class="result-score">Cosine Score: <b>{score:.4f}</b></div>
                </div>
                """, unsafe_allow_html=True)

        with col_right:
            st.markdown("#### 🤖 Neural Embeddings + Cosine Similarity")
            st.markdown('<div class="section-desc">Modern — matches by semantic meaning</div>',
                        unsafe_allow_html=True)
            for rank, idx in enumerate(emb_top, 1):
                score    = emb_scores[idx]
                category = df.iloc[idx]['emoji_label']
                preview  = df.iloc[idx]['text'][:150].replace('\n', ' ').strip()
                st.markdown(f"""
                <div class="result-card result-card-neural">
                    <div class="result-category" style="color:#f7516a">{category} &nbsp;·&nbsp; #{rank}</div>
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div class="result-text">{preview}...</div>
                    </div>
                    <div class="result-score">Cosine Score: <b>{score:.4f}</b></div>
                </div>
                """, unsafe_allow_html=True)

        # ── Score comparison chart ─────────────────────────────────────────
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown("#### 📈 Score Comparison")

        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='TF-IDF',
            x=[f"#{i+1}" for i in range(top_k)],
            y=[tfidf_scores[i] for i in tfidf_top],
            marker_color='#7c6af7',
            opacity=0.85
        ))
        fig.add_trace(go.Bar(
            name='Neural Embeddings',
            x=[f"#{i+1}" for i in range(top_k)],
            y=[emb_scores[i] for i in emb_top],
            marker_color='#f7516a',
            opacity=0.85
        ))

        fig.update_layout(
            barmode='group',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#444', size=12),
            legend=dict(
                bgcolor='rgba(0,0,0,0)',
                font=dict(color='#444')
            ),
            xaxis=dict(gridcolor='#e8e8f0', title='Result Rank'),
            yaxis=dict(gridcolor='#e8e8f0', title='Cosine Similarity Score', range=[0, 1]),
            margin=dict(l=20, r=20, t=20, b=20),
            height=300
        )

        st.plotly_chart(fig, use_container_width=True)

        # ── Key insight ────────────────────────────────────────────────────
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:0.85rem; color:#888; margin-bottom:0.5rem">💡 Key Insight</div>
            <div style="font-size:0.95rem; color:#333; line-height:1.6">
                Both methods use <b style="color:#7c6af7">Cosine Similarity</b> — the exact same math 
                invented in 1972. The only difference is the <b>quality of the vectors</b>. 
                TF-IDF counts word frequency. Neural embeddings learn semantic meaning from 
                billions of sentences. Same formula, smarter input.
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── Empty state ────────────────────────────────────────────────────
        st.markdown("""
        <div style="text-align:center; padding: 4rem 2rem; color:#444;">
            <div style="font-size:3rem; margin-bottom:1rem">🔍</div>
            <div style="font-size:1.1rem; margin-bottom:0.5rem; color:#666">Type a query above to search</div>
            <div style="font-size:0.85rem; color:#444">
                Try: "space shuttle launch" · "hockey playoffs" · "computer graphics software" · "gun control laws"
            </div>
        </div>
        """, unsafe_allow_html=True)
        
# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — CLUSTER EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗂️ Cluster Explorer":

    st.markdown('<div class="section-header">🗂️ Cluster Explorer</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-desc">K-Means clustering applied to 500 articles — no labels given. The algorithm discovered topic groups purely from TF-IDF vectors. Compare with neural embedding space.</div>',
                unsafe_allow_html=True)

    # ─── Toggle: TF-IDF vs Embeddings ─────────────────────────────────────────
    space = st.pills(
        label="Vector space",
        options=["📊 TF-IDF Space", "🤖 Neural Embedding Space"],
        default="📊 TF-IDF Space",
        label_visibility="visible"
    )

    coords = tfidf_2d if space == "📊 TF-IDF Space" else emb_2d
    space_label = "TF-IDF" if space == "📊 TF-IDF Space" else "Neural Embeddings"

    # ─── Toggle: color by cluster or true category ────────────────────────────
    color_by = st.pills(
        label="Color by",
        options=["🎯 True Category", "🔵 K-Means Cluster"],
        default="🎯 True Category",
        label_visibility="visible"
    )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ─── Build plot dataframe ──────────────────────────────────────────────────
    plot_df = pd.DataFrame({
        'x'        : coords[:, 0],
        'y'        : coords[:, 1],
        'category' : df['category'],
        'label'    : df['emoji_label'],
        'cluster'  : [f'Cluster {c}' for c in tfidf_clusters],
        'preview'  : df['text'].str[:120].str.replace('\n', ' ') + '...'
    })

    if color_by == "🎯 True Category":
        color_col   = 'label'
        color_map   = {v: CATEGORY_COLORS[k] for k, v in CATEGORY_LABELS.items()}
        plot_title  = f"True Categories in {space_label} Space"
    else:
        color_col  = 'cluster'
        color_map  = {
            'Cluster 0': '#7c6af7',
            'Cluster 1': '#f7516a',
            'Cluster 2': '#36d399',
            'Cluster 3': '#f4a62a'
        }
        plot_title = f"K-Means Clusters in {space_label} Space"

    # ─── Scatter plot ──────────────────────────────────────────────────────────
    fig = px.scatter(
        plot_df,
        x='x', y='y',
        color=color_col,
        hover_data={'preview': True, 'x': False, 'y': False,
                    'label': True, 'cluster': True},
        color_discrete_map=color_map,
        opacity=0.75,
        title=plot_title
    )

    fig.update_traces(
        marker=dict(size=7, line=dict(width=0.3, color='white')),
        hovertemplate="<b>%{customdata[2]}</b><br>Cluster: %{customdata[3]}<br><br>%{customdata[0]}<extra></extra>"
    )

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#444', size=12),
        legend=dict(
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#e0e0ee',
            borderwidth=1,
            font=dict(color='#444')
        ),
        xaxis=dict(
            gridcolor='#e8e8f0',
            zeroline=False,
            title='PCA Component 1'
        ),
        yaxis=dict(
            gridcolor='#e8e8f0',
            zeroline=False,
            title='PCA Component 2'
        ),
        title=dict(font=dict(color='#1a1a2e', size=14)),
        margin=dict(l=20, r=20, t=50, b=20),
        height=520
    )

    st.plotly_chart(fig, use_container_width=True)

    # ─── Cluster stats ─────────────────────────────────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("#### 📊 Cluster Composition")
    st.markdown('<div class="section-desc">How well did K-Means recover the true categories?</div>',
                unsafe_allow_html=True)

    cols = st.columns(4)
    for cluster_id in range(4):
        mask         = tfidf_clusters == cluster_id
        cluster_cats = df[mask]['category'].value_counts()
        dominant     = cluster_cats.index[0]
        dominant_pct = cluster_cats.iloc[0] / mask.sum() * 100
        total        = mask.sum()

        with cols[cluster_id]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Cluster {cluster_id}</div>
                <div class="metric-value">{CATEGORY_LABELS[dominant]}</div>
                <div style="font-size:0.85rem; color:#888; margin-top:0.3rem">
                    {dominant_pct:.0f}% dominant · {total} articles
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ─── Key insight ───────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size:0.85rem; color:#888; margin-bottom:0.5rem">💡 Key Insight</div>
        <div style="font-size:0.95rem; color:#333; line-height:1.6">
            K-Means was given <b style="color:#7c6af7">zero labels</b> — it never saw 
            "space", "hockey", "graphics", or "politics". It discovered these groups 
            purely by finding mathematical clusters in TF-IDF vector space. 
            This same mechanism organizes the internal knowledge of every LLM — 
            "Paris" clusters near "London", "king" clusters near "queen", 
            without anyone explicitly teaching it those relationships.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    
    # ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — NAÏVE BAYES CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎲 Naïve Bayes Classifier":

    st.markdown('<div class="section-header">🎲 Naïve Bayes Classifier</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Type any sentence and the model predicts which newsgroup category it belongs to — with confidence scores. Trained on 400 articles, tested on 100.</div>',
                unsafe_allow_html=True)

    # ─── Live classifier ───────────────────────────────────────────────────────
    user_input = st.text_area(
        label="Input text",
        placeholder="e.g. The NASA shuttle launched successfully into orbit yesterday...",
        height=120,
        label_visibility="collapsed"
    )

    predict_btn = st.button("🎲 Classify", type="primary")

    if predict_btn and user_input.strip():
        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # transform input and predict
        input_vec   = tfidf.transform([user_input])
        probs       = nb_model.predict_proba(input_vec)[0]
        pred_label  = nb_model.predict(input_vec)[0]
        pred_cat    = label_names[pred_label]
        pred_emoji  = CATEGORY_LABELS[pred_cat]

        # ── Winner card ────────────────────────────────────────────────────
        st.markdown(f"""
        <div class="metric-card" style="border-left: 3px solid #7c6af7; margin-bottom:1.5rem">
            <div style="font-size:0.8rem; color:#888; margin-bottom:0.3rem; 
                        text-transform:uppercase; letter-spacing:0.05em">
                Predicted Category
            </div>
            <div style="font-size:2.2rem; font-weight:800; color:#1a1a2e">
                {pred_emoji}
            </div>
            <div style="font-size:0.85rem; color:#555; margin-top:0.3rem">
                {pred_cat}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Confidence bars ────────────────────────────────────────────────
        st.markdown("#### 📊 Confidence Scores")

        sorted_indices = np.argsort(probs)[::-1]

        for idx in sorted_indices:
            cat      = label_names[idx]
            emoji    = CATEGORY_LABELS[cat]
            prob     = probs[idx]
            is_top   = idx == pred_label
            color    = '#7c6af7' if is_top else '#2a2a3a'
            txtcolor = '#e8e8e8' if is_top else '#666'

            st.markdown(f"""
            <div class="pred-bar-container" 
                 style="border-left: 3px solid {color}">
                <div style="display:flex; justify-content:space-between; 
                            margin-bottom:0.4rem">
                    <span style="color:{txtcolor}; font-weight:{'700' if is_top else '400'}">
                        {emoji}
                    </span>
                    <span style="color:{txtcolor}; font-weight:{'700' if is_top else '400'}">
                        {prob*100:.1f}%
                    </span>
                </div>
                <div style="background:#e8e8f0; border-radius:999px; 
                            height:6px; overflow:hidden">
                    <div style="background:{color}; width:{prob*100}%; 
                                height:100%; border-radius:999px;
                                transition: width 0.5s ease">
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── TF-IDF word scores for this input ──────────────────────────────
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown("#### 🔢 TF-IDF Scores for Your Input")
        st.markdown('<div class="section-desc">These are the words Naïve Bayes actually saw — scored by TF-IDF importance.</div>',
                    unsafe_allow_html=True)

        input_array  = input_vec.toarray()[0]
        nonzero_idx  = input_array.nonzero()[0]

        if len(nonzero_idx) > 0:
            word_scores = [(feature_names[i], input_array[i]) 
                           for i in nonzero_idx]
            word_scores.sort(key=lambda x: x[1], reverse=True)
            top_words   = word_scores[:15]

            fig = go.Figure(go.Bar(
                x=[s for _, s in top_words],
                y=[w for w, _ in top_words],
                orientation='h',
                marker_color='#7c6af7',
                opacity=0.85
            ))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#444', size=11),
                xaxis=dict(
                    gridcolor='#e8e8f0',
                    title='TF-IDF Score'
                ),
                yaxis=dict(
                    gridcolor='#e8e8f0',
                    autorange='reversed'
                ),
                margin=dict(l=20, r=20, t=10, b=20),
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)

            # show as badges too
            st.markdown("**Detected terms:**")
            badges = ''.join([
                f'<span class="badge">{w} ({s:.3f})</span>'
                for w, s in top_words
            ])
            st.markdown(badges, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="color:#555; font-size:0.9rem; padding:1rem">
                ⚠️ No known TF-IDF terms found in your input. 
                Try using more common words related to the categories.
            </div>
            """, unsafe_allow_html=True)

    elif predict_btn and not user_input.strip():
        st.warning("Please enter some text first.")

    else:
        # ── Empty state ────────────────────────────────────────────────────
        st.markdown("""
        <div style="text-align:center; padding: 4rem 2rem; color:#444;">
            <div style="font-size:3rem; margin-bottom:1rem">🎲</div>
            <div style="font-size:1.1rem; margin-bottom:0.5rem; color:#666">
                Type any text and click Classify
            </div>
            <div style="font-size:0.85rem; color:#444">
                Try: "The astronaut performed a spacewalk outside the ISS" · 
                "He scored the winning goal in overtime" · 
                "The shader renders realistic lighting effects"
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ─── Model performance stats ───────────────────────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("#### 🏆 Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Accuracy</div>
            <div class="metric-value">{nb_accuracy*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Training Articles</div>
            <div class="metric-value">400</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Test Articles</div>
            <div class="metric-value">100</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Categories</div>
            <div class="metric-value">4</div>
        </div>
        """, unsafe_allow_html=True)

    # ─── Confusion matrix ──────────────────────────────────────────────────────
    st.markdown("#### 🎯 Confusion Matrix")

    cm_norm = nb_cm.astype('float') / nb_cm.sum(axis=1)[:, np.newaxis]
    labels  = [CATEGORY_LABELS[c] for c in CATEGORIES]

    fig = px.imshow(
        cm_norm,
        labels=dict(x="Predicted", y="True", color="Score"),
        x=labels, y=labels,
        color_continuous_scale=[
            [0, '#0f0f13'],
            [0.5, '#3a2a6a'],
            [1, '#7c6af7']
        ],
        zmin=0, zmax=1,
        text_auto='.0%'
    )

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#444', size=11),
        margin=dict(l=20, r=20, t=20, b=20),
        height=380,
        coloraxis_showscale=False
    )
    fig.update_traces(textfont=dict(color='#1a1a2e', size=13))

    st.plotly_chart(fig, use_container_width=True)

    # ─── Key insight ───────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size:0.85rem; color:#888; margin-bottom:0.5rem">💡 Key Insight</div>
        <div style="font-size:0.95rem; color:#333; line-height:1.6">
            Naïve Bayes achieves <b style="color:#7c6af7">{nb_accuracy*100:.1f}% accuracy</b> 
            using nothing but word probabilities — no neural network, no GPU, no 
            billions of parameters. Gmail's spam filter still uses this exact algorithm 
            today. Modern LLMs extend this idea: instead of P(category | words), 
            they compute P(next word | all previous words) — same probabilistic 
            foundation, vastly larger scale.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    
    # ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — HOW IT WORKS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📖 How It Works":

    st.markdown('<div class="section-header">📖 How It Works</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-desc">The complete story — from classical data mining algorithms invented in the 1960s–70s to the AI systems running in production today.</div>',
                unsafe_allow_html=True)

    # ─── Timeline ─────────────────────────────────────────────────────────────
    st.markdown("#### 🕐 The Evolution Timeline")

    timeline = [
        ("1960s", "Naïve Bayes", "Probabilistic text classification invented. Still powers Gmail spam filter today.", "#f4a62a"),
        ("1967", "K-Means", "Unsupervised clustering algorithm published. Still organizes LLM embedding spaces today.", "#36d399"),
        ("1972", "TF-IDF", "Term frequency weighting invented for information retrieval. Foundation of all text vectorization.", "#7c6af7"),
        ("1972", "Cosine Similarity", "Angle-based similarity measure for vectors. Still the standard metric in every vector database.", "#f7516a"),
        ("2013", "Word2Vec", "Neural embeddings — same cosine similarity, but vectors learned from billions of words.", "#7c6af7"),
        ("2017", "Transformers", "Attention mechanism = weighted cosine similarity at massive scale. Powers GPT, BERT, Claude.", "#f7516a"),
        ("2020", "RAG Systems", "Retrieval Augmented Generation = TF-IDF pipeline + neural embeddings + LLM generation.", "#36d399"),
        ("2024+", "Modern LLMs", "Every query = K-Means space + Cosine Sim retrieval + Probabilistic next-token prediction.", "#f4a62a"),
    ]

    for year, name, desc, color in timeline:
        st.markdown(f"""
        <div style="display:flex; gap:1.2rem; margin-bottom:0.8rem; align-items:flex-start">
            <div style="min-width:52px; text-align:right">
                <span style="font-size:0.75rem; color:#555; font-weight:600">{year}</span>
            </div>
            <div style="width:2px; background:{color}; min-height:50px; 
                        border-radius:2px; margin-top:4px; flex-shrink:0"></div>
            <div class="metric-card" style="flex:1; margin-bottom:0; 
                         border-left: 3px solid {color}">
                <div style="font-size:0.95rem; font-weight:700; 
                            color:#1a1a2e; margin-bottom:0.2rem">{name}</div>
                <div style="font-size:0.85rem; color:#888; line-height:1.5">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ─── Algorithm explainers ──────────────────────────────────────────────────
    st.markdown("#### 🔬 Algorithm Deep Dive")

    algo = st.pills(
        label="Select algorithm",
        options=["TF-IDF", "Cosine Similarity", "K-Means", "Naïve Bayes"],
        default="TF-IDF",
        label_visibility="visible"
    )

    if algo == "TF-IDF":
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">What it does</div>
                <div style="color:#333; font-size:0.9rem; margin-top:0.5rem; line-height:1.7">
                    Converts raw text into a vector of numbers where each number 
                    represents how <b style="color:#7c6af7">important</b> a word is 
                    to a specific document.<br><br>
                    <b>TF</b> — how often the word appears in this document<br>
                    <b>IDF</b> — how rare the word is across all documents<br>
                    <b>TF-IDF</b> = TF × IDF → important AND rare = high score
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Modern counterpart</div>
                <div style="color:#333; font-size:0.9rem; margin-top:0.5rem; line-height:1.7">
                    <b style="color:#7c6af7">Neural Embeddings</b> (Word2Vec, BERT, GPT)<br><br>
                    Same goal: turn words into numbers.<br>
                    But instead of counting frequency, a neural network 
                    <b>learns</b> from billions of sentences what "similar" 
                    actually means in context.<br><br>
                    "rocket" and "spacecraft" get similar vectors even though 
                    they share zero characters.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # live TF-IDF demo
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown("#### 🧪 Live TF-IDF Demo")
        st.markdown('<div class="section-desc">Type a sentence and see TF-IDF scores in real time.</div>',
                    unsafe_allow_html=True)

        demo_text = st.text_input(
            "Enter a sentence",
            placeholder="e.g. NASA launched a rocket to the moon",
            label_visibility="collapsed"
        )

        if demo_text.strip():
            vec         = tfidf.transform([demo_text])
            arr         = vec.toarray()[0]
            nonzero     = arr.nonzero()[0]

            if len(nonzero) > 0:
                word_scores = sorted(
                    [(feature_names[i], arr[i]) for i in nonzero],
                    key=lambda x: x[1], reverse=True
                )[:12]

                fig = go.Figure(go.Bar(
                    x=[w for w, _ in word_scores],
                    y=[s for _, s in word_scores],
                    marker_color='#7c6af7',
                    opacity=0.85
                ))
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#444', size=11),
                    xaxis=dict(gridcolor='#e8e8f0'),
                    yaxis=dict(gridcolor='#e8e8f0', title='TF-IDF Score'),
                    margin=dict(l=20, r=20, t=10, b=20),
                    height=280
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("**Words recognized in our vocabulary:**")
                badges = ''.join([
                    f'<span class="badge">{w} {s:.3f}</span>'
                    for w, s in word_scores
                ])
                st.markdown(badges, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="color:#555; padding:1rem; font-size:0.9rem">
                    ⚠️ No known vocabulary terms found. 
                    Try words related to space, hockey, graphics, or politics.
                </div>
                """, unsafe_allow_html=True)

    elif algo == "Cosine Similarity":
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">What it does</div>
                <div style="color:#333; font-size:0.9rem; margin-top:0.5rem; line-height:1.7">
                    Measures the <b style="color:#f7516a">angle</b> between two vectors 
                    instead of their distance — making it length-independent.<br><br>
                    <b>Score = 1.0</b> → identical direction → identical meaning<br>
                    <b>Score = 0.0</b> → 90° angle → completely unrelated<br>
                    <b>Score = -1.0</b> → opposite direction → opposite meaning<br><br>
                    Formula: cos(θ) = (A · B) / (|A| × |B|)
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Modern counterpart</div>
                <div style="color:#f7516a; font-size:0.9rem; margin-top:0.5rem; line-height:1.7">
                    <b>Still cosine similarity.</b> Exactly.<br><br>
                    <span style="color:#333">
                    pgvector uses it. Pinecone uses it. Weaviate uses it. 
                    ChromaDB uses it. Every vector database today computes 
                    the same formula from 1972 to find similar documents.<br><br>
                    When ChatGPT retrieves context for RAG, it's running 
                    cosine similarity between your question vector and 
                    millions of stored document vectors.
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # visual demo
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown("#### 🧪 Live Similarity Demo")

        c1, c2 = st.columns(2)
        with c1:
            text_a = st.text_input("Text A", 
                                   value="NASA launched a rocket to orbit",
                                   label_visibility="visible")
        with c2:
            text_b = st.text_input("Text B",
                                   value="The space shuttle reached the ISS",
                                   label_visibility="visible")

        if text_a.strip() and text_b.strip():
            vecs  = tfidf.transform([text_a, text_b])
            vecs  = normalize(vecs, norm='l2')
            score = cosine_similarity(vecs[0], vecs[1])[0][0]

            color = '#36d399' if score > 0.3 else '#f7516a' if score < 0.1 else '#f4a62a'
            label = 'Very Similar' if score > 0.3 else 'Unrelated' if score < 0.1 else 'Somewhat Related'

            st.markdown(f"""
            <div class="metric-card" style="text-align:center; border-left: 3px solid {color}">
                <div class="metric-label">Cosine Similarity Score</div>
                <div style="font-size:3rem; font-weight:800; color:{color}">
                    {score:.4f}
                </div>
                <div style="font-size:0.9rem; color:{color}; margin-top:0.3rem">
                    {label}
                </div>
            </div>
            """, unsafe_allow_html=True)

    elif algo == "K-Means":
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">What it does</div>
                <div style="color:#333; font-size:0.9rem; margin-top:0.5rem; line-height:1.7">
                    Finds <b style="color:#36d399">K natural groups</b> in data 
                    without any labels — purely unsupervised.<br><br>
                    <b>Step 1</b> → Pick K random centroids<br>
                    <b>Step 2</b> → Assign every point to nearest centroid<br>
                    <b>Step 3</b> → Move centroid to average of its points<br>
                    <b>Step 4</b> → Repeat until stable<br><br>
                    Result: natural topic groups emerge from raw numbers.
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Modern counterpart</div>
                <div style="color:#333; font-size:0.9rem; margin-top:0.5rem; line-height:1.7">
                    <b style="color:#36d399">LLM Embedding Space Organization</b><br><br>
                    Inside every LLM, the vector space organizes itself exactly 
                    like K-Means clusters. "Paris", "London", "Tokyo" cluster 
                    together. "happy", "joyful", "excited" cluster together.<br><br>
                    Researchers use K-Means to <b>visualize and analyze</b> 
                    what concepts an LLM has learned. The TensorFlow Embedding 
                    Projector is K-Means visualization applied to neural embeddings.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # clustering metrics
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown("#### 📊 Clustering Quality Metrics")

        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        from sklearn.cluster import KMeans as KM

        emb_kmeans     = KM(n_clusters=4, random_state=RANDOM_STATE, n_init=10)
        emb_clusters   = emb_kmeans.fit_predict(embeddings)

        ari_tfidf = adjusted_rand_score(df['label'].values, tfidf_clusters)
        ari_emb   = adjusted_rand_score(df['label'].values, emb_clusters)
        nmi_tfidf = normalized_mutual_info_score(df['label'].values, tfidf_clusters)
        nmi_emb   = normalized_mutual_info_score(df['label'].values, emb_clusters)

        mc1, mc2, mc3, mc4 = st.columns(4)
        for col, label, val, color in [
            (mc1, "ARI — TF-IDF",   ari_tfidf, '#7c6af7'),
            (mc2, "ARI — Neural",   ari_emb,   '#f7516a'),
            (mc3, "NMI — TF-IDF",   nmi_tfidf, '#7c6af7'),
            (mc4, "NMI — Neural",   nmi_emb,   '#f7516a'),
        ]:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value" style="color:{color}">{val:.3f}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("""
        <div style="font-size:0.85rem; color:#555; margin-top:0.5rem">
            ARI & NMI range 0→1. Higher = clusters better match true categories.
            Neural embeddings produce tighter, more meaningful clusters.
        </div>
        """, unsafe_allow_html=True)

    elif algo == "Naïve Bayes":
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">What it does</div>
                <div style="color:#333; font-size:0.9rem; margin-top:0.5rem; line-height:1.7">
                    Classifies text using <b style="color:#f4a62a">word probabilities</b> 
                    learned from training data.<br><br>
                    For each category it learns:<br>
                    P("NASA" | space) = 0.80<br>
                    P("NASA" | hockey) = 0.02<br><br>
                    For a new article it computes:<br>
                    P(space | article) vs P(hockey | article)<br>
                    → picks the highest probability category
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Modern counterpart</div>
                <div style="color:#333; font-size:0.9rem; margin-top:0.5rem; line-height:1.7">
                    <b style="color:#f4a62a">Probabilistic Language Models</b><br><br>
                    GPT predicts the next token by computing:<br>
                    P(next_word | all_previous_words)<br><br>
                    Naïve Bayes computes:<br>
                    P(category | words)<br><br>
                    Same probabilistic foundation. GPT just conditions on 
                    the entire context window instead of assuming independence — 
                    50 years and billions of parameters later.
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ─── Final conclusion ──────────────────────────────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card" style="border-left: 3px solid #7c6af7; 
                                     background: linear-gradient(135deg, #f0eeff, #fff0f3)">
        <div style="font-size:0.8rem; color:#555; margin-bottom:0.8rem; 
                    text-transform:uppercase; letter-spacing:0.08em">
            Core Argument
        </div>
        <div style="font-size:1.1rem; color:#1a1a2e; line-height:1.8; font-weight:500">
            "We spent this semester learning K-Means, Naïve Bayes, cosine similarity, 
            and association rules. These aren't legacy concepts we'll forget after the exam. 
            They are the architecture that every AI company in the world is running 
            in production right now. <b style="color:#7c6af7">We didn't study the past — 
            we studied the foundation.</b>"
        </div>
        <div style="font-size:0.85rem; color:#555; margin-top:1rem">
            — 林宇辰 (Niko Yoga Pranata) · L25020019
        </div>
    </div>
    """, unsafe_allow_html=True)