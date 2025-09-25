# Part 4: Simple Streamlit Application
# This follows the example structure provided in the assignment

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import re

# Configure page
st.set_page_config(
    page_title="CORD-19 Data Explorer",
    page_icon="🦠",
    layout="wide"
)

# Title and description
st.title("🦠 CORD-19 Data Explorer")
st.write("Simple exploration of COVID-19 research papers")

# Add some introductory text
st.markdown("""
This application provides a simple interface to explore the CORD-19 dataset, 
which contains metadata about COVID-19 research papers. Use the interactive 
elements below to filter and explore the data.
""")

@st.cache_data
def load_data():
    """Load the cleaned dataset"""
    try:
        # Try to load the cleaned data from previous parts
        df = pd.read_csv('cord19_cleaned.csv')
        df['publish_time'] = pd.to_datetime(df['publish_time'])
        return df
    except FileNotFoundError:
        # Create sample data if file doesn't exist
        st.warning("Cleaned data file not found. Using sample data for demonstration.")
        
        np.random.seed(42)
        n_samples = 5000
        
        df = pd.DataFrame({
            'cord_uid': [f'cord-{i:07d}' for i in range(n_samples)],
            'source_x': np.random.choice(['PMC', 'medRxiv', 'bioRxiv', 'Elsevier'], n_samples),
            'title': [f'COVID-19 study on {np.random.choice(["vaccines", "treatments", "transmission", "variants", "prevention"])}' 
                      for _ in range(n_samples)],
            'abstract': [f'This study examines COVID-19 research. ' * np.random.randint(20, 80) 
                         for _ in range(n_samples)],
            'publish_time': pd.date_range(start='2020-01-01', end='2023-12-31', periods=n_samples),
            'authors': [f'Author{i%20}, B.; Researcher{(i+1)%15}, C.' for i in range(n_samples)],
            'journal': np.random.choice([
                'Nature', 'Science', 'Cell', 'The Lancet', 'NEJM', 'PLOS ONE', 
                'Journal of Virology', 'PNAS', 'BMJ', 'medRxiv', 'bioRxiv'
            ], n_samples),
            'year': np.random.choice(range(2020, 2024), n_samples),
            'abstract_word_count': np.random.randint(100, 500, n_samples),
            'title_word_count': np.random.randint(5, 20, n_samples),
            'author_count': np.random.randint(1, 8, n_samples),
            'source_category': np.random.choice(['Published', 'Preprint'], n_samples)
        })
        
        return df

# Load data
with st.spinner('Loading data...'):
    df = load_data()

st.success(f"✅ Data loaded successfully! {len(df):,} papers in dataset")

# Sidebar for filters
st.sidebar.header("📊 Filters")

# Year range slider (following the example structure)
min_year = int(df['year'].min())
max_year = int(df['year'].max())

year_range = st.sidebar.slider(
    "Select year range", 
    min_year, 
    max_year, 
    (min_year, max_year)
)

# Journal selector
journals = sorted(df['journal'].unique())
selected_journals = st.sidebar.multiselect(
    "Select journals",
    journals,
    default=journals[:5]  # Default to first 5 journals
)

# Source filter
sources = sorted(df['source_x'].unique())
selected_sources = st.sidebar.multiselect(
    "Select sources",
    sources,
    default=sources
)

# Filter the data based on selections
filtered_df = df[
    (df['year'] >= year_range[0]) & 
    (df['year'] <= year_range[1]) &
    (df['journal'].isin(selected_journals) if selected_journals else True) &
    (df['source_x'].isin(selected_sources) if selected_sources else True)
]

st.sidebar.write(f"Filtered dataset: {len(filtered_df):,} papers")

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.metric(
        label="Total Papers",
        value=f"{len(filtered_df):,}",
        delta=f"{len(filtered_df) - len(df):,}" if len(filtered_df) != len(df) else None
    )

with col2:
    st.metric(
        label="Date Range",
        value=f"{filtered_df['year'].min()} - {filtered_df['year'].max()}"
    )

# Visualization 1: Publications by year
st.subheader("📈 Publications Over Time")

if not filtered_df.empty:
    yearly_counts = filtered_df['year'].value_counts().sort_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    yearly_counts.plot(kind='bar', ax=ax, color='skyblue', alpha=0.8)
    ax.set_title('Number of Publications by Year', fontsize=14, fontweight='bold')
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Publications')
    ax.tick_params(axis='x', rotation=0)
    plt.grid(axis='y', alpha=0.3)
    st.pyplot(fig)
    plt.close()
else:
    st.warning("No data to display with current filters.")

# Visualization 2: Top journals
st.subheader("🏆 Top Publishing Journals")

if not filtered_df.empty:
    top_journals = filtered_df['journal'].value_counts().head(10)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    top_journals.plot(kind='barh', ax=ax, color='lightcoral', alpha=0.8)
    ax.set_title('Top 10 Journals', fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Publications')
    plt.grid(axis='x', alpha=0.3)
    st.pyplot(fig)
    plt.close()

# Visualization 3: Source distribution
st.subheader("📊 Publication Sources")

if not filtered_df.empty:
    source_counts = filtered_df['source_x'].value_counts()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(source_counts.values, labels=source_counts.index, autopct='%1.1f%%', startangle=90)
    ax.set_title('Distribution by Source', fontsize=14, fontweight='bold')
    st.pyplot(fig)
    plt.close()

# Abstract length analysis
st.subheader("📝 Abstract Length Analysis")

if not filtered_df.empty and 'abstract_word_count' in filtered_df.columns:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Average Length",
            value=f"{filtered_df['abstract_word_count'].mean():.0f} words"
        )
    
    with col2:
        st.metric(
            label="Median Length",
            value=f"{filtered_df['abstract_word_count'].median():.0f} words"
        )
    
    with col3:
        st.metric(
            label="Max Length",
            value=f"{filtered_df['abstract_word_count'].max():.0f} words"
        )
    
    # Histogram of abstract lengths
    fig, ax = plt.subplots(figsize=(10, 6))
    filtered_df['abstract_word_count'].hist(bins=30, ax=ax, color='lightgreen', alpha=0.7)
    ax.axvline(filtered_df['abstract_word_count'].mean(), color='red', linestyle='--', 
               label=f'Mean: {filtered_df["abstract_word_count"].mean():.0f}')
    ax.set_title('Distribution of Abstract Lengths', fontsize=14, fontweight='bold')
    ax.set_xlabel('Word Count')
    ax.set_ylabel('Frequency')
    ax.legend()
    plt.grid(axis='y', alpha=0.3)
    st.pyplot(fig)
    plt.close()

# Simple keyword analysis
st.subheader("🔍 Common Keywords in Titles")

if not filtered_df.empty:
    def extract_keywords(titles, n=15):
        """Extract common keywords from titles"""
        all_text = ' '.join(titles.fillna('').astype(str).str.lower())
        words = re.findall(r'\b[a-zA-Z]{4,}\b', all_text)
        
        # Remove common stop words
        stop_words = {
            'covid', 'coronavirus', 'sars', 'pandemic', 'study', 'analysis', 'research',
            'paper', 'article', 'with', 'from', 'this', 'that', 'have', 'been', 'were'
        }
        
        words = [w for w in words if w not in stop_words]
        return Counter(words).most_common(n)
    
    keywords = extract_keywords(filtered_df['title'])
    
    if keywords:
        keywords_df = pd.DataFrame(keywords, columns=['Keyword', 'Frequency'])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(keywords_df['Keyword'], keywords_df['Frequency'], color='mediumseagreen', alpha=0.8)
        ax.set_title('Most Common Keywords in Titles', fontsize=14, fontweight='bold')
        ax.set_xlabel('Frequency')
        ax.invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        st.pyplot(fig)
        plt.close()
        
        # Show keywords table
        st.dataframe(keywords_df, use_container_width=True)

# Sample data display
st.subheader("📄 Sample Data")

if not filtered_df.empty:
    # Show a sample of the filtered data
    sample_columns = ['title', 'journal', 'year', 'source_x']
    if 'abstract_word_count' in filtered_df.columns:
        sample_columns.append('abstract_word_count')
    
    st.write(f"Showing 10 random papers from the filtered dataset:")
    sample_data = filtered_df[sample_columns].sample(min(10, len(filtered_df)))
    st.dataframe(sample_data, use_container_width=True)

# Summary statistics
st.subheader("📊 Summary Statistics")

if not filtered_df.empty:
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Dataset Summary:**")
        st.write(f"• Total papers: {len(filtered_df):,}")
        st.write(f"• Unique journals: {filtered_df['journal'].nunique()}")
        st.write(f"• Unique sources: {filtered_df['source_x'].nunique()}")
        st.write(f"• Time span: {filtered_df['year'].max() - filtered_df['year'].min() + 1} years")
        
        if 'author_count' in filtered_df.columns:
            st.write(f"• Avg authors per paper: {filtered_df['author_count'].mean():.1f}")
    
    with col2:
        st.write("**Top Contributors:**")
        top_journal = filtered_df['journal'].mode().iloc[0] if not filtered_df['journal'].empty else "N/A"
        st.write(f"• Most active journal: {top_journal}")
        
        top_source = filtered_df['source_x'].mode().iloc[0] if not filtered_df['source_x'].empty else "N/A"
        st.write(f"• Most common source: {top_source}")
        
        peak_year = filtered_df['year'].mode().iloc[0] if not filtered_df['year'].empty else "N/A"
        st.write(f"• Peak publication year: {peak_year}")

# Download section
st.subheader("💾 Download Data")

if not filtered_df.empty:
    # Create downloadable CSV
    csv_data = filtered_df.to_csv(index=False)
    
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv_data,
        file_name=f"cord19_filtered_data_{year_range[0]}_{year_range[1]}.csv",
        mime="text/csv"
    )
    
    # Create summary report
    summary_stats = {
        'Metric': [
            'Total Papers',
            'Date Range', 
            'Unique Journals',
            'Unique Sources',
            'Average Abstract Length (words)',
            'Most Common Journal',
            'Most Common Source'
        ],
        'Value': [
            len(filtered_df),
            f"{filtered_df['year'].min()} - {filtered_df['year'].max()}",
            filtered_df['journal'].nunique(),
            filtered_df['source_x'].nunique(),
            f"{filtered_df['abstract_word_count'].mean():.0f}" if 'abstract_word_count' in filtered_df.columns else 'N/A',
            filtered_df['journal'].mode().iloc[0] if not filtered_df['journal'].empty else 'N/A',
            filtered_df['source_x'].mode().iloc[0] if not filtered_df['source_x'].empty else 'N/A'
        ]
    }
    
    summary_df = pd.DataFrame(summary_stats)
    summary_csv = summary_df.to_csv(index=False)
    
    st.download_button(
        label="Download Summary Report",
        data=summary_csv,
        file_name="analysis_summary.csv",
        mime="text/csv"
    )

# Interactive features section
st.subheader("🎯 Key Insights")

if not filtered_df.empty:
    insights = []
    
    # Generate dynamic insights based on filtered data
    total_papers = len(filtered_df)
    date_span = filtered_df['year'].max() - filtered_df['year'].min() + 1
    
    if total_papers > 1000:
        insights.append(f"📊 Large dataset: {total_papers:,} papers analyzed")
    
    if date_span > 1:
        yearly_trend = filtered_df['year'].value_counts().sort_index()
        if len(yearly_trend) > 1:
            growth_rate = ((yearly_trend.iloc[-1] - yearly_trend.iloc[0]) / yearly_trend.iloc[0]) * 100
            if growth_rate > 20:
                insights.append(f"📈 Strong growth: {growth_rate:.1f}% increase from first to last year")
            elif growth_rate < -20:
                insights.append(f"📉 Declining trend: {abs(growth_rate):.1f}% decrease from first to last year")
    
    top_journal = filtered_df['journal'].value_counts().iloc[0]
    journal_dominance = (top_journal / total_papers) * 100
    if journal_dominance > 15:
        insights.append(f"🏆 Journal concentration: Top journal publishes {journal_dominance:.1f}% of papers")
    
    if 'abstract_word_count' in filtered_df.columns:
        avg_length = filtered_df['abstract_word_count'].mean()
        if avg_length > 300:
            insights.append(f"📝 Comprehensive abstracts: Average of {avg_length:.0f} words")
        elif avg_length < 150:
            insights.append(f"📝 Concise abstracts: Average of {avg_length:.0f} words")
    
    if 'source_category' in filtered_df.columns:
        preprint_pct = (filtered_df['source_category'] == 'Preprint').mean() * 100
        if preprint_pct > 40:
            insights.append(f"⚡ High preprint activity: {preprint_pct:.1f}% are preprints")
    
    # Display insights
    for insight in insights[:5]:  # Show top 5 insights
        st.info(insight)

else:
    st.warning("⚠️ No data matches your current filter selection. Please adjust the filters.")

# Help section
with st.expander("ℹ️ How to Use This App"):
    st.markdown("""
    **Getting Started:**
    1. Use the sidebar filters to narrow down the data
    2. Adjust the year range slider to focus on specific time periods
    3. Select specific journals or sources you're interested in
    
    **Understanding the Visualizations:**
    - **Publications Over Time**: Shows publication trends by year
    - **Top Journals**: Identifies the most active publishers
    - **Source Distribution**: Shows the breakdown between different publication sources
    - **Abstract Length**: Analyzes the comprehensiveness of research papers
    
    **Interactive Features:**
    - All charts update automatically when you change filters
    - Download filtered data or summary reports using the download buttons
    - View sample papers in the data table
    
    **Tips:**
    - Start with broad filters and gradually narrow down
    - Compare different time periods by adjusting the year range
    - Look for patterns in journal preferences and publication trends
    """)

# Technical details
with st.expander("🔧 Technical Details"):
    st.markdown(f"""
    **Dataset Information:**
    - Original dataset: CORD-19 COVID-19 Research Dataset
    - Total records processed: {len(df):,}
    - Data processing: Cleaned and standardized in previous analysis steps
    - Last updated: Real-time filtering based on your selections
    
    **Application Details:**
    - Built with: Streamlit {st.__version__}
    - Visualization: Matplotlib and Seaborn
    - Data processing: Pandas
    - Caching: Enabled for optimal performance
    
    **Data Quality:**
    - Missing values handled through preprocessing
    - Date validation applied
    - Text standardization performed
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>CORD-19 Data Explorer | Built with Streamlit | COVID-19 Research Analysis</p>
    <p>Assignment: Frameworks and Data Analysis | Step-by-step implementation</p>
</div>
""", unsafe_allow_html=True)

# Additional interactive widget example (following assignment structure)
st.sidebar.markdown("---")
st.sidebar.subheader("📈 Quick Stats")

if not filtered_df.empty:
    # Display quick statistics in sidebar
    st.sidebar.metric("Papers in Selection", len(filtered_df))
    
    if 'abstract_word_count' in filtered_df.columns:
        avg_abstract = filtered_df['abstract_word_count'].mean()
        st.sidebar.metric("Avg Abstract Length", f"{avg_abstract:.0f} words")
    
    journals_count = filtered_df['journal'].nunique()
    st.sidebar.metric("Unique Journals", journals_count)
    
    # Simple progress bar showing data completeness
    completeness = (len(filtered_df) / len(df)) * 100
    st.sidebar.progress(completeness / 100)
    st.sidebar.caption(f"Using {completeness:.1f}% of total dataset")

else:
    st.sidebar.warning("No data in current selection")
