# Part 3: Data Analysis and Visualization
# Time estimate: 3-4 hours

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

print("CORD-19 Dataset Analysis - Part 3: Data Analysis and Visualization")
print("=" * 75)

# Load cleaned data from Part 2
print("Loading cleaned data from Part 2...")

try:
    df = pd.read_csv('cord19_cleaned.csv')
    df['publish_time'] = pd.to_datetime(df['publish_time'])
    print(f"✓ Cleaned data loaded: {df.shape}")
except FileNotFoundError:
    print("Cleaned data not found. Creating sample data...")
    # Recreate sample data if file doesn't exist
    np.random.seed(42)
    n_samples = 8000  # After cleaning
    
    df = pd.DataFrame({
        'cord_uid': [f'cord-{i:07d}' for i in range(n_samples)],
        'source_x': np.random.choice(['PMC', 'medRxiv', 'bioRxiv', 'Elsevier'], n_samples),
        'title': [f'COVID-19 research on {np.random.choice(["vaccine efficacy", "treatment protocols", "epidemiological analysis", "public health measures", "genetic variants"])}' 
                  for _ in range(n_samples)],
        'abstract': [f'This comprehensive study examines COVID-19 {np.random.choice(["vaccines", "treatments", "transmission", "variants", "prevention"])}. ' * np.random.randint(20, 50) 
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
    print(f"✓ Sample data created: {df.shape}")

# Task 1: Perform basic analysis
print("\n1. BASIC ANALYSIS")
print("-" * 18)

# Count papers by publication year
print("Papers by publication year:")
yearly_counts = df['year'].value_counts().sort_index()
print(yearly_counts)

# Create a simple bar plot for yearly publications
plt.figure(figsize=(10, 6))
yearly_counts.plot(kind='bar', color='skyblue', alpha=0.8)
plt.title('Number of COVID-19 Papers Published by Year', fontsize=16, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Number of Papers', fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('papers_by_year.png', dpi=300, bbox_inches='tight')
plt.show()

# Identify top journals publishing COVID-19 research
print("\nTop 10 journals publishing COVID-19 research:")
top_journals = df['journal'].value_counts().head(10)
print(top_journals)

# Create visualization for top journals
plt.figure(figsize=(12, 8))
top_journals.plot(kind='barh', color='lightcoral', alpha=0.8)
plt.title('Top 10 Journals Publishing COVID-19 Research', fontsize=16, fontweight='bold')
plt.xlabel('Number of Papers', fontsize=12)
plt.ylabel('Journal', fontsize=12)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('top_journals.png', dpi=300, bbox_inches='tight')
plt.show()

# Find most frequent words in titles
print("\nAnalyzing title keywords...")

def extract_title_keywords(titles, top_n=20):
    """Extract most common keywords from titles"""
    # Combine all titles
    all_titles = ' '.join(titles.fillna('').astype(str))
    
    # Simple text preprocessing
    words = re.findall(r'\b[a-zA-Z]{3,}\b', all_titles.lower())
    
    # Remove common stop words
    stop_words = {
        'and', 'the', 'of', 'in', 'to', 'for', 'with', 'on', 'by', 'from', 'up', 'about', 'into', 
        'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among', 'under',
        'covid', 'coronavirus', 'sars', 'cov', 'pandemic', 'research', 'study', 'analysis'
    }
    
    words = [word for word in words if word not in stop_words and len(word) > 3]
    return Counter(words).most_common(top_n)

title_keywords = extract_title_keywords(df['title'])
print("Most frequent keywords in titles:")
for word, count in title_keywords[:15]:
    print(f"{word}: {count}")

# Task 2: Create visualizations
print("\n2. CREATING VISUALIZATIONS")
print("-" * 27)

# Visualization 1: Number of publications over time (monthly)
print("Creating publication timeline visualization...")

df['year_month'] = df['publish_time'].dt.to_period('M')
monthly_counts = df.groupby('year_month').size()

plt.figure(figsize=(15, 6))
monthly_counts.plot(kind='line', marker='o', linewidth=2, markersize=4, color='darkblue')
plt.title('COVID-19 Research Publications Over Time (Monthly)', fontsize=16, fontweight='bold')
plt.xlabel('Time Period', fontsize=12)
plt.ylabel('Number of Publications', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('publications_timeline.png', dpi=300, bbox_inches='tight')
plt.show()

# Visualization 2: Create a bar chart of top publishing journals (enhanced)
print("Creating enhanced journal distribution...")

plt.figure(figsize=(14, 8))
top_12_journals = df['journal'].value_counts().head(12)
colors = plt.cm.Set3(np.linspace(0, 1, len(top_12_journals)))

bars = plt.barh(range(len(top_12_journals)), top_12_journals.values, color=colors, alpha=0.8)
plt.yticks(range(len(top_12_journals)), top_12_journals.index)
plt.xlabel('Number of Publications', fontsize=12)
plt.title('Top 12 Journals Publishing COVID-19 Research', fontsize=16, fontweight='bold')
plt.grid(axis='x', alpha=0.3)

# Add value labels on bars
for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(width + 5, bar.get_y() + bar.get_height()/2, 
             f'{int(width)}', ha='left', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('journal_distribution_enhanced.png', dpi=300, bbox_inches='tight')
plt.show()

# Visualization 3: Generate a word cloud of paper titles
print("Creating word cloud of paper titles...")

try:
    # Combine all titles
    title_text = ' '.join(df['title'].fillna('').astype(str))
    
    # Create word cloud
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        colormap='viridis',
        max_words=100,
        stopwords=set(['covid', 'coronavirus', 'sars', 'cov', 'pandemic', 'research', 'study', 'analysis', 'paper'])
    ).generate(title_text)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of COVID-19 Paper Titles', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('title_wordcloud.png', dpi=300, bbox_inches='tight')
    plt.show()
    
except ImportError:
    print("WordCloud library not available. Creating alternative visualization...")
    
    # Alternative: Bar chart of top words
    plt.figure(figsize=(12, 6))
    words, counts = zip(*title_keywords[:15])
    plt.barh(range(len(words)), counts, color='mediumseagreen', alpha=0.8)
    plt.yticks(range(len(words)), words)
    plt.xlabel('Frequency', fontsize=12)
    plt.title('Most Common Words in COVID-19 Paper Titles', fontsize=16, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('title_keywords.png', dpi=300, bbox_inches='tight')
    plt.show()

# Visualization 4: Plot distribution of paper counts by source
print("Creating source distribution visualization...")

source_counts = df['source_x'].value_counts()

# Pie chart
plt.figure(figsize=(10, 8))
colors = plt.cm.Pastel1(np.linspace(0, 1, len(source_counts)))
wedges, texts, autotexts = plt.pie(source_counts.values, labels=source_counts.index, 
                                   autopct='%1.1f%%', colors=colors, startangle=90)
plt.title('Distribution of COVID-19 Papers by Source', fontsize=16, fontweight='bold')

# Enhance text
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

plt.axis('equal')
plt.tight_layout()
plt.savefig('source_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Additional Analysis and Visualizations
print("\n3. ADDITIONAL ANALYSIS")
print("-" * 23)

# Abstract length analysis
print("Abstract length analysis:")
print(f"Average abstract length: {df['abstract_word_count'].mean():.1f} words")
print(f"Median abstract length: {df['abstract_word_count'].median():.1f} words")
print(f"Standard deviation: {df['abstract_word_count'].std():.1f} words")

# Abstract length distribution
plt.figure(figsize=(12, 6))
plt.hist(df['abstract_word_count'], bins=50, color='lightblue', alpha=0.7, edgecolor='black')
plt.axvline(df['abstract_word_count'].mean(), color='red', linestyle='--', 
           label=f'Mean: {df["abstract_word_count"].mean():.1f}')
plt.axvline(df['abstract_word_count'].median(), color='green', linestyle='--', 
           label=f'Median: {df["abstract_word_count"].median():.1f}')
plt.xlabel('Abstract Word Count', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Abstract Lengths', fontsize=16, fontweight='bold')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('abstract_length_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Correlation analysis between numerical variables
print("\nCorrelation analysis:")
numerical_cols = ['abstract_word_count', 'title_word_count', 'author_count', 'year']
correlation_matrix = df[numerical_cols].corr()
print(correlation_matrix)

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5)
plt.title('Correlation Matrix of Numerical Variables', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# Publication trends by source category
print("\nPublication trends by source category:")
source_yearly = pd.crosstab(df['year'], df['source_category'])
print(source_yearly)

plt.figure(figsize=(12, 6))
source_yearly.plot(kind='bar', stacked=True, color=['lightcoral', 'lightblue'])
plt.title('Publication Trends by Source Category', fontsize=16, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Number of Publications', fontsize=12)
plt.legend(title='Source Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('source_category_trends.png', dpi=300, bbox_inches='tight')
plt.show()

# Author collaboration analysis
print("\nAuthor collaboration analysis:")
print(f"Average authors per paper: {df['author_count'].mean():.2f}")
print(f"Papers with single author: {(df['author_count'] == 1).sum()} ({(df['author_count'] == 1).mean()*100:.1f}%)")
print(f"Papers with 5+ authors: {(df['author_count'] >= 5).sum()} ({(df['author_count'] >= 5).mean()*100:.1f}%)")

# Author count distribution
plt.figure(figsize=(10, 6))
author_dist = df['author_count'].value_counts().sort_index()
plt.bar(author_dist.index, author_dist.values, color='gold', alpha=0.8, edgecolor='black')
plt.xlabel('Number of Authors', fontsize=12)
plt.ylabel('Number of Papers', fontsize=12)
plt.title('Distribution of Author Count per Paper', fontsize=16, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('author_count_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Task 3: Summary insights
print("\n4. KEY INSIGHTS AND FINDINGS")
print("-" * 32)

insights = []

# Publication trends
yearly_growth = yearly_counts.pct_change().dropna()
if len(yearly_growth) > 0:
    avg_growth = yearly_growth.mean() * 100
    if avg_growth > 5:
        insights.append(f"📈 Strong growth trend: Average year-over-year growth of {avg_growth:.1f}%")
    elif avg_growth < -5:
        insights.append(f"📉 Declining trend: Average year-over-year decline of {abs(avg_growth):.1f}%")
    else:
        insights.append(f"📊 Stable publication rate with moderate fluctuations")

# Journal insights
top_journal = top_journals.index[0]
top_journal_share = (top_journals.iloc[0] / len(df)) * 100
insights.append(f"🏆 '{top_journal}' leads with {top_journals.iloc[0]} papers ({top_journal_share:.1f}% of total)")

# Abstract insights
avg_abstract_length = df['abstract_word_count'].mean()
if avg_abstract_length > 300:
    insights.append(f"📝 Research papers tend to be comprehensive (avg: {avg_abstract_length:.0f} words per abstract)")
else:
    insights.append(f"📝 Research papers are concise (avg: {avg_abstract_length:.0f} words per abstract)")

# Collaboration insights
avg_authors = df['author_count'].mean()
if avg_authors > 4:
    insights.append(f"👥 High collaboration: Average of {avg_authors:.1f} authors per paper")
else:
    insights.append(f"👥 Moderate collaboration: Average of {avg_authors:.1f} authors per paper")

# Source insights
preprint_share = (df['source_category'] == 'Preprint').mean() * 100
if preprint_share > 30:
    insights.append(f"⚡ High preprint activity: {preprint_share:.1f}% of papers are preprints")
else:
    insights.append(f"📚 Emphasis on peer review: {100-preprint_share:.1f}% are published papers")

# Display insights
print("Key findings from the analysis:")
for i, insight in enumerate(insights, 1):
    print(f"{i}. {insight}")

# Save analysis results
print("\n5. SAVING ANALYSIS RESULTS")
print("-" * 29)

# Create comprehensive summary
analysis_summary = {
    'Total_Papers': len(df),
    'Date_Range': f"{df['year'].min()} - {df['year'].max()}",
    'Top_Journal': top_journals.index[0],
    'Top_Journal_Papers': int(top_journals.iloc[0]),
    'Avg_Abstract_Length': f"{df['abstract_word_count'].mean():.1f} words",
    'Avg_Authors_Per_Paper': f"{df['author_count'].mean():.1f}",
    'Preprint_Percentage': f"{preprint_share:.1f}%",
    'Most_Common_Title_Word': title_keywords[0][0] if title_keywords else 'N/A',
    'Peak_Publication_Year': int(yearly_counts.idxmax()),
    'Peak_Year_Papers': int(yearly_counts.max())
}

# Save summary to CSV
summary_df = pd.DataFrame(list(analysis_summary.items()), columns=['Metric', 'Value'])
summary_df.to_csv('part3_analysis_summary.csv', index=False)

# Save detailed results
yearly_counts.to_csv('yearly_publication_counts.csv', header=['Count'])
top_journals.to_csv('top_journals_analysis.csv', header=['Paper_Count'])

# Save keyword analysis
keywords_df = pd.DataFrame(title_keywords, columns=['Keyword', 'Frequency'])
keywords_df.to_csv('title_keywords_analysis.csv', index=False)

print("✓ Analysis summary saved to: part3_analysis_summary.csv")
print("✓ Yearly counts saved to: yearly_publication_counts.csv")
print("✓ Journal analysis saved to: top_journals_analysis.csv")
print("✓ Keyword analysis saved to: title_keywords_analysis.csv")
print("✓ All visualizations saved as PNG files")

print("\n" + "="*75)
print("PART 3 COMPLETED SUCCESSFULLY!")
print("Visualizations created:")
print("• papers_by_year.png - Annual publication trends")
print("• top_journals.png - Leading journals in COVID-19 research")
print("• publications_timeline.png - Monthly publication timeline")
print("• title_wordcloud.png (or title_keywords.png) - Common title words")
print("• source_distribution.png - Publication source breakdown")
print("• abstract_length_distribution.png - Abstract length patterns")
print("• correlation_heatmap.png - Variable correlations")
print("• source_category_trends.png - Preprint vs published trends")
print("• author_count_distribution.png - Collaboration patterns")
print("Ready to proceed to Part 4: Streamlit Application")
print("="*75)
