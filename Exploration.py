# Part 1: Data Loading and Basic Exploration
# Time estimate: 2-3 hours

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("CORD-19 Dataset Analysis - Part 1: Data Loading and Basic Exploration")
print("=" * 70)

# Task 1: Download and load the data
print("\n1. LOADING THE DATA")
print("-" * 30)

# Load the metadata.csv file
# Note: Replace 'metadata.csv' with the actual path to your downloaded file
try:
    # For demonstration, we'll create sample data that matches CORD-19 structure
    # In your actual implementation, use: df = pd.read_csv('metadata.csv')
    print("Loading CORD-19 metadata.csv...")
    
    # Creating sample data that matches the actual CORD-19 structure
    np.random.seed(42)
    n_samples = 10000  # Reduced for demonstration
    
    # Sample data mimicking CORD-19 metadata structure
    df = pd.DataFrame({
        'cord_uid': [f'cord-{i:07d}' for i in range(n_samples)],
        'sha': [f'sha{i:07d}' if np.random.random() > 0.3 else np.nan for i in range(n_samples)],
        'source_x': np.random.choice(['PMC', 'medRxiv', 'bioRxiv', 'Elsevier', 'WHO', 'arXiv'], n_samples),
        'title': [f'COVID-19 research paper about {np.random.choice(["vaccines", "treatments", "epidemiology", "public health", "genetics"])}' 
                  for _ in range(n_samples)],
        'doi': [f'10.1000/{i}' if np.random.random() > 0.2 else np.nan for i in range(n_samples)],
        'pmcid': [f'PMC{i}' if np.random.random() > 0.4 else np.nan for i in range(n_samples)],
        'pubmed_id': [f'{1000000 + i}' if np.random.random() > 0.5 else np.nan for i in range(n_samples)],
        'license': np.random.choice(['cc-by', 'cc-by-nc', 'no-cc', np.nan], n_samples),
        'abstract': [f'This paper investigates COVID-19 research topic {i}. ' * np.random.randint(5, 25) 
                     if np.random.random() > 0.1 else np.nan for i in range(n_samples)],
        'publish_time': pd.date_range(start='2019-12-01', end='2023-12-31', periods=n_samples),
        'authors': [f'Smith, J.; Doe, A.{"; Author" + str(i%5) if np.random.random() > 0.3 else ""}' 
                    for i in range(n_samples)],
        'journal': np.random.choice([
            'Nature', 'Science', 'Cell', 'The Lancet', 'NEJM', 'PLOS ONE', 
            'Journal of Virology', 'PNAS', 'BMJ', 'medRxiv', 'bioRxiv'
        ], n_samples),
        'mag_id': [f'mag{i}' if np.random.random() > 0.6 else np.nan for i in range(n_samples)],
        'who_covidence_id': [f'who{i}' if np.random.random() > 0.8 else np.nan for i in range(n_samples)],
        'arxiv_id': [f'arXiv:{i}' if np.random.random() > 0.9 else np.nan for i in range(n_samples)],
        'pdf_json_files': [f'pdf_files_{i}.json' if np.random.random() > 0.7 else np.nan for i in range(n_samples)],
        'pmc_json_files': [f'pmc_files_{i}.json' if np.random.random() > 0.8 else np.nan for i in range(n_samples)],
        'url': [f'https://example.com/paper/{i}' if np.random.random() > 0.3 else np.nan for i in range(n_samples)],
        's2_id': [f's2_{i}' if np.random.random() > 0.4 else np.nan for i in range(n_samples)]
    })
    
    print(f"✓ Data loaded successfully!")
    print(f"✓ Using sample data that mimics CORD-19 structure")
    
    # In your actual implementation, uncomment the following line:
    # df = pd.read_csv('metadata.csv')
    
except FileNotFoundError:
    print("❌ Error: metadata.csv file not found!")
    print("Please download the CORD-19 dataset from Kaggle and place metadata.csv in your working directory")
    exit()

# Examine the first few rows and data structure
print("\n2. EXAMINING THE DATA STRUCTURE")
print("-" * 35)

print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset columns:")
print(df.columns.tolist())

print(f"\nDataset shape: {df.shape[0]} rows × {df.shape[1]} columns")

# Task 2: Basic data exploration
print("\n3. BASIC DATA EXPLORATION")
print("-" * 30)

# Check DataFrame dimensions
print(f"Dataset dimensions: {df.shape}")
print(f"Total number of papers: {df.shape[0]:,}")
print(f"Number of columns: {df.shape[1]}")

# Identify data types of each column
print("\nData types for each column:")
print(df.dtypes)

# Check for missing values in important columns
print("\nMissing values analysis:")
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100

missing_df = pd.DataFrame({
    'Column': missing_values.index,
    'Missing Count': missing_values.values,
    'Missing Percentage': missing_percentage.values
}).sort_values('Missing Percentage', ascending=False)

print(missing_df)

# Identify columns with high missing values (>50%)
high_missing = missing_df[missing_df['Missing Percentage'] > 50]
if not high_missing.empty:
    print(f"\nColumns with >50% missing values:")
    print(high_missing)

# Generate basic statistics for numerical columns
print("\n4. BASIC STATISTICS")
print("-" * 20)

# Convert publish_time to datetime for better analysis
df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')

# Add derived columns for analysis
df['abstract_length'] = df['abstract'].fillna('').str.len()
df['title_length'] = df['title'].fillna('').str.len()
df['year'] = df['publish_time'].dt.year

print("Basic statistics for numerical columns:")
numerical_cols = ['abstract_length', 'title_length', 'year']
print(df[numerical_cols].describe())

# Additional exploratory analysis
print("\n5. ADDITIONAL EXPLORATION")
print("-" * 28)

print("Unique values in key categorical columns:")
categorical_cols = ['source_x', 'license', 'journal']
for col in categorical_cols:
    unique_count = df[col].nunique()
    print(f"{col}: {unique_count} unique values")
    if unique_count <= 10:  # Show values if not too many
        print(f"  Values: {df[col].value_counts().head().to_dict()}")

# Time range analysis
print(f"\nTime range of publications:")
print(f"Earliest: {df['publish_time'].min()}")
print(f"Latest: {df['publish_time'].max()}")
print(f"Time span: {(df['publish_time'].max() - df['publish_time'].min()).days} days")

# Quick data quality checks
print("\n6. DATA QUALITY CHECKS")
print("-" * 25)

print("Data quality summary:")
print(f"• Duplicate cord_uid entries: {df['cord_uid'].duplicated().sum()}")
print(f"• Papers with titles: {df['title'].notna().sum():,} ({(df['title'].notna().sum()/len(df)*100):.1f}%)")
print(f"• Papers with abstracts: {df['abstract'].notna().sum():,} ({(df['abstract'].notna().sum()/len(df)*100):.1f}%)")
print(f"• Papers with valid dates: {df['publish_time'].notna().sum():,} ({(df['publish_time'].notna().sum()/len(df)*100):.1f}%)")

# Memory usage
print(f"\nMemory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# Save initial exploration results
print("\n7. SAVING EXPLORATION RESULTS")
print("-" * 32)

# Create summary report
exploration_summary = {
    'Total Records': len(df),
    'Total Columns': len(df.columns),
    'Date Range': f"{df['publish_time'].min().date()} to {df['publish_time'].max().date()}",
    'Papers with Abstracts': f"{df['abstract'].notna().sum()} ({df['abstract'].notna().sum()/len(df)*100:.1f}%)",
    'Unique Journals': df['journal'].nunique(),
    'Unique Sources': df['source_x'].nunique(),
    'Average Abstract Length': f"{df['abstract_length'].mean():.0f} characters",
    'Memory Usage (MB)': f"{df.memory_usage(deep=True).sum() / 1024**2:.1f}"
}

exploration_df = pd.DataFrame(list(exploration_summary.items()), columns=['Metric', 'Value'])
exploration_df.to_csv('part1_exploration_summary.csv', index=False)
missing_df.to_csv('part1_missing_values.csv', index=False)

print("✓ Exploration summary saved to: part1_exploration_summary.csv")
print("✓ Missing values analysis saved to: part1_missing_values.csv")

print("\n" + "="*70)
print("PART 1 COMPLETED SUCCESSFULLY!")
print("Key findings:")
print(f"• Dataset contains {len(df):,} research papers")
print(f"• Data spans from {df['year'].min()} to {df['year'].max()}")
print(f"• {df['abstract'].notna().sum():,} papers have abstracts ({df['abstract'].notna().sum()/len(df)*100:.1f}%)")
print(f"• {df['journal'].nunique()} unique journals represented")
print("• Ready to proceed to Part 2: Data Cleaning and Preparation")
print("="*70)
