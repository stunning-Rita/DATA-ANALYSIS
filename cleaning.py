# Part 2: Data Cleaning and Preparation
# Time estimate: 2-3 hours

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

print("CORD-19 Dataset Analysis - Part 2: Data Cleaning and Preparation")
print("=" * 70)

# Load data from Part 1 (or reload if running independently)
print("Loading data from Part 1...")

# If running this independently, you'll need to load the data again
# For this example, we'll recreate the sample data from Part 1
np.random.seed(42)
n_samples = 10000

df_original = pd.DataFrame({
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

print(f"✓ Original dataset loaded: {df_original.shape}")

# Task 1: Handle missing data
print("\n1. HANDLING MISSING DATA")
print("-" * 28)

# Identify columns with many missing values
missing_analysis = df_original.isnull().sum()
missing_percentage = (missing_analysis / len(df_original)) * 100

missing_summary = pd.DataFrame({
    'Column': missing_analysis.index,
    'Missing_Count': missing_analysis.values,
    'Missing_Percentage': missing_percentage.values
}).sort_values('Missing_Percentage', ascending=False)

print("Missing values analysis:")
print(missing_summary)

# Define thresholds for handling missing data
HIGH_MISSING_THRESHOLD = 70  # Drop columns with >70% missing
MODERATE_MISSING_THRESHOLD = 30  # Consider special handling for 30-70% missing

# Categorize columns by missing data severity
high_missing_cols = missing_summary[missing_summary['Missing_Percentage'] > HIGH_MISSING_THRESHOLD]['Column'].tolist()
moderate_missing_cols = missing_summary[
    (missing_summary['Missing_Percentage'] > MODERATE_MISSING_THRESHOLD) & 
    (missing_summary['Missing_Percentage'] <= HIGH_MISSING_THRESHOLD)
]['Column'].tolist()
low_missing_cols = missing_summary[missing_summary['Missing_Percentage'] <= MODERATE_MISSING_THRESHOLD]['Column'].tolist()

print(f"\nColumns with >70% missing (will consider dropping): {len(high_missing_cols)}")
print(high_missing_cols)

print(f"\nColumns with 30-70% missing (special handling): {len(moderate_missing_cols)}")
print(moderate_missing_cols)

print(f"\nColumns with <30% missing (keep and clean): {len(low_missing_cols)}")

# Decision making for missing data handling
print("\n2. MISSING DATA DECISIONS")
print("-" * 28)

# Create a copy for cleaning
df_cleaned = df_original.copy()

# Strategy 1: Drop columns with excessive missing data (>70%)
columns_to_drop = [col for col in high_missing_cols if col not in ['abstract']]  # Keep abstract even if high missing
if columns_to_drop:
    print(f"Dropping columns with >70% missing data: {columns_to_drop}")
    df_cleaned = df_cleaned.drop(columns=columns_to_drop)

# Strategy 2: Handle important columns with moderate missing data
print("\nHandling moderate missing data:")

# For abstract: Keep papers with abstracts (important for analysis)
if 'abstract' in df_cleaned.columns:
    before_abstract_filter = len(df_cleaned)
    df_cleaned = df_cleaned.dropna(subset=['abstract'])
    after_abstract_filter = len(df_cleaned)
    print(f"• Removed {before_abstract_filter - after_abstract_filter} papers without abstracts")
    print(f"• Remaining papers: {after_abstract_filter}")

# For other moderate missing columns, fill with appropriate values
for col in moderate_missing_cols:
    if col in df_cleaned.columns and col != 'abstract':
        if df_cleaned[col].dtype == 'object':
            df_cleaned[col] = df_cleaned[col].fillna('Unknown')
            print(f"• Filled missing {col} with 'Unknown'")
        else:
            df_cleaned[col] = df_cleaned[col].fillna(0)
            print(f"• Filled missing {col} with 0")

# Strategy 3: Handle low missing data columns
print("\nHandling low missing data:")
for col in low_missing_cols:
    if col in df_cleaned.columns:
        missing_count = df_cleaned[col].isnull().sum()
        if missing_count > 0:
            if col in ['title', 'journal', 'authors']:
                # For important text fields, drop rows with missing values
                df_cleaned = df_cleaned.dropna(subset=[col])
                print(f"• Removed {missing_count} rows with missing {col}")
            elif df_cleaned[col].dtype == 'object':
                df_cleaned[col] = df_cleaned[col].fillna('Unknown')
                print(f"• Filled {missing_count} missing {col} with 'Unknown'")

print(f"\nDataset size after missing data handling: {df_cleaned.shape}")

# Task 2: Prepare data for analysis
print("\n3. DATA PREPARATION FOR ANALYSIS")
print("-" * 35)

# Convert date columns to datetime format
print("Converting date columns to datetime format:")
if 'publish_time' in df_cleaned.columns:
    df_cleaned['publish_time'] = pd.to_datetime(df_cleaned['publish_time'], errors='coerce')
    
    # Remove rows with invalid dates
    invalid_dates = df_cleaned['publish_time'].isnull().sum()
    if invalid_dates > 0:
        df_cleaned = df_cleaned.dropna(subset=['publish_time'])
        print(f"• Removed {invalid_dates} rows with invalid publish_time")
    
    print(f"• Date range: {df_cleaned['publish_time'].min()} to {df_cleaned['publish_time'].max()}")

# Extract year from publication date
print("\nExtracting temporal features:")
df_cleaned['year'] = df_cleaned['publish_time'].dt.year
df_cleaned['month'] = df_cleaned['publish_time'].dt.month
df_cleaned['day_of_year'] = df_cleaned['publish_time'].dt.dayofyear
df_cleaned['quarter'] = df_cleaned['publish_time'].dt.quarter

print("• Added columns: year, month, day_of_year, quarter")

# Create new columns for analysis
print("\nCreating derived columns:")

# Abstract word count and character count
if 'abstract' in df_cleaned.columns:
    df_cleaned['abstract_word_count'] = df_cleaned['abstract'].fillna('').str.split().str.len()
    df_cleaned['abstract_char_count'] = df_cleaned['abstract'].fillna('').str.len()
    print("• Added abstract_word_count and abstract_char_count")

# Title word count
if 'title' in df_cleaned.columns:
    df_cleaned['title_word_count'] = df_cleaned['title'].fillna('').str.split().str.len()
    df_cleaned['title_char_count'] = df_cleaned['title'].fillna('').str.len()
    print("• Added title_word_count and title_char_count")

# Author count (simple count by splitting on semicolons)
if 'authors' in df_cleaned.columns:
    df_cleaned['author_count'] = df_cleaned['authors'].fillna('').str.split(';').str.len()
    # Handle cases where there are no authors
    df_cleaned['author_count'] = df_cleaned['author_count'].replace(1, 0).where(df_cleaned['authors'].fillna('') != '', 0)
    print("• Added author_count")

# Source category (group similar sources)
if 'source_x' in df_cleaned.columns:
    def categorize_source(source):
        if pd.isna(source):
            return 'Unknown'
        source = source.lower()
        if 'arxiv' in source:
            return 'Preprint'
        elif 'medrxiv' in source or 'biorxiv' in source:
            return 'Preprint'
        elif 'pmc' in source:
            return 'Published'
        else:
            return 'Published'
    
    df_cleaned['source_category'] = df_cleaned['source_x'].apply(categorize_source)
    print("• Added source_category (Preprint/Published/Unknown)")

# Clean text columns
print("\nCleaning text data:")
text_columns = ['title', 'abstract', 'authors', 'journal']
for col in text_columns:
    if col in df_cleaned.columns:
        # Remove extra whitespace and standardize
        df_cleaned[col] = df_cleaned[col].fillna('').str.strip()
        df_cleaned[col] = df_cleaned[col].replace('', np.nan)  # Convert empty strings back to NaN
        print(f"• Cleaned whitespace in {col}")

# Handle duplicates
print("\n4. HANDLING DUPLICATES")
print("-" * 22)

# Check for duplicates based on cord_uid (should be unique)
duplicate_uids = df_cleaned.duplicated(subset=['cord_uid']).sum()
print(f"Duplicate cord_uid entries: {duplicate_uids}")

if duplicate_uids > 0:
    df_cleaned = df_cleaned.drop_duplicates(subset=['cord_uid'], keep='first')
    print(f"• Removed {duplicate_uids} duplicate entries")

# Check for potential duplicate papers (same title)
duplicate_titles = df_cleaned.duplicated(subset=['title'], keep=False).sum()
print(f"Papers with duplicate titles: {duplicate_titles}")

if duplicate_titles > 0 and duplicate_titles < 100:  # Only if manageable number
    # Keep the one with most complete information
    df_cleaned = df_cleaned.sort_values('abstract_char_count', ascending=False)
    df_cleaned = df_cleaned.drop_duplicates(subset=['title'], keep='first')
    print(f"• Removed duplicate titles, kept most complete entries")

# Data validation
print("\n5. DATA VALIDATION")
print("-" * 19)

# Validate date ranges (should be reasonable for COVID-19 research)
min_date = pd.to_datetime('2019-01-01')  # COVID-19 emerged late 2019
max_date = pd.to_datetime('2024-12-31')  # Reasonable future limit

invalid_dates = (df_cleaned['publish_time'] < min_date) | (df_cleaned['publish_time'] > max_date)
invalid_date_count = invalid_dates.sum()

if invalid_date_count > 0:
    print(f"Found {invalid_date_count} papers with dates outside expected range")
    # Optionally remove or flag these
    df_cleaned = df_cleaned[~invalid_dates]
    print(f"• Removed papers with invalid date ranges")

# Validate text lengths (remove extremely short or long entries that might be errors)
if 'abstract_word_count' in df_cleaned.columns:
    # Remove abstracts that are too short (likely incomplete) or too long (likely errors)
    valid_abstract_length = (df_cleaned['abstract_word_count'] >= 10) & (df_cleaned['abstract_word_count'] <= 1000)
    invalid_abstracts = (~valid_abstract_length).sum()
    
    if invalid_abstracts > 0:
        print(f"Found {invalid_abstracts} papers with invalid abstract lengths")
        df_cleaned = df_cleaned[valid_abstract_length]
        print(f"• Removed papers with abstracts <10 or >1000 words")

# Final data summary
print("\n6. CLEANED DATASET SUMMARY")
print("-" * 28)

print(f"Original dataset size: {df_original.shape}")
print(f"Cleaned dataset size: {df_cleaned.shape}")
print(f"Reduction: {df_original.shape[0] - df_cleaned.shape[0]:,} rows ({((df_original.shape[0] - df_cleaned.shape[0])/df_original.shape[0]*100):.1f}%)")

print("\nFinal data quality metrics:")
print(f"• Papers with abstracts: {df_cleaned['abstract'].notna().sum()} (100%)")
print(f"• Papers with titles: {df_cleaned['title'].notna().sum()} (100%)")
print(f"• Papers with valid dates: {df_cleaned['publish_time'].notna().sum()} (100%)")
print(f"• Average abstract length: {df_cleaned['abstract_word_count'].mean():.1f} words")
print(f"• Date range: {df_cleaned['year'].min()} - {df_cleaned['year'].max()}")

# Save cleaned dataset
print("\n7. SAVING CLEANED DATASET")
print("-" * 27)

# Save the cleaned dataset
df_cleaned.to_csv('cord19_cleaned.csv', index=False)
print("✓ Cleaned dataset saved to: cord19_cleaned.csv")

# Save cleaning report
cleaning_report = {
    'Original_Records': len(df_original),
    'Final_Records': len(df_cleaned),
    'Records_Removed': len(df_original) - len(df_cleaned),
    'Reduction_Percentage': f"{((len(df_original) - len(df_cleaned))/len(df_original)*100):.1f}%",
    'Columns_Original': len(df_original.columns),
    'Columns_Final': len(df_cleaned.columns),
    'Date_Range': f"{df_cleaned['year'].min()} - {df_cleaned['year'].max()}",
    'Avg_Abstract_Words': f"{df_cleaned['abstract_word_count'].mean():.1f}",
    'Quality_Score': 'High'  # All key fields now complete
}

cleaning_report_df = pd.DataFrame(list(cleaning_report.items()), columns=['Metric', 'Value'])
cleaning_report_df.to_csv('part2_cleaning_report.csv', index=False)
print("✓ Cleaning report saved to: part2_cleaning_report.csv")

# Show sample of cleaned data
print("\nSample of cleaned data:")
print(df_cleaned[['cord_uid', 'title', 'journal', 'year', 'abstract_word_count', 'source_category']].head())

print("\n" + "="*70)
print("PART 2 COMPLETED SUCCESSFULLY!")
print("Key achievements:")
print(f"• Cleaned dataset from {len(df_original):,} to {len(df_cleaned):,} records")
print("• Handled missing values appropriately")
print("• Added temporal and derived features")
print("• Ensured data quality and consistency")
print("• Ready to proceed to Part 3: Data Analysis and Visualization")
print("="*70)
