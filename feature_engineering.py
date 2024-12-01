import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time  

# Define base directory
base_dir = os.path.expanduser('~/Desktop/Fraud_Yelper')

# Load the business, reviews, and users data
business_csv_path = os.path.join(base_dir, 'cleaned_datasets/business_data.csv')
reviews_csv_path = os.path.join(base_dir, 'cleaned_datasets/review_data.csv')
users_csv_path = os.path.join(base_dir, 'cleaned_datasets/user_data.csv')

business_csv = pd.read_csv(business_csv_path)
reviews_csv = pd.read_csv(reviews_csv_path)
users_csv = pd.read_csv(users_csv_path)

# Start processing and track time
start_time = time.strftime('%Y-%m-%d %H:%M:%S')
print(f"{start_time} - Process started.")

# Check and add standard deviation and review percentages
if 'stars_std_dev' not in business_csv.columns:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Calculating and merging standard deviation and review percentages...")

    # Rename the stars column in the reviews
    reviews_csv.rename(columns={'stars': 'review_stars'}, inplace=True)

    # Merge both CSVs on 'business_id'
    merged_df = pd.merge(business_csv, reviews_csv, on='business_id')

    # Group by 'business_id' and calculate standard deviation for 'review_stars'
    std_deviation = merged_df.groupby('business_id')['review_stars'].std().reset_index()
    std_deviation.columns = ['business_id', 'stars_std_dev']

    # Calculate the percentage of reviews for each star rating (1* to 5*)
    review_counts = merged_df.groupby(['business_id', 'review_stars']).size().unstack(fill_value=0)
    total_reviews = review_counts.sum(axis=1)

    # Calculate percentages and add them as new columns
    for star in range(1, 6):
        col_name = f'percentage_{star}_star'
        review_counts[col_name] = (review_counts.get(star, 0) / total_reviews) * 100

    # Keep only percentage columns and reset index
    percentage_columns = review_counts[[f'percentage_{star}_star' for star in range(1, 6)]].reset_index()

    # Merge the std deviation and percentage columns back into the business CSV
    business_csv = pd.merge(business_csv, std_deviation, on='business_id', how='left')
    business_csv = pd.merge(business_csv, percentage_columns, on='business_id', how='left')

    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Standard deviation and review percentages added.")
else:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Standard deviation and review percentages column already exists. No changes made.")

# Check and add average timestamp gap
if 'avg_timestamp_gap' not in business_csv.columns:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Calculating and merging average timestamp gap...")

    # Convert the 'date' column to datetime format
    reviews_csv['date'] = pd.to_datetime(reviews_csv['date'])

    # Sort reviews by business_id and date to ensure gaps are calculated correctly
    reviews_csv = reviews_csv.sort_values(by=['business_id', 'date'])

    # Calculate the time gap between consecutive reviews for each business
    reviews_csv['timestamp_gap'] = reviews_csv.groupby('business_id')['date'].diff().dt.total_seconds()

    # Calculate the average timestamp gap (in seconds) for each business_id
    avg_timestamp_gap = reviews_csv.groupby('business_id')['timestamp_gap'].mean().reset_index()

    # Rename columns for clarity
    avg_timestamp_gap.columns = ['business_id', 'avg_timestamp_gap']

    # Merge the average timestamp gap back into the business CSV
    business_csv = pd.merge(business_csv, avg_timestamp_gap, on='business_id', how='left')

    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Average timestamp gap added.")
else:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Average timestamp gap column already exists. No changes made.")

# Check and add text similarity scores
if 'text_similarity_score' not in business_csv.columns:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Calculating and merging text similarity scores...")

    # Define a function to calculate average pairwise similarity for each business
    def calculate_average_similarity(business_id):
        business_reviews = reviews_csv[reviews_csv['business_id'] == business_id]['text']
        if len(business_reviews) <= 1:
            return np.nan  # Perfect similarity if only one review or none
        tfidf_matrix = TfidfVectorizer(stop_words='english').fit_transform(business_reviews)
        pairwise_similarities = cosine_similarity(tfidf_matrix)
        triu_indices = np.triu_indices(pairwise_similarities.shape[0], k=1)
        avg_similarity = pairwise_similarities[triu_indices].mean()
        return avg_similarity

    # Calculate the similarity score for each business
    business_csv['text_similarity_score'] = business_csv['business_id'].apply(calculate_average_similarity)

    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Text similarity scores added.")
else:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Text similarity score column already exists. No changes made.")

# Check if yelping experience columns already exist
experience_columns = ['%_under_3_months', '%_3_months_to_1_year', '%_over_1_year']

if not all(col in business_csv.columns for col in experience_columns):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Calculating yelping experience brackets...")

    # Convert relevant columns to datetime
    reviews_csv['date'] = pd.to_datetime(reviews_csv['date'])
    users_csv['yelping_since'] = pd.to_datetime(users_csv['yelping_since'])

    # Merge reviews with users to get yelping_since
    reviews_with_users = pd.merge(reviews_csv, users_csv[['user_id', 'yelping_since']], on='user_id', how='left')

    # Calculate the difference in days between review date and yelping_since
    reviews_with_users['yelping_duration_days'] = (reviews_with_users['date'] - reviews_with_users['yelping_since']).dt.days

    # Classify into experience brackets
    reviews_with_users['experience_bracket'] = pd.cut(
        reviews_with_users['yelping_duration_days'],
        bins=[-np.inf, 90, 365, np.inf],
        labels=['under_3_months', '3_months_to_1_year', 'over_1_year']
    )

    # Group by business_id and experience bracket, and calculate percentages
    experience_counts = reviews_with_users.groupby(['business_id', 'experience_bracket'], observed=True).size().unstack(fill_value=0)
    total_reviewers = experience_counts.sum(axis=1)

    for bracket in ['under_3_months', '3_months_to_1_year', 'over_1_year']:
        col_name = f'%_{bracket}'
        experience_counts[col_name] = (experience_counts[bracket] / total_reviewers) * 100

    # Keep only percentage columns and reset index
    experience_percentage = experience_counts[[f'%_{bracket}' for bracket in ['under_3_months', '3_months_to_1_year', 'over_1_year']]].reset_index()

    # Merge experience percentages into the business CSV
    business_csv = pd.merge(business_csv, experience_percentage, on='business_id', how='left')

    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Yelping experience brackets calculated and merged.")
else:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Yelping experience columns already exist. No changes made.")

# Check if the column already exists
if '%_reviewers_with_less_than_10_reviews' not in business_csv.columns:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Calculating percentage of reviewers with <10 reviews...")

    # Merge reviews with users to get review counts
    reviews_with_users = pd.merge(reviews_csv, users_csv[['user_id', 'review_count']], on='user_id', how='left')

    # Count reviewers with fewer than 10 reviews for each business
    reviews_with_users['low_review_count'] = np.where(reviews_with_users['review_count'] < 10, 1, 0)
    low_review_count_by_business = reviews_with_users.groupby('business_id')['low_review_count'].sum().reset_index()

    # Calculate the percentage of reviewers with <10 reviews
    business_review_counts = reviews_csv.groupby('business_id').size().reset_index(name='total_reviews')
    reviewers_percentage = pd.merge(low_review_count_by_business, business_review_counts, on='business_id')
    reviewers_percentage['%_reviewers_with_less_than_10_reviews'] = (reviewers_percentage['low_review_count'] / reviewers_percentage['total_reviews']) * 100
    
    # Merge the percentage data back into the business CSV
    business_csv = pd.merge(business_csv, reviewers_percentage[['business_id', '%_reviewers_with_less_than_10_reviews']], on='business_id', how='left')

    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Percentage of reviewers with <10 reviews added and file updated.")
else:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Column '%_reviewers_with_less_than_10_reviews' already exists. No changes made.")

# Benchmarking user behavior, a lower value suggests suspicious activity 
if 'avg_star_deviation_squared' not in business_csv.columns:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Calculating average star deviation squared...")

    # Merge reviews with users to get user average_stars
    reviews_with_users = pd.merge(reviews_csv, users_csv[['user_id', 'average_stars']], on='user_id', how='left')

    # Calculate (average_stars - review_stars)^2
    reviews_with_users['star_deviation_squared'] = (reviews_with_users['average_stars'] - reviews_with_users['stars']) ** 2

    # Group by business_id and calculate the sum of squared deviations
    star_deviation_sum_by_business = reviews_with_users.groupby('business_id')['star_deviation_squared'].sum().reset_index()

    # Merge this with business_csv to use the total review count from business_csv
    deviation_with_review_count = pd.merge(star_deviation_sum_by_business, business_csv[['business_id', 'review_count']], on='business_id')

    # Calculate the average star deviation squared
    deviation_with_review_count['avg_star_deviation_squared'] = deviation_with_review_count['star_deviation_squared'] / deviation_with_review_count['review_count']

    # Merge the result back into business_csv
    business_csv = pd.merge(business_csv, deviation_with_review_count[['business_id', 'avg_star_deviation_squared']], on='business_id', how='left')

    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Average star deviation squared added and file updated.")

else:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Column 'avg_star_deviation_squared' already exists. No changes made.")

# Save the updated CSV to file
business_csv.to_csv(business_csv_path, index=False)
end_time = time.strftime('%Y-%m-%d %H:%M:%S')
print(f"{end_time} - Process completed and file saved.")

# Show details about the updated CSV
print(f"Updated CSV saved at: {business_csv_path}")
print(f"Number of records in the updated business CSV: {business_csv.shape[0]}")
print(f"Max/Min for stars_std_dev: {business_csv['stars_std_dev'].max()}, {business_csv['stars_std_dev'].min()}")
print(f"Max/Min for avg_timestamp_gap: {business_csv['avg_timestamp_gap'].max()}, {business_csv['avg_timestamp_gap'].min()}")
print(f"Max/Min for text_similarity_score: {business_csv['text_similarity_score'].max()}, {business_csv['text_similarity_score'].min()}")
print(f"Max/Min for %_reviewers_with_less_than_10_reviews: {business_csv['%_reviewers_with_less_than_10_reviews'].max()}, {business_csv['%_reviewers_with_less_than_10_reviews'].min()}")
print(f"Max/Min for avg_star_deviation_squared: {business_csv['avg_star_deviation_squared'].max()}, {business_csv['avg_star_deviation_squared'].min()}")
