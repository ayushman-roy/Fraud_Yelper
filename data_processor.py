import csv
import json
import os
import random
import datetime

# Utility function for timestamped logging
def log(message):
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

# Paths
base_dir = os.path.expanduser('~/Desktop/Fraud_Yelper')
csv_business_path = os.path.join(base_dir, 'cleaned_datasets/business_data.csv')
csv_review_path = os.path.join(base_dir, 'cleaned_datasets/review_data.csv')
csv_user_path = os.path.join(base_dir, 'cleaned_datasets/user_data.csv')

checkin_json_path = os.path.join(base_dir, 'Yelp_JSON/yelp_academic_dataset_checkin.json')
business_json_path = os.path.join(base_dir, 'Yelp_JSON/yelp_academic_dataset_business.json')
review_json_path = os.path.join(base_dir, 'Yelp_JSON/yelp_academic_dataset_review.json')
user_json_path = os.path.join(base_dir, 'Yelp_JSON/yelp_academic_dataset_user.json')

random.seed(47)

# Step 1: Load Business Data
log("Loading business data...")
with open(business_json_path, 'r') as json_file:
    data = [json.loads(line) for line in json_file]

sampled_data = random.sample(data, min(1000, len(data)))
log(f"Sampled {len(sampled_data)} businesses.")

sampled_business_ids = {business['business_id'] for business in sampled_data}
checkin_counts = {}

# Step 2: Process Check-in Data
log("Processing check-in data...")
with open(checkin_json_path, 'r') as checkin_file:
    for line in checkin_file:
        checkin_record = json.loads(line)
        business_id = checkin_record['business_id']
        if business_id in sampled_business_ids:
            checkin_dates = checkin_record.get('date', "")
            checkin_count = len([d for d in checkin_dates.split(',') if d.strip()]) if checkin_dates else 0
            checkin_counts[business_id] = checkin_count

# Step 3: Write Business Data with Check-ins
log("Writing business data to CSV...")
with open(csv_business_path, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['business_id', 'postal_code', 'stars', 'review_count', 'is_open', 'categories', 'checkin_count'])
    for record in sampled_data:
        checkin_count = checkin_counts.get(record['business_id'], 0)
        writer.writerow([
            record['business_id'],
            record['postal_code'],
            record['stars'],
            record['review_count'],
            record['is_open'],
            record['categories'],
            checkin_count
        ])
log(f"Business data saved to {csv_business_path}.")

# Step 4: Extract Relevant Reviews
log("Extracting reviews for sampled businesses...")
business_ids = set()
with open(csv_business_path, 'r') as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        business_ids.add(row['business_id'])

filtered_entries = []
with open(review_json_path, 'r') as json_file:
    for line in json_file:
        record = json.loads(line)
        if record['business_id'] in business_ids:
            filtered_entries.append(record)

log(f"Extracted {len(filtered_entries)} reviews.")

# Step 5: Write Review Data
log("Writing review data to CSV...")
with open(csv_review_path, 'w', newline='') as output_csv:
    if filtered_entries:
        writer = csv.DictWriter(output_csv, fieldnames=filtered_entries[0].keys())
        writer.writeheader()
        writer.writerows(filtered_entries)
log(f"Review data saved to {csv_review_path}.")

# Step 6: Process User Data
log("Processing user data...")
user_ids = set()
with open(csv_review_path, 'r') as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        user_ids.add(row['user_id'])

filtered_entries = []
compliment_fields = [
    'compliment_hot', 'compliment_more', 'compliment_profile',
    'compliment_cute', 'compliment_list', 'compliment_note',
    'compliment_plain', 'compliment_cool', 'compliment_funny',
    'compliment_writer'
]

with open(user_json_path, 'r') as json_file:
    for line in json_file:
        record = json.loads(line)
        if record['user_id'] in user_ids:
            aggregate_compliment_score = sum(record.get(field, 0) for field in compliment_fields)
            friends_list = [friend.strip() for friend in record.get('friends', "").split(',') if friend.strip()]
            elite_list = [elite.strip() for elite in record.get('elite', "").split(',') if elite.strip()]
            record['friend_count'] = len(friends_list)
            record['elite_count'] = len(elite_list)
            record['aggregate_compliments'] = aggregate_compliment_score
            filtered_entries.append(record)

log(f"Processed {len(filtered_entries)} users.")

# Step 7: Write User Data
log("Writing user data to CSV...")
desired_columns = ['user_id', 'review_count', 'yelping_since', 'friend_count', 'fans', 'elite_count', 'average_stars', 'aggregate_compliments', 'compliment_photos']
with open(csv_user_path, 'w', newline='') as output_csv:
    writer = csv.writer(output_csv)
    writer.writerow(desired_columns)
    for record in filtered_entries:
        writer.writerow([record.get(column, '') for column in desired_columns])
log(f"User data saved to {csv_user_path}.")
