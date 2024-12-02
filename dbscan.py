import pandas as pd
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import plotly.express as px
import numpy as np

def load_data():
    base_path = os.path.dirname(os.path.abspath(__file__))
    business_data_path = os.path.join(base_path, 'cleaned_datasets', 'business_data.csv')
    df = pd.read_csv(business_data_path)
    return df

def perform_dbscan(X, eps=0.5, min_samples=5):
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)
    return dbscan, labels

def plot_dbscan_results(X, labels, ids):
    # Add cluster labels to the dataframe
    X['Cluster'] = labels
    X['Business ID'] = ids  # Add the Business ID for hover information

    # Assign meaningful labels for noise and clusters
    X['Cluster Label'] = X['Cluster'].apply(lambda x: f'Cluster {x}' if x != -1 else 'Noise')

    # Use Plotly for an interactive scatter plot
    fig = px.scatter(
        X,
        x='Feature 1',  # Assuming the first column in X after scaling is Feature 1
        y='Feature 2',  # Assuming the second column is Feature 2
        color='Cluster Label',
        hover_data=['Business ID', 'Cluster Label'],  # Add Business ID to hover info
        title='DBSCAN Clustering Results',
        labels={'Cluster': 'Cluster Label'}
    )

    # Save interactive plot as an HTML file
    output_path = os.path.join('output', 'dbscan_results.html')
    os.makedirs('output', exist_ok=True)
    fig.write_html(output_path)
    print(f"Interactive DBSCAN visualization saved at: {output_path}")

def save_anomalous_ids(df, predictions, file_name, id_column='business_id'):
    """
    Save anomalous business IDs to a CSV file.
    
    Parameters:
    - df: DataFrame containing the data and business IDs.
    - predictions: Array of anomaly labels (DBSCAN: -1 for noise, Isolation Forest: -1 for anomalies).
    - file_name: Name of the output CSV file.
    - id_column: Column name for business IDs in the DataFrame.
    """
    # Extract anomalies
    anomalous_data = df[predictions == -1]
    
    # Select only the business IDs
    anomalous_ids = anomalous_data[[id_column]]
    
    # Define the output path
    output_path = os.path.join('output', file_name)
    os.makedirs('output', exist_ok=True)
    
    # Save to CSV
    anomalous_ids.to_csv(output_path, index=False)
    print(f"Anomalous business IDs saved at: {output_path}")

def main():
    df = load_data()
    numerical_columns = ['stars', 'review_count', 'checkin_count', 'stars_std_dev',
                         'percentage_1_star', 'percentage_2_star', 'percentage_3_star',
                         'percentage_4_star', 'percentage_5_star', 'avg_timestamp_gap']
    X = df[numerical_columns]
    business_ids = df['business_id']  # Assuming 'business_id' column exists in the data
    dbscan, labels = perform_dbscan(X)
    X = pd.DataFrame(X, columns=numerical_columns)  # Convert X back to a DataFrame for visualization
    X['Feature 1'] = X.iloc[:, 0]  # Explicitly set Feature 1 for visualization
    X['Feature 2'] = X.iloc[:, 1]  # Explicitly set Feature 2 for visualization
    plot_dbscan_results(X, labels, business_ids)
    # Save anomalous IDs to a CSV
    save_anomalous_ids(df, labels, 'dbscan_anomalous_ids.csv')


if __name__ == "__main__":
    main()
