import pandas as pd
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import plotly.express as px

def load_data():
    """
    Load the data file using the relative path.
    """
    # Set relative path
    base_path = os.path.dirname(os.path.abspath(__file__))
    business_data_path = os.path.join(base_path, 'cleaned_datasets', 'business_data.csv')
    
    # Load data into a pandas DataFrame
    df = pd.read_csv(business_data_path)
    return df

def perform_isolation_forest(X, contamination=0.1):
    """
    Perform anomaly detection using Isolation Forest.
    """
    # Impute missing values with the mean
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Fit Isolation Forest
    isolation_forest = IsolationForest(contamination=contamination, random_state=42)
    predictions = isolation_forest.fit_predict(X_scaled)
    
    return isolation_forest, predictions

def plot_anomalies(X, predictions, ids):
    """
    Visualize the data with anomalies highlighted using Plotly.
    """
    # Add predictions and IDs to the DataFrame
    X['Anomaly'] = predictions
    X['Business ID'] = ids
    X['Anomaly Label'] = X['Anomaly'].apply(lambda x: 'Normal' if x == 1 else 'Anomaly')

    # Use Plotly for an interactive scatter plot
    fig = px.scatter(
        X,
        x='Feature 1',  # Assuming first column as Feature 1
        y='Feature 2',  # Assuming second column as Feature 2
        color='Anomaly Label',
        hover_data=['Business ID', 'Anomaly Label'],  # Add Business ID and anomaly label to hover info
        title='Isolation Forest Anomaly Detection',
        labels={'Anomaly': 'Anomaly Label'}
    )

    # Save interactive plot as an HTML file
    output_path = os.path.join('output', 'isolation_forest_results.html')
    os.makedirs('output', exist_ok=True)
    fig.write_html(output_path)
    print(f"Interactive Isolation Forest visualization saved at: {output_path}")

import os

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
    """
    Main function to execute the Isolation Forest pipeline.
    """
    # Load data
    df = load_data()
    
    # Select numerical features for Isolation Forest
    numerical_columns = [
        'stars', 'review_count', 'checkin_count', 'stars_std_dev',
        'percentage_1_star', 'percentage_2_star', 'percentage_3_star',
        'percentage_4_star', 'percentage_5_star', 'avg_timestamp_gap',
        'text_similarity_score', '%_under_3_months', '%_3_months_to_1_year',
        '%_over_1_year', '%_reviewers_with_less_than_10_reviews',
        'avg_star_deviation_squared'
    ]
    X = df[numerical_columns]
    business_ids = df['business_id']  # Assuming 'business_id' column exists in the data

    # Check for missing values
    print("Missing values before imputation:")
    print(X.isnull().sum())

    # Perform Isolation Forest
    isolation_forest, predictions = perform_isolation_forest(X)

    # Convert X back to a DataFrame for visualization
    X = pd.DataFrame(X, columns=numerical_columns)
    X['Feature 1'] = X.iloc[:, 0]  # Explicitly set Feature 1 for visualization
    X['Feature 2'] = X.iloc[:, 1]  # Explicitly set Feature 2 for visualization

    # Visualize anomalies
    plot_anomalies(X, predictions, business_ids)

    # Save anomalous IDs to a CSV
    save_anomalous_ids(df, predictions, 'isolation_forest_anomalous_ids.csv')


if __name__ == "__main__":
    main()
