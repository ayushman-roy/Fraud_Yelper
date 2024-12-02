import pandas as pd
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import numpy as np

def load_data():
    base_path = os.path.dirname(os.path.abspath(__file__))
    business_data_path = os.path.join(base_path, 'cleaned_datasets', 'business_data.csv')
    df = pd.read_csv(business_data_path)
    return df

def perform_pca(X, n_components=3):
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    return pca, X_pca, X_scaled

def analyze_clusters(df, X_scaled, fraud_criteria):
    """
    Analyzes fraud and non-fraud cases based on given criteria.
    Assigns fraud and non-fraud labels.
    """
    fraud_condition = ~(
        (df['percentage_4_star'] + df['percentage_5_star'] > df['percentage_1_star'] + df['percentage_2_star'] * 2) |
        (df['avg_timestamp_gap'] < 10) |
        (df['text_similarity_score'] > 0.7) |
        (df['%_under_3_months'] > 50) |
        (df['%_reviewers_with_less_than_10_reviews'] > 50)
    )
    df['Fraud_Label'] = np.where(fraud_condition, 'Fraud', 'Not Fraud')
    return df['Fraud_Label']

def plot_pca_interactive(df, X_pca, labels):
    """
    Creates an interactive 3D PCA visualization using Plotly.
    """
    plot_df = pd.DataFrame(X_pca, columns=["PCA1", "PCA2", "PCA3"])
    plot_df['Fraud_Status'] = labels
    plot_df['business_id'] = df['business_id']

    fig = px.scatter_3d(
        plot_df, x="PCA1", y="PCA2", z="PCA3", color="Fraud_Status",
        hover_data=["business_id"], title="PCA 3D Visualization - Fraud Detection"
    )
    output_path = "output/pca_interactive.html"
    os.makedirs("output", exist_ok=True)
    fig.write_html(output_path)
    print(f"Interactive PCA visualization saved at: {output_path}")

def save_fraud_business_ids(df):
    """
    Saves all fraud business IDs to a CSV file.
    """
    fraud_ids = df[df['Fraud_Label'] == 'Fraud']['business_id']
    output_path = "output/fraud_business_ids.csv"
    fraud_ids.to_csv(output_path, index=False, header=["business_id"])
    print(f"Fraud business IDs saved at: {output_path}")

def main():
    df = load_data()
    numerical_columns = [
        'stars', 'review_count', 'checkin_count', 'stars_std_dev',
        'percentage_1_star', 'percentage_2_star', 'percentage_3_star',
        'percentage_4_star', 'percentage_5_star', 'avg_timestamp_gap',
        'text_similarity_score', '%_under_3_months', '%_3_months_to_1_year',
        '%_over_1_year', '%_reviewers_with_less_than_10_reviews',
        'avg_star_deviation_squared'
    ]
    X = df[numerical_columns]

    # Perform PCA
    pca, X_pca, X_scaled = perform_pca(X)

    # Analyze fraud and assign labels
    labels = analyze_clusters(df, X_scaled, fraud_criteria=None)

    # Plot interactive PCA
    plot_pca_interactive(df, X_pca, labels)

    # Save fraud business IDs
    save_fraud_business_ids(df)

if __name__ == "__main__":
    main()
