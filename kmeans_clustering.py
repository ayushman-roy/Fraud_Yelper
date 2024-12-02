import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_classif
import plotly.express as px
from sklearn.decomposition import PCA

def load_data():
    base_path = os.path.dirname(os.path.abspath(__file__))
    business_data_path = os.path.join(base_path, 'cleaned_datasets', 'business_data.csv')
    df = pd.read_csv(business_data_path)
    return df

def select_important_features(X, percentile=70):
    # Generate a dummy target variable to compute mutual information scores
    y_dummy = np.random.randint(0, 2, size=len(X))
    mi_scores = mutual_info_classif(X, y_dummy, discrete_features=False)
    top_feature_indices = np.argsort(mi_scores)[-int(len(mi_scores) * percentile / 100):]
    return X.columns[top_feature_indices]

def perform_kmeans(X):
    # Impute and scale
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Select most important features
    X_df = pd.DataFrame(X_scaled, columns=X.columns)
    important_feature_names = select_important_features(X_df)
    X_important = X_df[important_feature_names].values
    
    # Perform K-Means
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
    kmeans.fit(X_important)
    
    return kmeans, X_scaled, important_feature_names

def analyze_clusters(df, labels):
    # Attach labels to the dataframe
    df['Cluster'] = labels
    
    # Compute average metrics for each cluster
    cluster_metrics = {}
    for cluster in [0, 1]:
        cluster_data = df[df['Cluster'] == cluster]
        cluster_metrics[cluster] = {
            'avg_timestamp_gap': cluster_data['avg_timestamp_gap'].mean(),
            'text_similarity_score': cluster_data['text_similarity_score'].mean(),
            '%_under_3_months': cluster_data['%_under_3_months'].mean(),
            '%_reviewers_with_less_than_10_reviews': cluster_data['%_reviewers_with_less_than_10_reviews'].mean(),
            'review_polarization': (cluster_data['percentage_4_star'] + cluster_data['percentage_5_star']).mean() / 
                                   (cluster_data['percentage_1_star'] + cluster_data['percentage_2_star'] + 1).mean()
        }
    
    # Determine fraud cluster based on multiple indicators
    fraud_indicators = [
        'avg_timestamp_gap',
        'text_similarity_score', 
        '%_under_3_months', 
        '%_reviewers_with_less_than_10_reviews',
        'review_polarization'
    ]
    
    fraud_cluster = max(
        [0, 1], 
        key=lambda cluster: sum(
            1 for indicator in fraud_indicators 
            if cluster_metrics[cluster][indicator] > cluster_metrics[1-cluster][indicator]
        )
    )
    
    return fraud_cluster, cluster_metrics

def plot_kmeans_interactive(df, X_scaled, labels, important_feature_names):
    # Identify fraud cluster
    fraud_cluster, cluster_metrics = analyze_clusters(df, labels)
    
    # Reduce dimensions for visualization
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X_scaled)
    
    # Create an interactive DataFrame for Plotly
    plot_df = pd.DataFrame(X_reduced, columns=["PCA1", "PCA2"])
    plot_df['Cluster'] = labels
    plot_df['business_id'] = df['business_id']
    plot_df['Fraud_Status'] = plot_df['Cluster'].map({
        fraud_cluster: 'Fraud', 
        1-fraud_cluster: 'Not Fraud'
    })
    
    # Create the plot
    fig = px.scatter(
        plot_df, x="PCA1", y="PCA2", 
        color='Fraud_Status',
        hover_data=["business_id"], 
        title="KMeans Clustering - Fraud Detection"
    )
    fig.write_html("output/kmeans_interactive.html")
    
    # Print analysis results
    print("\nFraud Detection Analysis:")
    print(f"Fraud Cluster: {fraud_cluster}")
    
    # Print cluster metrics
    for cluster, metrics in cluster_metrics.items():
        print(f"\nCluster {cluster} Indicators:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
    
    print("\nInteractive visualization saved at: output/kmeans_interactive.html")
    
    return fraud_cluster

def save_fraud_business_ids(df, labels, fraud_cluster):
    """
    Saves the business IDs of fraud businesses to a CSV file.
    
    Parameters:
        df (pd.DataFrame): Original dataframe containing business data.
        labels (np.ndarray): Clustering labels.
        fraud_cluster (int): Label of the fraud cluster.
    """
    # Filter fraud businesses
    fraud_businesses = df[labels == fraud_cluster]
    
    # Select only business IDs
    fraud_ids = fraud_businesses['business_id']
    
    # Save to CSV
    output_path = os.path.join("output", "fraud_business_ids.csv")
    os.makedirs("output", exist_ok=True)
    fraud_ids.to_csv(output_path, index=False, header=["business_id"])
    
    print(f"Fraud business IDs saved to: {output_path}")


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
    
    # Perform K-Means
    kmeans, X_scaled, important_feature_names = perform_kmeans(X)
    
    # Plot and analyze results
    fraud_cluster = plot_kmeans_interactive(df, X_scaled, kmeans.labels_, important_feature_names)

    # Save fraud business IDs
    save_fraud_business_ids(df, kmeans.labels_, fraud_cluster)

if __name__ == "__main__":
    main()
