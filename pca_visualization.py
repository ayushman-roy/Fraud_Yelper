import pandas as pd
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    return pca, X_pca

def plot_pca_results(X_pca, labels=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], 
                          c=labels, cmap='viridis', s=50, alpha=0.8)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    plt.colorbar(scatter, ax=ax, label='Cluster Label' if labels is not None else 'Points')
    plt.title('PCA 3D Visualization', fontsize=14)
    output_path = os.path.join('output', 'pca_results.png')
    os.makedirs('output', exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"PCA visualization saved at: {output_path}")

def main():
    df = load_data()
    numerical_columns = ['stars', 'review_count', 'checkin_count', 'stars_std_dev',
                         'percentage_1_star', 'percentage_2_star', 'percentage_3_star',
                         'percentage_4_star', 'percentage_5_star', 'avg_timestamp_gap']
    X = df[numerical_columns]
    pca, X_pca = perform_pca(X)
    plot_pca_results(X_pca)

if __name__ == "__main__":
    main()
