# Fraud Yelper
## Fraud Anomaly Detection in Yelp Reviews Using Unsupervised Learning
We propose a machine learning model that segregates businesses that have potentially used fraudulent reviews from legitimate businesses by purely looking at quantitative metrics.

We propose that removing the human subjectivity from distinguishing between legitimate businesses and businesses that potentially use fraudulent reviews allows companies and review sites like Yelp to make somewhat objective decisions about potentially suspicious businesses and protect users or at least warn them of what they might expect.

Contributors: Roshan Pathak, Ayushman Roy

## Table of Contents
- [Setup Instructions](#setup-instructions)
- [Dependency Installation](#dependency-installation)
- [Preprocessing Steps](#preprocessing-steps)
- [Anomaly Detection Scripts](#anomaly-detection-scripts)
  - [KMeans](#kmeans)
  - [PCA](#pca)
  - [DBSCAN](#dbscan)
  - [Isolation Forest](#isolation-forest)
- [Output Details](#output-details)

## Setup Instructions

1. **Download the Yelp dataset**:  
   Obtain the dataset from the official [Yelp Dataset Page](https://www.yelp.com/dataset).  

2. **Clone the repository**:  
   Clone this GitHub repository to your local machine:  
   ```bash
   git clone https://github.com/ayushman-roy/Fraud_Yelper
   cd Fraud_Yelper
   ```

3. **Organize the dataset**:  
   Rename the downloaded Yelp dataset folder to `Yelp_JSON` and move it under the repository directory:  
   ```
   Fraud_Yelper/
   ├── Yelp_JSON/
   ├── data_processor.py
   ├── feature_engineering.py
   └── ...
   ```

## Dependency Installation

To ensure a smooth setup, we've provided a `requirements.txt` file with all necessary Python libraries. Follow these steps to install dependencies:

1. **Navigate to the project directory**:
   ```bash
   cd Fraud_Yelper
   ```

2. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify library installation**:
   ```bash
   pip list
   ```

### `requirements.txt` Contents
```
pandas==1.5.3
numpy==1.24.3
matplotlib==3.7.2
scikit-learn==1.2.2
seaborn==0.12.2
plotly==5.17.0
os-sys==2.1.4
```

**Note**: It's recommended to use a virtual environment to prevent potential conflicts with existing Python packages.

4. **Update paths if necessary**:  
   Open `data_processor.py` and `feature_engineering.py`, and verify that file paths match your local directory structure. Modify any paths if required.

5. **Customize sample size** (optional):  
   To limit the number of businesses processed, edit line 29 in `data_processor.py`:  
   ```python
   sampled_data = data.sample(n=5000)  # Change 5000 to your desired sample size
   ```

6. **Run preprocessing scripts sequentially**:  
   Execute the following scripts in order:
   ```bash
   python data_processor.py
   python feature_engineering.py
   ```

## Anomaly Detection Scripts

### **KMeans**
1. Navigate to the KMeans script:  
   ```bash
   cd scripts
   python kmeans_analysis.py
   ```

2. **Expected Output**:  
   - Visualization of clusters based on the selected features.
   - Saved cluster visualizations in the `output/` directory.
   - Clustering insights (e.g., number of clusters, anomalies, etc.) logged to the console.

### **PCA**
1. Run the PCA analysis script:  
   ```bash
   python pca_analysis.py
   ```

2. **Expected Output**:  
   - Dimensionality reduction results in a 3D interactive plot.
   - Anomaly identification in reduced space.
   - Output files and visualizations saved under `output/`.

### **DBSCAN**
1. Execute the DBSCAN clustering script:  
   ```bash
   python dbscan_analysis.py
   ```

2. **Expected Output**:  
   - DBSCAN clustering visualizations with noise points identified.
   - Outliers highlighted in the scatter plot. Hovering over points displays business IDs.
   - Anomalous business IDs saved to `output/dbscan_anomalous_ids.csv`.

### **Isolation Forest**
1. Run the Isolation Forest anomaly detection script:  
   ```bash
   python isolation_forest_analysis.py
   ```

2. **Expected Output**:  
   - Visualization of data points with anomalies highlighted.
   - Hovering over anomalies displays corresponding business IDs.
   - Anomalous business IDs saved to `output/isolation_forest_anomalous_ids.csv`.

## Output Details

1. **Visualizations**:
   - Enhanced, colorful plots for each algorithm are saved in the `output/` directory.
   - Hoverable tooltips display business IDs for anomalous points (DBSCAN & Isolation Forest).

2. **CSV Files**:
   - DBSCAN anomalies: `output/dbscan_anomalous_ids.csv`
   - Isolation Forest anomalies: `output/isolation_forest_anomalous_ids.csv`

Feel free to explore the scripts and customize parameters for further experimentation. For detailed explanations of the algorithms and the project, refer to the [Documentation](./docs.txt).
