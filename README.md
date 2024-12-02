# Fraud Yelper
## Fraud Anomaly Detection in Yelp Reviews Using Unsupervised Learning
We propose a machine learning model that segregates businesses that have potentially used fraudulent reviews from legitimate businesses by purely looking at quantitative metrics.

We propose that removing the human subjectivity from distinguishing between legitimate businesses and businesses that potentially use fraudulent reviews allows companies and review sites like Yelp to make somewhat objective decisions about potentially suspicious businesses and protect users or at least warn them of what they might expect.

## Usage Instructions
1. Download the Yelp dataset (https://www.yelp.com/dataset).
2. Download the GitHub repository (https://github.com/ayushman-roy/Fraud_Yelper).
3. Rename the Yelp dataset as Yelp_JSON and move it under the repository directory.
4. Go to data_processor.py and feature_engineering.py and change any file paths if required.
5. You may change the number of businesses to classify through line 29 @ data_processor.py by changing the sample size.
6. Run data_processor.py and feature_engineering.py sequentially. 

## Requirements
1. Python
2. pandas
3. NumPy
4. scikit-learn
