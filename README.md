
House Price Prediction - Machine Learning Project
Project Overview:
A comprehensive machine learning pipeline for predicting residential property prices using Linear Regression. This project implements a complete data science workflow from data preprocessing to model deployment with evaluation.

Business Objective:
Develop an accurate pricing model for real estate properties to assist buyers, sellers, and real estate professionals in making informed decisions.

Dataset Information:
- Source: Kaggle House Price Prediction Dataset
- Size: 545 property listings
- Features: 12 property attributes
- Target Variable: price (in Indian Rupees)
- Price Range: ₹1.75M -  ₹13.3M
Data Pipeline Architecture
Raw Data -> Cleaning -> Feature Engineering -> Model Training -> Evaluation -> Deployment
1. Data Preprocessing
- Missing Value Handling: Complete case analysis (listings with missing data removed)
- Feature Encoding:
  - Binary features (mainroad, guestroom, etc.): Label encoding (yes=1, no=0)
  - Ordinal feature (furnishingstatus): Manual mapping (0-2 scale)
- Column Standardization: Lowercase conversion and whitespace removal

2. Model Development
- Algorithm: Linear Regression
- Train-Test Split: 80-20 stratified split with random seed 42
- Feature Set: All available property characteristics excluding target
- Validation: Hold-out validation strategy

3. Performance Metrics

Mean Absolute Error (MAE):  ₹979,680 - This shows the average prediction deviation.

Root Mean Squared Error (RMSE): ₹1,331,071 - This indicates the error magnitude with outlier penalty.

Model Insights
- Feature Importance: Property area and bedroom count show highest correlation with price
- Location Premium: Properties with mainroad access and prefarea designation command higher prices
- Amenity Value: Air conditioning and parking facilities significantly impact valuation

Usage Instructions
Prerequisites:
Python 3.8+
pip install pandas scikit-learn matplotlib
Execution:
1. Clone repository
git clone https://github.com/Araiba-Suhail/House-pirce-prediction-model.py
2. Navigate to directory
cd House_price_prediction_Model.py
3. Run model
python House_price_prediction_model.py

Methodology Details
Feature Engineering Strategy
1. Numerical Features: Used as-is (area, bedrooms, bathrooms, stories, parking)
2. Binary Features: Boolean conversion (amenities, location factors)
3. Ordinal Features: Rank-based encoding (furnishing quality)
Evaluation Protocol
1. Training Phase: 80% data for parameter estimation
2. Testing Phase: 20% unseen data for unbiased evaluation
3. Error Metrics: MAE for business interpretation, RMSE for model refinement

Visual Analytics
1. Prediction Accuracy Scatter Plot
2. Error Distribution Analysis Historgram

License
MIT License - See LICENSE file for details

Contributing
1. Fork the repository
2. Create a feature branch
3. Commit changes with descriptive messages
4. Push to the branch
5. Open a Pull Request




