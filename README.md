
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

Metric
Value
Interpretation
Mean Absolute Error (MAE)
 ₹ 979,680
Average prediction deviation
Root Mean Squared Error (RMSE)
 ₹ 1,331,071
Error magnitude with outlier penalty


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
cd house-price-prediction

4. Run model
python house_price_prediction.py

Making Predictions
from house_price_prediction import predict_house_price
Example: Predict price for a 3BHK semi-furnished apartment
prediction = predict_house_price(
    area=3000,          # Square feet
    bedrooms=3,         # Number of bedrooms
    bathrooms=2,        # Number of bathrooms
    stories=2,          # Building stories
    mainroad='yes',     # Main road access
    guestroom='no',     # Guest room availability
    basement='yes',     # Basement present
    hotwaterheating='no', # Water heating
    airconditioning='yes', # AC availability
    parking=2,          # Parking spaces
    prefarea='yes',     # Preferred area
    furnishingstatus='semi-furnished' # Furnishing level
)

print(f"Estimated Market Value: {prediction}")
Output: Estimated Market Value: ₹5,420,000

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
1. Prediction Accuracy Plot
- X-axis: Actual market prices
- Y-axis: Model predictions  
- Red Line: Perfect prediction line (y=x)
- Interpretation: Cluster proximity to diagonal indicates prediction accuracy

2. Error Distribution Analysis
- Distribution: Approximately normal with mean near zero
- Variance: Error spread indicates prediction consistency
- Outliers: Few extreme errors suggest data anomalies or model limitations
License
MIT License - See LICENSE file for details

Contributing
1. Fork the repository
2. Create a feature branch
3. Commit changes with descriptive messages
4. Push to the branch
5. Open a Pull Request




