import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import matplotlib.pyplot as plt

#loading data
df= pd.read_csv(r'C:\Users\HAIER\Downloads\Housing.csv')

#removing missing values
df= df.dropna()

# Clean column names
df.columns= df.columns.str.strip().str.lower()

# Convert yes/no to 1/0
columns_to_convert= ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                  'airconditioning', 'prefarea']
for col in columns_to_convert:
        df[col] = df[col].astype(str).map({'yes': 1, 'no': 0})

# Convert furnishingstatus
df['furnishingstatus'] = df['furnishingstatus'].map ({'unfurnished': 0, 'semi-furnished': 1, 'furnished': 2})

# Prepare features and target
X= df.drop('price', axis = 1)
y= df['price']

# Split data
X_train,X_test, y_train,y_test= train_test_split (X,y, test_size= 0.2, random_state= 42)

# Create and train model
print ("Training Model")
model= LinearRegression()
model.fit( X_train, y_train)
print ("Model Training Completed✔")

# Make predictions
print ("Making predictions>>>")
y_pred= model.predict (X_test)
print (f'Model prediction completed✔')

#Model Evaluation
print ("Evaluating Model")
print ("Model Evaluation Completed")

mae= mean_absolute_error(y_test, y_pred)
rmse= root_mean_squared_error (y_test, y_pred)

# Print results
print ("Results")
print(f'MAE: Rs.{mae:,.0f}')
print(f'RMSE: Rs.{rmse:,.0f}')

#Predictions
def predict_house_price(area, bedrooms, bathrooms, stories, mainroad, guestroom, 
                       basement, hotwaterheating, airconditioning, parking, 
                       prefarea, furnishingstatus):

    mainroad = 1 if mainroad == 'yes' else 0
    guestroom = 1 if guestroom == 'yes' else 0
    basement = 1 if basement == 'yes' else 0
    hotwaterheating = 1 if hotwaterheating == 'yes' else 0
    airconditioning = 1 if airconditioning == 'yes' else 0
    prefarea = 1 if prefarea == 'yes' else 0
    
    furnishing_map = {'unfurnished': 0, 'semi-furnished': 1, 'furnished': 2}
    furnishingstatus = furnishing_map[furnishingstatus]

    input_data = [[area, bedrooms, bathrooms, stories, mainroad, guestroom,
                   basement, hotwaterheating, airconditioning, parking, 
                   prefarea, furnishingstatus]]
    
    prediction = model.predict(input_data)[0]
    return f"Rs.{prediction:,.0f}"

print("\nExample Prediction:")
example_price = predict_house_price(
    area=3000, bedrooms=3, bathrooms=2, stories=2,
    mainroad='yes', guestroom='no', basement='yes',
    hotwaterheating='no', airconditioning='yes', 
    parking=2, prefarea='yes', furnishingstatus='semi-furnished'
)

print(f"Predicted Price: {example_price}")


# Visulaization
# plotting
print ("Scatter Plot✔")
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha= 0.2)
plt.plot([y_test.min(), y_test.max()], [y_test.min(),y_test.max()], 'r--', lw= 2)
plt.xlabel('Actual Price (Rupees)')
plt.ylabel('Predicted Price(Rupees)')
plt.title(('Actual vs Predicted House Prices'))
plt.grid(True, alpha= 0.3)
plt.show()

#Histogram
print ("Histogram ✔")
plt.figure(figsize=(8,6))
errors= y_test- y_pred
plt.hist(errors, bins=30, edgecolor= "black")
plt.axvline(x= 0, color= 'red', linestyle= '--')
plt.xlabel('Prediction Error (Actual - Predicted)')
plt.ylabel('Frequency')
plt.title("Prediction Error Distribution")
plt.grid(True, alpha= 0.3)
plt.show()