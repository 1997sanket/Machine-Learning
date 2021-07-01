# ML program to find price of a house where area of the house is given

#Libraries required
import pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model



data_frame = pd.read_csv("home_prices.csv") #using Pandas to take data from CSV to our program
# print(data_frame)

reg = linear_model.LinearRegression()   #Getting linear regression model
reg.fit(data_frame[['area']],data_frame.price) #Putting our data in Linear Regression

print("For the line y = mx + b, slope(m) = ", reg.coef_)
print("For the line y = mx + b, intercept(b) = ", reg.intercept_)

#Prediction for 3300 sq ft of house
print("Prediction price for 3300 sq ft of house is : ", reg.predict([[3300]]))  #For some reason its only accepts 2D array

#Reading a CSV file with only areas
df2 = pd.read_csv("areas.csv")
results = reg.predict(df2)  #Storing predicted results

#Adding new coloum 'prices' to csv file
df2['price'] = results

#Creating a new csv file without index and storing the predicted prices
df2.to_csv("prediction.csv", index=False)

predicted = pd.read_csv("prediction.csv")

plt.xlabel('area', fontsize=20)
plt.ylabel('price', fontsize=20)
plt.scatter(data_frame.area, data_frame.price, color='red', marker='+')
plt.plot(data_frame.area, reg.predict(data_frame[['area']]), color='blue')
plt.show()
