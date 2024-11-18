import app as py
import pandas as pd
import matplotlib.pyplot as plt
import os
import subprocess

# sales = pd.read_csv(
#      'data/sales_data.csv',
#  parse_dates=['Date'])

# sales= pd.read_csv('data/sales_data.csv', parse_dates=['Date'])
# # print(sales['Unit_Cost'].median())
# # print(sales['Unit_Cost'].plot(kind='box', vert=False, figsize=(14,6)))
# # print(sales['Customer_Age'].plot(kind='kde', figsize=(14,6)))
# # print(sales['Month'].value_counts().plot(kind='bar', figsize=(14,6)))
# print(sales['Year'].value_counts())
# year_counts = sales['Year'].value_counts()

# # Plot the pie chart
# year_counts.plot(kind='pie', figsize=(6,6), autopct='%1.1f%%')  # autopct adds percentage labels

# # Display the plot
# plt.title("Distribution of Sales by Year")
# plt.show()# print(sales['Customer_Age'].plot(kind='kde', figsize=(14,6)))

df = pd.read_excel('data/product.xlsx')
print(df.head())
