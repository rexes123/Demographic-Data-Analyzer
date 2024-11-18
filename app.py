# Numpy

import numpy as np
# Syetem-specific parameters and functions
import sys
import pandas as pd
import matplotlib.pyplot as pltf
from decimal import Decimal
import sqlite3
import requests

# a = np.array([1,2,3], dtype='int32')
# print(a)

# b= np.array([[9.0,8.0, 7.0], [6.0,5.0, 4.0]])
# print(a)

# Get Dimension
# ndim is number of dimensions(axes) of the ndarray


# # Get shape
# print(a.ndim)
# # Get dimentsion
# print(a.shape)
# # Get data type
# print(a.dtype)

# # Get size
# print(a.itemsize)

# Get total size
# print(a.size * a.itemsize)
# print(a.nbytes)



# c =np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
# print(c)

# --------------------------Accessing/Changing specific element, rows, columns, etc---------------------------------
 
# a = np.array([[1,2,3,4,5,6,7], [8,9,10,11,12,13,14]])
# print(a.shape)

# Get spcific element [r,c]
# print(a[1,-1])

# Get a specific row
# : is get everything from the row
# print(a[0,:])

# Get a specific column
# print(a[:, 2])

# Getting a little more fancy[startIndex: endIndex: stepSize]
# print(a[0, 1:-1:2])

# a[1,5] = 21
# print(a)

# a[:,2] = 5
# print(a)

# 3-d example
# b= np.array([[[1,2], [3,4], [5,6], [7,8]]])


# Get specific element (work outside in)
# print(b)
# print(b[:,1,:])

# replace
# b[:,1,:] = [10,10]
# print(b)


# --------------------------------------Initializing difference Arrays-------------------------------------------------
# All 0s matrix
# np.zeros(5)
# print(np.zeros([2, 3]))

# All is matrix
# print(np.ones((4,2,1), dtype='int32'))

# Any other number  
# print(np.full((2,5,2), 1, dtype='float16'))

# output
#  [
#   [
#      [1 1]
#      [1 1]
#      [1 1]
#      [1 1]
#      [1 1]
#   ]
#   [
#      [1 1]
#      [1 1]
#      [1 1]
#      [1 1]
#      [1 1]
#   ]
#  ]

# Any other number (full_like)
# np.full(a.shape)
# a = np.array([[1,2,3,4,5,6,7,10], [8,9,10,11,12,13,14,16]])
# print(a.shape)
# Output
# (2,8)
# a = np.full_like(a, 4)


# Random decimal numbers
# a = np.random.rand(2,2)
# print(a)

# Random Integer values
# randint is used to generate random integers within a specified range
# a = np.random.randint(10, size=(3,3))
# print(a)

# a =  np.identity(5)
# print(a)


# arr = np.array([[1, 2, 3]])
# # print(arr)
# r1 = np.repeat(arr, 3, axis=0)
# print(r1)

# output = np.ones((5,5))
# print(output)

# Creating z
# z = np.zeros((3,3))
# print(z)

# Modify z
# z[0,2] = 9

# Modify output
# 1: -1 mean select element starting from index 1 up to (but not including) the last element
# Effectively leaving out the first and last rows and columns
# [1:-1,1:-1]
# Select the middle 3x3 portion
# output[1:-1,1:-1] = z 
# # print(output)

# a=np.array([1,2,3])
# b=a.copy()
# b[0] = 100


# print(b)

# a = np.array([1,2,3,4])
# # print(a+2)
# # print(a-2)
# # print(a*2)
# # print(a/2)

# b = np.array([1,0,1,0])
# a+b
# print(b)
# print(a+b)

# print(a**2)

# #Take the sin
# print(np.sin(a))


# For a lot more (https://docs.scipy.org/doc/numpy/reference/routines.math.html)

# Linear Algrebra
# a = np.ones((2,3))

# b = np.full((3,3), 2)
# # print(a)
# # print(b)

# matmul = np.matmul(a,b)
# print(matmul)


# Find the determinant
# c = np.identity(5)
# # print(c)
# np.linalg.det(c)

# print(np.linalg.det(c))

# https://numpy.org/doc/2.1/reference/routines.linalg.html
# Determinant
# Trace
# Singular Vector Decompositiom
# Eigenvalus
# Matrix Norm
# Inverse
# Etc

# Correct array initialization
# The determinant of a 2-D array [[a,b],[c,d]] is ad - bc
# a = np.array([[1,2], [3,4]])
# # [[a,b], [c,d]]
# # Compute the determinant
# b = np.linalg.det(a)
# print(b)

# (1)(4) - (2)(3) = 4-6
# output is -2

# print(np.linalg.det(a))

# Computing determinants for a stack of matrices(2)
# Stack of square matrices(2x2)
           # [[[a,b], [c,d]], [[e,f], [g,h]], [[i,j], [k,l]]]
# a = np.array([[[1,2], [3,4]], 
#               [[1,2], [2,1]], 
#               [[1,3], [3,1]]])
# print(a)
# print(a.shape)
# det(A) = a(ei - fh) - b(di-fg) +c(dh-eg)
# Matrix 1
# 12
# 34
# (a.b) - (b-c)
# (1.4) - (2.3) = 4-6 = -2

# Matrix 2
# 12
# 21
# (a.d) - (b-c)
# 1 - 4 = -3

# Matrix 3
# 13
# 31
# (a.d) - (d-c)
# 1-9 = -8



# determinants = np.linalg.det(a)
# print(determinants)
# output[-2, -3, -8]


# stats = np.array([[1,2,3], [4,5,6]])
# # print(stats)

# min = np.min(stats, axis=1)
# # print(min)

# max = np.max(stats, axis=1)
# # print(max)

# sum = np.sum(stats, axis= 1)
# print(sum)


# a = np.array(([1,2,3,4,5], [6, 7, 8, 9 , 10]))
# b = np.max(a, axis=1).sum()
# print(b)

# axis = 0, is apply each column
# axis  = 1, is apply each row

# before = np.array([[1,2,3,4], [5,6,7,8]])
# # print(before)
# # print(before.shape)

# after = before.reshape((2,2,2))
# print(after)

# Vertically stacking vertors
# v1 = np.array([1,2,3,4])
# v2 = np.array([5,6,7,8])
# # print(v1)
# # print(v2)

# v3 = np.vstack([v1,v2,v1,v2])
# # print(v3)

# # Horizontal stack
# h1 = np.ones((2,4))
# h2 = np.zeros((2,4))

# print(h1)
# print(h2)
# np.hstack((h1, h2))
# print(np.hstack((h1, h2)))
 
#  [
#  [1,1,1,1], [1,1,1,1]
#  ]

# a = np.ones((2, 4))
# b = a.reshape((4, 2))
# print(a)
# print(b)

# Load data from file
# Example data
# 1, 13, 21, 11, 196, 75, 4, 3, 34, 6, 7, 8, 0, 1, 2, 3, 4, 5 
# 3, 42, 12, 33, 766, 75, 4, 55, 6, 4, 3, 4, 5, 6, 7, 0, 11, 12
# 1, 22, 33, 11, 999, 11, 2, 1, 78, 0, 1, 2, 9, 8, 7, 1, 76, 88

# loadData = np.genfromtxt('data.txt', delimiter=',')
# loadData.astype('int16')
# print(loadData)

# Boolean masking and advanced indexing
# print(loadData > 50)

# You can index with a list in Numpy
# a = np.array([1,2,3,4,5,6,7,8,9])
# print(a)
# print(a[[1, 2, 8]])
# print(a[[1, 2, 8]])
# print(a[starting index, ending index])
# print(a[2:4, 0:2])

# print(a[2:4])

# a = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
# print(a)
# getData = np.all(loadData > 50, axis=0)
# print(getData)

# axis=0, add each column
# axis=1, add each row

# print(loadData > 50)
# print((loadData > 50) & (loadData < 100)) 


# loadData = np.genfromtxt('data.txt', delimiter=',')
# output = loadData[loadData < 50]
# print(output)


# Basic Numpy arrays
# a = np.array([1,2,3,4])
# # print(a)

# print(a[0], a[1])
# print(a[2:])
# print(a[0:2])
  


# print(2**2)

# 8 bits = 1 byte

# print(np.int8)

# print(2**7)


# print(np.int8)

# a = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.float64)
# print(a)

# c = np.array(['a','b','c'])
# print(c.dtype)

# print(a[0], a[2], a[3])

# b= np.array([0., 0.2, 1., 1.2, 2.])
# print(b.dtype)

# Dimensions and shapes
# ndim is Number of array dimensions
# A = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
# print(A.shape)
# print(A.ndim)
# print(A.size)

# B = np.array([
#     [
#     [12, 11, 10],
#     [9,8,7],
#     ],
#     [
#         [6,5,4],
#         [3,2,1]
#     ]
# ]
# )

# print(type(B[0]))


# Square matrix
# A = np.array([
#     [1,2,3],
#     [4,5,6],
#     [7,8,9]
# ])
# A[1] = np.array([10, 10, 10])
# A[2] = 100
# print(A)
# print[A[1]]

# a = np.array([1, 2, 3, 4, 6])
# print(a.sum())
# print(a.mean())
# print(a.std())

# Mean = (1+2+3+4+6)/5 = 3.2

# Variance = (1-3.2)^2 + (2-3.2)^2 + (3-3.2)^2 + (4-3.2)^2 + (6-3.2)^2  / 5
#          =2.96

# Standard deviation = 1.72

# b =np.array([
#     [1,2,3],
#     [4,5,6],
#     [7,8,9]
# ]
# )

# All value that occur vertically
# print(b.sum(axis=0))

# # All value that occur horinzontally
# print(b.sum(axis=1))

# print(b.std(axis=0))
# print(b.std(axis=1))

# a = np.array([
#     ['a','b','c'],
#     ['d','e','f'],
#     ['g','h','i']
# ])

# print(a[:,:2])

# Broadcasting and Vectorized operations
# a += 100
# print(a)

# l = [0,1,2,4]
# print([i * 10 for i in l])

# a = np.arange(5)
# 0,1,2,3
# print(a)
# print(a+20)
# print(a[0], a[-1])

# print(a.mean())
# mean = 2.5
# print(a[a>a.mean()])
# [2 3]

# ~ is convert each true to false, each false to true
# print(~(a > a.mean()))

# || logical or
# print(a)
# print(a[0])
# print(a[(a==1)])
# if true shown 1, if false shown 0
# print(a[(a==0)] | [a==1] | [a==2] | [a==3])
# print(a[(a <= 2) & (a % 2==0)])

# Return random integers from low(inclusive) to high(exclusive)
# A = np.random.randint(100, size=(3, 3))

# print(A)

# print(A[np.array([
#     [True, False, True],
#     [False, True, False],
#     [True, False, True]
# ])])

# print(A > 30)
# print(A[A>30])

# a = np.arange(5)
# print(a)

# print(a <= 3)

# Linear Algebra

# A = np.array([
#      [1, 2, 3],
#      [4, 5, 6],
#      [7, 8, 9]
#  ])

# # print(A)


# B = np.array([
#     [6,5],
#     [4,3],
#     [2,1]
# ])

# print(B)

# Numpy Algebra and Size
# print(A.dot(B))

# print(np.dot(3,4))

# dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k, :, m])

# np.dot([2j, 3j], [2j, 3j])


# Size of objects in Memory
# Int, floats
# An integer in Python is > 24bytes
# print(sys.getsizeof(1))
# Long are even larger
# sys - Sytem-specific parameters and function
# print(sys.getsizeof(10**100))


# print(np.dtype(int).itemsize)

# print(np.dtype(np.int8).itemsize)
# print(np.dtype(float).itemsize)



# List are even larger
# print(sys.getsizeof([1]))

# a = np.array([1]).nbytes
# print(a)

# And performance is also importance
# l = list(range(100))
# print(l)

# a = np.arange(1000)
# print(a)
# b = np.sum(a**2)
# print(b)


# a = np.arange(1000)
# print(a)

# start_time = time.time()
# print(start_time)
# l = list(range(1000))
# a = np.arange(1000)
# print(np.sum(a **2))
# print(sum([x**2 for x in l]))


# Useful numpy functions
# random
# Random Uniform distribution between 0 and 1
# Normal Gaussian distribution with custom mean(loc) and standard deviation(scale)
# print(np.random.random(size=2))
# print(np.random.normal(size=2))

# arange
# print(np.arange(5, 10))
# print(np.arange(0, 1, .1))


# reshape
# print(np.arange(10).reshape(5, 2))


# linspace
# print(np.linspace(0,1, 10, False))


# print(np.empty((2, 3)))


# look like a list
# a = pd.Series([35, 457, 63.121, 60.12, 12.12, 12.64, 124.12])
# a.name = 'G7 population in mil'
# l = ['a','b','c']

# a.index = [
#     'Canada',
#     'France',
#     'Germany',
#     'Italy',
#     'Japan',
#     'United Kingdom',
#     'United States'
# ]

# print(a.index)

# # Convert the series to a DataFrame for a tabular display
# # df = a.reset_index()
# # print(df)
# print(a.series)

# a = pd.Series([35.467, 63.951, 80.940, 60.665, 127.061, 64.511, 318.523])
# a.name = 'G7 Population in millions'

# print(a.values)
# print(type(a))
# print(a[0])
# print(a[1])
# print(a.index)

# l = ['a','b','c']
# print(l)

# a.index = [
#      'Canada',
#      'France',
#      'Germany',
#      'Italy',
#      'Japan',
#      'Uk',
#      'US'
#  ]





# print(a['Canada'])
# print(a['Japan'])


# print(pd.Series({
#     'Canada': 35.467,
#     'France': 63.951,
#     'Germany': 80.940,
#     'Italy': 80.940,
#     'Japan': 127.061,
#     'Germany': 80.940,
#     'Italy': 60.665,
#     'Japan': 1270.061,
#     'Uk': 64.411,
#     'Us': 318.523
# }, name='G7 Population in mil'))


# iloc stand for integer location

# Numeric position can also be used, with the iloc attibute:
# print(a.iloc[1])
# print(a.iloc[0])

# print(a['Canada': 'Italy'])

# l = ['a','b','c']
# print(l)
# print(l[:2])
# print(l[1:])


# Conditiona selection(boolean arrays)
# The same boolean array technique we saw applied to numpy arrays can be used for Pandas Series:
# print(a > 70)
# print(a[a > 70])
# print(a.mean())
# print(a > a.mean())
# print(a.std())
# print[((a > a.mean() - a.std()/2) | (a > a.mean() + a.std()/2))]
# result = ((a > a.mean() - a.std()/2) | (a < a.mean() + a.std()/2))
# print(result)

# print(a * 1000000)
# print(a.mean())
# print(a > a.mean())
# print(a.std())



# print(a.mean())
# print(a > a.mean())
# print(a.std())

# ~not
# !or
# &and


# result = ((a > a.mean() - a.std()/2) | (a < a.mean() + a.std()/2))
# print(result)

# Operation methods
# print(a)
# print(a*1_000_000)

# print(a.mean())


# log_a = np.log(a)
# print(log_a)

# print(a.mean())
# print(np.log(a))

# print(np.log10(10))
# print(a.mean())
# print(a['France': 'US'].mean())

# Boolean array
# print(a > 80)
# print(a[(a > 80 ) | (a < 40)])
# print(a[(a > 80) & (a < 200)])
# print(a)

# print(a)

# print(a.mean())

# print(a['France':'Italy'].mean())

# print(a > 80)
# print(a)
# print((a > 80) | (a <40))


# Modifying series
# b = a['Canada']=40.5
# a['Canada']=40.5
# print(a)



# iloc stand for integer location
# a[-1]=500
# print(a)
# a.iloc[-1]=500
# print(a)

# a[a < 70]=99.99
# print(a)

# certificats_earned = pd.Series(
#     [8, 2, 5, 6],
#     index=['Tom', 'Kris', 'Ahmad', 'Beau']
# )

# print(certificats_earned[certificats_earned > 5])

# dataFrame is df



# print(df.columns)
# print(df.index)

# print(df.info())
# print(df.size)
# print(df.shape)
# print(df.describe())
# print(df.dtypes)
# print(df.dtypes.value_counts())

# Indexing, selection and slicing
# print(df['Population'])
# print(df['Population'].to_frame())


# print(df['Population'])
# print(df['GDP'])

# print(df[0:3])
# print(df.loc['France' : 'Itally'])
# print(df.loc['France':'Italy','Population'])
# print(df.loc['France', 'Itally'])4
# print(df.loc['France': 'Italy', ['Population', 'GDP']])

# print(df)

# iloc stand for integer location
# print(df.iloc[0])
# print(df.iloc[1]
# print(df.iloc[[0, 1, -2]])

# print(df.iloc[1:5])
# print(df.iloc[1: 3])
# # print(df.iloc[1: 3, 3])
# print(df.iloc[1:3, [1,3]])
# print(df.iloc[1:3, 1:3])



# certificates_earned = pd.DataFrame({
#     'Certificates': [8, 2, 5, 6],
#     'Time (in months)': [16, 5, 9, 12]
# })

# certificates_earned.index = ['Tom', 'Kris', 'Ahmad', 'Beau']
# print(certificates_earned.iloc[2])

# Conditional Selection (boolean arrays)
# We saw conditional selection applied to Series and it'll work in the same way for DataFrame. After all, a DataFrame is a collection of Series:
# print(df)
# print(df)
# The boolean matching is done at index level, so can filter by any row, as long as it contain the right indexes. Column selection will works as expected:


# print(df.loc[df['Population'] > 70])


# print(df['Population'])
# print(df['GDP'])

# print(df['Population'] > 70)


# print(df['Population'] > 70)
# print(df.loc[df['Population']> 70])
# print(df.loc[df['Population']> 70, ['Population', 'GDP']])

# Dropping stuff
# df is data frame
# print(df.drop(['Canada','Italy']))
# print(df.columns)


# axis=1 is total cols, axis=0 is total rows
# axis=1 is total cols
# print(df)
# print(df.drop(['Italy', 'Canada'], axis=0))
# print(df.drop(['Population', 'HDI']), aixs=1)
# crisis = pd.Series([-1_000_000, -0.3], index=['GDP', 'HDI'])
# print(df[['GDP', 'HDI']] + crisis)


# Adding new column.


# df = pd.DataFrame({
    # 'Population': [35.467, 63.951, 80.94, 60.665, 126.061, 64.611, 318.523],
    # 'GDP': [
    #     1785387,
    #     2833687,
    #     3874437,
    #     2167744,
    #     4602367,
    #     2950039,
    #     17348075
    # ],
    # 'Surface Area':[
    #     9984670,
    #     640679,
    #     357114,
    #     301336,
    #     377930,
    #     242495,
    #     9525067
    # ],
    # 'HDI':[
    #     0.913,
    #     0.888,
    #     0.916,
    #     0.873,
    #     0.891,
    #     0.907,
    #     0.915
    # ],
    # 'Continent':[
    #     'America',
    #     'Europe',
    #     'Europe',
    #     'Europe',
    #     'Asia',
    #     'Europe',
    #     'America'
    # ]
# }, columns=['Population', 'GDP','Surface Area','HDI', 'Continent'])



# df.index = [
#     'Canada',
#     'France',
#     'Germany',
#     'Italy',
#     'Japan',
#     'UK',
#     'US'
# ]


# Modifiying DataFrames
# langs = pd.Series(['French', 'German','Italian'],
#               index=['France','Germany','Italy'],
#               name='Language'
#               )

# df['Language'] = langs

# df['Language'] = 'English'
# print(df)



# Renaming Columns
# data frame

# df.rename(
#     columns ={
#         'HDI': 'Human development index',
#         'Anual Popcorn Consumption' : 'APC'
#     }, index={
#         'United States': 'USA',
#         'United Kingdom': 'UK',
#         'Argentina': 'AR' 
#     }
# )

# print(df)
# print(df.rename(index=str.upper))
# A way to create small, anounymous function on the fly, function like map, filter 
# print(df.rename(index=lambda x: x.lower()))


# double = lambda x:x *2
# print(double(5))

# certificates_earned = pd.DataFrame({
#     'Certificates': [8, 2 ,5, 6],
#     'Time (in months)': [16, 5, 9, 12]
# })

# names= ['Tom', 'Kris','Ahmad','Beau']

# certificates_earned.index = names
# longest_streak = pd.Series([13, 11, 9, 7], index=names)
# certificates_earned['Longest streak'] = longest_streak

# print(certificates_earned)




# print(pd.isnull(np.nan))
# print(pd.isnull(None))
# print(pd.notnull(
# None))
# print(pd.notnull(3))
# print(pd.notnull(np.nan))
# isna is null or NA(Not available)

# These function also work with Series and DataFrame


# Missing data
# print(pd.notnull(2))

# Function also work with Series and DataFrame
# print(pd.notnull(pd.Series([1, np.nan, 7])))

# a = pd.isnull(pd.DataFrame({
#     'Column A': [1, np.nan, 7],
#     'Column B':[np.nan, 2, 3],
#     'Column C':[np.nan, 2, np.nan]
# }))

# print(a)

# Pandas Operation with Missing Values
# print(pd.Series([1, 2, np.nan]).mean())


# a = pd.Series([1, 2, 3, np.nan, np.nan, 4])
# print(a)

# b = np.array([1,2,3,4])
# print(b)
# print(pd.notnull(a))

# print(a)
# print(a.dropna())


# df = pd.DataFrame({
#      'Column A': [1, 3, 2, np.nan],
#      'Column B': [2, 4, 2, np.nan],
#      'Column C': [9, np.nan, 100, 20],
#      'Column D': [5, 8, 34, 110],
#  })
# print(df.shape)
# print(df.info)
# print(df.isnull().sum())

# print(df.dropna(how='all'))
# print(df.dropna(how='any'))

# # Drop rows with any NaN values
# print(df.dropna())

# s = pd.Series(['a',3 ,np.nan, 1, np.nan])
# print(s.dropna())


# Use thresh parameter to indicate a threshold(a minimum number) of non-null values for the row/column to be kept.
# print(df.dropna(thresh=1))
# print(df.dropna(thresh=3))


# print(df)
# print(s)

# Filling nulls with arbitrary value
# s = pd.Series([1,2 ,3, np.nan, np.nan, 4])
# Convert to numeric, ignoring non-numeric entries
# mean = pd.to_numeric(s, errors='coerce').mean()

# print(s.fillna(0))
# print(s.fillna(0))
# print(s.fillna(s.mean()))
# ffill() function is used to fill missing value in the dataframe. 'ffill' stands for 'forward fill'
# print(s.fillna(method='ffill'))


# bfill() method, replace the NULL values from the next row
# s = pd.Series(['a',3 ,np.nan, 1, np.nan])
# print(s.fillna(method='bfill'))


# Filling null valus on DataFrames
# np.array
# pd.Series


# forwardFill = pd.Series([np.nan, 3, np.nan, 9]).fillna(method='ffill')
# print(forwardFill)


# backwardFill = pd.Series([np.nan, 3, np.nan, 9]).fillna(method='bfill')
# print(backwardFill)

# Filling null values on DataFrames

# df = pd.DataFrame({
#      'Column A': [1, 3, 2, np.nan],
#      'Column B': [2, 4, 2, np.nan],
#      'Column C': [9, 32, 100, 20],
#      'Column D': [5, 8, 34, 110],
#  })
# print(df.fillna({'Column A': 0, 'Column B': 10, 'Column C': df['Column C'].mean()}))

# col -axis=1
# row -axis=0
# print(df)
# print(df.fillna(method='ffill', axis=1))
# print(df.fillna(method='ffill', axis=0))

# Checking if there are NAs
# The question is: Does this Series or DataFrame contain any missing value? The answer should be yes or not: True or False.

# Checking the length
# print(s)
# print(s.dropna().count())

# len() return the number of rows(or elements in the case of a series)

# s.dropna(): Create a s new Series without NaN values
# missing_values = len(s.dropna()) !=len(s)
# print(missing_values)


# More Pythonic solution any
# print(pd.Series([True, False, True]).any())
# print(pd.Series([True, True, True]).all())
# s = pd.Series(['a',3 ,np.nan, 1, np.nan])

# print(s.isnull())


# Can just use any method with the boolean array returned
# a = pd.Series([1, np.nan]).isnull().any()
# print(a)

# s = pd.Series(['a',3 ,np.nan, 1, np.nan])


# print(s)



# Cleaning not-null values
# After dealing with many datasets 
# Missing data is not such a big deal
# Clearly see values like np.nan
# The only thins need to do is just use method like is null and fillna/ dropna
# And pandas will take care of the rest
# But sometime, can have invalid values that not just "missing data"(None, or nan).

# print(df)

# print(df.fillna({
#     'Column A': 0,
#     'Column B': 2,
#     'Column C': df['Column C'].mean()
# }))



# print('Hi')
# df = pd.DataFrame({
#     'name': ['A','B','C','?'],
#     'age':[12, 20 ,30, 350]
# })
# print(df['name'].unique())
# print(df['name'].value_counts())
# print(df['name'].replace({'C':'F', '?':'M'}))
# print(df.replace({
#     'name': {
#         'C':'F',
#         '?':'M'
#     },
#     'age' : {
#         350: 35
#     }
# }))

# print(df.loc[df['Age'] > 100])

# print(df)

# df.loc[df['age'] > 100, 'age'] = df.loc[df['age'] > 100, 'age'] /10
# print(df)

# Duplicates
# ambassadors = pd.Series([
#     'France',
#     'UK',
#     'UK',
#     'Italy',
#     'Germany',
#     'Germany',
#     'Germany'
# ], index=[
#     'A',
#     'B',
#     'C',
#     'D',
#     'E',
#     'F',
#     'E'
# ])

# print(ambassadors)
# print(ambassadors.duplicated())
# print(ambassadors.duplicated(keep=False))


# Duplicates in DataFrames
# Conceptually speaking, duplicates in DataFrame happen at "row" level

# players = pd.DataFrame({
#     'Name': [
#         'Kobe Bryant',
#         'LeBron James',
#         'Kobe Bryant',
#         'Carmelo Anthony',
#         'Kobe Bryant',
#     ],
#     'Pos':[
#         'SG',
#         'SF',
#         'SG',
#         'SF',
#         'SF'
#     ]
# })

# # print(players)
# print(players.duplicated())
# print(players.duplicated(subset=['Name'], keep='last'))


# df = pd.DataFrame({
#     'Data': [
#         '1987_M_US_1',
#         '1990?_M_UK_1',
#         '1992_F_US_2',
#         '1970?_M_    IT    _1',
#         '1985_F_   IT_2'
#     ]
# })

# df = df['Data'].str.split('_', expand=True)

# df.columns = ['Year','Gender', 'Country', ' No Children']
# # print(df)

# print(df['Year'].str.contains('\?'))
# print(df['Country'].str.contains('U'))




# txt = "     banana    "
# print(txt.strip())


# text = 'I am Chin Hong'
# x = text.replace("Chin Hong", "CH")
# print(x)

# print(df['Country'])
# df = df['Country'].str.strip()
# print(df['Year'].str.replace(r'(?P<year>\d{4}\?)', lambda m: m.group('year')))


# Global API
# x = np.arange(-10, 11)
# plt.figure(figsize=(10, 16))
# plt.title('Nice plot')
# plt.plot(x, x ** 2)
# plt.plot(x, -x ** 2)
# plt.show()


# plt.figure(figsize=(12, 6))
# plt.title('My Nice Plot')
# # Rows, columns, panel selected
# # plt.subplot(1, 2, 1)
# plt.plot([0, 0, 0], [-10, 0, 100])
# plt.show()


# Get a Job

# Matplolib
# x = np.arange(-10, 11)
# plt.figure(figsize=(12, 6))
# plt.title('Nice Plot')

# # Rows, columns, panel selected
# # Have 1 row and 2 columns, therefore have 2 subplot, then active the first one
# plt.subplot(1, 2, 1)
# plt.plot(x, x**2)
# plt.plot(x, -1*(x**2))
# plt.plot([0, 0, 0], [-10, 0, 100])
# plt.xlabel('X')
# plt.ylabel('Y')

# plt.subplot(1, 2, 2)
# plt.plot(x, -1 *(x**2))
# plt.plot([-10, 0, 10], [-50, -50, -50])
# # # Three point
# # # (-10, -50)
# # # (0, -50)
# # # (10, -50)

# plt.legend(['X^2', 'Vertical Line'])
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.show()

# OOP in javascript
# Class Student{
#     # Data Properties
#     # Name
#     # Age
#     # Standard

#     # Method(action)
#     study(){

#     }

#     Play(){

#     }

#     doHomeWork(){

#     }
# }


# class MyClass:
#  x = 5

# # Create an instance of MyClass
# p1 = MyClass()
    
# print(p1.x)

# OOP interface
# fig, axes =plt.subplot(fiqsize=(12, 6))
# axes.plot(
#     x, (x**2), color='red', linewWidth=3,
#     marker = 'o', markersize= 8, label='X^2'
# )

# Corrected np.linespace
# x = np.linspace(-10, 10, 100)
# # Create the figure and axes
# fig, axes = plt.subplots(figsize=(12, 6))

# # Plot diffrent line with varying styles
# axes.plot(x, x+0, linestyle='solid', label='Solid Line')
# axes.plot(x, x+1, linestyle='dashed', label='Dashed Line')
# axes.plot(x, x+2, linestyle='dashdot', label='Dash-Dot Line')
# axes.plot(x, x+3, linestyle='dotted', label='Dotted Line')

# # Plot -X^2 with blue dashed line
# # '-': Solid line
# # ':'" Dotted line
# # '-.': Dash-dot line
# # 'b--': Dast-dot line
# axes.set_title('My Nice Plot')
# # Show Legend
# axes.legend()

# # Show the plot
# plt.show()


# Other types of plots
# We call the subplots() function get a tuple containing Figure and a axes element

# Axes are the lines that used to measure data on graph and grids
# There are two type of axes:  The vertical axis and horizontal axis

# plot_object = plt.subplots()
# fig, ax = plot_object
# # ax.plot([x-axis], [y-axis])
# ax.plot([1, 2, 3], [1,2,3])
# plt.show()

# plot_objects = plt.subplots(nrows= 2, ncols=2, figsize=(14,6))
# fig, ((ax1, ax2), (ax3, ax4)) = plot_objects
# plot_objects

# # x=0,10,20,30,40,50
# # y=1.45, 1.869

# ax1.plot(np.random.randn(50), c='red', linestyle='--')
# ax2.plot(np.random.randn(50), c='green', linestyle=':')
# ax3.plot(np.random.randn(50), c='green', marker='o', linewidth=3.0)
# ax4.plot(np.random.randn(50), c='yellow')

# plt.show()


# Histograms
# values = np.random.randn(1000)
# plt.subplots(figsize=(12, 6))
# plt.hist(values, bins=100, alpha=0.8,
# histtype='bar', color='steelblue',
# edgecolor='green'
# )
# plt.xlim(xmin=-5, xmax=5)

# plt.show()

# # Rows, columns, panel selected
# # plt.subplot(1, 2, 1)

# plt.subplot(2,2,1)
# plt.show()

# def addNumber(x, y):
#     return x + y

# print(addNumber(1, 2))

# language = 'Javascript'

# if language == 'Python':
#  print('Let get started')
# else: 
#  print('Let javascript')


# Variable
# name = "Chin Hong"
# age = 30
# print(type(name))
# print(type(age))

# print(Decimal('0.1') * 3)
# print((0.1)*3)

# print(len('Hello'))


# print(type(False))
# x = None
# print(x)
# print(type(None))
# print(type(13)==int)

# def greeting():
#     return 'Hello'
# print(greeting())

# def empty():
#     return 3

# result = empty()
# print(result)


# print(2**5)

# print(4>3)

# print(True and False)
# print(not False)
# print(False or True)

# days_subsribed = 28
# if (days_subsribed > 30):
#     print("Loyal customer")
# elif(days_subsribed >= 15 and days_subsribed <30):
#     print("Halfway customer")
# else :
#     print("New customer")


# names = ['A','B','C','D']

# for name in names:
#     print(name)

# While loop 
# count = 0
# while count <3:
#     print("Counting...")
#     count +=1

# Collections
# Lists
# Tuples
# Dictionaries
# Sets

# l = [3, "Hello", True]
# print(l)
# print(l[0])

# l.append('Hi')
# print(l)
# print("Hello" in l)
# print(l[0])

# Dictionaries
# user = {
#      "name":"Chin Hong",
#      "email":"rexes_123hotmail.com",
#      "age":30
#  }

# print(user)
# print(user['email'])
# print('age' in user)

# Sets
# s = {3, 1, 3, 7, 8, 10}
# s.add(11)
# # s.pop()
# print(s)
# print(5 in s)

# list =("a","b","c")
# list.remove("a")
# print(list)

# Iterating collections

# l = [3, 'Hllo', True]
# print(l)

# list is mutable
# immutable is immutable

# Iterating collections

# l = [3, 'Hello', True]
# # print(l)

# for el in l:
#     print(el)

# for key in user:
#     print(key.title(), '=>', user(key))

# for in is to print all the element from list



# print(user.values())
# for (value in user.values()):
# print(value)
# print(user)
# for key in user:
#     # print(key)
#     # print(key.title(), '=>', user[key])
#     print(user[key])

# Modules
# user = {
#      "name":"Chin Hong",
#      "email":"rexes_123hotmail.com",
#      "age":30
#  }
# print(user.values())
# for key, values in user.items():
#     print(key, values)
# for i in range(5):
#     print(i)

# x = user.items()
# print(x)


# Modules
# import random

# # randint(begineer, end)
# print(random.randint(0, 99))

# Exceptions
# age = "30"

# if age > 21:
#   print("Allow to entrance")
# else:
#   print("Not allow to entrace")

# a = 33
# b = 200

# if b > a:
#     print('B is greater a A')

# age = "30"

# try:
#   if age > 21:
#     print("Allow to entrance")
# except:
#    print("Something went wrong")


# Reading CSV and TXT files
#  Rather than creating Series or DataFrames structure from scatch

# Use of pandas is based on the loading of information from files or sources information
# for further exploration, transformation and analysis

# comma-separated values files(.csv)
# raw text files(.txt) into pandas DataFrame s.

# Read data with Python

# with open('data/btc-market-price.csv', 'r') as fp:
#     # print(fp)
#     for index, line in enumerate(fp.readlines()):
#         if (index < 100):
#             timestamp, price = line.split(',')
#             # print(f"{timestamp}, ${price}")
#             print(index, line)

# exam_review.csv

# !head exam_review.csv
# csv is command-seperated values

# with open('data/exam_review.csv', 'r') as file:
#     for index, line in enumerate(file):
#         if index < 10:
#             print(index, line)


# print(pd.read_csv)
# csv= comma-separated value
# df = pd.read_csv('data/btc-market-price.csv', 
#                  header=None,
#                  na_values=['','?','-']
#                  )
# print(df.head())
# # print(df.info())
# # print(df)
# # print(df.dtypes)
# print(df.dtypes)

# df is dataframe

# ex_df = pd.read_csv('data/exam_review.csv', sep='>')
# print(ex_df)

# df = pd.read_csv('data/btc-market-price.csv',
#                  na_values=['','?','-'],
#                  names=['Timestamp','Price'],
#                  dtype={'Price': 'float'}
#                  )
# print(df.dtypes)

# print('A','B','C', sep='>')


# exam_df = pd.read_csv('data/exam_review.csv', sep='>', skip_blank_lines=False)
# print(exam_df)

# # Save to CSV file
# # print(pd.read_csv('out.csv'))
# # print(exam_df.to_csv('out.csv'))
# print(pd.read_csv('out.csv'))


# Reading data from relational databases
# Read SQL queries and relational database tables into DataFrame objects using pandas
# Different techniques to persist that pandas DataFrame objects to datbase tables

# Read data from SQL database
# print('Hi')

# conn = sqlite3.connect('chinook.db')
# print(conn)

# cur=conn.cursor()
# print(cur)

# cur.execute('SELECT * FROM employees')
# results = cur.fetchall()
# print(results)

# Parsing HTML tables from the web
# html_url = "https://en.wikipedia.org/wiki/The_Simpsons"
# simpsons = pd.read_html(html_url)
# print(len(wiki_tables[0]))

# print(len(wiki_tables[1]))
# print(len(wiki_tables))
# print(wiki_tables)
# print(simpsons.head())
# print(simpsons)

# read_excel

df = pd.read_excel('products.xlsx')
print(df.head())
