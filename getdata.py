import pandas as pd
import numpy as np
import string

#load file
data = pd.read_csv('data.xlsx')

#Create an empty DataFrame
df = pd.DataFrame({"y": [], "X": []})

#Padding with required columns
for i in range(0, 2):
    df['column_' + str(i)] = 0
    
#Concat data and column names
data.columns = df.columns

#Return object type
for column_astype in df.columns:
    df[column_astype] = df[column_astype].astype(str)
