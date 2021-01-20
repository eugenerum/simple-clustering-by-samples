import pandas as pd
import numpy as np
import string

index_row = int(-1)
#Reading every string 
for row in data.itertuples():
    index_row = index_row + 1
    #readig columns 
    for i, j in zip(range(1, 4, 2), range(2, 4, 2)):
        #Is null next field?
        if pd.isnull(data.iloc[index_row][i]) == True:
            break
        else:
            #Required attributes to form a data structure
            df.loc[len(df)] = np.nan
            df = df.shift() 
            df.at[0,'y'] = data.iloc[index_row][0]
            df.at[0,'X'] = data.iloc[index_row][1]

#Drop usless colums         
df.drop(df.iloc[:, 2:], inplace = True, axis = 1)