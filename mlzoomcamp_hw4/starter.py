#!/usr/bin/env python
# coding: utf-8

# In[70]:


import pickle
import pandas as pd
import sklearn
import sys


# In[58]:


year = int(sys.argv[1])
month = int(sys.argv[2])


# In[59]:


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[60]:


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# In[61]:


f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'


# In[62]:


df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet')


# In[63]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


# In[64]:


y_pred.std()


# In[69]:


print(y_pred.mean())


# In[65]:


df_result = pd.DataFrame()


# In[66]:


df_result['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')


# In[67]:


df_result['pred'] = y_pred


# In[68]:


df_result.to_parquet(
    f"output_{year:04d}-{month:02d}",
    engine='pyarrow',
    compression=None,
    index=False
)

