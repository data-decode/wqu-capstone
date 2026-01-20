#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install nolds')
get_ipython().system('pip install hmmlearn')
get_ipython().system('pip install yfinance')
get_ipython().system('pip install statsmodels')


# In[3]:


import yfinance as yf
import datetime
from numpy import *
from pylab import plot, show

import pandas as pd
import numpy as np

import numpy as np
from pathlib import Path
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
#from google.colab import drive


# In[17]:


#Umcomment this method when running from google colab
#def load_data_from_drive(start_date, end_date):
  #drive.mount('/content/drive', force_remount=True)
  #base_path = Path("/content/drive/My Drive/WQU/Capstone/data/parquet_out")
  #dfs = []
  #for d in sorted(base_path.iterdir()):
      #if d.is_dir() and start_date <= d.name <= end_date:
          #dfs.append(pd.read_parquet(d))

  #return pd.concat(dfs, ignore_index=True)

def load_data(start_date, end_date):
  base_path = Path("./data/parquet_out")
  dfs = []
  for d in sorted(base_path.iterdir()):
      if d.is_dir() and start_date <= d.name <= end_date:
          dfs.append(pd.read_parquet(d))

  return pd.concat(dfs, ignore_index=True)




# In[5]:


df = load_data("2026-01-01", "2026-01-12")
print(df.head())
print(df.tail())


# **Function to get 1 min data**

# In[6]:


def fill_intraday_minutes(day_df):
    day = day_df.index[0].date()
    full_index = pd.date_range(
        start=pd.Timestamp(f"{day} 09:15"),
        end=pd.Timestamp(f"{day} 15:30"),
        freq="1min"
    )
    return day_df.reindex(full_index).ffill()

def get_data_one_min_freq(start_date, end_date):
  df = load_data(start_date=start_date, end_date=end_date)
  df.drop(columns=['time', 'symbol', 'security_id', 'level2', 'bcast_time'], inplace=True)
  df['ltt'] = pd.to_datetime(df['ltt'])
  df = df.sort_values('ltt').set_index('ltt')
  df_1min = (
    df
    .groupby(df.index.date, group_keys=False)
    .apply(fill_intraday_minutes)
  )
  return df_1min



# In[7]:


df_1min = get_data_one_min_freq("2026-01-01", "2026-01-12")
print(df_1min.index.to_series().diff().value_counts())
print(df_1min.groupby(df_1min.index.date).size().value_counts())


# **Function to get 5 mins data**

# In[8]:


def fill_intraday_5min(day_df):
    day = day_df.index[0].date()

    # Create fixed 5-min intraday grid
    full_index = pd.date_range(
        start=pd.Timestamp(f"{day} 09:15"),
        end=pd.Timestamp(f"{day} 15:30"),
        freq="5min"
    )

    # Resample to 5-min bars (last price per bar)
    day_5min = (
        day_df
        .resample("5min")
        .last()
    )

    # Reindex to full grid and forward-fill within day
    return day_5min.reindex(full_index).ffill()


def get_data_5min_freq(start_date, end_date):
    df = load_data(start_date=start_date, end_date=end_date)

    df.drop(
        columns=['time', 'symbol', 'security_id', 'level2', 'bcast_time'],
        inplace=True
    )

    df['ltt'] = pd.to_datetime(df['ltt'])
    df = df.sort_values('ltt').set_index('ltt')

    df = df[~df.index.isna()]

    df_5min = (
        df
        .groupby(df.index.date, group_keys=False)
        .apply(fill_intraday_5min)
    )

    return df_5min


# In[9]:


df_5min = get_data_5min_freq("2026-01-01", "2026-01-12")
print(df_5min.index.to_series().diff().value_counts())
print(df_5min.groupby(df_5min.index.date).size().value_counts())


# In[10]:


df_5min.groupby(df_5min.index.date).size().head()


# **Function to get 15 mins data**

# In[11]:


def fill_intraday_15min(day_df):
    day = day_df.index[0].date()

    # Create fixed 15-min intraday grid
    full_index = pd.date_range(
        start=pd.Timestamp(f"{day} 09:15"),
        end=pd.Timestamp(f"{day} 15:30"),
        freq="15min"
    )

    # Resample to 5-min bars (last price per bar)
    day_15min = (
        day_df
        .resample("15min")
        .last()
    )

    # Reindex to full grid and forward-fill within day
    return day_15min.reindex(full_index).ffill()


def get_data_15min_freq(start_date, end_date):
    df = load_data(start_date=start_date, end_date=end_date)

    df.drop(
        columns=['time', 'symbol', 'security_id', 'level2', 'bcast_time'],
        inplace=True
    )

    df['ltt'] = pd.to_datetime(df['ltt'])
    df = df.sort_values('ltt').set_index('ltt')

    df = df[~df.index.isna()]

    df_15min = (
        df
        .groupby(df.index.date, group_keys=False)
        .apply(fill_intraday_15min)
    )

    return df_15min


# In[12]:


df_15min = get_data_15min_freq("2026-01-01", "2026-01-12")
print(df_15min.index.to_series().diff().value_counts())
print(df_15min.groupby(df_15min.index.date).size().value_counts())


# In[13]:


df_15min.tail()


# In[ ]:


df_15min.groupby(df_15min.index.date).size().head()


# Function to get 60 mins data

# In[14]:


NUMERIC_COLS = [
    "tot_vol", "ask", "bid", "oi", "askqty", "bidqty",
    "tot_buyqty", "tot_sellqty", "ltq", "atp"
]

for col in NUMERIC_COLS:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")



PRICE_COL = "ltp"
VOLUME_COL = "tot_vol"

LAST_COLS = [
    "ask", "bid", "oi", "askqty", "bidqty",
    "tot_buyqty", "tot_sellqty", "ltq", "atp", "recv_time"
]



def build_agg_dict(df):
  agg = {
      PRICE_COL: ["first", "max", "min", "last"],
      VOLUME_COL: "sum"
  }

  for col in LAST_COLS:
    if col in df.columns:
      agg[col] = "last"
  return agg


def get_data_60min_freq(start_date, end_date):
    df = load_data(start_date=start_date, end_date=end_date)

    df.drop(
        columns=['time', 'symbol', 'security_id', 'level2', 'bcast_time'],
        inplace=True
    )


    NUMERIC_COLS = [
        "tot_vol", "ask", "bid", "oi",
        "askqty", "bidqty", "tot_buyqty",
        "tot_sellqty", "ltq", "atp"
    ]

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df['ltt'] = pd.to_datetime(df['ltt'], errors="coerce")
    df = df.sort_values('ltt').set_index('ltt')
    df = df[~df.index.isna()]

    df_60min = (
        df
        .groupby(df.index.date, group_keys=False)
        .apply(fill_intraday_60min)
    )

    return df_60min


def fill_intraday_60min(day_df):
    day = day_df.index[0].date()

    full_index = pd.date_range(
        start=pd.Timestamp(f"{day} 09:15"),
        end=pd.Timestamp(f"{day} 15:15"),
        freq="60min"
    )

    agg_dict = build_agg_dict(day_df)

    day_60min = (
    day_df
    .resample(
        "60min",
        origin=pd.Timestamp(f"{day} 09:15"),
        label="right",
        closed="right"
    )
    .agg(agg_dict)
    )

    day_60min.columns = [
        f"{c[0]}_{c[1]}" if isinstance(c, tuple) else c
        for c in day_60min.columns
    ]

    return day_60min.reindex(full_index).ffill()



# In[15]:


df_60min = get_data_60min_freq("2026-01-01", "2026-01-12")
print(df_60min.index.to_series().diff().value_counts())
print(df_60min.groupby(df_60min.index.date).size().value_counts())


# In[16]:


df_60min.index.name = 'time'
df_60min.tail()

if __name__ == "__main__":
    main()

