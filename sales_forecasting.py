#!/usr/bin/env python
# coding: utf-8

# In[7]:


# #Installing Packages
# !pip install prophet


# In[8]:


#Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import plot_plotly
import numpy as n
from plotly.offline import iplot

# import warnings
# warnings.filterwarnings('ignore') 


# In[9]:


#Reading and cleaning the Dataset
try:
    dataset = pd.read_csv('data/superstore_final_dataset.csv', encoding='latin1')
except UnicodeDecodeError:
    print("Dataset Reading Failed")

try:
    dataset['Sales'] = pd.to_numeric(dataset['Sales'], errors='coerce')
    dataset = dataset.dropna(subset=['Sales'])
    print(f"Cleaned DataFrame has {len(dataset)} rows after dropping rows with non-numeric sales.")
except Exception as e:
    print(f"Warning: Could not convert Sales to numeric. Error: {e}")

#Feature Engineering: Extract Year and Month for time-series analysis
dataset['Order_Date'] = pd.to_datetime(
    dataset['Order_Date'], 
    format='%d/%m/%Y',
    errors='coerce' # Keep errors='coerce' to turn bad dates into NaT
)

dataset['Order_Year'] = dataset['Order_Date'].dt.year
dataset['Order_Month_Year'] = dataset['Order_Date'].dt.to_period('M')


# In[10]:


ts_df = dataset[['Order_Date', 'Sales']].copy()
ts_df = ts_df.sort_values('Order_Date')
    
# Aggregate data to Month End frequency
monthly_sales = ts_df.set_index('Order_Date').resample('MS')['Sales'].sum().reset_index()
    
# Prophet requires columns to be named 'ds' (datestamp) and 'y' (value)
monthly_sales.columns = ['ds', 'y']
    
print(f"Aggregated Data ready for Prophet. Total months: {len(monthly_sales)}")
print(f"Time Range: {monthly_sales['ds'].min().strftime('%Y-%m')} to {monthly_sales['ds'].max().strftime('%Y-%m')}")
print("-" * 50)


# In[13]:


model = Prophet(yearly_seasonality=True,daily_seasonality=False,weekly_seasonality=False,seasonality_mode='multiplicative')

# Fit the model to the historical monthly sales data
model.fit(monthly_sales)
print("Prophet model training complete.")


# In[16]:


# Create a DataFrame with future dates (next 12 months)
future = model.make_future_dataframe(periods=12, freq='MS')
    
# Generate the forecast
forecast = model.predict(future)
    
print(f"Forecast generated for {len(forecast) - len(monthly_sales)} future periods (12 months).")

# Display key forecast columns (ds, yhat - prediction, yhat_lower/upper - confidence interval)
print("\nNext 12 Months Sales Forecast:")
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12).to_string(index=False))


# In[17]:


fig_forecast = plot_plotly(model, forecast)
fig_forecast.update_layout(
        title='Superstore Sales Forecast (Prophet)',
        xaxis_title='Date',
        yaxis_title='Sales ($)',
        margin=dict(l=20, r=20, t=50, b=20)
    )
# Use iplot for Jupyter/Colab display. Use fig.show() in standard Python environments
iplot(fig_forecast)


# In[18]:


# Plot the components to understand trends and seasonality
print("\nVisualizing Forecast Components ---")
fig_components = model.plot_components(forecast)
plt.show() # Display Matplotlib components plot


# In[ ]:




