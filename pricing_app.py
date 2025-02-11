import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Set page config
st.set_page_config(page_title="Pricing Analysis Dashboard", layout="wide")

# Add title and description
st.title("Pricing Analysis - Anthony")
st.markdown("Upload your pricing data to analyze trends and comparisons")

# File upload
uploaded_file = st.file_uploader("Upload File", type=["csv"])

# Sample data generation if no file is uploaded
if uploaded_file is None:
    st.info("No file uploaded. Using sample data...")
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    
    data = pd.DataFrame({
        'Date': dates,
        'Product': np.random.choice(['Product A', 'Product B', 'Product C'], size=len(dates)),
        'Price': np.random.normal(100, 15, size=len(dates)),
        'Units_Sold': np.random.randint(50, 200, size=len(dates)),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], size=len(dates))
    })
else:
    data = pd.read_csv(uploaded_file)

# Sidebar for filters
st.sidebar.header("Filters")

# Date range selector if 'Date' column exists
if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'])
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(data['Date'].min(), data['Date'].max()),
        min_value=data['Date'].min().date(),
        max_value=data['Date'].max().date()
    )
    data = data[(data['Date'].dt.date >= date_range[0]) & 
                (data['Date'].dt.date <= date_range[1])]

# Product filter if 'Product' column exists
if 'Product' in data.columns:
    selected_products = st.sidebar.multiselect(
        "Select Products",
        options=data['Product'].unique(),
        default=data['Product'].unique()
    )
    data = data[data['Product'].isin(selected_products)]

# Region filter if 'Region' column exists
if 'Region' in data.columns:
    selected_regions = st.sidebar.multiselect(
        "Select Regions",
        options=data['Region'].unique(),
        default=data['Region'].unique()
    )
    data = data[data['Region'].isin(selected_regions)]

# Create main dashboard
col1, col2 = st.columns(2)

with col1:
    st.subheader("Price Trends Over Time")
    if 'Date' in data.columns and 'Price' in data.columns:
        fig_trend = px.line(
            data.groupby('Date')['Price'].mean().reset_index(),
            x='Date',
            y='Price',
            title='Average Price Trend'
        )
        st.plotly_chart(fig_trend, use_container_width=True)

with col2:
    st.subheader("Price Distribution")
    if 'Price' in data.columns:
        fig_dist = px.histogram(
            data,
            x='Price',
            title='Price Distribution',
            nbins=30
        )
        st.plotly_chart(fig_dist, use_container_width=True)

# Product comparison if relevant columns exist
if 'Product' in data.columns and 'Price' in data.columns:
    st.subheader("Product Price Comparison")
    fig_box = px.box(
        data,
        x='Product',
        y='Price',
        title='Price Distribution by Product'
    )
    st.plotly_chart(fig_box, use_container_width=True)

# Sales analysis if units sold data exists
if 'Units_Sold' in data.columns and 'Price' in data.columns:
    st.subheader("Price vs. Sales Analysis")
    fig_scatter = px.scatter(
        data,
        x='Price',
        y='Units_Sold',
        color='Product' if 'Product' in data.columns else None,
        title='Price vs. Units Sold'
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# Summary statistics
st.subheader("Summary Statistics")
col3, col4, col5 = st.columns(3)

with col3:
    st.metric("Average Price", f"${data['Price'].mean():.2f}")
    
with col4:
    st.metric("Minimum Price", f"${data['Price'].min():.2f}")
    
with col5:
    st.metric("Maximum Price", f"${data['Price'].max():.2f}")

# Data table view
st.subheader("Raw Data")
st.dataframe(data)