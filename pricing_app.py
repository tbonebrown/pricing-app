import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

# Set page config with improved styling
st.set_page_config(
    page_title="Advanced Pricing Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main > div {
        padding: 2rem;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Add title and description with more context
st.title("🎯 Dane's Wonderful World of Pricing")
st.markdown("""
    This dashboard provides comprehensive pricing analytics including trend analysis, 
    distribution patterns, sales correlations, and price elasticity calculations. 
    Upload your CSV file with pricing data or use our sample dataset to explore the features.
""")

def generate_sample_data():
    """Generate realistic sample data with seasonal patterns"""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    products = ['Premium Widget', 'Basic Widget', 'Deluxe Widget']
    regions = ['North America', 'Europe', 'Asia Pacific', 'Latin America']
    
    base_prices = {'Premium Widget': 150, 'Basic Widget': 75, 'Deluxe Widget': 200}
    
    data = []
    for date in dates:
        for product in products:
            base_price = base_prices[product]
            seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * date.dayofyear / 365)
            price = base_price * seasonal_factor * (1 + np.random.normal(0, 0.05))
            
            for region in regions:
                units = int(np.random.normal(100, 20) * (1.2 if date.weekday() < 5 else 0.8))
                data.append({
                    'Date': date,
                    'Product': product,
                    'Price': price,
                    'Units_Sold': max(0, units),
                    'Region': region,
                    'Revenue': price * units
                })
    
    return pd.DataFrame(data)

def calculate_elasticity(group):
    """Calculate price elasticity of demand for a group of data"""
    price_pct_change = group['Price'].pct_change()
    demand_pct_change = group['Units_Sold'].pct_change()
    
    mask = (price_pct_change != 0) & (price_pct_change.notna()) & (demand_pct_change.notna())
    elasticities = demand_pct_change[mask] / price_pct_change[mask]
    
    elasticities = elasticities[elasticities.abs() < 10]  # Remove extreme outliers
    return elasticities.mean()

# Enhanced file upload with error handling
uploaded_file = st.file_uploader("Upload Your Pricing Data (CSV)", type=["csv"])

# Load data with improved error handling
try:
    if uploaded_file is None:
        st.info("👉 No file uploaded. Using sample data with realistic pricing patterns...")
        data = generate_sample_data()
    else:
        data = pd.read_csv(uploaded_file)
        required_columns = ['Date', 'Product', 'Price', 'Units_Sold', 'Region']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            st.warning(f"⚠️ Missing columns: {', '.join(missing_columns)}. Some features may be limited.")
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Enhanced sidebar with more analytical options
st.sidebar.header("📊 Analysis Controls")

# Advanced date filtering with proper datetime handling
if 'Date' in data.columns:
    # Ensure Date column is datetime
    data['Date'] = pd.to_datetime(data['Date'])
    min_date = data['Date'].min()
    max_date = data['Date'].max()
    
    date_range = st.sidebar.date_input(
        "📅 Date Range",
        value=(min_date.date(), max_date.date()),
        min_value=min_date.date(),
        max_value=max_date.date()
    )
    
    # Add quick date range selectors
    quick_ranges = st.sidebar.selectbox(
        "Quick Date Ranges",
        ["Custom", "Last 7 Days", "Last 30 Days", "Last Quarter", "Year to Date"]
    )
    
    if quick_ranges != "Custom":
        end_date = pd.to_datetime(data['Date'].max())
        if quick_ranges == "Last 7 Days":
            start_date = end_date - pd.Timedelta(days=7)
        elif quick_ranges == "Last 30 Days":
            start_date = end_date - pd.Timedelta(days=30)
        elif quick_ranges == "Last Quarter":
            start_date = end_date - pd.Timedelta(days=90)
        else:  # Year to Date
            start_date = pd.to_datetime(f"{end_date.year}-01-01")
        date_range = (start_date.date(), end_date.date())

    # Convert date_range to pandas datetime for filtering
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])
    data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]

# Enhanced filters with select all/none options
if 'Product' in data.columns:
    st.sidebar.subheader("🏷️ Product Selection")
    select_all_products = st.sidebar.checkbox("Select All Products", True)
    if select_all_products:
        selected_products = data['Product'].unique()
    else:
        selected_products = st.sidebar.multiselect(
            "Select Products",
            options=data['Product'].unique(),
            default=data['Product'].unique()
        )
    data = data[data['Product'].isin(selected_products)]

if 'Region' in data.columns:
    st.sidebar.subheader("🌍 Region Selection")
    select_all_regions = st.sidebar.checkbox("Select All Regions", True)
    if select_all_regions:
        selected_regions = data['Region'].unique()
    else:
        selected_regions = st.sidebar.multiselect(
            "Select Regions",
            options=data['Region'].unique(),
            default=data['Region'].unique()
        )
    data = data[data['Region'].isin(selected_regions)]

# Advanced Analysis Options
st.sidebar.subheader("📈 Analysis Options")
show_trend_line = st.sidebar.checkbox("Show Trend Line", True)
show_annotations = st.sidebar.checkbox("Show Data Points Annotations", False)
aggregation = st.sidebar.selectbox(
    "Time Aggregation",
    ["Daily", "Weekly", "Monthly"]
)

# Main dashboard sections
tab1, tab2, tab3 = st.tabs(["📊 Price Analysis", "💰 Sales Analysis", "📐 Price Elasticity"])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Price Trends Over Time")
        if 'Date' in data.columns and 'Price' in data.columns:
            if aggregation == "Weekly":
                trend_data = data.groupby([pd.Grouper(key='Date', freq='W')])['Price'].mean().reset_index()
            elif aggregation == "Monthly":
                trend_data = data.groupby([pd.Grouper(key='Date', freq='M')])['Price'].mean().reset_index()
            else:
                trend_data = data.groupby('Date')['Price'].mean().reset_index()
                
            fig_trend = px.line(
                trend_data,
                x='Date',
                y='Price',
                title='Average Price Trend'
            )
            
            if show_trend_line:
                fig_trend.add_trace(
                    go.Scatter(
                        x=trend_data['Date'],
                        y=trend_data['Price'].rolling(window=7).mean(),
                        name='7-day Moving Average',
                        line=dict(dash='dash')
                    )
                )
                
            fig_trend.update_layout(
                xaxis_title="Date",
                yaxis_title="Average Price ($)",
                hovermode='x unified'
            )
            st.plotly_chart(fig_trend, use_container_width=True)

    with col2:
        st.subheader("📊 Price Distribution Analysis")
        if 'Price' in data.columns:
            fig_dist = px.histogram(
                data,
                x='Price',
                color='Product' if 'Product' in data.columns else None,
                title='Price Distribution by Product',
                nbins=30,
                opacity=0.7
            )
            fig_dist.update_layout(
                xaxis_title="Price ($)",
                yaxis_title="Count",
                bargap=0.1
            )
            st.plotly_chart(fig_dist, use_container_width=True)

    if 'Product' in data.columns and 'Price' in data.columns:
        st.subheader("📦 Product Performance Analysis")
        
        col3, col4 = st.columns(2)
        
        with col3:
            fig_box = px.box(
                data,
                x='Product',
                y='Price',
                title='Price Distribution by Product',
                points="outliers"
            )
            fig_box.update_layout(
                xaxis_title="Product",
                yaxis_title="Price ($)"
            )
            st.plotly_chart(fig_box, use_container_width=True)
        
        with col4:
            if 'Revenue' in data.columns:
                product_revenue = data.groupby('Product')['Revenue'].sum().reset_index()
                fig_revenue = px.pie(
                    product_revenue,
                    values='Revenue',
                    names='Product',
                    title='Revenue Distribution by Product'
                )
                st.plotly_chart(fig_revenue, use_container_width=True)

with tab2:
    if 'Units_Sold' in data.columns and 'Price' in data.columns:
        st.subheader("💰 Price-Sales Relationship Analysis")
        
        col5, col6 = st.columns(2)
        
        with col5:
            fig_scatter = px.scatter(
                data,
                x='Price',
                y='Units_Sold',
                color='Product' if 'Product' in data.columns else None,
                size='Revenue' if 'Revenue' in data.columns else None,
                title='Price vs. Units Sold',
                trendline="ols" if show_trend_line else None
            )
            
            if show_annotations:
                fig_scatter.update_traces(
                    hovertemplate="<br>".join([
                        "Price: $%{x:.2f}",
                        "Units Sold: %{y}",
                        "Product: %{customdata[0]}",
                        "Region: %{customdata[1]}"
                    ]),
                    customdata=data[['Product', 'Region']]
                )
            
            fig_scatter.update_layout(
                xaxis_title="Price ($)",
                yaxis_title="Units Sold"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col6:
            if 'Date' in data.columns:
                sales_trend = data.groupby('Date')['Units_Sold'].sum().reset_index()
                fig_sales = px.line(
                    sales_trend,
                    x='Date',
                    y='Units_Sold',
                    title='Daily Sales Trend'
                )
                st.plotly_chart(fig_sales, use_container_width=True)

with tab3:
    if all(col in data.columns for col in ['Price', 'Units_Sold']):
        col_elasticity1, col_elasticity2 = st.columns(2)
        
        with col_elasticity1:
            st.markdown("### Historical Price Elasticity")
            
            if 'Product' in data.columns:
                elasticities = data.groupby('Product').apply(calculate_elasticity)
                
                fig_elasticity = go.Figure()
                
                for product in elasticities.index:
                    elasticity = elasticities[product]
                    if pd.notna(elasticity):
                        fig_elasticity.add_trace(go.Bar(
                            x=[product],
                            y=[elasticity],
                            name=product,
                            text=[f'{elasticity:.2f}'],
                            textposition='auto',
                        ))
                
                fig_elasticity.update_layout(
                    title='Price Elasticity by Product',
                    yaxis_title='Price Elasticity of Demand',
                    showlegend=False
                )
                
                fig_elasticity.add_hline(y=-1, line_dash="dash", line_color="red",
                                       annotation_text="Unit Elasticity")
                
                st.plotly_chart(fig_elasticity, use_container_width=True)
                
                for product in elasticities.index:
                    elasticity = elasticities[product]
                    if pd.notna(elasticity):
                        if abs(elasticity) > 1:
                            elastic_status = "elastic (price sensitive)"
                        elif abs(elasticity) < 1:
                            elastic_status = "inelastic (price insensitive)"
                        else:
                            elastic_status = "unit elastic"
                        
                        st.markdown(f"- **{product}**: Elasticity = {elasticity:.2f} ({elastic_status})")

        with col_elasticity2:
            st.markdown("### Revenue Impact Calculator")
            
            selected_product = st.selectbox(
                "Select Product",
                options=data['Product'].unique() if 'Product' in data.columns else ['All Products']
            )
            
            price_change_pct = st.slider(
                "Price Change (%)",
                min_value=-50,
                max_value=50,
                value=0,
                step=1
            )
            
            if selected_product == 'All Products':
                current_price = data['Price'].mean()
                current_demand = data['Units_Sold'].mean()
                elasticity = calculate_elasticity(data)
            else:
                product_data = data[data['Product'] == selected_product]
                current_price = product_data['Price'].mean()
                current_demand = product_data['Units_Sold'].mean()
                elasticity = calculate_elasticity(product_data)
            
            if pd.notna(elasticity):
                price_change_decimal = price_change_pct / 100
                demand_change_decimal = elasticity * price_change_decimal
                
                new_price = current_price * (1 + price_change_decimal)
                new_demand = current_demand * (1 + demand_change_decimal)
                
                current_revenue = current_price * current_demand
                new_revenue = new_price * new_demand
                revenue_change_pct = ((new_revenue - current_revenue) / current_revenue) * 100
                
                st.markdown("#### Impact Analysis")
                
                metrics_col1, metrics_col2 = st.columns(2)
                
                with metrics_col1:
                    st.metric(
                        "New Average Price",
                        f"${new_price:.2f}",
                        f"{price_change_pct:+.1f}%"
                    )
                    st.metric(
                        "New Daily Demand",
                        f"{new_demand:.0f}",
                        f"{demand_change_decimal*100:+.1f}%"
                    )
                
                with metrics_col2:
                    st.metric(
                        "New Daily Revenue",
                        f"${new_revenue:.2f}",
                        f"{revenue_change_pct:+.1f}%"
                    )
                
                fig_revenue_impact = go.Figure()
                
                fig_revenue_impact.add_trace(go.Bar(
                    x=['Current Revenue', 'Projected Revenue'],
                    y=[current_revenue, new_revenue],
                    text=[f'${current_revenue:,.2f}', f'${new_revenue:,.2f}'],
                    textposition='auto',
                ))
                
                fig_revenue_impact.update_layout(
                    title='Revenue Impact Comparison',
                    yaxis_title='Daily Revenue ($)',
                    showlegend=False
                )
                
                st.plotly_chart(fig_revenue_impact, use_container_width=True)
                
                st.markdown("#### Recommendation")
                if revenue_change_pct > 0:
                    st.success(f"✅ The proposed price change is projected to increase revenue by {revenue_change_pct:.1f}%")
                elif revenue_change_pct < 0:
                    st.error(f"⚠️ The proposed price change is projected to decrease revenue by {abs(revenue_change_pct):.1f}%")
                else:
                    st.info("The proposed price change is projected to have no significant impact on revenue")
            else:
                st.warning("Unable to calculate elasticity with current data. Please ensure you have sufficient price and demand variation in your dataset.")

# Add Key Performance Metrics section
st.subheader("📊 Key Performance Metrics")
col7, col8, col9, col10 = st.columns(4)

with col7:
    avg_price = data['Price'].mean()
    if 'Date' in data.columns:
        comparison_date = pd.to_datetime(date_range[0])
        previous_data = data[data['Date'].dt.normalize() < comparison_date]
        prev_avg_price = previous_data['Price'].mean() if not previous_data.empty else avg_price
    else:
        prev_avg_price = avg_price
    price_delta = ((avg_price - prev_avg_price) / prev_avg_price * 100) if prev_avg_price != 0 else 0
    st.metric(
        "Average Price",
        f"${avg_price:.2f}",
        f"{price_delta:+.1f}%" if price_delta != 0 else None
    )

with col8:
    total_revenue = (data['Price'] * data['Units_Sold']).sum() if 'Units_Sold' in data.columns else None
    st.metric(
        "Total Revenue",
        f"${total_revenue:,.2f}" if total_revenue is not None else "N/A"
    )

with col9:
    total_units = data['Units_Sold'].sum() if 'Units_Sold' in data.columns else None
    st.metric(
        "Total Units Sold",
        f"{total_units:,.0f}" if total_units is not None else "N/A"
    )

with col10:
    price_variance = data['Price'].std()
    st.metric(
        "Price Variance",
        f"${price_variance:.2f}"
    )

# Detailed statistics table
st.subheader("📑 Detailed Statistics")
if 'Product' in data.columns:
    detailed_stats = data.groupby('Product').agg({
        'Price': ['mean', 'min', 'max', 'std'],
        'Units_Sold': ['sum', 'mean'] if 'Units_Sold' in data.columns else None,
        'Revenue': ['sum'] if 'Revenue' in data.columns else None
    }).round(2)
    
    detailed_stats.columns = [f"{col[0]}_{col[1]}" for col in detailed_stats.columns]
    st.dataframe(detailed_stats)

# Raw data view with improved filtering
st.subheader("🔍 Raw Data Explorer")
if st.checkbox("Show Raw Data"):
    search_term = st.text_input("Search in data", "")
    if search_term:
        filtered_data = data[data.astype(str).apply(lambda x: x.str.contains(search_term, case=False)).any(axis=1)]
    else:
        filtered_data = data
    
    st.dataframe(filtered_data)