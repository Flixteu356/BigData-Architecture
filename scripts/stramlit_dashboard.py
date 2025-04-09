import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psycopg2 
from psycopg2.extras import RealDictCursor 
from datetime import datetime
import time
import us  # For state abbreviations 

# Application title and description
st.set_page_config(page_title="COVID-19 Risk Predictions", layout="wide")
st.title("COVID-19 Risk Predictions Dashboard")
st.markdown("Analysis of COVID-19 risk prediction data")

# Define risk category mapping
RISK_CATEGORIES = {
    0: "Very Low",
    1: "Low",
    2: "Medium",
    3: "High",
    4: "Very High"
}

# Color scheme for risk categories
RISK_COLORS = {
    "Very Low": "#00CC96",
    "Low": "#90EE90",
    "Medium": "#FFD700",
    "High": "#FF7F50",
    "Very High": "#FF4500"
}

# Dictionary to convert state names to their two-letter codes
STATE_CODES = {}
for state in us.states.STATES:
    STATE_CODES[state.name] = state.abbr

# Initialize session state for data storage
if 'predictions' not in st.session_state:
    st.session_state.predictions = pd.DataFrame()
    st.session_state.data_loaded = False

# PostgreSQL database connection parameters
DB_PARAMS = {
    "dbname": "pandemic_db",
    "user": "spark_user",
    "password": "1234",
    "host": "hadoop-master",
    "port": "5432"
}

# Function to load data from PostgreSQL
@st.cache_data
def load_postgresql_data():
    try:
        st.info("Loading data from PostgreSQL database...")

        # Establish connection to PostgreSQL
        conn = psycopg2.connect(**DB_PARAMS)
        # Ensure the date column is cast to a date type before applying TO_CHAR
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("ALTER TABLE risk_predictions ALTER COLUMN date TYPE DATE USING date::DATE")
        conn.commit()
        # Query to fetch all data from risk_predictions table
        query = "SELECT state, county, TO_CHAR(date, 'YYYY-MM-DD') as date, risk_category, predicted_risk_category FROM risk_predictions"

        # Load data into pandas DataFrame
        df = pd.read_sql_query(query, conn)

        # Close connection
        conn.close()

        if not df.empty:
            # Add timestamp for when data was loaded
            df['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Convert numeric risk categories to descriptive text
            if 'risk_category' in df.columns:
                df['risk_category_text'] = df['risk_category'].map(RISK_CATEGORIES)

            if 'predicted_risk_category' in df.columns:
                df['predicted_risk_category_text'] = df['predicted_risk_category'].map(RISK_CATEGORIES)

            # Convert date to datetime - using a specific format to avoid parsing issues
            if 'date' in df.columns:
                # Parse date string to datetime object
                df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

            # Add state code column for mapping
            if 'state' in df.columns:
                df['state_code'] = df['state'].map(STATE_CODES)

            st.success(f"Successfully loaded {len(df)} records from PostgreSQL database")
            return df
        else:
            st.warning("No data found in the PostgreSQL database")
            return pd.DataFrame()

    except Exception as e:
        st.error(f"Error loading PostgreSQL data: {str(e)}")
        return pd.DataFrame()

# Load button
if not st.session_state.data_loaded:
    if st.button("Load Data from PostgreSQL"):
        df = load_postgresql_data()
        if not df.empty:
            st.session_state.predictions = df
            st.session_state.data_loaded = True
            st.rerun()
else:
    # Create dashboard tabs
    tab1, tab2, tab3 = st.tabs(["Risk Analysis", "Geographic Distribution", "Detailed Data"])

    with tab1:
        st.header("COVID-19 Risk Analysis")

        if not st.session_state.predictions.empty:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Risk Category Distribution")
                if 'risk_category_text' in st.session_state.predictions.columns:
                    field = 'risk_category_text'
                else:
                    field = 'risk_category'

                risk_counts = st.session_state.predictions[field].value_counts().reset_index()
                risk_counts.columns = ['Risk Category', 'Count']

                fig = px.pie(
                    risk_counts,
                    values='Count',
                    names='Risk Category',
                    color='Risk Category',
                    color_discrete_map=RISK_COLORS
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Actual vs Predicted Risk Category")

                if 'risk_category' in st.session_state.predictions.columns and 'predicted_risk_category' in st.session_state.predictions.columns:
                    comparison_df = st.session_state.predictions.groupby(['risk_category', 'predicted_risk_category']).size().reset_index()
                    comparison_df.columns = ['Actual Risk', 'Predicted Risk', 'Count']

                    # Create a heatmap for comparison
                    fig = px.density_heatmap(
                        comparison_df,
                        x='Actual Risk',
                        y='Predicted Risk',
                        z='Count',
                        color_continuous_scale='YlOrRd',
                        title='Actual vs Predicted Risk Categories'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Risk category comparison data not available")

            # Risk distribution by date
            if 'date' in st.session_state.predictions.columns:
                st.subheader("Risk Categories Over Time")

                # Group by date and risk category
                if 'risk_category_text' in st.session_state.predictions.columns:
                    time_data = st.session_state.predictions.groupby([pd.Grouper(key='date', freq='D'), 'risk_category_text']).size().reset_index()
                    time_data.columns = ['Date', 'Risk Category', 'Count']
                else:
                    time_data = st.session_state.predictions.groupby([pd.Grouper(key='date', freq='D'), 'risk_category']).size().reset_index()
                    time_data.columns = ['Date', 'Risk Category', 'Count']

                fig = px.line(
                    time_data,
                    x='Date',
                    y='Count',
                    color='Risk Category',
                    title='Risk Categories Distribution Over Time',
                    color_discrete_map=RISK_COLORS
                )
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("No prediction data available for analysis.")

    with tab2:
        st.header("Geographic Risk Distribution")

        if not st.session_state.predictions.empty:
            # Create a choropleth map for risk by state
            st.subheader("Average Risk Level by State")

            # Calculate average risk by state
            if 'risk_category' in st.session_state.predictions.columns:
                # Check if state_code column exists
                if 'state_code' in st.session_state.predictions.columns:
                    # Prepare data for choropleth map
                    state_risk = st.session_state.predictions.groupby(['state', 'state_code'])['risk_category'].mean().reset_index()

                    # Create the choropleth map
                    fig = px.choropleth(
                        state_risk,
                        locations='state_code',  
                        locationmode='USA-states',
                        color='risk_category',
                        scope='usa',
                        color_continuous_scale='RdYlGn_r',  # Red for high risk, green for low
                        range_color=[0, 4],  # Range of risk categories
                        labels={'risk_category': 'Risk Level'},
                        hover_data=['state']
                    )

                    fig.update_layout(
                        title_text='Average Risk Level by State',
                        geo=dict(
                            showlakes=True,
                            lakecolor='rgb(255, 255, 255)'
                        )
                    )

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("State code data is missing. Can't create choropleth map.")

                    # Fall back to a bar chart instead
                    st.subheader("Average Risk by State (Bar Chart)")
                    state_risk = st.session_state.predictions.groupby('state')['risk_category'].mean().reset_index()
                    state_risk = state_risk.sort_values('risk_category', ascending=False)

                    fig = px.bar(
                        state_risk,
                        x='state',
                        y='risk_category',
                        title='Average Risk Level by State',
                        color='risk_category',
                        color_continuous_scale='RdYlGn_r'
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Top states by high risk counties
            st.subheader("States with Most High-Risk Counties")

            # Filter for high risk counties (categories 3 & 4)
            high_risk_df = st.session_state.predictions[st.session_state.predictions['risk_category'] >= 3]

            if not high_risk_df.empty:
                state_counts = high_risk_df['state'].value_counts().reset_index()
                state_counts.columns = ['State', 'High Risk Counties']
                state_counts = state_counts.sort_values('High Risk Counties', ascending=False).head(10)

                fig = px.bar(
                    state_counts,
                    x='State',
                    y='High Risk Counties',
                    title='Top 10 States by Number of High Risk Counties',
                    color='High Risk Counties',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig, use_container_width=True)

                # Table of high risk counties
                st.subheader("High Risk Counties")

                high_risk_table = high_risk_df[['state', 'county', 'date', 'risk_category']]
                if 'risk_category_text' in high_risk_df.columns:
                    high_risk_table['Risk Category'] = high_risk_df['risk_category_text']
                else:
                    high_risk_table['Risk Category'] = high_risk_df['risk_category'].map(RISK_CATEGORIES)

                high_risk_table = high_risk_table[['state', 'county', 'date', 'Risk Category']]
                high_risk_table.columns = ['State', 'County', 'Date', 'Risk Category']
                st.dataframe(high_risk_table.sort_values(['State', 'County']), use_container_width=True)
            else:
                st.info("No high risk counties in the current dataset.")
        else:
            st.info("No geographic data available for analysis.")

    with tab3:
        st.header("Detailed Prediction Data")

        if not st.session_state.predictions.empty:
            # Add filters for the data
            col1, col2, col3 = st.columns(3)

            with col1:
                # Filter by state
                states = ['All'] + sorted(st.session_state.predictions['state'].unique().tolist())
                selected_state = st.selectbox("Filter by State", states)

            with col2:
                # Filter by risk category
                if 'risk_category_text' in st.session_state.predictions.columns:
                    risk_categories = ['All'] + sorted(st.session_state.predictions['risk_category_text'].unique().tolist())
                    selected_risk = st.selectbox("Filter by Risk Category", risk_categories)
                else:
                    risk_categories = ['All'] + [RISK_CATEGORIES[x] for x in sorted(st.session_state.predictions['risk_category'].unique().tolist())]
                    selected_risk = st.selectbox("Filter by Risk Category", risk_categories)

            with col3:
                # Date range filter
                if 'date' in st.session_state.predictions.columns:
                    min_date = st.session_state.predictions['date'].min()
                    max_date = st.session_state.predictions['date'].max()
                    date_range = st.date_input(
                        "Select Date Range",
                        value=(min_date, max_date),
                        min_value=min_date,
                        max_value=max_date
                    )

            # Apply filters
            filtered_df = st.session_state.predictions.copy()

            if selected_state != 'All':
                filtered_df = filtered_df[filtered_df['state'] == selected_state]

            if selected_risk != 'All':
                if 'risk_category_text' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['risk_category_text'] == selected_risk]
                else:
                    risk_value = [k for k, v in RISK_CATEGORIES.items() if v == selected_risk][0]
                    filtered_df = filtered_df[filtered_df['risk_category'] == risk_value]

            if 'date' in filtered_df.columns and len(date_range) == 2:
                filtered_df = filtered_df[(filtered_df['date'] >= pd.Timestamp(date_range[0])) &
                                        (filtered_df['date'] <= pd.Timestamp(date_range[1]))]

            # Display the filtered data
            st.subheader(f"Filtered Data ({len(filtered_df)} records)")

            # Enhance the display data
            display_df = filtered_df.copy()

            # Add text descriptions for risk categories if not already present
            if 'risk_category_text' not in display_df.columns and 'risk_category' in display_df.columns:
                display_df['Risk Category'] = display_df['risk_category'].map(RISK_CATEGORIES)
            else:
                display_df['Risk Category'] = display_df['risk_category_text']

            if 'predicted_risk_category_text' not in display_df.columns and 'predicted_risk_category' in display_df.columns:
                display_df['Predicted Risk'] = display_df['predicted_risk_category'].map(RISK_CATEGORIES)
            else:
                display_df['Predicted Risk'] = display_df['predicted_risk_category_text'] if 'predicted_risk_category_text' in display_df.columns else None

            # Select and rename columns for display
            cols_to_display = ['state', 'county', 'date']
            col_names = ['State', 'County', 'Date']

            # Add available metrics
            if 'Risk Category' in display_df.columns:
                cols_to_display.append('Risk Category')
                col_names.append('Risk Category')

            if 'Predicted Risk' in display_df.columns:
                cols_to_display.append('Predicted Risk')
                col_names.append('Predicted Risk')

            # Create final display dataframe
            final_display = display_df[cols_to_display].copy()
            final_display.columns = col_names

            st.dataframe(final_display, use_container_width=True)

            # Download CSV option
            csv = final_display.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Filtered Data as CSV",
                csv,
                "covid_risk_predictions.csv",
                "text/csv",
                key='download-csv'
            )
        else:
            st.info("No data available for display.")

    # Reset button
    if st.button("Reset Data"):
        st.session_state.data_loaded = False
        st.session_state.predictions = pd.DataFrame()
        st.rerun()

    # Manual refresh button
    if st.button("Refresh Data"):
        df = load_postgresql_data()
        if not df.empty:
            st.session_state.predictions = df
            st.rerun()