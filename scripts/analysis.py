import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

def top_10_handsets(data, column='Handset Type'):
    # Check if the column exists
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in the DataFrame.")
    
    # Count occurrences of each handset type
    top_handsets = data[column].value_counts().head(10)
    
    # Print the results
    print("Top 10 Handsets Used by Customers:")
    print(top_handsets)
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    top_handsets.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title("Top 10 Handsets Used by Customers", fontsize=14)
    plt.xlabel("Handset Type", fontsize=12)
    plt.ylabel("Number of Users", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
    
    # Return the top handsets as a DataFrame for further analysis
    return top_handsets.reset_index().rename(columns={'index': column, column: 'Count'})

def top_3_handset_manufacturers(data, column='Handset Manufacturer'):
    # Check if the column exists
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in the DataFrame.")
    
    # Count occurrences of each handset manufacturer
    top_manufacturers = data[column].value_counts().head(3)
    
    # Print the results
    print("Top 3 Handset Manufacturers:")
    print(top_manufacturers)
    
    # Plot the results
    plt.figure(figsize=(8, 5))
    top_manufacturers.plot(kind='bar', color='orange', edgecolor='black')
    plt.title("Top 3 Handset Manufacturers", fontsize=14)
    plt.xlabel("Manufacturer", fontsize=12)
    plt.ylabel("Number of Users", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
    
    # Return the top manufacturers as a DataFrame for further analysis
    return top_manufacturers.reset_index().rename(columns={'index': column, column: 'Count'})


def top_5_handsets_per_top_3_manufacturers(data, manufacturer_column='Handset Manufacturer', handset_column='Handset Type'):
    # Check if the necessary columns exist
    if manufacturer_column not in data.columns or handset_column not in data.columns:
        raise ValueError(f"One or both columns ('{manufacturer_column}', '{handset_column}') not found in the DataFrame.")
    
    # Identify the top 3 manufacturers
    top_manufacturers = data[manufacturer_column].value_counts().head(3).index
    
    # Filter the dataset for only the top 3 manufacturers
    filtered_data = data[data[manufacturer_column].isin(top_manufacturers)]
    
    # Group by manufacturer and handset type, and count occurrences
    grouped = (
        filtered_data.groupby([manufacturer_column, handset_column])
        .size()
        .reset_index(name='Count')
    )
    
    # Identify the top 5 handsets per manufacturer
    top_5_handsets = (
        grouped.groupby(manufacturer_column, group_keys=False)
        .apply(lambda x: x.nlargest(5, 'Count'))
    )
    
    # Visualization
    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=top_5_handsets,
        x='Count',
        y=handset_column,
        hue=manufacturer_column,
        dodge=False,
        palette='Set2'
    )
    plt.title("Top 5 Handsets per Top 3 Manufacturers", fontsize=14)
    plt.xlabel("Count", fontsize=12)
    plt.ylabel("Handset Type", fontsize=12)
    plt.legend(title="Manufacturer", loc='upper right')
    plt.tight_layout()
    plt.show()
    
    return top_5_handsets


def aggregate_user_behavior(df):
    # Convert 'Start' and 'End' columns to datetime (if they are not already in datetime format)
    df['Start'] = pd.to_datetime(df['Start'])
    df['End'] = pd.to_datetime(df['End'])

    # Calculate the session duration in seconds (assuming 'Dur. (ms)' is in milliseconds)
    df['Duration (s)'] = df['Dur. (ms)'] / 1000

    # Group by user (e.g., IMSI or MSISDN)
    grouped = df.groupby('IMSI').agg(
        # Number of sessions per user
        num_sessions=('IMSI', 'count'),
        
        # Total session duration per user (in seconds)
        total_duration=('Duration (s)', 'sum'),
        
        # Total download and upload data per user for each application
        total_social_media_dl=('Social Media DL (Bytes)', 'sum'),
        total_social_media_ul=('Social Media UL (Bytes)', 'sum'),
        
        total_google_dl=('Google DL (Bytes)', 'sum'),
        total_google_ul=('Google UL (Bytes)', 'sum'),
        
        total_email_dl=('Email DL (Bytes)', 'sum'),
        total_email_ul=('Email UL (Bytes)', 'sum'),
        
        total_youtube_dl=('Youtube DL (Bytes)', 'sum'),
        total_youtube_ul=('Youtube UL (Bytes)', 'sum'),
        
        total_netflix_dl=('Netflix DL (Bytes)', 'sum'),
        total_netflix_ul=('Netflix UL (Bytes)', 'sum'),
        
        total_gaming_dl=('Gaming DL (Bytes)', 'sum'),
        total_gaming_ul=('Gaming UL (Bytes)', 'sum'),
        
        total_other_dl=('Other DL (Bytes)', 'sum'),
        total_other_ul=('Other UL (Bytes)', 'sum'),
        
        # Total download and upload data overall
        total_dl=('Total DL (Bytes)', 'sum'),
        total_ul=('Total UL (Bytes)', 'sum')
    ).reset_index()

    # Calculate total data volume for each user (total DL + UL)
    grouped['total_data_volume'] = grouped['total_dl'] + grouped['total_ul']

    # Task 1.1: Aggregated user behavior overview for each application
    apps = [
        'social_media', 'google', 'email', 'youtube', 'netflix', 'gaming', 'other'
    ]

    for app in apps:
        grouped[f'{app}_total_dl_ul'] = grouped[f'total_{app}_dl'] + grouped[f'total_{app}_ul']

    # Plot bar chart for total data volume per user for each application
    apps_dl_ul = [
        'social_media_total_dl_ul', 'google_total_dl_ul', 'email_total_dl_ul', 
        'youtube_total_dl_ul', 'netflix_total_dl_ul', 'gaming_total_dl_ul', 'other_total_dl_ul'
    ]

    grouped_apps_dl_ul = grouped[apps_dl_ul].sum()
    grouped_apps_dl_ul.plot(kind='bar', figsize=(10, 6), color='skyblue')
    plt.title("Total Download and Upload Data Volume per Application")
    plt.ylabel("Data Volume (Bytes)")
    plt.xticks(rotation=45)
    plt.show()

    # Pie chart for distribution of total data volume across applications
    grouped_apps_dl_ul.plot(kind='pie', figsize=(8, 8), autopct='%1.1f%%', startangle=90)
    plt.title("Data Volume Distribution per Application")
    plt.ylabel("")  # Hide the y-label
    plt.show()

    return grouped



def segment_users_by_decile(data_tel):
    
    # Calculate total duration per user (sum of 'Dur. (ms)' per user)
    data_tel['Total Duration'] = data_tel.groupby('IMSI')['Dur. (ms)'].transform('sum')
    
    # Remove duplicates to keep only one row per user
    user_data = data_tel[['IMSI', 'Total Duration', 'Total DL (Bytes)', 'Total UL (Bytes)']].drop_duplicates()
    
    # Create decile classes based on the total duration
    user_data['Decile'] = pd.qcut(user_data['Total Duration'], 10, labels=False) + 1  # Decile ranges from 1 to 10
    
    # Get the top 5 deciles (i.e., 6th to 10th deciles)
    top_deciles = user_data[user_data['Decile'] > 5]
    
    # Calculate total data per decile (DL + UL)
    top_deciles.loc[:, 'Total Data'] = top_deciles['Total DL (Bytes)'] + top_deciles['Total UL (Bytes)']

    
    # Group by decile and calculate total data for each decile
    decile_data = top_deciles.groupby('Decile')['Total Data'].sum().reset_index()
    
    return decile_data

def calculate_basic_metrics(data):
    
    # Identify numeric columns
    numeric_columns = data.select_dtypes(include=['number']).columns
    
    # Initialize an empty dictionary to store the metrics
    metrics = {}
    
    # Calculate metrics for each numeric column
    for col in numeric_columns:
        metrics[col] = {
            'mean': data[col].mean(),
            'median': data[col].median(),
            'std': data[col].std(),
            'min': data[col].min(),
            'max': data[col].max(),
            'range': data[col].max() - data[col].min(),
            'skewness': data[col].skew(),
            '25th_percentile': data[col].quantile(0.25),
            '75th_percentile': data[col].quantile(0.75)
        }
    
    # Convert the metrics dictionary to a DataFrame for better presentation
    metrics_df = pd.DataFrame(metrics).T
    return metrics_df

def analyze_opportunities(data_tel):
    opportunities = {}
    
    # Segment the data by deciles
    decile_data = segment_users_by_decile(data_tel)
    
    # Example opportunity: High data usage segments (top 3 deciles)
    high_usage_segment = decile_data.iloc[-3:]  # Get the top 3 deciles (high data usage)
    opportunities['High Usage Segment'] = high_usage_segment
    
    # Example opportunity: Low data usage segment (bottom 3 deciles)
    low_usage_segment = decile_data.iloc[:3]  # Get the bottom 3 deciles (low data usage)
    opportunities['Low Usage Segment'] = low_usage_segment
    
    return opportunities
def compute_dispersion(data, column):
    
    variance = data[column].var()
    standard_deviation = data[column].std()
    iqr = data[column].quantile(0.75) - data[column].quantile(0.25)
    data_range = data[column].max() - data[column].min()
    
    return {
        'Variance': variance,
        'Standard Deviation': standard_deviation,
        'Interquartile Range (IQR)': iqr,
        'Range': data_range
    }

# Function to analyze the dispersion for all quantitative variables in the dataset
def analyze_dispersion(data):
    
    # Select quantitative columns (numeric columns)
    quantitative_columns = data.select_dtypes(include=['number']).columns

    # Initialize a dictionary to store the results
    dispersion_params = {}

    # Loop through each quantitative column to compute the dispersion parameters
    for col in quantitative_columns:
        dispersion_params[col] = compute_dispersion(data, col)

    # Convert the results into a DataFrame for easy interpretation
    dispersion_df = pd.DataFrame(dispersion_params).T

    return dispersion_df

def compute_dispersion(data, column):
    variance = data[column].var()
    standard_deviation = data[column].std()
    iqr = data[column].quantile(0.75) - data[column].quantile(0.25)
    data_range = data[column].max() - data[column].min()
    
    return {
        'Variance': variance,
        'Standard Deviation': standard_deviation,
        'Interquartile Range (IQR)': iqr,
        'Range': data_range
    }

# Function to analyze the dispersion for all quantitative variables in the dataset
def analyze_dispersion(data):
  
    # Select quantitative columns (numeric columns)
    quantitative_columns = data.select_dtypes(include=['number']).columns

    # Initialize a dictionary to store the results
    dispersion_params = {}

    # Loop through each quantitative column to compute the dispersion parameters
    for col in quantitative_columns:
        dispersion_params[col] = compute_dispersion(data, col)

    # Convert the results into a DataFrame for easy interpretation
    dispersion_df = pd.DataFrame(dispersion_params).T

    return dispersion_df

def compute_dispersion(data, column):
   
    variance = data[column].var()
    standard_deviation = data[column].std()
    iqr = data[column].quantile(0.75) - data[column].quantile(0.25)
    data_range = data[column].max() - data[column].min()
    
    return {
        'Variance': variance,
        'Standard Deviation': standard_deviation,
        'Interquartile Range (IQR)': iqr,
        'Range': data_range
    }


def preprocess_data(df, numerical_cols, categorical_cols):
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(), categorical_cols)
        ])
    
    return preprocessor

# Step 3: Apply PCA to the Preprocessed Data
def apply_pca(df, n_components=2):
    # Drop rows with any missing values
    df_cleaned = df.dropna()
    
    # Identify numerical and categorical columns
    numerical_cols = df_cleaned.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df_cleaned.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Preprocess the data
    preprocessor = preprocess_data(df_cleaned, numerical_cols, categorical_cols)
    
    # Create a pipeline with the preprocessor and PCA
    pca_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('pca', PCA(n_components=n_components))
    ])
    
    # Apply the pipeline to the data
    pca_result = pca_pipeline.fit_transform(df_cleaned)
    
    return pca_result

import pandas as pd

def calculate_total_traffic(df):
    """
    Adds a 'Total Traffic (Bytes)' column to the DataFrame by summing
    'Total UL (Bytes)' and 'Total DL (Bytes)'.
    """
    df['Total Traffic (Bytes)'] = df['Total UL (Bytes)'] + df['Total DL (Bytes)']
    return df

def aggregate_metrics(df):
    """
    Aggregates metrics per customer (MSISDN/Number) to calculate:
    - Total session duration (Dur. (ms)).
    - Total traffic (Total Traffic (Bytes)).
    - Session frequency (count of sessions).
    """
    aggregated = df.groupby('MSISDN/Number').agg({
        'Dur. (ms)': 'sum',
        'Total Traffic (Bytes)': 'sum',
        'MSISDN/Number': 'count'
    }).rename(columns={
        'MSISDN/Number': 'Session Frequency'
    })
    return aggregated

def get_top_10_by_metric(aggregated_df, metric):
    """
    Sorts the aggregated DataFrame by the specified metric and returns the top 10 customers.
    """
    return aggregated_df.sort_values(metric, ascending=False).head(10)

def print_top_customers(top_customers, metric_name):
    """
    Prints the top 10 customers for a specific metric.
    """
    print(f"Top 10 Customers by {metric_name}:")
    print(top_customers)
    print("\n")










