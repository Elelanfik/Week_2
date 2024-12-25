import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.ensemble import RandomForestRegressor
import psycopg2 
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


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
import pandas as pd
import numpy as np
import scipy.stats 
from scipy.stats import zscore

def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # dtype of missing values
    mis_val_dtype = df.dtypes

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent, mis_val_dtype], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
    columns={0: 'Missing Values', 1: '% of Total Values', 2: 'Dtype'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
          "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns


# Define a function to convert milliseconds to seconds and drop the original ms columns
def apply_ms_to_sec_and_drop(columns, df):
    for col in columns:
        new_col = col.replace('ms', 'sec')
        df[new_col] = df[col].apply(ms_to_sec)  # Convert ms to sec
        df.drop(col, axis=1, inplace=True)  # Drop the original ms column
    return df

def ms_to_sec(ms):
    return ms / 1000.0


def convert_bytes_to_megabytes(df, bytes_data):
    megabyte = 1 * 10e+5
    df[bytes_data] = df[bytes_data] / megabyte
    return df[bytes_data]

# Define a function to convert bytes to megabytes for multiple columns
def convert_columns_to_mb(columns, df):
    for col in columns:
        new_col = col.replace('Bytes', 'MB')
        df[new_col] = convert_bytes_to_megabytes(df, col)
        df.drop(col, axis=1, inplace=True)  # Drop the original ms column
    return df

def handeling_missing_data(data):
    missing_data = data.isnull().sum().sort_values(ascending=False)
    missing_percentage = (missing_data / len(data)) * 100
    data_cleaned = data.drop(columns=missing_data[missing_percentage > 90].index)

    # Step 3: Impute missing values in numeric columns using the mean
    numeric_cols = data_cleaned.select_dtypes(include=['float64', 'int64']).columns
    data_cleaned[numeric_cols] = data_cleaned[numeric_cols].fillna(0)

    # Step 4: Impute missing values in categorical columns using 'unknown'
    categorical_cols = data_cleaned.select_dtypes(include=['object']).columns
    data_cleaned[categorical_cols] = data_cleaned[categorical_cols].fillna('unknown')
    return data_cleaned



def fix_outlier(df, column):
    df[column] = np.where(df[column] > df[column].quantile(0.95), df[column].median(), df[column])
    return df[column]

def remove_outliers(df, column_to_process, z_threshold=3):
    # Apply outlier removal to the specified column
    z_scores = zscore(df[column_to_process])
    outlier_column = column_to_process + '_Outlier'
    df[outlier_column] = (np.abs(z_scores) > z_threshold).astype(int)
    df = df[df[outlier_column] == 0]  # Keep rows without outliers

    # Drop the outlier column as it's no longer needed
    df = df.drop(columns=[outlier_column], errors='ignore')

    return df


class ExperienceAnalyzer:
    """
    Analyzes user experience based on network parameters and handset type.
    """

    def __init__(self, df):
        """
        Initializes the analyzer with the input DataFrame.

        Args:
            df: pandas DataFrame with the required columns.
        """
        self.df = df.copy()
        self.user_agg = None
        self.experience_cluster_centers_ = None

    def _fill_missing_values(self):
        """
        Fills missing values in the DataFrame using mean for numerical columns
        and mode for categorical columns.
        """
        fill_values = {
            'TCP DL Retrans. Vol (Bytes)': self.df['TCP DL Retrans. Vol (Bytes)'].mean(),
            'Avg RTT DL (ms)': self.df['Avg RTT DL (ms)'].mean(),
            'Avg Bearer TP DL (kbps)': self.df['Avg Bearer TP DL (kbps)'].mean(),
            'Handset Type': self.df['Handset Type'].mode()[0]
        }
        self.df.fillna(fill_values, inplace=True)

    def aggregate_user_data(self):
        """
        Aggregates network parameters and handset type per customer.
        """
        # Fill missing values
        self._fill_missing_values()

        # Aggregate data per customer
        self.user_agg = (
            self.df.groupby('MSISDN/Number')
            .agg({
                'TCP DL Retrans. Vol (Bytes)': 'mean',
                'Avg RTT DL (ms)': 'mean',
                'Handset Type': 'first',
                'Avg Bearer TP DL (kbps)': 'mean'
            })
            .reset_index()
        )

    def analyze_columns(self, columns_to_analyze, top_n=10):
        """
        Analyzes specified columns for top, bottom, and most frequent values.

        Args:
            columns_to_analyze: List of columns to analyze.
            top_n: Number of values to extract for each metric.

        Returns:
            Dictionary containing analysis results for each column.
        """
        if self.user_agg is None:
            raise ValueError("Data has not been aggregated. Call aggregate_user_data first.")

        def get_top_bottom_frequent(data, column, top_n):
            top_values = data[column].nlargest(top_n)
            bottom_values = data[column].nsmallest(top_n)
            frequent_values = data[column].value_counts().head(top_n)
            return top_values, bottom_values, frequent_values

        # Dictionary to store the results
        analysis_results = {}

        # Analyze each column
        for column in columns_to_analyze:
            top_values, bottom_values, frequent_values = get_top_bottom_frequent(self.user_agg, column, top_n)
            analysis_results[column] = {
                'Top 10': top_values,
                'Bottom 10': bottom_values,
                'Most Frequent 10': frequent_values
            }

        return analysis_results


class Experiencevisualizer:
    def __init__(self, data):
        # Initialize the class with the DataFrame
        self.df = data

    def plot_throughput_distribution_by_handset(self):
        """
        Analyzes and visualizes the distribution of average throughput per handset type.
        """
        # Group by handset type and calculate average throughput
        throughput_per_handset = self.df.groupby('Handset Type')['Avg Bearer TP DL (kbps)'].mean()
        print("\nAverage Throughput per Handset Type:")
        print(throughput_per_handset)
        
        # Visualize distribution (bar plot)
        plt.figure(figsize=(14, 8))
        throughput_per_handset.plot(kind='barh', color='skyblue', edgecolor='blue')
        plt.ylabel('Handset Type')
        plt.xlabel('Average Throughput (kbps)')
        plt.title('Average Throughput per Handset Type')
        plt.tight_layout()
        plt.show()

    def plot_retransmission_distribution_by_handset(self):
        """
        Analyzes the average TCP retransmission per handset type.
        """
        # Group by handset type and calculate average retransmission
        retransmission_per_handset = self.df.groupby('Handset Type')['TCP DL Retrans. Vol (Bytes)'].mean()
        print("\nAverage TCP Retransmission per Handset Type:")
        print(retransmission_per_handset)
        
        # Visualize distribution (bar plot)
        plt.figure(figsize=(14, 8))
        retransmission_per_handset.plot(kind='barh', color='salmon', edgecolor='blue')
        plt.ylabel('Handset Type')
        plt.xlabel('Average TCP Retransmission (Bytes)')
        plt.title('Average TCP Retransmission per Handset Type')
        plt.tight_layout()
        plt.show()


# Define functions to modularize the code

def standardize_features(data, features):
    """
    Standardizes the features for clustering.

    Args:
        data (DataFrame): Input dataset.
        features (list): List of feature column names to standardize.

    Returns:
        ndarray: Standardized feature matrix.
        StandardScaler: Fitted scaler for inverse transformation.
    """
    scaler = StandardScaler()
    transformed_data = scaler.fit_transform(data[features])
    return transformed_data, scaler


def apply_kmeans(data, transformed_data, n_clusters=3, random_state=42):
    """
    Applies K-means clustering to the data.

    Args:
        data (DataFrame): Input dataset.
        transformed_data (ndarray): Standardized feature matrix.
        n_clusters (int): Number of clusters.
        random_state (int): Random state for reproducibility.

    Returns:
        DataFrame: Dataset with added cluster labels.
        KMeans: Fitted KMeans model.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    data['Cluster'] = kmeans.fit_predict(transformed_data)
    return data, kmeans


def analyze_clusters(kmeans, scaler, features):
    """
    Analyzes the cluster centers.

    Args:
        kmeans (KMeans): Fitted KMeans model.
        scaler (StandardScaler): Fitted scaler for inverse transformation.
        features (list): List of feature column names.

    Returns:
        DataFrame: Cluster centers in original feature space.
    """
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    return pd.DataFrame(cluster_centers, columns=features)


def map_clusters(data, cluster_mapping):
    """
    Maps cluster numbers to descriptive labels.

    Args:
        data (DataFrame): Dataset with cluster labels.
        cluster_mapping (dict): Mapping of cluster numbers to descriptive labels.

    Returns:
        DataFrame: Dataset with added descriptive cluster labels.
    """
    data['ClusterNo'] = data['Cluster'].map(cluster_mapping)
    return data


def plot_clusters(data, x_feature, y_feature, cluster_mapping, colors):
    """
    Plots the clusters in a scatter plot.

    Args:
        data (DataFrame): Dataset with cluster labels.
        x_feature (str): Feature name for x-axis.
        y_feature (str): Feature name for y-axis.
        cluster_mapping (dict): Mapping of cluster numbers to descriptive labels.
        colors (list): List of colors for clusters.
    """
    plt.figure(figsize=(8, 6))
    for cluster_id in data['Cluster'].unique():
        cluster_data = data[data['Cluster'] == cluster_id]
        plt.scatter(cluster_data[x_feature], cluster_data[y_feature],
                    c=colors[cluster_id], label=cluster_mapping[cluster_id], s=100, alpha=0.7)
    plt.title('User Clusters Based on Experience Metrics')
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.legend(title="Cluster")
    plt.show()



class SatisfactionAnalyzer:
    """
    Analyzes user satisfaction based on engagement and experience scores.
    """
    def cluster_satisfaction(self, n_clusters=2):
        """
        Performs K-Means clustering on satisfaction scores (Engagement and Experience Scores).

        Args:
            n_clusters: Number of clusters to form.
        """
        # Ensure that satisfaction scores are calculated before clustering
        if 'Engagement_Score' not in self.user_agg or 'Experience_Score' not in self.user_agg:
            raise ValueError("Satisfaction scores (Engagement_Score and Experience_Score) must be calculated before clustering.")

        # K-Means clustering on satisfaction scores
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.user_agg['Satisfaction_Cluster'] = kmeans.fit_predict(
            self.user_agg[['Engagement_Score', 'Experience_Score']]
        )
        print(f"Satisfaction clustering completed with {n_clusters} clusters.")

    def cluster_satisfaction(self, n_clusters=2):
        """
        Performs K-Means clustering on satisfaction scores (Engagement and Experience Scores).

        Args:
            n_clusters: Number of clusters to form.
        """
        # Ensure that satisfaction scores are calculated before clustering
        if 'Engagement_Score' not in self.user_agg or 'Experience_Score' not in self.user_agg:
            raise ValueError("Satisfaction scores (Engagement_Score and Experience_Score) must be calculated before clustering.")

        # K-Means clustering on satisfaction scores
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.user_agg['Satisfaction_Cluster'] = kmeans.fit_predict(
            self.user_agg[['Engagement_Score', 'Experience_Score']]
        )
        print(f"Satisfaction clustering completed with {n_clusters} clusters.")    

    def __init__(self, df):
        """
        Initializes the analyzer with the input DataFrame.

        Args:
            df: pandas DataFrame with the required columns.
        """
        self.df = df.copy()
        self.engagement_centers_ = None
        self.experience_centers_ = None

    def preprocess_data(self):
        """
        Prepares and aggregates network parameters and handset type per customer,
        handling missing values using mean/mode.
        """
        required_columns = [
            'TCP DL Retrans. Vol (Bytes)', 'Avg RTT DL (ms)', 'Avg Bearer TP DL (kbps)',
            'Handset Type', 'MSISDN/Number', 'Dur. (ms)', 'Total UL (Bytes)', 'Total DL (Bytes)'
        ]

        # Validate presence of required columns
        for col in required_columns:
            if col not in self.df.columns:
                raise KeyError(f"Column '{col}' is missing from the DataFrame.")

        # Calculate additional features
        self.df['Session_Frequency'] = self.df.groupby('MSISDN/Number')['MSISDN/Number'].transform('count')
        self.df['Total_Session_Duration'] = self.df.groupby('MSISDN/Number')['Dur. (ms)'].transform('sum')
        self.df['Total_Traffic'] = self.df['Total UL (Bytes)'] + self.df['Total DL (Bytes)']

        fill_values = {
            'TCP DL Retrans. Vol (Bytes)': self.df['TCP DL Retrans. Vol (Bytes)'].mean(),
            'Avg RTT DL (ms)': self.df['Avg RTT DL (ms)'].mean(),
            'Avg Bearer TP DL (kbps)': self.df['Avg Bearer TP DL (kbps)'].mean(),
            'Handset Type': self.df['Handset Type'].mode()[0]
        }
        self.df.fillna(fill_values, inplace=True)

        self.user_agg = self.df.groupby('MSISDN/Number').agg({
            'TCP DL Retrans. Vol (Bytes)': 'mean',
            'Avg RTT DL (ms)': 'mean',
            'Handset Type': 'first',
            'Avg Bearer TP DL (kbps)': 'mean',
            'Session_Frequency': 'first',
            'Total_Session_Duration': 'first',
            'Total_Traffic': 'first'
        }).reset_index()

    def get_top_satisfied_customers(self, n=10):
        """
        Finds the top n most satisfied customers and generates a bar plot of their satisfaction scores.

        Args:
            n: Number of top satisfied customers to return.
        """
        top_customers = self.user_agg.nlargest(n, 'Satisfaction_Score')
        
        print(f"\nTop {n} Most Satisfied Customers:")
        print(top_customers[['MSISDN/Number', 'Satisfaction_Score']])
        
        # Plotting bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(top_customers['MSISDN/Number'].astype(str), top_customers['Satisfaction_Score'], color='skyblue')
        plt.xlabel('MSISDN/Number')
        plt.ylabel('Satisfaction Score')
        plt.title(f'Top {n} Most Satisfied Customers')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()  # To prevent overlap of labels
        plt.show()   

    def cluster_engagement(self, n_clusters=3):
        """
        Clusters user engagement using K-Means.

        Args:
            n_clusters: Number of clusters.
        """
        scaler = StandardScaler()
        normalized_engagement = scaler.fit_transform(self.user_agg[['Session_Frequency', 
                                                                     'Total_Session_Duration', 
                                                                     'Total_Traffic']])
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.user_agg['Engagement_Cluster'] = kmeans.fit_predict(normalized_engagement)
        self.engagement_centers_ = kmeans.cluster_centers_

    def cluster_experience(self, n_clusters=3):
        """
        Clusters user experience using K-Means.

        Args:
            n_clusters: Number of clusters.
        """
        scaler = StandardScaler()
        normalized_experience = scaler.fit_transform(self.user_agg[['TCP DL Retrans. Vol (Bytes)',
                                                                     'Avg RTT DL (ms)',
                                                                     'Avg Bearer TP DL (kbps)']])
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.user_agg['Experience_Cluster'] = kmeans.fit_predict(normalized_experience)
        self.experience_centers_ = kmeans.cluster_centers_

    def compute_engagement_score(self):
        """
        Calculates engagement scores based on distances to the least engaged cluster.
        """
        if self.engagement_centers_ is None:
            raise ValueError("Cluster engagement first using `cluster_engagement`.")

        least_engaged_center = self.engagement_centers_[0]
        scaler = StandardScaler()
        normalized_engagement = scaler.fit_transform(self.user_agg[['Session_Frequency', 
                                                                     'Total_Session_Duration', 
                                                                     'Total_Traffic']])
        self.user_agg['Engagement_Score'] = euclidean_distances(normalized_engagement, 
                                                                least_engaged_center.reshape(1, -1))[:, 0]

    def compute_experience_score(self):
        """
        Calculates experience scores based on distances to the worst experience cluster.
        """
        if self.experience_centers_ is None:
            raise ValueError("Cluster experience first using `cluster_experience`.")

        worst_experience_center = self.experience_centers_[-1]
        scaler = StandardScaler()
        normalized_experience = scaler.fit_transform(self.user_agg[['TCP DL Retrans. Vol (Bytes)', 
                                                                     'Avg RTT DL (ms)', 
                                                                     'Avg Bearer TP DL (kbps)']])
        self.user_agg['Experience_Score'] = euclidean_distances(normalized_experience, 
                                                                worst_experience_center.reshape(1, -1))[:, 0]

    def compute_satisfaction_score(self):
        """
        Calculates satisfaction scores as the average of engagement and experience scores.
        """
        self.user_agg['Satisfaction_Score'] = (self.user_agg['Engagement_Score'] + 
                                               self.user_agg['Experience_Score']) / 2

    def predict_satisfaction_with_rf(self):
        """
        Predicts satisfaction scores using Random Forest Regression.
        """
        X = self.user_agg[['Session_Frequency', 'Total_Session_Duration', 
                        'Total_Traffic', 'TCP DL Retrans. Vol (Bytes)', 
                        'Avg RTT DL (ms)', 'Avg Bearer TP DL (kbps)']]
        y = self.user_agg['Satisfaction_Score']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize Random Forest Regressor
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        # Make predictions
        y_pred = rf_model.predict(X_test)

        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print("Random Forest Regression RMSE:", rmse)

        # Calculate R² score
        r2 = r2_score(y_test, y_pred)
        print("R² Score:", r2)

        # Plot predicted vs true values
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)  # Diagonal line
        plt.title("Predicted vs True Satisfaction Scores")
        plt.xlabel("True Satisfaction Score")
        plt.ylabel("Predicted Satisfaction Score")
        plt.show()

    def analyze_satisfaction_clusters(self, n_clusters=2):
        """
        Analyzes satisfaction clusters using K-Means clustering and visualizes the results.
        """
        # Extract the relevant features for clustering
        X = self.user_agg[['Session_Frequency', 'Total_Session_Duration', 
                        'Total_Traffic', 'TCP DL Retrans. Vol (Bytes)', 
                        'Avg RTT DL (ms)', 'Avg Bearer TP DL (kbps)']]

        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.user_agg['Cluster'] = kmeans.fit_predict(X)

        # Getting the cluster centers
        cluster_centers = kmeans.cluster_centers_

        # Visualize the clusters using 'Session Frequency' and 'Total Session Duration'
        plt.figure(figsize=(8, 6))
        plt.scatter(self.user_agg['Session_Frequency'], self.user_agg['Total_Session_Duration'], 
                    c=self.user_agg['Cluster'], cmap='coolwarm', marker='o')  # Using 'coolwarm' color map
        plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=200, c='lime', label='Centroids', marker='X')
        plt.title('K-Means Clustering on Session Frequency & Total Session Duration (k={})'.format(n_clusters))
        plt.xlabel('Session Frequency')
        plt.ylabel('Total Session Duration')
        plt.legend()
        plt.show()

        # Optionally, display the first few rows of the cluster assignments
        print(self.user_agg[['Session_Frequency', 'Total_Session_Duration', 'Cluster']].head())

        # Optional: Add a bar plot showing the number of customers in each cluster
        cluster_counts = self.user_agg['Cluster'].value_counts()
        plt.figure(figsize=(6, 4))
        cluster_counts.plot(kind='bar', color='skyblue')
        plt.title('Number of Customers in Each Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Number of Customers')
        plt.xticks(rotation=0)
        plt.show()

    def export_to_postgresql(self, host, user, password, database, table_name, port=5432):
        """
        Exports the final table to a PostgreSQL database.

        Args:
            host: PostgreSQL host address.
            user: PostgreSQL username.
            password: PostgreSQL password.
            database: PostgreSQL database name.
            table_name: Name of the table to create in the database.
            port: PostgreSQL port number (default is 5432).
        """
        try:
            # Connect to PostgreSQL
            conn = psycopg2.connect(
                host=host,
                user=user,
                password=password,
                dbname=database,
                port=port
            )
            cursor = conn.cursor()

            # Create table in PostgreSQL (if it doesn't already exist)
            create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                MSISDN_Number VARCHAR(255),
                Engagement_Score FLOAT,
                Experience_Score FLOAT,
                Satisfaction_Score FLOAT,
                Cluster INT
            )
            """
            cursor.execute(create_table_query)

            # Insert data into PostgreSQL
            for index, row in self.user_agg.iterrows():
                msisdn = row['MSISDN/Number']  # Ensure this column name exists in your DataFrame
                engagement_score = row['Engagement_Score']
                experience_score = row['Experience_Score']
                satisfaction_score = row['Satisfaction_Score']
                cluster = row['Cluster']  # Ensure this column name matches

                insert_query = f"""
                INSERT INTO {table_name} (MSISDN_Number, Engagement_Score, Experience_Score, Satisfaction_Score, Cluster)
                VALUES (%s, %s, %s, %s, %s)
                """
                cursor.execute(insert_query, (msisdn, engagement_score, experience_score, satisfaction_score, cluster))

            conn.commit()
            cursor.close()
            conn.close()
            print(f"Data successfully exported to {table_name} table in {database} database.")

        except psycopg2.Error as err:
            print(f"Error: {err}")

