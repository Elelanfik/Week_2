import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

    return grouped




