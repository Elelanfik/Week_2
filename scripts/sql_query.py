import psycopg2
import pandas as pd

def get_user_data_query():
    # Establish the database connection with correct string values
    connection = psycopg2.connect(
        host="localhost",           # Host should be a string
        port=5432,                  # Port is already a number (no need for quotes)
        database="telecome_data",   # Database name should be a string
        user="postgres",            # User should be a string
        password="blessed"          # Password should be a string
    )

    # SQL query to aggregate data
    query = """
    SELECT 
        "MSISDN/Number" AS user_id,
        COUNT("Bearer Id") AS number_of_xDR_sessions,  
        SUM("Dur. (ms)") AS total_session_duration,    
        SUM("Total DL (Bytes)") AS total_download_data, 
        SUM("Total UL (Bytes)") AS total_upload_data,   
        (SUM("Total DL (Bytes)") + SUM("Total UL (Bytes)")) AS total_data_volume,  
        SUM("Social Media DL (Bytes)") AS social_media_data_volume,  
        SUM("Google DL (Bytes)") AS google_data_volume,  
        SUM("Email DL (Bytes)") AS email_data_volume,    
        SUM("Youtube DL (Bytes)") AS youtube_data_volume,  
        SUM("Netflix DL (Bytes)") AS netflix_data_volume,  
        SUM("Gaming DL (Bytes)") AS gaming_data_volume,    
        SUM("Other DL (Bytes)") AS other_data_volume       
    FROM 
        xdr_data  
    GROUP BY 
        "MSISDN/Number"
    ORDER BY 
        user_id;
    """
    return query
