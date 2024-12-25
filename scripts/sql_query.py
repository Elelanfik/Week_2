import psycopg2

# Function to return the SQL query
def get_user_data_query():
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

def load_data_from_postgres(query):
    try:
        # Define your PostgreSQL connection parameters
        connection = psycopg2.connect(
            host="db",  # Replace with your host
            port=5432,  # Ensure this is an integer, not a string
            database="xdr_data",  # Replace with your database name
            user="postgres",  # Replace with your PostgreSQL username
            password="1234"  # Replace with your password
        )
        
        # Load data into a pandas DataFrame
        df = pd.read_sql(query, connection)
        
        # Close the connection
        connection.close()
        
        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None