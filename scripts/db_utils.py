import psycopg2
import pandas as pd

def fetch_data_from_db(query, host="localhost", port="5432", database="telecome_data", user="postgres", password="blessed"):
   
    try:
        # Connect to the PostgreSQL database
        connection = psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password
        )

        # Execute the query and load the data into a Pandas DataFrame
        data = pd.read_sql_query(query, connection)
        return data

    except Exception as e:
        print("Error:", e)
        return None

    finally:
        if connection:
            connection.close()