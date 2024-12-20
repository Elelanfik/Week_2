import psycopg2
import pandas as pd

def fetch_data_from_db(query, host="localhost", port="5432", database="telecome_data", user="postgres", password="blessed"):
    """
    Fetch data from PostgreSQL database using a SQL query and return it as a Pandas DataFrame.

    Parameters:
    - query (str): The SQL query to execute.
    - host (str): Hostname of the database server.
    - port (str): Port number of the database server.
    - database (str): Name of the database.
    - user (str): Username for the database.
    - password (str): Password for the database.

    Returns:
    - pd.DataFrame: The query result as a Pandas DataFrame.
    """
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