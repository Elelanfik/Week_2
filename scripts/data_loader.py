import pandas as pd
import sqlite3

def load_data(file_path):
    return pd.read_csv(file_path)

def load_sql_data(sql_file_path, db_path):
    # Create a database connection
    conn = sqlite3.connect(db_path)
    
    # Read the SQL queries from the file
    with open(sql_file_path, 'r') as f:
        sql_queries = f.read()
    
    # Execute the SQL query and load the data into a DataFrame
    df = pd.read_sql_query(sql_queries, conn)
    
    # Close the connection
    conn.close()
    
    return df


def load_excel_data(file_path, sheet_name=0):
    
    return pd.read_excel(file_path, sheet_name=sheet_name)


