import pandas as pd
import psycopg2
from config.setttings import get_settings

settings = get_settings()


def query(sql: str) -> pd.DataFrame:
    """
    Query the PostgreSQL database and return the result as a DataFrame.

    Args:
        sql (str): The SQL query to execute.

    Returns:
        pd.DataFrame: The result of the query as a DataFrame.
    """
    # Establish a connection to the PostgreSQL database
    conn = psycopg2.connect(settings.database.service_url)

    # Create a cursor object
    with conn:
        with conn.cursor() as cur:
            # Execute the SQL query
            cur.execute(sql)

            # Fetch all rows from the executed query
            rows = cur.fetchall()

            # Get column names from the cursor description
            colnames = [desc[0] for desc in cur.description]

            # Convert rows to a DataFrame
            df = pd.DataFrame(rows, columns=colnames)

    conn.close()
    return df


def create_database(db_name: str):
    """
    Create a new database with the given name.

    Args:
        db_name (str): The name of the database to create.
    """
    conn = psycopg2.connect(settings.database.service_url)
    conn.autocommit = True
    with conn.cursor() as c:
        c.execute(f"DROP DATABASE IF EXISTS {db_name}")
        c.execute(f"CREATE DATABASE {db_name}")
    conn.close()
