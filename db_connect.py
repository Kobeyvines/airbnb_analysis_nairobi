import os
import psycopg2
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()


def connect_to_db():
    """
    Establishes and returns a secure connection to the PostgreSQL database.
    """
    try:
        connection = psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
        )
        print("Database connection successful!")
        return connection
    except Exception as error:
        print("Error connecting to the database:", error)
        return None
