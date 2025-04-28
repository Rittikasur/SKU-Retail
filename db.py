import psycopg2

# Database connection config
DB_NAME = "skur"
DB_USER = "postgres"
DB_PASSWORD = "postgres"
DB_HOST = "10.4.1.66"
DB_PORT = "5432"

# SQL to create extension and table
CREATE_EXTENSION_SQL = "CREATE EXTENSION IF NOT EXISTS vector;"
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS items (
    id SERIAL PRIMARY KEY,
    itemcode TEXT,
    itemdesc TEXT,
    company TEXT,
    brand TEXT,
    packaging TEXT,
    pack_size TEXT,
    qty INT,
    uom TEXT,
    unit TEXT,
    filtered_itemdesc_embedding VECTOR(1024),
    itemdesc_embedding VECTOR(1024),
    company_embedding VECTOR(512),
    brand_embedding VECTOR(512),
    packaging_embedding VECTOR(512),
    pack_size_embedding VECTOR(512)
);
"""

def get_connection():
    conn = None
    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
    except:
        print("Connection Failed")
    return conn

def connect_and_create():
    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        conn.autocommit = True  # Required to run CREATE EXTENSION
        cur = conn.cursor()

        # Create pgvector extension
        cur.execute(CREATE_EXTENSION_SQL)
        print("✅ pgvector extension checked/created.")

        # Create table
        cur.execute(CREATE_TABLE_SQL)
        print("✅ Table 'items' created or already exists.")

        cur.close()
        conn.close()

    except Exception as e:
        print("❌ Error:", e)

if __name__ == "__main__":
    connect_and_create()
