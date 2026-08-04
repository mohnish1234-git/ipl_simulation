import os
import pandas as pd
import psycopg2
from io import StringIO
from dotenv import load_dotenv

##############################################################
# LOAD ENVIRONMENT
##############################################################

load_dotenv("database/.env")

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

##############################################################
# CONNECT
##############################################################

conn = psycopg2.connect(
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT,
)

cur = conn.cursor()

##############################################################
# LOAD CSV
##############################################################

CSV_PATH = "data/processed/fantasy_classifier_dataset.csv"

df = pd.read_csv(CSV_PATH)

print(f"Loaded {len(df)} rows × {len(df.columns)} columns")

##############################################################
# CREATE TABLE
##############################################################

cur.execute("DROP TABLE IF EXISTS fantasy_classifier_dataset")

columns = []

for col, dtype in zip(df.columns, df.dtypes):
    if "int" in str(dtype):
        sql_type = "BIGINT"
    elif "float" in str(dtype):
        sql_type = "DOUBLE PRECISION"
    elif "bool" in str(dtype):
        sql_type = "BOOLEAN"
    else:
        sql_type = "TEXT"

    columns.append(f'"{col}" {sql_type}')

create_query = f"""
CREATE TABLE fantasy_classifier_dataset (
    {', '.join(columns)}
)
"""

cur.execute(create_query)

##############################################################
# BULK INSERT
##############################################################

buffer = StringIO()
df.to_csv(buffer, index=False, header=False)
buffer.seek(0)

cur.copy_expert(
    f"""
    COPY fantasy_classifier_dataset
    FROM STDIN
    WITH CSV
    """,
    buffer,
)

conn.commit()

print("\nSuccessfully uploaded fantasy_classifier_dataset")
print(f"Rows inserted: {len(df)}")

cur.close()
conn.close()