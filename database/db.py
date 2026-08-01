from dotenv import load_dotenv
from psycopg2.pool import SimpleConnectionPool

import os

load_dotenv("database/.env")

pool = SimpleConnectionPool(

    1,

    10,

    dbname=os.getenv("DB_NAME"),

    user=os.getenv("DB_USER"),

    password=os.getenv("DB_PASSWORD"),

    host=os.getenv("DB_HOST"),

    port=os.getenv("DB_PORT")

)

def get_connection():

    return pool.getconn()


def return_connection(conn):

    pool.putconn(conn)


def close_pool():

    pool.closeall()