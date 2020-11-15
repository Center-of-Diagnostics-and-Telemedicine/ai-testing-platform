import sqlite3
from sqlite3 import Error


def create_connection(path='aitesting.db'):
    connection = None
    try:
        connection = sqlite3.connect(path)
        print("Connection to SQLite DB successful")
    except Error as e:
        print(f"The error '{e}' occurred")

    return connection

def execute_query(connection, query):
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        connection.commit()
        print("Query executed successfully: " + query)
    except Error as e:
        print(f"The error '{e}' occurred")

    return cursor.lastrowid

def execute_read_query(connection, query):
    cursor = connection.cursor()
    result = None
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        return result
    except:
        return("The error occurred")


def empty_db():
    with create_connection() as conn:
        execute_query(conn, 'DROP TABLE IF EXISTS services;')
        execute_query(conn, 'DROP TABLE IF EXISTS session_tokens;')
        execute_query(conn, 'DROP TABLE IF EXISTS testing;')
        execute_query(conn, 'DROP TABLE IF EXISTS datasets;')

        execute_query(conn, 'CREATE TABLE services (id INTEGER NOT NULL PRIMARY KEY, name VARCHAR(100), token VARCHAR(64));')
        execute_query(conn, 'CREATE TABLE session_tokens (id INTEGER NOT NULL PRIMARY KEY, service INTEGER, session_token VARCHAR(64), issue_date TIMESTAMP, expiry_date TIMESTAMP, active BOOLEAN);')
        execute_query(conn, 'CREATE TABLE testing (id INTEGER NOT NULL PRIMARY KEY, session INTEGER, dataset_title TEXT, dataset_file_id INTEGER, created TIMESTAMP, retrieved TIMESTAMP, received TIMESTAMP, ai_ct INTEGER, ai_left_affected_part FLOAT, ai_left_total_volume FLOAT, ai_left_affected_volume FLOAT, ai_right_affected_part FLOAT, ai_right_total_volume FLOAT, ai_right_affected_volume FLOAT, viewer_url VARCHAR(200), description VARCHAR(300), requests INTEGER);')
        execute_query(conn, 'CREATE TABLE datasets(id INTEGER NOT NULL PRIMARY KEY, title TEXT, filename TEXT, var1 VARCHAR(30), var2 VARCHAR(30), var3 VARCHAR(30), var4 VARCHAR(30), var5 VARCHAR(30), added TIMESTAMP);')




