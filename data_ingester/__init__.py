import psycopg2
import os.path
import pi_eye
import datetime

dbname = 'pieyeDB'
user = 'username'
password = 'pass'
def connect():
    connection_string = f"host='127.0.0.1' dbname='{dbname}' user='{user}' password='{password}' port=5488"
    print(f"connecting {connection_string}")
    return psycopg2.connect(connection_string)

def insert(object_class, creation_date):
    connection = connect()
    cursor = connection.cursor()
    sql_insert_query = """ INSERT INTO observation (class, observed_on) 
                           VALUES (%s,%s) """
    cursor.execute(sql_insert_query, (object_class, creation_date))
    connection.commit()


def list():
    connection = connect()
    cursor = connection.cursor()
    # Execute a query
    cursor.execute("SELECT * FROM observation")

    # Retrieve query results
    records = cursor.fetchall()
    print(records)
    return records;

# returns list of files with classifications
def load_input_files(upload_dir):
    result = {}
    list_of_files_to_process = os.listdir(upload_dir)
    for file in list_of_files_to_process:
        absolute_path = os.path.abspath(upload_dir + "/" + file)
        class_of_object = pi_eye.classify(absolute_path)
        result[file] = class_of_object
        file_creation_date = os.path.getmtime(absolute_path)
        insert(class_of_object, datetime.datetime.fromtimestamp(file_creation_date))
        os.remove(absolute_path)
