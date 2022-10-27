import psycopg2

dbname = 'pieyeDB'
user = 'username'
password = 'pass'
# 'postgresql://postgres@127.0.0.1:5488/test'
def connect():
    connection_string = f"host='127.0.0.1' dbname='{dbname}' user='{user}' password='{password}' port=5488"
    print(f"connecting {connection_string}")
    return psycopg2.connect(connection_string)

def insert(object_class):
    connection = connect()
    cursor = connection.cursor()
    sql_insert_query = """ INSERT INTO observation (class, observed_on) 
                           VALUES (%s,now()) """
    record_to_insert = [object_class]
    cursor.execute(sql_insert_query, record_to_insert)
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
