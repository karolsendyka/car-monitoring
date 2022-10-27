import psycopg2

# Connect to your postgres DB
connection_string = "host='127.0.0.1' dbname='pieyeDB' user='username' password='pass' port=5488"
conn = psycopg2.connect(connection_string)

# Open a cursor to perform database operations
cur = conn.cursor()

# Execute a query
cur.execute("SELECT * FROM observation")

# Retrieve query results
records = cur.fetchall()

print(records)

