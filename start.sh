# runs project locally
pip3.9 install -r requirements.txt

docker run --name pi-eye-postgres -p 5488:5432 -e POSTGRES_USER=username -e POSTGRES_PASSWORD=pass -e POSTGRES_DB=pieyeDB -d postgres

flask run

add nwe module for postgres stuff. or maybe keep in one fiile for now??
installwd binary !!
pip install psycopg2-binary

eg:

