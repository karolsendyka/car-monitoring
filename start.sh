# runs project locally
pip3.9 install -r requirements.txt
pip3.9 install psycopg2-binary

docker run --name pi-eye-postgres -p 5488:5432 -e POSTGRES_USER=username -e POSTGRES_PASSWORD=pass -e POSTGRES_DB=pieyeDB -d postgres
scp pi@raspberrypi.local:/motion/*.jpg ./uploads/

flask run
