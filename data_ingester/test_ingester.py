import datetime
import unittest
import data_ingester
import testing.postgresql
import tempfile
import shutil
import os


class DataIngesterTests(unittest.TestCase):
    UPLOAD_DIR = "./test-data/golf/"

    def setUp(self):
        self.postgresql = testing.postgresql.Postgresql(port=5488)
        print(self.postgresql.url())
        data_ingester.dbname = "test"
        data_ingester.user = "postgres"
        data_ingester.password = None

        db_con = data_ingester.connect()
        with db_con.cursor() as cur:
            cur.execute(read_file('./data_ingester/storage/migrations/V1_create_observations_table.sql'))
            db_con.commit()

    def tearDown(self):
        self.postgresql.stop()

    def test_no_results_when_empty_database(self):
        self.assertEqual(data_ingester.list(), [])

    def test_that_can_retrieve_after_insert(self):
        creationdate = datetime.datetime(1985, 11, 4, 1, 5)
        data_ingester.insert("class_abc", creationdate)

        result = data_ingester.list()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][1], "class_abc")
        self.assertEqual(result[0][2], creationdate)

    def test_load_dir_and_read_data(self):
        temporary_dir = tempfile.mkdtemp()
        destination_file = f"{temporary_dir}/04-20210108112804-01.jpg"
        shutil.copyfile(f"{self.UPLOAD_DIR}/04-20210108112804-01.jpg", destination_file)
        file_creation_date = datetime.datetime.fromtimestamp(os.path.getmtime(destination_file))

        data_ingester.load_input_files(temporary_dir)
        self.assertEqual([(1, 'srebrny golf', file_creation_date)] , data_ingester.list())


def read_file(path):
    """ Reads and returns the entire contents of a file """
    with open(path, 'r') as f:
        return f.read()


if __name__ == '__main__':
    unittest.main()
