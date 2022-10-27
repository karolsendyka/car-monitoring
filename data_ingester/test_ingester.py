import unittest
import data_ingester
import testing.postgresql

class DataIngesterTests(unittest.TestCase):

    # import testing.postgresql
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

    def assert_empty_database(self):
        self.assertEqual(data_ingester.list(), [])


def read_file(path):
    """ Reads and returns the entire contents of a file """
    with open(path, 'r') as f:
        return f.read()

if __name__ == '__main__':
    unittest.main()
