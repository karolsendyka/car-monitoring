import unittest
import numpy as np
# import os.path
import tflite_runtime.interpreter as tflite


class PiEyeTests(unittest.TestCase):

    def test_example_photos(self):
        print(np)
        print(tflite)
        self.assertEqual(1, 2)


if __name__ == '__main__':
    unittest.main()
