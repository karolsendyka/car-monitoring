import unittest
import pi_eye


class PiEyeTests(unittest.TestCase):

    # available classes ['Blekitne suzuki', 'bialy bus', 'chrupek', 'czarne suzuki', 'czerwona mazda', 'srebrny golf']
    CLASS_CHRUPEK = 'chrupek'
    CLASS_GOLF = 'srebrny golf'

    CHRUPEK_PHOTO = "./test-data/chrupek/03-20210109122815-00.jpg"
    GOLF_PHOTO = "./test-data/golf/04-20210108112804-01.jpg"
    UPLOAD_DIR = "./test-data/golf/"

    target = pi_eye

    def test_example_photos(self):
        self.assertEqual(pi_eye.classify(self.CHRUPEK_PHOTO), self.CLASS_CHRUPEK)
        self.assertEqual(pi_eye.classify(self.GOLF_PHOTO), self.CLASS_GOLF)

    def test_load_dir(self):
        self.assertEqual(pi_eye.load_input_files(self.UPLOAD_DIR), {'04-20210108112804-01.jpg': 'srebrny golf'})

if __name__ == '__main__':
    unittest.main()
