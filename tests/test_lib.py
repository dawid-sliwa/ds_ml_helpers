import unittest
import torch
import os
import io
from unittest.mock import patch
from pathlib import Path
from ds_ml_data_kit.functions import walk_through_dir, accuracy_fn, download_data
import shutil


class TestDsMlDataHelpers(unittest.TestCase):
    def test_walk_through_dir_output(self):
        test_dir = "test_directory"
        os.mkdir(test_dir)
        with open(os.path.join(test_dir, "test_file.txt"), "w") as f:
            f.write("Test")

        expected_output = f"There are 0 directories and 1 images in '{test_dir}'.\n"
        with unittest.mock.patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            walk_through_dir(test_dir)
            self.assertEqual(mock_stdout.getvalue(), expected_output)

        os.remove(os.path.join(test_dir, "test_file.txt"))
        os.rmdir(test_dir)
         
    
    def test_accuracy_calculation(self):
        y_true = torch.tensor([1, 0, 1, 1])
        y_pred = torch.tensor([1, 0, 0, 1])

        expected_accuracy = 75.0
        accuracy = accuracy_fn(y_true, y_pred)
        self.assertEqual(accuracy, expected_accuracy)
        
    
    def test_download_and_extraction(self):
        test_url = "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
        test_destination = "test_data"

        expected_path = Path("data") / test_destination

        result_path = download_data(test_url, test_destination)

        self.assertTrue(expected_path.is_dir())

        shutil.rmtree(Path("data"))



if __name__ == '__main__':
    unittest.main()
