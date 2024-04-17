'''
test_normalize_coordinates: This test case verifies the correctness of the normalize_coordinates function. It sets up a simplified hand joint data structure, a palm coordinate, and a scale factor. Then, it calls the normalize_coordinates function with these inputs and compares the result with the expected output.

test_read_json_file: This test case checks the read_json_file function. It uses the @patch decorator to mock the open function and simulate reading a JSON file. The test calls the read_json_file function with a test file name and compares the result with the expected JSON data.

test_calculate_scale_factor: This test case validates the calculate_scale_factor function. It creates a hand frame dictionary with palm and middle fingertip coordinates. The expected scale factor is calculated using the Euclidean distance formula. The test calls the calculate_scale_factor function and compares the result with the expected value.

test_aggregate_gesture_data: This test case tests the aggregate_gesture_data function. It mocks the normalize_file function using the @patch decorator and provides an empty NumPy array as the return value. The test creates a list of file paths and calls the aggregate_gesture_data function with these paths. It then checks if the result is an instance of a NumPy array.

test_get_file_paths_and_save_filtered_counts: This test case verifies the get_file_paths_and_save_filtered_counts function. It mocks the glob.glob function to return a list of file paths. The test calls the get_file_paths_and_save_filtered_counts function and checks if the result is an instance of a dictionary.

test_process_datasets: This test case tests the process_datasets function. It mocks multiple functions using the @patch decorator to control their behavior. The test sets up the mock to return a dictionary with file paths. Then, it calls the process_datasets function with base and output directories. Finally, it checks if the get_dataset_json function was called twice.
'''
import unittest
import json
import os
import numpy as np
from unittest.mock import patch, mock_open
from normalize_data import (normalize_coordinates, read_json_file, calculate_scale_factor,
                            process_frames, normalize_file, aggregate_gesture_data, 
                            get_file_paths_and_save_filtered_counts, process_datasets)

class TestNormalizationFunctions(unittest.TestCase):
    def test_normalize_coordinates(self):
        hand_joints = {
            "thumb": {
                "distal": [100, 200, 300],
                "intermediate": [150, 250, 350],
                "proximal": [200, 300, 400],
                "metacarpal": [250, 350, 450]
            },
            "index": {
                "distal": [110, 210, 310],
                "intermediate": [160, 260, 360],
                "proximal": [210, 310, 410],
                "metacarpal": [260, 360, 460]
            }
        }
        palm_coord = [100, 200, 300]
        scale_factor = 10
        expected = {
            "thumb": {
                "distal": [0, 0, 0],
                "intermediate": [5, 5, 5],
                "proximal": [10, 10, 10],
                "metacarpal": [15, 15, 15]
            },
            "index": {
                "distal": [1, 1, 1],
                "intermediate": [6, 6, 6],
                "proximal": [11, 11, 11],
                "metacarpal": [16, 16, 16]
            }
        }
        result = normalize_coordinates(hand_joints, palm_coord, scale_factor)
        self.assertEqual(result, expected)

    @patch("builtins.open", new_callable=mock_open, read_data='{"name": "test"}')
    def test_read_json_file(self, mock_file):
        result = read_json_file('test_file.json')
        self.assertEqual(result, {"name": "test"})

    def test_calculate_scale_factor(self):
        hand_frame = {'palm': [0, 0, 0], 'middle_fingertip': [3, 4, 0]}
        expected = 5  # sqrt(3^2 + 4^2)
        result = calculate_scale_factor(hand_frame['palm'], hand_frame['middle_fingertip'])
        self.assertEqual(result, expected)

    @patch("normalize_data.normalize_file", return_value=np.array([]))
    def test_aggregate_gesture_data(self, mock_normalize_file):
        file_paths = ['file1.json', 'file2.json']
        result = aggregate_gesture_data(file_paths)
        self.assertIsInstance(result, np.ndarray)

    @patch("glob.glob", return_value=['gesture1_file1.json', 'gesture1_file2.json', 'gesture2_file1.json'])
    def test_get_file_paths_and_save_filtered_counts(self, mock_glob):
        result = get_file_paths_and_save_filtered_counts()
        self.assertIsInstance(result, dict)  # Check if it returns a dictionary

    @patch("normalize_data.get_dataset_json")
    @patch("normalize_data.get_file_paths_and_save_filtered_counts1")  # Correct function name
    @patch("builtins.open", new_callable=mock_open, read_data="dummy data")
    @patch("os.path.exists", return_value=True)
    def test_process_datasets(self, mock_exists, mock_file, mock_get_files, mock_get_dataset_json):
        mock_get_files.return_value = {'gesture1': ['file1.json', 'file2.json']}
        base_dir = '/path/to/dataset'
        output_dir = '/path/to/output'
        process_datasets(base_dir, output_dir)
        self.assertEqual(mock_get_dataset_json.call_count, 2)

def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestNormalizationFunctions)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print(f"\nRun: {result.testsRun}")
    print(f"Errors: {len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Successes: {result.testsRun - len(result.errors) - len(result.failures)}")

if __name__ == '__main__':
    run_tests()