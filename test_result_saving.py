#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script to verify result saving functionality
"""

import os
from result_utils import create_test_result_folder

def test_folder_creation():
    """Test if test result folder is created correctly"""
    print("Testing test result folder creation...")
    print("=" * 80)
    
    # Test folder creation
    base_path = './saved_models'
    dataset_name = 'IEMOCAP'
    
    test_result_path = create_test_result_folder(base_path, dataset_name)
    
    print(f"\nCreated folder: {test_result_path}")
    print(f"Folder exists: {os.path.exists(test_result_path)}")
    print(f"Is directory: {os.path.isdir(test_result_path)}")
    
    # Test file saving
    print("\n" + "=" * 80)
    print("Testing file saving to the folder...")
    print("=" * 80)
    
    # Create a test file
    test_file_path = os.path.join(test_result_path, 'test_file.txt')
    with open(test_file_path, 'w') as f:
        f.write("This is a test file.\n")
    
    print(f"\nTest file created: {test_file_path}")
    print(f"File exists: {os.path.exists(test_file_path)}")
    
    # List all files in the folder
    print("\n" + "=" * 80)
    print("Files in the test result folder:")
    print("=" * 80)
    
    if os.path.exists(test_result_path):
        files = os.listdir(test_result_path)
        if files:
            for f in files:
                print(f"  - {f}")
        else:
            print("  (empty)")
    else:
        print("  Folder does not exist!")
    
    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)
    
    return test_result_path


if __name__ == '__main__':
    test_folder_creation()

