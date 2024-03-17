import os
import sys
"""
This module is used to add the root directory to the sys.path.
Useful to import a module from a branch of directories into another branch of directories.
"""


def add_root(nested_level=0):
    """
    Add the path of the root directory to the sys.path
    """

    # Initialize the parent_path with the current script's directory
    parent_path = os.path.dirname(__file__)

    # Go up the directory hierarchy based on nested_level
    for _ in range(nested_level):
        parent_path = os.path.dirname(parent_path)

    if not parent_path in sys.path:
        sys.path.append(parent_path)
        print("Added to the sys.path:", parent_path)



def get_root_path(nested_level=0):
    """
    Get the path of the root directory
    """
    
    # Initialize the parent_path with the current script's directory
    parent_path = os.path.dirname(__file__)

    # Go up the directory hierarchy based on nested_level
    for _ in range(nested_level):
        parent_path = os.path.dirname(parent_path)

    return parent_path