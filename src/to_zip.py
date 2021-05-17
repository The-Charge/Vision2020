"""Simple utility to create a zip file from the vision module."""


import shutil


if __name__ == '__main__':
    shutil.make_archive('vision', 'zip', '../src', 'vision')
