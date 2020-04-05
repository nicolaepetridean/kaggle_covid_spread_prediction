import os, glob
import re


def get_files_for_indexing(folder):
    if not os.path.exists(folder):
        print('could not find {} . Please check the provided path exists!'.format(folder))
        return

    files = []
    for f in glob.glob(os.path.join(folder, '**/*.csv'), recursive=True):
        values = {}
        f = os.path.normpath(f)
        file_name = f.split(os.sep)[-1]
        values['file_id'] = file_name
        values['file_path'] = f
        files.append(values)
    return files