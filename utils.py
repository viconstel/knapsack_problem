import os
import csv
import numpy as np


def test_files_generator(dir_path):
    for file in os.listdir(dir_path):
        if file.startswith('test_'):
            data = []
            answers = []
            with open(os.path.join(dir_path, file)) as fin:
                file_obj = csv.reader(fin, delimiter=';')
                for row in file_obj:
                    data.append(list(map(float, row)))

            num_tests = file.split('_')[-1].split('.')[0]
            res_file = 'bpresult_' + num_tests + '.csv'
            if res_file in os.listdir(dir_path):
                with open(os.path.join(dir_path, res_file)) as fin:
                    file_obj = csv.reader(fin, delimiter=';')
                    answers = list(map(int, next(file_obj)))

            data = np.array([np.array(i) for i in data])
            answers = np.array(answers)
            yield file, data, answers