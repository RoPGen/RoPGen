import os
import sys


def delete_successful_files(test, precess_path):
    for root, dirs, files in os.walk(test):
        for dir in dirs:
            path = os.path.join(test, dir)
            for root1, dirs1, files1 in os.walk(test):
                for file1 in files1:
                    file = file1.split('.')[0]
                    for root2, dirs2, files2 in os.walk(os.path.join(precess_path, dir)):
                        for file2 in files2:
                            if file == file2.split('##')[0]:
                                print("successful deleted: ", os.path.join(root, file))
                                os.remove(os.path.join(path, file1))
    delete_empty(test)


def delete_empty(large_path):
    for root, dirs, files in os.walk(large_path):
        for dir in dirs:
            path = os.path.join(large_path,dir)
            for root, dirs, files in os.walk(path):
                length_file = len(files)
                if length_file == 0:
                    os.rmdir(os.path.join(path))


def del_duplicate_file(small_path):
    for root, dirs, files in os.walk(small_path):
        if len(files) != 0:
            file_correct = []
            file_correct.append(files[0].split('##')[0])
            for file in files[1:]:
                if file.split('##')[0] in file_correct:
                    print("successful deleted: ", os.path.join(root, file))
                    os.remove(os.path.join(root, file))
                else:
                    file_correct.append(file.split('##')[0])


if __name__ == '__main__':
    test = '../program_file/test'
    precess_path = './data_processing'
    flag = sys.argv[1]
    if flag == '1':
        # delete successful test sets
        delete_successful_files(test, precess_path)
    elif flag == '2':
        # delete duplicate files
        del_duplicate_file(precess_path)
