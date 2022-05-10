"""
    instruction:
    Start directional transformation
    This is the project's second step

    input:'./program_file/test'
    output:'./program_file/directional_file'

    step:
    1. get into test_transform directory,run 'python directional_transform.py'
"""
import os
import sys
import traceback
import random
cur_path = os.path.abspath('.')
up_path = os.path.dirname(cur_path)
sys.path.append(up_path)
sys.path.append(cur_path)
from test_transform import transform
from test_transform import scan


# start to transform
def scan_style(program_path, author_path):
    # every program starts to transform
    for root, sub_dirs, files in os.walk(program_path):
        for sub_dir in sub_dirs:
            # each author's program path to be transformed
            sub_path = os.path.join(root, sub_dir)
            file_list = os.listdir(sub_path) if os.path.isdir(sub_path) else [sub_path]
            for file_name in file_list:
                # every program's path to be transformed
                file_path = os.path.join(root, sub_dir, file_name)
                author_list = os.listdir(author_path) if os.path.isdir(author_path) else [author_path]
                author_list = random.sample(author_list, 100)
                print(author_list)
                # convert to all target authors
                for author_name in author_list:
                    if not author_name.endswith('.txt'):
                        continue
                    author_name = author_name.split('.')[0]
                    # if the names of two authors are the same, it means the same author without conversion
                    if sub_dir == author_name:
                        continue
                    try:
                        # start to transform
                        transform.start_trans(file_path, author_name)
                    except:
                        error_file = open('./style/error.txt', 'a')
                        error_file.write(traceback.format_exc()+'\n\n')
                        error_file.close()
                        scan.cmd('rm -rf ./style/transform/*')
                        continue
                    # move each converted program to the final directory
                    move_change(file_name, sub_dir, author_name)


# move each converted program to the final directory
def move_change(file_name, pre_author_name, author_name):
    path = './program_file/directional_file/'+author_name
    program_name = ''
    program_name = file_name.split('.')[0] + '##' + pre_author_name + '###' + author_name + \
                   '.' + file_name.split('.')[-1]
    change_name = 'mv ' + path + '/' + file_name + ' ' + path + '/' + program_name
    if not os.path.exists(path):
        os.mkdir(path)
    scan.cmd(command='mv ./style/transform/* '+path)
    scan.cmd(change_name)


if __name__ == '__main__':
    # author program to be transformed
    program_path = './program_file/test'
    # author's style path
    author_path = './author_style'
    # save path after transformation
    transform_file = 'program_file/directional_file'
    # start to transform
    scan_style(program_path, author_path)
