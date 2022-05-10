"""
Count the number of rows per program
    input: './test'
    output: './result'
"""

import os


def line_cout(path):
    for root, sub_dirs, files in os.walk(path):
        sum_lines = 0
        for sub_dir in sub_dirs:
            autline_cout = 0
            f = open('./result/'+sub_dir+'.txt', 'w')
            file_list = os.listdir(os.path.join(root, sub_dir))
            for file in file_list:
                cout = 0
                for file_line in open(os.path.join(root,sub_dir,file), 'rb').readlines():
                    if file_line != '' and file_line != '\n':
                        cout += 1
                autline_cout += cout
                f.write(str(file)+': '+str(cout)+'\n')
            avg = round(autline_cout/len(file_list), 1)
            f.write(str(avg)+'\n')
            sum_lines += avg
        f = open('./result/sum.text', 'w')

        f.write(str(round(sum_lines/len(sub_dirs), 1))+'\n')
        break


if __name__ == '__main__':
    path = './test'
    line_cout(path)