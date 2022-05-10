"""
Divide the data set into ten parts
    input:'./test' original data set
    output:'./test' data set after partition
"""

import del_dot
import os


def separate_10(path1, path2):
    for auts_files in os.listdir(path1):
        auts_path = os.path.join(path2,auts_files)
        if not os.path.exists(auts_path):
            os.mkdir(auts_path)
        files_path = os.path.join(path1,auts_files)
        num1 = int(len(os.listdir(files_path))/10)
        a = len(os.listdir(files_path))/10
        num2 = int(a*10-num1*10)
        for i in range(1, num2+1):
            num_path = os.path.join(auts_path, str(i))
            if not os.path.exists(num_path):
                os.mkdir(num_path)
            cout = 0
            for file in os.listdir(files_path):
                if cout == num1 + 1:break
                command = 'mv '+os.path.join(files_path,file)+' '+os.path.join(num_path,file)
                del_dot.cmd(command)
                print(command)
                cout += 1
        for j in range(num2+1, 11):
            num_path = os.path.join(auts_path, str(j))
            if not os.path.exists(num_path):
                os.mkdir(num_path)
            cout = 0
            for file in os.listdir(files_path):
                if cout == num1: break
                command = 'mv '+os.path.join(files_path, file)+' '+os.path.join(num_path,file)
                del_dot.cmd(command)
                print(command)
                cout+=1


def separate_8(path1,path2):
    for auts_files in os.listdir(path1):
        auts_path = os.path.join(path2,auts_files)
        if not os.path.exists(auts_path):
            os.mkdir(auts_path)
        files_path = os.path.join(path1,auts_files)
        a = len(os.listdir(files_path))/10
        for i in range(1,9):
            num_path = os.path.join(auts_path,str(i))
            if not os.path.exists(num_path):
                os.mkdir(num_path)
            file_list = os.listdir(files_path)
            file_list.sort(key=lambda x:int(x.split('_')[0]+x.split('_')[1]))
            for file in file_list:
                command = 'mv '+os.path.join(files_path,file)+' '+os.path.join(num_path,file)
                del_dot.cmd(command)
                print(command)
                break


def combine(path1,path2):
    del_dot.cmd('rm -rf ./'+path2+'/*')
    for i in range(1, 11):
        test = 'test_'+str(i)
        train = 'train_'+str(i)
        for auts_files in os.listdir(path1):
            test_path = os.path.join(path2,test)
            train_path = os.path.join(path2,train)
            if not os.path.exists(test_path):
                os.mkdir(test_path)
            if not os.path.exists(train_path):
                os.mkdir(train_path)
            aur_test_path = os.path.join(test_path,auts_files)
            aur_train_path = os.path.join(train_path,auts_files)
            if not os.path.exists(aur_test_path):
                os.mkdir(aur_test_path)
            if not os.path.exists(aur_train_path):
                os.mkdir(aur_train_path)
            auts_path = os.path.join(path1,auts_files)
            for num_dst in os.listdir(auts_path):
                if str(num_dst) == str(i):
                    command = 'cp '+os.path.join(auts_path,num_dst)+'/* '+aur_test_path
                    del_dot.cmd(command)
                else:
                    command = 'cp '+os.path.join(auts_path,num_dst)+'/* '+aur_train_path
                    del_dot.cmd(command)


if __name__ == '__main__':
    path1 = './test'
    path2 = './train'
    separate_10(path1, path2)
    combine(path2, path1)
