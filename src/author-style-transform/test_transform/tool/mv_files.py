"""
input: input txt file,Txt file each line is the author's name,
output: './result'

"""
import sys
import del_dot

def mv_file(path1,path2):
    with open(path1,'r') as f:
        for line in f.readlines():
            file_name = line.strip().rstrip('\'').lstrip('\'')
            if file_name == '':continue
            command = 'mv ./test/'+file_name+' ./result'
            del_dot.cmd(command)


if __name__ == '__main__':
    path1 = sys.argv[1]
    path2 = './test'
    mv_file(path1, path2)
