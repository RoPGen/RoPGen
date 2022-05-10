"""

    instruction:
    The original program of each author is transformed into an XML file,
    and the proportion of each type of each author is calculated

    input:'./program_file/target_author_file':target author program
    output:'./author_style':author style
            './xml_file':author program's xml file

    step:
    1. convert source program to XML file
    2. get author's style
"""
import os
import subprocess
import re
import sys

cur_path = os.path.abspath('.')
up_path = os.path.dirname(cur_path)
sys.path.append(up_path)
sys.path.append(cur_path)


from test_transform import transform
from test_transform.get_transform import (transform_1, transform_4, transform_5, transform_6, transform_7, transform_8, transform_9,
                           transform_10, transform_11, transform_13, transform_14, transform_15,
                           transform_16, transform_17, transform_18,transform_19)
flag = True  # indicates whether the shell command runs successfully
program_style_flag = 'c'  # program's style c++/java/c


# express shell command
def cmd(command):
    global flag
    flag = True
    subp = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
    subp.wait(10)
    if subp.poll() == 0:
        flag = True
    else:
        print("False!")
        flag = False


# convert original program to xml file
def program_to_xml(pre_file, xml_file):
    global program_style_flag
    for root, sub_dirs, file in os.walk(pre_file):
        for sub_dir in sub_dirs:
            pre_java = os.path.join(root, sub_dir)
            # create the corresponding XML file directory
            if not os.path.exists(os.path.join(xml_file, sub_dir)):
                os.mkdir(os.path.join(xml_file, sub_dir))
            for root1, sub_dirs1, file1 in os.walk(pre_java):
                for file1_elem in file1:
                    # get program name (without suffix)
                    name = re.findall(r'(.+?)\.', file1_elem)
                    srcml_program_xml(os.path.join(pre_java, file1_elem), os.path.join(xml_file, sub_dir, name[0]))
                    if flag is False:
                        continue


# convert program to xml
def srcml_program_xml(pre_path, xml_path):
    str = 'srcml \"'+ pre_path +'\" -o \"'+ xml_path +'.xml\" --position --src-encoding UTF-8'
    cmd(str)


# convert xml to program
def srcml_xml_program(pre_path, xml_path):
    str = "srcml \""+ pre_path +'\" -o \"'+ xml_path +"\" --src-encoding UTF-8"
    cmd(str)


# get all author's style and store them to './author_style'
def get_xml(xml_file):
    for root, sub_dirs, files in os.walk(xml_file):
        if len(sub_dirs) == 0:
            break
        for sub_dir in sub_dirs:
            #############################################
            list_4 = {'1.1': 0, '1.2': 0, '1.3': 0, '1.4': 0, '1.5': 0}
            list_5 = {'4.1': 0, '4.2': 0}
            list_6 = {'5.1': 0, '5.2': 0}
            list_7 = {'6.1': 0, '6.2': 0}
            list_8 = {'7.1': 0, '7.2': 0}
            list_9 = {'8.1': 0, '8.2': 0}
            list_10 = {'9.1': 0, '9.2': 0, '9.3': 0, '9.4': 0}
            list_11 = {'10.1': 0, '10.2': 0}
            list_12 = {'11.1': 0, '11.2': 0}
            list_13 = {'13.1': 0, '13.2': 0}
            list_14 = {'14.1': 0, '14.2': 0}
            list_15 = {'15.1': 0, '15.2': 0}
            list_16 = {'16.1': 0, '16.2': 0}
            list_17 = {'17.1': 0, '17.2': 0}
            list_18 = {'18.1': 0, '18.2': 0}
            list_19 = {'19': [0, 0]}
            num_19 = 0
            # create author style file
            f = open(os.path.join('./author_style', sub_dir + '.txt'), 'w')
            for root1, sub_dirs1, files1 in os.walk(os.path.join(root, sub_dir)):
                if len(files1) == 0:
                    break
                # traverse each XML file
                for file1 in files1:
                    # style 19 need to calculate the number of xml files
                    num_19 += 1
                    # get the path of each XML file
                    xml_file_path = os.path.join(root, sub_dir, file1)
                    # calculate every author's style ratio
                    style_list = get_style(xml_file_path)

                    for i in range(3, 19):
                        dict_style = style_list[i]
                        # style 19
                        if i == 18:
                            for j in range(0, 2):
                                list_19['19'][j] += dict_style['19'][j]
                            continue
                        for key in dict_style:
                            eval('list_'+str(i+1))[key] += dict_style[key]
                #
                list_19['19'][0] = round(list_19['19'][0]/num_19, 1)
                list_19['19'][1] = round(list_19['19'][1] / num_19, 1)
                # calculate ratio

                for i in range(3, 18):
                    dict_style = eval('list_' + str(i + 1))
                    sum1 = sum(dict_style.values())
                    for key in dict_style:
                        if sum1 != 0:
                            # style 4
                            if key == '4.1' and dict_style[key] > 0:
                                dict_style[key] = 100.0
                                dict_style['4.2'] = 0.0
                                break
                            # style 10
                            elif key == '10.1' and dict_style[key] > 0:
                                dict_style[key] = 100.0
                                dict_style['10.2'] = 0.0
                                break
                            # style 11
                            elif key == '11.1' and dict_style[key] > 0:
                                dict_style[key] = 100.0
                                dict_style['11.2'] = 0.0
                                break
                            else:
                                dict_style[key] = round((dict_style[key] / sum1) * 100, 1)
                print("----------------")
                print("Author " + sub_dir + " proportion of all programs:")
                #############################################
                for i in range(3, 19):
                    print(eval('list_'+str(i+1)))
            # count the type proportion of each author
            # style 2
            list_20 = transform.get_style2(author_name=sub_dir, tmp=1)
            # style 3
            list_21 = transform.get_style2(author_name=sub_dir, tmp=2)
            # style 12
            list_22 = transform.get_style12(author_name=sub_dir)
            for i in range(3, 22):
                list_elem = eval('list_'+str(i+1))
                f.write(str(list_elem)+'\n')
            f.close()


# calculate every author's style ratio
def get_style(xml_file_path):
    global program_style_flag
    style_list = [0, 0, 0]
    # 1
    doc = eval('transform_1')
    style_list.append(doc.get_program_style(xml_file_path))
    # 4
    doc = eval('transform_4')
    style_list.append(doc.get_program_style(xml_file_path))
    # 5
    doc = eval('transform_5')
    style_list.append(doc.get_program_style(xml_file_path))
    # 6
    doc = eval('transform_6')
    style_list.append(doc.get_program_style(xml_file_path))
    # 7
    doc = eval('transform_7')
    style_list.append(doc.get_program_style(xml_file_path))
    # 8
    if program_style_flag == 'cpp' or program_style_flag == 'c':
        doc = eval('transform_8')
        style_list.append(doc.get_program_style(xml_file_path))
    else:
        style_list.append({'8.1': 0, '8.2': 0})
    # 9
    doc = eval('transform_9')
    style_list.append(doc.get_program_style(xml_file_path))
    # 10
    if program_style_flag == 'cpp' or program_style_flag == 'c':
        doc = eval('transform_10')
        style_list.append(doc.get_program_style(xml_file_path))
    else:
        style_list.append({'10.1': 0, '10.2': 0})
    # 11
    if program_style_flag == 'cpp' or program_style_flag == 'c':
        doc = eval('transform_11')
        style_list.append(doc.get_program_style(xml_file_path))
    else:
        style_list.append({'11.1': 0, '11.2': 0})
    for i in range(13, 20):
        if i == 13 and program_style_flag == 'java':
            style_list.append({'13.1': 0, '13.2': 0})
            continue
        if i == 14 and program_style_flag == 'java':
            style_list.append({'14.1': 0, '14.2': 0})
            continue
        if i == 19 and program_style_flag == 'java':
            style_list.append(transform_19.get_program_style_java(xml_file_path))
            continue
        doc = eval('transform_'+str(i))
        style_list.append(doc.get_program_style(xml_file_path))
    return style_list


# the project's input port
if __name__ == '__main__':
    # all untransformed original program
    pre_file = "program_file/target_author_file"
    # store XML file of all source programs
    xml_file = "./xml_file"
    # convert source program to XML file
    program_to_xml(pre_file, xml_file)
    # get all author's style and store them to './author_style'
    get_xml(xml_file)
