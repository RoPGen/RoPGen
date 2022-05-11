"""
    1. Input: input program and target author
    2. Output: get author's coding style
    3. Start 23 types of transformations
    4. Compare the different styles of the program and the target author,
    and transform all the different styles into the style of the target author

"""


import os
import re
from transform import get_style
from transform.get_transform import (transform_1, transform_2, transform_3, transform_4, transform_5, transform_6, transform_7, transform_8, transform_9,
                                     transform_10, transform_11, transform_12, transform_13, transform_14, transform_15, transform_16, transform_17, transform_18, transform_19,
                                     transform_20, transform_21, transform_22, transform_23)

auth_name = ''
path_program = ''
change_file = 'srcml ./style/a.'+path_program.split('.')[-1]+' -o '
program_styles = []


# start to convert
def get_input(path_program, auth_name, style_path):
    print("-----------------------")
    print("author's name：", auth_name)
    print("program's name: ", path_program.split('/')[-1].split('.')[0])
    print("-----------------------")

    for root, sub_dirs, files in os.walk(style_path):
        auth_file = ''
        for file in files:
            name = re.findall(r'(.+?)\.', file)[0]
            if auth_name == name:
                auth_file = os.path.join(root, file)
        # get author's style
        if auth_file != '':
            tar_auth_styles = get_author_style(auth_file)
            # the program to be transformed is compared with the style of the target author
            transform_P_to_auth(tar_auth_styles, path_program)



# get author's style
def get_author_style(auth_file):
    f_auth = open(auth_file, 'r')
    data_auth = f_auth.readlines()
    # list_auth the mainstream style of every author
    list_auth = []
    for i in range(0, 23):
        dict_auth = eval(data_auth[i])
        for key in dict_auth:
            # style 21
            if key == '21.1' or key == '21.2':
                if dict_auth[key] > 0:
                    list_auth.append(key)
            # style 23
            elif key == '23':
                list_auth.append({'23': dict_auth[key]})
            elif len(dict_auth) == 2:
                if dict_auth[key] >= 70:
                    list_auth.append(key)
            elif len(dict_auth) == 4:
                if dict_auth[key] >= 50:
                    list_auth.append(key)
            elif len(dict_auth) == 5:
                if dict_auth[key] >= 50:
                    list_auth.append(key)
    print("author's style："+str(list_auth))
    return list_auth


# get the author's path
def get_author_path(pre_auth_name):
    path = './xml_file'
    for root, sub_dirs, files in os.walk(path):
        for sub_dir in sub_dirs:
            if pre_auth_name == sub_dir:
                return os.path.join(path, sub_dir)

# compare program to target author and complete the transformation
def transform_P_to_auth(tar_auth_styles, path_program):

    #
    # the style to be converted
    converted_styles = []
    # get program's name
    if path_program.endswith('.cpp'):
        get_style.file_type = 'cpp'
    if path_program.endswith('.java'):
        get_style.file_type = 'java'
    program_name = path_program.split('/')[-1]
    if len(os.listdir('./style/transform')) > 0:
        get_style.cmd('rm -rf ./style/transform/*')
    get_style.srcml_program_xml(path_program, './style/style')
    # when the program is converting to XML file, if it fails, the next program will be converted
    if get_style.flag == False:
        return
    # get target author's path
    path_author = get_author_path(auth_name)
    print("-----------------------")
    print("the style of program is ：")
    global program_styles
    program_styles = get_style.get_style('./style/style.xml')
    get_style.srcml_xml_program('./style/style.xml', os.path.join('./style/transform', program_name))
    # begin to transform to the style of the target author
    for auth_style in tar_auth_styles:
        for program_style in program_styles[1:]:
            # style 23
            if type(auth_style) == dict and '23' in program_style.keys():
                get_style.srcml_program_xml(os.path.join('./style/transform', program_name), './style/style')
                transform_23.check_transform(auth_style, program_style, './style/style.xml', path_author, program_name, converted_styles)
                get_style.srcml_xml_program('./style/transform.xml', os.path.join('./style/transform', program_name))
            elif type(auth_style) != dict and auth_style in program_style.keys():
                # for other classes, check whether conversion is required
                get_style.srcml_program_xml(os.path.join('./style/transform', program_name), './style/style')
                eval('transform_'+auth_style.split('.')[0]).check_transform(auth_style, program_style, './style/style.xml', path_author, converted_styles)
                get_style.srcml_xml_program('./style/style.xml', os.path.join('./style/transform', program_name))
    
    if len(converted_styles) != 0:
        print("The style to be transformed is:")
        print(converted_styles)
        print("-----------------------")
        print("Successful transformation!!")
    else:
        print("-----------------------")
        print("This program has no style can be converted!!")
        return None
    
    get_style.cmd('rm -rf ./style/style.xml ./style/transform.bak ./style/transform.xml')
    

def start_trans(program_path, author_name):
    # scan each author's TXT file
    style_path = './author_style'
    global auth_name, path_program
    path_program = program_path
    # auth_name: express the author of this program
    auth_name = author_name
    # start to convert
    get_input(path_program, auth_name, style_path)
