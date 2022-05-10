"""
    instruction:
    transformed code

    step:
    1. input program and target author
    2. get author's style, get program's style
    3. Start 19 kinds of conversion
    4. Compare the different styles of the program and the target author,
    and transform all the different styles into the style of the target author

"""


import os
import re
from test_transform import scan
from test_transform.get_transform import transform_13
from test_transform.py import (if_spilt, if_combine, switch_if, ternary, while_for, for_while, assign_value, assign_combine,
                re_temp, temporary_var, init_declaration,java_split_function,java_merge_function,
                var_init_split, array_to_pointer, pointer_to_array,java_import,tmp_in_name,
                static_dyn_mem, dyn_static_mem, cpp_lib_to_c, c_lib_to_cpp,
                typedef, type11_def, type_define, retypedef, const_vars, incr_opr_prepost,
                var_init_pos, var_init_merge,var_name_style,select_tmp_id_names,select_nontmp_id_names,split_function,include)
auth_name = ''
path_program = ''
change_file = 'srcml ./style/a.'+path_program.split('.')[-1]+' -o '
style_list = []


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
            list_auth = get_author_style(auth_file)
            # the program to be transformed is compared with the style of the target author
            transform_P_to_auth(list_auth, path_program)


# get author's style
def get_author_style(auth_file):
    f_auth = open(auth_file, 'r')
    data_auth = f_auth.readlines()
    # list_auth the mainstream style of every author
    list_auth = []
    for i in range(0, 16):
        dict_auth = eval(data_auth[i])
        style_1(dict_auth, list_auth)
    print("author's style："+str(list_auth))
    return list_auth


# style 2 and 3
def get_style2(author_name='', tmp=1):
    if tmp == 1:
        path_author = get_author_path(author_name)
        return select_tmp_id_names.get_vars_cnt_by_author(path_author, tmp_only=True)
    elif tmp == 2:
        path_author = get_author_path(author_name)
        return select_tmp_id_names.get_vars_cnt_by_author(path_author,tmp_only=False)


# style 12
def get_style12(path_program='', author_name = ''):
    path_author = get_author_path(author_name)
    a, b, c, auth_list_keys1, d = include.transform_include(path_program, path_author)
    a, b, c, auth_list_keys2, d = java_import.transform_include(path_program, path_author)
    return auth_list_keys1 if auth_list_keys2 == [] else auth_list_keys2


# compare program to target author and complete the transformation
def transform_P_to_auth(list_a, path_program):

    # the program's all style
    list_p = []
    # the program's main style
    list_p_main = []
    # get program's name
    if path_program.endswith('.cpp'):
        scan.program_style_flag = 'cpp'
    if path_program.endswith('.java'):
        scan.program_style_flag = 'java'
    program_name = path_program.split('/')[-1]
    scan.srcml_program_xml(path_program, './style/style')
    # when the program is converting to XML file, if it fails, the next program will be converted
    if scan.flag == False:
        return
    # get target author's path
    path_author = get_author_path(auth_name)
    print("-----------------------")
    print("the style of program is ：")
    global style_list
    style_list = scan.get_style('./style/style.xml')
    scan.srcml_xml_program('./style/style.xml', os.path.join('./style/transform', program_name))
    # get the style ratio of program
    for i in range(3, 19):
        style_2(style_list[i], list_p)
        
    for elem_p in list_p:
        str_p = [0]
        if type(elem_p) == str:
            str_p = elem_p.split('.')
        # 类别17
        if elem_p == '17.1':
            list_p_main.append('17.1')
        if elem_p == '17.2':
            list_p_main.append('17.2')
        for elem_a in list_a:
            str_a = [0]
            if type(elem_a) == str:
                str_a = elem_a.split('.')
            # transform style 1
            if str_p[0] == '1' and str_p[0] == str_a[0] and str_p[1] != str_a[1]:
                scan.srcml_program_xml(os.path.join('./style/transform', program_name), './style/style')
                var_name_style.program_transform('./style/style.xml', elem_p, elem_a)
                scan.srcml_xml_program('./style/style.xml', os.path.join('./style/transform', program_name))
                list_p_main.append(str_p[0])
            # store style 4
            elif elem_p == '4.1' or elem_p == '4.2':
                if elem_a == '4.1' and elem_a == elem_p:
                    list_p_main.append(elem_p)
            # transform style 9
            elif str_p[0] == '9' and str_p[0] == str_a[0] and str_p[1] != str_a[1]:
                scan.srcml_program_xml(os.path.join('./style/transform', program_name), './style/style')
                incr_opr_prepost.program_transform('./style/style.xml', elem_p, elem_a)
                scan.srcml_xml_program('./style/style.xml', os.path.join('./style/transform', program_name))
                list_p_main.append(str_p[0])
            # transform style 19
            elif type(elem_p) == dict and type(elem_a) == dict:
                if abs(elem_p['19'][0] - elem_a['19'][0]) >= 10:
                    list_p_main.append('19')
                    scan.srcml_program_xml(os.path.join('./style/transform', program_name), './style/style')
                    if program_name.endswith('.java'):
                        java_split_function.transform_by_line_cnt('./style/style.xml', path_author,
                                                         srccode_path=os.path.join('./style/transform', program_name),
                                                         save_to='./style/transform.xml')

                    if program_name.endswith('.cpp') or program_name.endswith('.c'):
                        split_function.transform_by_line_cnt('./style/style.xml', path_author,
                                                         srccode_path=os.path.join('./style/transform', program_name),
                                                         save_to='./style/transform.xml')
                    scan.srcml_xml_program('./style/transform.xml', os.path.join('./style/transform', program_name))
            # store style 10
            elif str_p[0] == '10' and str_a[0] == '10' and str_p[1] == str_a[1] =='1':
                list_p_main.append('10.2')
            # store style 5 6 7 8 11 13 14 15 16 18
            elif str_p[0] == str_a[0] and str_p[1] != str_a[1]:
                list_p_main.append(elem_p)
            # store style 17
            if elem_a == '17.1':
                if '17.1' in list_p_main:
                    list_p_main.remove('17.1')
            if elem_a == '17.2':
                if '17.2' in list_p_main:
                    list_p_main.remove('17.2')
    # transform style 13.3
    if '13.1' in list_p and '13.1' not in list_a and '13.2' not in list_a:
        cpp_lib_to_c.program_transform('./style/style.xml','13.1','13.3')
    elif '13.2' in list_p and '13.1' not in list_a and '13.2' not in list_a:
        cpp_lib_to_c.program_transform('./style/style.xml', '13.2', '13.3')
    elif '13.1' in list_a and '13.1' not in list_p and '13.2' not in list_p:
        cpp_lib_to_c.program_transform('./style/style.xml', '13.3', '13.1')
    elif '13.2' in list_a and '13.1' not in list_p and '13.2' not in list_p:
        cpp_lib_to_c.program_transform('./style/style.xml', '13.3', '13.2')
    # transform style 12
    scan.srcml_program_xml(os.path.join('./style/transform', program_name), './style/style')
    a_12, b_12, c_12, auth_list_keys, pro_list_key= include.transform_include('./style/style.xml', path_author)
    c_12 = 1
    a_12_java,b_12_java,c_12_java,auth_list_keys_java,pro_list_key_java = java_import.transform_include('./style/style.xml', path_author)
    
    # style 12:C++/c
    if scan.program_style_flag == 'cpp' or scan.program_style_flag == 'c':
        list_p_main.append('12.1')
        style_list.append(pro_list_key)

    # style 12:java
    if scan.program_style_flag == 'java':
        list_p_main.append('12.2')
        style_list.append(pro_list_key_java)

    # store style 2
    list_p_main.append('2')
    # store style 3
    list_p_main.append('3')

    if len(list_p_main) > 0:
        py_list = get_eval(list_p_main)
        for elem in py_list:
            # style 1 already transformed,so just skip it
            if elem == 'var_name_style':
                continue
            # transform style 2
            elif elem == 'select_tmp_id_names':
                scan.srcml_program_xml(os.path.join('./style/transform', program_name), './style/style')
                if select_tmp_id_names.is_transformable('./style/style.xml', path_author):
                    style_list.append(select_tmp_id_names.get_vars_cnt_by_author('./style/style.xml',tmp_only=True))
                    select_tmp_id_names.transform_tmp_id_names('./style/style.xml', path_author, ignore_list= [], save_to= './style/transform.xml')
                    scan.srcml_xml_program('./style/transform.xml', os.path.join('./style/transform', program_name))
                else:
                    list_p_main.remove('2')
            # transform style 3
            elif elem == 'select_nontmp_id_names':
                if select_nontmp_id_names.is_transformable('./style/style.xml', path_author):
                    scan.srcml_program_xml(os.path.join('./style/transform', program_name), './style/style')
                    style_list.append(select_tmp_id_names.get_vars_cnt_by_author('./style/style.xml',tmp_only=False))
                    select_nontmp_id_names.transform_nontmp_id_names('./style/style.xml', path_author, save_to='./style/transform.xml')
                    scan.srcml_xml_program('./style/transform.xml', os.path.join('./style/transform', program_name))
                else:
                    list_p_main.remove('3')
            # transform style 4
            elif elem == 'const_vars':
                scan.srcml_program_xml(os.path.join('./style/transform', program_name), './style/style')
                eval(elem).program_transform('./style/style.xml', path_author)
                scan.srcml_xml_program('./style/style.xml', os.path.join('./style/transform', program_name))
            # style 9 already transformed,so just skip it
            elif elem == 'incr_opr_prepost':
                continue
            # transform style 10
            elif elem == 'retypedef':
                retypedef.ls.clear()
                scan.srcml_program_xml(os.path.join('./style/transform', program_name), './style/style')
                eval(elem).program_transform('./style/style.xml', path_author)
                scan.srcml_xml_program('./style/style.xml', os.path.join('./style/transform', program_name))
            # transform style 11
            elif elem == 'type11_def':
                scan.srcml_program_xml(os.path.join('./style/transform', program_name), './style/style')
                eval(elem).program_transform('./style/style.xml', path_author)
                scan.srcml_xml_program('./style/style.xml', os.path.join('./style/transform', program_name))
            # transform style 12 c++
            elif elem == 'include':
                scan.srcml_program_xml(os.path.join('./style/transform', program_name), './style/style')
                eval(elem).program_transform('./style/style.xml', path_author)
                scan.srcml_xml_program('./style/style.xml', os.path.join('./style/transform', program_name))
            # transform style 12 java
            elif elem == 'java_import':
                scan.srcml_program_xml(os.path.join('./style/transform', program_name), './style/style')
                eval(elem).program_transform('./style/style.xml', path_author)
                scan.srcml_xml_program('./style/style.xml', os.path.join('./style/transform', program_name))
            # style 19 already transformed,so just skip it
            elif elem == 'split_function':
                continue
            # transform style 5 6 7 8 13 14 15 16 18
            else:
                scan.srcml_program_xml(os.path.join('./style/transform', program_name), './style/style')
                eval(elem).program_transform('./style/style.xml')
                scan.srcml_xml_program('./style/style.xml', os.path.join('./style/transform', program_name))
        # transform style 2 ：the name of a temporary variable defined in a for statement
        scan.srcml_program_xml(os.path.join('./style/transform', program_name), './style/style')
        tmp_in_name.trans_tree('./style/style.xml', path_author)
        scan.srcml_xml_program('./style/style.xml', os.path.join('./style/transform', program_name))
        if len(list_p_main) != 0:
            print("The style to be transformed is：")
            print(list_p_main)
            print("-----------------------")
            print("Successful transformation！！")
            return list_p_main
    if len(list_p_main) == 0:
        print("-----------------------")
        print("This program has no style can be converted!!")
        return None


# get the author's path
def get_author_path(pre_auth_name):
    path = './xml_file'
    for root, sub_dirs, files in os.walk(path):
        for sub_dir in sub_dirs:
            if pre_auth_name == sub_dir:
                return os.path.join(path, sub_dir)


# get percent of style
def get_percent(style_list):
    for i in range(3, 18):
        dict_style = style_list[i]
        sum1 = sum(dict_style.values())
        for key in dict_style:
            if sum1 != 0:
                dict_style[key] = round((dict_style[key] / sum1) * 100, 1)


# get the main style of the program
def style_2(dict, list):
    for key in dict:
        # style 17
        if key == '17.1' or key == '17.2':
            if dict[key] > 0:
                list.append(key)
        elif key == '19':
            list.append({'19':dict[key]})
        elif len(dict) == 2:
            if dict[key] > 0:
                list.append(key)
        else:
            if dict[key] > 0 and dict[key] == max(dict.values()):
                list.append(key)


#  the mainstream style of every author
def style_1(dict, list):
    for key in dict:
        # style 17
        if key == '17.1' or key == '17.2':
            if dict[key] > 0:
                list.append(key)
        # style 19
        elif key == '19':
            list.append({'19': dict[key]})
        elif len(dict) == 2:
            if dict[key] >= 70:
                list.append(key)
        elif len(dict) == 4:
            if dict[key] >= 50:
                list.append(key)
        elif len(dict) == 5:
            if dict[key] >= 50:
                list.append(key)


# get style's object
def get_eval(list_p):
    py_list = []
    # one Python object for each style
    f = open('./style/style.txt', 'r')
    style_list = eval(f.readline())
    for key in list_p:
        py_list.append((style_list[key]))
    return py_list


def start_trans(program_path, author_name):
    # scan each author's TXT file
    style_path = './author_style'
    global auth_name, path_program
    path_program = program_path
    # auth_name: express the author of this program
    auth_name = author_name
    # start to convert
    get_input(path_program, auth_name, style_path)
