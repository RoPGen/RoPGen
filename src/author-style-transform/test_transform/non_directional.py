"""
    instruction:
    Start non directional transformation
    This is the project's second step

    input:'./program_file/test'
    output:'./program_file/nondirectional_file'

    step:
    1. run scan_style(program_path, author_flag='1')  find the appropriate target author,leave the non directional
    success program after model testing
    2. put the non directed attack success in the './tool/data_precessing'
    3. run deal_non_dir.py:delete_successful_files()
    4. run scan_style(program_path, author_flag='all') transform to all target authors,leave the non directional
    success program after model testing
    5. put the non directed attack success in the './tool/data_precessing'
    6. run deal_non_dir.py:del_duplicate_file() each program has more than one target author. If the attack is
    successful,only one needs to be left, and all others need to be deleted
"""
import os
import random
import sys
cur_path = os.path.abspath('.')
up_path = os.path.dirname(cur_path)
sys.path.append(up_path)
sys.path.append(cur_path)
from test_transform.py import (if_spilt, if_combine, switch_if, ternary, while_for, for_while, assign_value, assign_combine,
                re_temp, temporary_var, init_declaration, java_split_function, java_merge_function,
                var_init_split, array_to_pointer, pointer_to_array, java_import, tmp_in_name,
                static_dyn_mem, dyn_static_mem, cpp_lib_to_c, c_lib_to_cpp,
                typedef, type11_def, type_define, retypedef, const_vars, incr_opr_prepost,
                var_init_pos, var_init_merge,var_name_style, select_tmp_id_names, select_nontmp_id_names, split_function,include)
from test_transform import transform, directional_transform, scan
xml_path = './style/style'  # the path for each program to transform XML
style_path = './author_style'  # every author's style
style_list = []   # every program's all style
program_name = ''  # target author's name
list_p = []   # the style that each program needs to transform
tmp_var_sums = None  # calculate the proportion of variable and class names for style 2 and style 3


# start to transform
def scan_style(program_path, author_flag='1'):
    # delete
    # transform_path = 'program_file/program_style'
    # f_type = open(transform_path+'/style.txt','w')
    # # 记录时间
    # f_type.write("开始时间："+str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+'\n\n\n\n'))
    # every program starts to transform
    for root, sub_dirs, files in os.walk(program_path):
        for sub_dir in sub_dirs:
            # each author's program path to be transformed
            sub_path = os.path.join(root, sub_dir)
            file_list = os.listdir(sub_path) if os.path.isdir(sub_path) else [sub_path]
            for file_name in file_list:
                # every program's path to be transformed
                file_path = os.path.join(root, sub_dir, file_name)
                # input,start to transform
                # return: target author
                get_input(file_path, file_name, sub_dir, author_flag)


# move each converted program to the final directory
def move_change(file_name, sub_dir, dst_aut):
    path = os.path.join(transform_file, sub_dir)
    if not os.path.exists(path):
        os.mkdir(path)
    change_name = 'mv ./style/transform/* ./style/transform/'+file_name.split('.')[0]+'##'+sub_dir+'###'+dst_aut + \
                  '.'+file_name.split('.')[-1]
    scan.cmd(change_name)
    scan.cmd(command='mv ./style/transform/* ' + path)


# input port, start to transform
# path_program:Program path to be transformed
# src_name: The original author of this program
def get_input(path_program, file_name, src_name, author_flag):
    author_path = './author_style'
    dst_aut = ''
    if path_program.endswith('.java'):
        scan.program_style_flag = 'java'
    if path_program.endswith('.c'):
        scan.program_style_flag = 'c'
    if path_program.endswith('.cpp'):
        scan.program_style_flag = 'cpp'
    # convert program to XML file
    scan.srcml_program_xml(path_program, './style/style')
    if author_flag == '1':
        # find the right author
        dst_aut, dst_auths = get_author('./style/style.xml', src_name, style_path)
        dst_aut = dst_aut.split('/')[-1]
        # convert the program to this target author
        transform.start_trans(path_program, dst_aut)
        # move each converted program to the final directory
        move_change(file_name, src_name, dst_aut)
    elif author_flag == 'all':
        # Convert to all authors
        author_list = os.listdir(author_path) if os.path.isdir(author_path) else [author_path]
        for author_name in author_list:
            if not author_name.endswith('.txt'):
                continue
            author_name = author_name.split('.')[0]
            # if the names of two authors are the same, it means the same author without conversion
            if src_name == author_name:
                continue
            # start to transform
            transform.start_trans(path_program, author_name)
            dst_aut = author_name
            # move each converted program to the final directory
            move_change(file_name, src_name, dst_aut)
    return dst_aut


def get_author(path_program, pre_name, style_path):
    sum_max = -1
    dst_max = ''
    dst_auths = []
    for file in os.listdir(style_path):
        file_name = file.split('.')[0]
        if pre_name != file_name:
            author_path = transform.get_author_path(file_name)
            list_12 = transform.get_style12(path_program, file_name)
            list_12 = sorted(set(list_12), key=list_12.index)
            len_12 = len(list_12)
            program_typedef = typedef.get_lentypedef_program(path_program)
            len_10 = typedef.get_lentypedef(program_typedef, author_path)
            program_define = type_define.get_define_program(path_program)
            len_11 = type_define.get_lendefine(program_define, author_path)
            # 选出最长的
            if len_10+len_11+len_12 > sum_max:
                sum_max = len_10+len_11+len_12
                dst_max = author_path

            if len_10+len_11+len_12 > 15:
                dst_auths.append(file_name)
    return dst_max, dst_auths


if __name__ == '__main__':
    # author program to be transformed
    program_path = './program_file/test'
    # save path after transformation
    transform_file = 'program_file/nondirectional_file'
    # delete
    # # calculate the proportion of variable and class names for style 2 and style 3
    # tmp_var_sums = count_names_ratio()

    # start to transform
    """
    First of all, find the appropriate target author, leave the non directional failure program after model testing, 
    and then convert these programs to more likely target authors, 
    and test the model one by one until the non directional successful author is found
    """
    # 1. author_flag='1' : means to find the right target author
    # 2. author_flag='all' : means to convert to more likely target authors
    #  scan_style(program_path, author_flag='1')
    scan_style(program_path, author_flag='all')
