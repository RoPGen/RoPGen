
"""

11：Macro
11.1   remove  define

"""

import os
import sys

import re
from lxml import etree
flag = False
ns = {'src': 'http://www.srcML.org/srcML/src',
   'cpp': 'http://www.srcML.org/srcML/cpp',
   'pos': 'http://www.srcML.org/srcML/position'}

save_xml_file = './transform_xml_file/def_var'
transform_java_file = './target_author_file/transform_java/def_var'
doc=None

def init_parse(file):
    global doc
    doc=etree.parse(file)
    e=etree.XPathEvaluator(doc,namespaces=ns)
    return e


def get_defines(e):
    return e('//cpp:define')


def get_allname(e):
    return e('//src:name')


def get_instances(e):
    instances = []
    defines=get_defines(e)
    names=get_allname(e)
    #all_define_value=[define[2] for define in defines]
    all_define_value = []
    for define in defines:
        if len(define)>3:
            all_define_value.append(define[2])
    for define in defines:
        if len(define)==3 and len(define[1])==1:
            define_name=define[1][0].text
            define_value=define[2].text
            value_name = define_name
            for alldefine in all_define_value:
                if len(define)>=len(alldefine.text):continue
                d = re.sub('([^\w])'+define_name+'([^\w]*)', r'\1 '+define_value+r'\2', alldefine.text)
                s = re.sub(define_name+'([^\w])', define_value+r'\1', alldefine.text[0:len(define_name)+1])
                alldefine.text=s+d[len(define_name)+1:]
                print(alldefine.text)
            for name in names:
                if len(name)!=0:continue
                if name.getparent().tag!='{http://www.srcML.org/srcML/cpp}macro'and name.text==define_name:
                    instances.append((name, define_value, value_name, define))
    return instances

def trans_define(e,ignore_list=[], instances=None):
    global flag
    flag=False
    #Get all the macro definition tags
    names = [get_instances(e) if instances is None else (instance[0] for instance in instances)]
    #Get all the name tags, as long as the name is the same as the name of the macro definition, replace the value in the macro definition
    
    tree_root=e('/*')[0].getroottree()
    new_ignore_list=[]
    

    for item in names:
        for inst_tuple in item:
            name = inst_tuple[0]
            define_value = inst_tuple[1]
            value_name = inst_tuple[2]
            define = inst_tuple[3]

            #Record the name of define and compare it with ignore_list. If there is a name with the same name, skip it
            if value_name in ignore_list: continue
            flag = True
            name.text=define_value

            #Record define name
            new_ignore_list.append(value_name)
            if define.getparent() is None:continue
            define.getparent().remove(define)
    return flag,tree_root,new_ignore_list
def get_style(xmlfilepath):
    e = init_parse(xmlfilepath)
    num=False
    defines = get_defines(e)
    # Get all the name tags, as long as the name is the same as the name of the macro definition, replace the value in the macro definition
    names = get_allname(e)
    for define in defines:
        if len(define) == 3 and len(define[1]) == 1:
            define_name = define[1][0].text
            define_value = define[2].text
            for name in names:
                if len(name)!=0:continue
                if name.getparent().tag != '{http://www.srcML.org/srcML/cpp}macro' and name.text == define_name:
                    num = True
    return num
def save_file(doc, file):
    with open(file,'w') as f:
        f.write(etree.tostring(doc).decode('utf-8'))
def program_transform(program_path):
    e = init_parse(program_path)
    trans_define(e)
    save_file(doc, './style/style.xml')
def get_lendefine(program_define, author_path):
    max_define = 0
    for file in os.listdir(author_path):
        program_def = program_define
        e = init_parse(os.path.join(author_path,file))
        author_defines = get_defines(e)
        for author_define in author_defines:
            if author_define not in program_def:
                program_def.append(author_define)
        len_define = len(program_def)
        if len_define > max_define:
            max_define = len_define
    if max_define >= 0:
        return max_define
def get_define_program(program_path):
    e = init_parse(program_path)
    return get_defines(e)
def xml_file_path(xml_path):
    global flag
    if not os.path.exists(transform_java_file):
        os.mkdir(transform_java_file)
    if not os.path.exists(save_xml_file):
        os.mkdir(save_xml_file)
    for xml_path_elem in xml_path:
        xmlfilepath = os.path.abspath(xml_path_elem)
        # 解析成树
        e = init_parse(xmlfilepath)
        # 转换
        flag = False
        print(xml_path_elem)
        trans_define(e)
        if flag == True:
            str = xml_path_elem.split('\\')[-1]
            sub_dir = xml_path_elem.split('\\')[-2]
            if not os.path.exists(os.path.join(save_xml_file, sub_dir)):
                os.mkdir(os.path.join(save_xml_file, sub_dir))
            save_file(doc, os.path.join(save_xml_file, sub_dir, str))
    return save_xml_file, transform_java_file