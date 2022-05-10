
"""

12：Header file

"""



import operator
import sys
import os  # 加上
from lxml import etree
from lxml.etree import Element

ns = {'src': 'http://www.srcML.org/srcML/src',
      'cpp': 'http://www.srcML.org/srcML/cpp',
      'pos': 'http://www.srcML.org/srcML/position'}
doc = None


def init_parser(file):
    global doc
    doc = etree.parse(file)
    e = etree.XPathEvaluator(doc)
    for k, v in ns.items():
        e.register_namespace(k, v)
    return e


def get_include(e):
    return e('//src:import')
def get_unit(e):
    return e('//src:unit')
def save_tree_to_file(tree, file):
    with open(file, 'w') as f:
        f.write(etree.tostring(tree).decode('utf8'))


# get all Library functions
def hunt(e):
    name_list = []
    get_elements = get_include(e)
    for get_element in get_elements:
        if len(get_element)==0:continue
        element = ''.join(get_element.itertext())
        name_list.append(element)
    return name_list


# find Library functions
def countnum(xml_path):
    xmlfilepath = os.path.abspath(xml_path)
    # 创建字典
    d = {}
    ele_list = []
    # os.walk 遍历作者包
    for root, sub_dirs, files in os.walk(xmlfilepath):
        # 遍历每个作者包下的程序，files是程序名
        for file1 in files:
            # 绝对路径
            file = os.path.join(root, file1)
            e = init_parser(file)
            get_elements = get_include(e)
            for get_element in get_elements:
                element = ''.join(get_element.itertext())
                if element in ele_list:continue
                ele_list.append(element)

    return ele_list

def transform_include(program_path, author_path):
    flag = 0
    #Statistics of incoming packets
    res = countnum(author_path)
    auth_list_keys = []
    # Converted to list form list(res.keys())
    auth_list_keys = res
    pro_list_keys = []
    if program_path != '':
        e = init_parser(program_path)
        pro_list_keys = hunt(e)
    #
    len1 = len(pro_list_keys)
    pro_list_key = pro_list_keys
    # If there is no library function in auth_list_keys in pro_list_keys, add it to pro_list_keys
    for element in pro_list_keys :
        if element not in auth_list_keys:
            flag = 1
            auth_list_keys.append(element)
    return auth_list_keys, len1, flag, auth_list_keys, pro_list_key
# Add fewer library functions to the xml of a given program
def transform1(program_path, author_path):
    new_ignore_list = []
    pro_list_keys, len1, flag, auth_list_keys, pro_list_key= transform_include(program_path, author_path)
    e = init_parser(program_path)
    include_first = get_include(e)


    includes = get_include(e)
    for include in includes:
        index = include.getparent().index(include)
        del include.getparent()[index]

    unit_elem = get_unit(e)[0]
    node = Element('import')
    unit_elem.insert(0, node)
    for elem in pro_list_keys:
        node1 = Element('name')
        node1.text = elem
        node1.tail = '\n'
        node.append(node1)
        flag = True
    return flag, new_ignore_list
def xml_file_path(xml_path):
    global flag
    save_xml_file = './transform_xml_file/import'
    transform_java_file = './target_author_file/transform_c/import'

    if not os.path.exists(transform_java_file):
        os.mkdir(transform_java_file)
    if not os.path.exists(save_xml_file):
        os.mkdir(save_xml_file)
    for xml_path_elem in xml_path:
        xmlfilepath = os.path.abspath(xml_path_elem)
        e = init_parser(xmlfilepath)
        flag = False
        if flag == True:
            str = xml_path_elem.split('\\')[-1]
            sub_dir = xml_path_elem.split('\\')[-2]
            path = os.path.join(save_xml_file, sub_dir)

            if not os.path.exists(os.path.join(save_xml_file, sub_dir)):
                os.mkdir(os.path.join(save_xml_file, sub_dir))
            save_tree_to_file(doc, os.path.join(path, str))
    return save_xml_file, transform_java_file

def program_transform(program_path, author_path):
    transform1(program_path, author_path)
    save_tree_to_file(doc, './style/style.xml')

import subprocess
def del_import(path):
    auths = os.listdir(path)
    for auth_elem in auths:
        files = os.listdir(os.path.join(path,auth_elem))
        for file_elem in files:
            srcml_java_xml(os.path.join(path,auth_elem,file_elem), '../style/style')
            xml_file = '../style/style.xml'
            e = init_parser(xml_file)
            import_list = get_include(e)
            #print(import_list)
            for import_elem in import_list:
                import_elem.getparent().remove(import_elem)
            save_tree_to_file(doc, '../style/style.xml')
            srcml_xml_java('../style/style.xml',os.path.join(path,auth_elem,file_elem))
def cmd(command):
    global flag
    flag = True
    subp = subprocess.Popen(command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,encoding="utf-8")
    subp.wait(10)
    if subp.poll() == 0:
        flag = True
    else:
        print("False!")
        flag = False
def srcml_java_xml(pre_path, xml_path):
    str = 'srcml \"'+ pre_path +'\" -o \"'+ xml_path +'.xml\" --position --src-encoding UTF-8'
    cmd(str)
def srcml_xml_java(pre_path, xml_path):
    str = "srcml \""+ pre_path +'\" -o \"'+ xml_path +"\" --src-encoding UTF-8"
    cmd(str)
if __name__ == '__main__':
    path1 = '../demo1.xml'
    path2 = '../xml_file/emigonza'
    program_transform(os.path.abspath(path1),path2)
    
    # del_import('../tool/test')