

"""

10：User-defined type
10.2  remove typedef

"""



import os
import sys
from copy import deepcopy

from lxml import etree

doc=None
flag = False
ns = {'src': 'http://www.srcML.org/srcML/src',
   'cpp': 'http://www.srcML.org/srcML/cpp',
   'pos': 'http://www.srcML.org/srcML/position'}

save_xml_file = './transform_xml_file/typedef_rep'
transform_java_file = './target_author_file/transform_java/typedef_rep'


def init_parse(file):
    global doc
    doc=etree.parse(file)
    e=etree.XPathEvaluator(doc,namespaces=ns)
    return e


def get_defines(e):
    return e('//src:typedef')


def get_allname(e):
    return e('//src:name')


def get_instances(e):
    instances = []
    defines=get_defines(e)
    names=get_allname(e)
    for define in defines:
        if len(define)<=1:continue
        #Get definition type is not struct
        if define[0][0].tag != '{http://www.srcML.org/srcML/src}struct' and define[0][0].tag != '{http://www.srcML.org/srcML/src}union' and define[0][0].tag != '{http://www.srcML.org/srcML/src}enum':
            define_name=define[1].text

            name_replace = ''.join(define[0].itertext())
            for name in names:
                if len(name)!=0:continue
                if  name.text==define_name and name.getparent().tag!='{http://www.srcML.org/srcML/src}typedef':
                    if name is None:continue
                    if name.getparent() is None:continue
                    if name.getparent().getparent() is None:continue
                    decl_stmt=name.getparent().getparent().getparent()
                    if len(name.getparent()) > 1:
                        argu_list = "".join(name.getparent()[1].itertext())
                    else:
                        argu_list = ''
                    if name.getparent().tag == '{http://www.srcML.org/srcML/src}call' and argu_list != "()":
                        name.text = '(' + name_replace + ')'
                    else:
                        name.text=name_replace

                    if  decl_stmt is not None and decl_stmt.tag=='{http://www.srcML.org/srcML/src}decl_stmt' and len(decl_stmt)!=1 :
                        decl_stmt[0].tail=';\n'
                        for decl in decl_stmt[1:]:
                            instances.append((decl, define, define_name))
                            
    return instances

def trans_define(e,ignore_list=[], instances=None):
    global flag
    flag = False

    defines = get_defines(e)
    decls = [get_instances(e) if instances is None else (instance[0] for instance in instances)]
    tree_root = e('/*')[0].getroottree()
    new_ignore_list = []
    removed_defines = []
    for item in decls:
        for inst_tuple in item:
            decl = inst_tuple[0]
            define = inst_tuple[1]
            define_name = inst_tuple[2]
            if define_name in ignore_list:continue
            flag=True
            if len(decl.getchildren()) == 0:continue
            decl.remove(decl.getchildren()[0])
            decl.insert(0,deepcopy(define[0]))
            decl.tail=';\n'

            new_ignore_list.append(define_name)
    for define in defines:
        if define[0][0].tag != '{http://www.srcML.org/srcML/src}struct' and define[0][0].tag != '{http://www.srcML.org/srcML/src}union' and define[0][0].tag != '{http://www.srcML.org/srcML/src}enum':
            if len(define) <= 1 or define[1].text==None: continue
            if define not in removed_defines and define[0][0].tag != '{http://www.srcML.org/srcML/src}struct':
                removed_defines.append(define)
                define.getparent().remove(define)

    return flag,tree_root,new_ignore_list

def save_file(doc, file):
    with open(file,'w') as f:
        f.write(etree.tostring(doc).decode('utf-8'))
def get_style(xml_path):
    e = init_parse(xml_path)
    defines = get_defines(e)
    num = False
    for define in defines:
        if len(define)<=1:continue
        if len(define[0][0]) == 0 and define[0][0].tag != '{http://www.srcML.org/srcML/src}struct' and define[0][0].tag != '{http://www.srcML.org/srcML/src}union' and define[0][0].tag != '{http://www.srcML.org/srcML/src}enum':
            define_name = define[1].text
            name_replace = ''
            for part_name in define[0]:
                name_replace = name_replace + " " + str(part_name.text)  # 直接累加名字拼凑成字符串
            num = True
    return num
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
        flag = False
        trans_define(e)
        if flag == True:
            str = xml_path_elem.split('\\')[-1]
            sub_dir = xml_path_elem.split('\\')[-2]
            if not os.path.exists(os.path.join(save_xml_file, sub_dir)):
                os.mkdir(os.path.join(save_xml_file, sub_dir))
            save_file(doc, os.path.join(save_xml_file, sub_dir, str))
    return save_xml_file, transform_java_file
def get_lentypedef(program_defines, author_path):
    max_typedef = 0
    for file in os.listdir(author_path):
        program_def = program_defines
        e = init_parse(os.path.join(author_path,file))
        author_defines = get_defines(e)
        for author_define in author_defines:
            if author_define not in program_def:
                program_def.append(author_define)
        len_typedef = len(program_def)
        if len_typedef > max_typedef:
            max_typedef = len_typedef
    if max_typedef >= 0:
        return max_typedef
def get_lentypedef_program(program_path):
    e = init_parse(program_path)
    return get_defines(e)
def program_transform(program_path):
    e = init_parse(program_path)
    trans_define(e)
    save_file(doc, './style/style.xml')
