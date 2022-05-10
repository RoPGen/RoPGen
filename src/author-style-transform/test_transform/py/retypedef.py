


"""

10：User-defined type
10.1  add other authors' typedef

"""

import sys
import glob
from copy import deepcopy

from lxml.etree import Element

from test_transform.py import typedef

path='.\\typ'
d_path='.\\des'

from lxml import etree
doc=None
ns = {'src': 'http://www.srcML.org/srcML/src',
   'cpp': 'http://www.srcML.org/srcML/cpp',
   'pos': 'http://www.srcML.org/srcML/position'}

ls=[] #The element is a list [1,2,3] 1 Number of occurrences of macro definition name 2 Macro definition name 3 Macro definition value (only the first one for different values)


key_words = ['int', 'long', 'union','enum','long long', 'double', 'char', 'string', 'float', 'vector','pair','bool','void','signed','unsigned','short','std']


def_Min=2 #Set macro definition threshold


def parse_src(file):
    f=etree.parse(file)
    e=etree.XPathEvaluator(f,namespaces=ns)
    return e
def get_var_name(e):
    return e.xpath('.//src:name',namespaces=ns)

def init_parse(file):
    global doc
    doc=etree.parse(file)
    e=etree.XPathEvaluator(doc,namespaces=ns)
    return e


def get_defines(e):
    return e('//src:typedef')


def get_allname(e):
    return e('//src:typedef/src:name')

def get_all_var_name(e):
    return e('//src:name')

def save_file(doc, f):
    with open(f,'w')as r:
        r.write(etree.tostring(doc).decode('utf-8'))


def trans_define(e,l,varnames):
    defines=get_defines(e)
    if l[1] in varnames:return
    for define in defines:
        if len(define)<=1:continue
        if define[1].text==l[1]:
            return
    if len(e('//src:using')) != 0:
        elem=e('//src:using')[-1]
        des=elem.getparent().index(elem)
        elem.getparent().insert(des+1,l[2])
    if len(e('//src:using')) == 0:
        if len(e('//cpp:include'))!=0:
            elem=e('//cpp:include')[-1]
            des=elem.getparent().index(elem)
            elem.getparent().insert(des+1,l[2])
        else:
            des=0
            print("文件既没有using也没有include")
            elem=e('//src:unit')[0]
            elem.insert(des,l[2])


def creat_def_list(e,src_var_names):
    global ls
    typedef_name=[]+key_words

    defines=get_defines(e)
    flag=0
    for define in defines:
        if len(define)<=1:continue
        if define[0][0].tag != '{http://www.srcML.org/srcML/src}struct' and define[0][0].tag != '{http://www.srcML.org/srcML/src}union' and define[0][0].tag != '{http://www.srcML.org/srcML/src}enum':
            for l in ls:
                if define[1].text==l[1]:
                    l[0]+=1
                    flag=1
                    break
            if flag==0 and define[0][0].tag!='{http://www.srcML.org/srcML/src}struct':
                var_names=[t.text for t in get_var_name(define) if len(t)==0 and t.text not in key_words]
                ff=False
                for var_name in var_names:
                    if var_name in src_var_names:
                        ff=True
                        break
                type_vars=[t.text for t in get_var_name(define[0]) if len(t)==0 and t.text not in typedef_name]
                if ff == False and type_vars == []:
                    ls.append(list([1, define[1].text, deepcopy(define)]))
                    typedef_name.append(define[1].text)
            flag=0

def del_typedef(del_ls, e,node_typ):
    #e If there are none in the del_ls list, delete the ones
    type_defines = get_defines(e)
    names = get_allname(e)
    for type_define in type_defines:
        if len(type_define)<=1:continue
        if type_define[0][0].tag != '{http://www.srcML.org/srcML/src}struct' and type_define[0][0].tag != '{http://www.srcML.org/srcML/src}union' and type_define[0][0].tag != '{http://www.srcML.org/srcML/src}enum':  # len(define[0][0]) == 0 and


            # Get definition type is not struct

            define_name = type_define[1].text  # Get the name of the typedef

            name_replace = ''.join(type_define[0].itertext())
            # for part_name in define[0]:
            #     name_replace = name_replace + " " + str(part_name.text)  # Directly accumulate names and piece them together into a string
            for name in names:
                if len(name)!=0:continue
                if name.text == define_name and name.getparent().tag != '{http://www.srcML.org/srcML/src}typedef':
                    if name is None: continue
                    if name.getparent() is None: continue
                    if name.getparent().getparent() is None: continue
                    decl_stmt = name.getparent().getparent().getparent()

                    if name.getparent().tag == '{http://www.srcML.org/srcML/src}call':
                        name.text = '(' + name_replace + ')'
                    else:
                        name.text = name_replace

                    if decl_stmt.tag == '{http://www.srcML.org/srcML/src}decl_stmt' and len(decl_stmt) != 1:
                        decl_stmt[0].tail = ';\n'
                        for decl in decl_stmt[1:]:
                            decl.remove(decl.getchildren()[0])
                            decl.insert(0, deepcopy(type_define[0]))
                            decl.tail = ';\n'

            type_define.getparent().remove(type_define)
    pass


def program_transform(program_path,author_path,ignore_list=[]):
    global ls
    files =[f for f in glob.glob(author_path+"**/*.xml",recursive=True)]
    src_e=parse_src(program_path)
    typedef.trans_define(src_e)
    src_var_names=[t.text for t in get_all_var_name(src_e) if len(t)==0 and t.text not in key_words]
    for f in files:
        e = init_parse(f)
        creat_def_list(e,src_var_names)
    des=[f for f in glob.glob(d_path+'**/*.xml',recursive=True)]
    e = init_parse(program_path)
    global flag
    flag = False
    tree_root = e('/*')[0].getroottree()
    new_ignore_list = []
    
    del_ls=[s[1] for s in ls]
    node_typ=[ "".join(s[2].itertext()) for s in ls]
    #del_typedef(del_ls, e,node_typ)
    typedef.trans_define(e)
    re_ls=ls[::-1]

    varnames_node=get_all_var_name(e)
    varnames=[node.text for node in varnames_node]
    for l in re_ls:
        value_name = l[1]
        if value_name in ignore_list: continue

        trans_define(e,l,varnames)

        flag = True
        value_name = l[1]
        # 记录typedef名字
        new_ignore_list.append(value_name)
    save_file(doc, './style/style.xml')
    ls=[]
    return flag, tree_root, new_ignore_list

