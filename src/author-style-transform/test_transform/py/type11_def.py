


"""

11：Macro
11.2  add other authors' define

"""

import sys
import glob

from lxml.etree import Element

path='.\\projects'
d_path='.\\dest'

from lxml import etree
doc=None
ns = {'src': 'http://www.srcML.org/srcML/src',
   'cpp': 'http://www.srcML.org/srcML/cpp',
   'pos': 'http://www.srcML.org/srcML/position'}

ls=[]

def_Min=2

def init_parse(file):
    global doc
    doc=etree.parse(file)
    e=etree.XPathEvaluator(doc,namespaces=ns)
    return e


def get_defines(e):
    return e('//cpp:define')


def get_allname(e):
    return e('//src:literal')
def get_allvarname(e):
    return e('//src:name')


def save_file(doc, f):
    with open(f,'w')as r:
        r.write(etree.tostring(doc).decode('utf-8'))

def trans_define(e,l):
    #Get all the macro definition tags
    defines=get_defines(e)
    all_names=get_allvarname(e)
    #First exclude the same define tag that already exists
    for name_var in all_names:
        if len(name_var) != 0: continue
        if name_var.text==l[1]:
            return
    for define in defines:
        if define[1][0].text==l[1]:
            return

    #Get all the name tags, as long as the name is the same as the 0name of the macro definition, replace the value in the macro definition
    # names=get_allname(e)
    # for name in names:
    #     if name.text==l[2]:
    #         name.text=l[1]
    #Add l as a macro definition to the file
    new_define=Element('define')
    new_define.text='#define '+str(l[1])+' '+str(l[2])+'\n'

    if len(e('//src:using')) != 0:
        elem=e('//src:using')[0]
        des=elem.getparent().index(elem)
        elem.getparent().insert(des+1,new_define)
    else:
        if len(e('//cpp:include'))!=0:
            elem=e('//cpp:include')[-1]
            des=elem.getparent().index(elem)
            elem.getparent().insert(des+1,new_define)
        else:
            des=0
            elem=e('//src:unit')[0]
            if elem.getparent() is not None:
                elem.getparent().insert(des+1,new_define)

def creat_def_list(e):
    global ls
    #Get define tag
    defines=get_defines(e)
    flag=0
    for define in defines:
        if len(define) == 3 and len(define[1]) == 1:
            for l in ls:
                if define[1][0].text==l[1]:  #Find the macro definition name
                    l[0]+=1
                    flag=1
                    break
            if flag==0:
                ls.append(list([1,define[1][0].text,define[2].text]))
            flag=0
def program_transform(program_path,author_path,ignore_list=[]):
    global ls
    files =[f for f in glob.glob(author_path+"**/*.xml",recursive=True)]
    for f in files:
        #Parse a list of files in a loop
        e = init_parse(f)
        #Record the macro definitions in a file every cycle
        creat_def_list(e)
    e = init_parse(program_path)
    global flag
    flag=False
    tree_root=e('/*')[0].getroottree()
    new_ignore_list=[]

    for l in ls:
        value_name=l[1]
        if value_name in ignore_list:continue

        #if l[0]>=def_Min:
        trans_define(e, l)
        flag=True
        value_name = l[1]
        # 记录define名字
        new_ignore_list.append(value_name)
    save_file(doc, './style/style.xml')
    ls=[]
    return flag,tree_root,new_ignore_list

    # des=[f for f in glob.glob(d_path+'**/*.xml',recursive=True)]
    # for f in des:
    #     e=init_parse(f)
    #     for l in ls:
    #         if l[0]>=def_Min:
    #             trans_define(e,l,f)

