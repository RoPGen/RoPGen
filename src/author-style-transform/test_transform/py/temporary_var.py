


"""

5：Local variable definition location
5.1  The beginning ->  First use

"""

import os
import sys
from copy import deepcopy

from lxml import etree
from lxml.etree import Element
doc=None
flag = False
ns = {'src': 'http://www.srcML.org/srcML/src',
   'cpp': 'http://www.srcML.org/srcML/cpp',
   'pos': 'http://www.srcML.org/srcML/position'}

save_xml_file = './transform_xml_file/temp_var_pre'
transform_java_file = './target_author_file/transform_java/temp_var_pre'

def init_parse(e):
    global doc
    doc=etree.parse(e)
    e=etree.XPathEvaluator(doc,namespaces=ns)
    return e


def get_block_cons(e):
    return e('//src:block_content')


def get_init_temps(block_con):
    return block_con.xpath('src:decl_stmt/src:decl',namespaces=ns)


def get_exprs(elem):
    return elem.xpath('src:expr_stmt/src:expr/src:name',namespaces=ns)


def get_var_name(b):
    return b.xpath('.//src:name',namespaces=ns)


def judge_ini(var_index, block_con, var_name):
    f=False
    b=None
    index=var_index
    for b in block_con[var_index+1:]:
        all_name=get_var_name(b)
        for name in all_name:
            if len(name)!=0:continue
            if name.text==var_name:
                f=True
                break
        if f==True:
            index=block_con.index(b)
            break
    return index,f,b


def get_instances(e):
    instances = []
    block_cons=get_block_cons(e)
    for block_con in block_cons:
        ls_decl=[] #Store the variables defined by the header
        #Get the variables defined by the head
        for decl_stml in block_con:
            if decl_stml.tag!='{http://www.srcML.org/srcML/src}decl_stmt':
                break
            #If the start is decl_stmt, get all the decl that meet the requirements
            for decl in decl_stml:
                if len(decl) == 2 or len(decl) == 3:  # and decl[2][0][0].tag=='{http://www.srcML.org/srcML/src}literal'):
                    modifier_num = decl[0].xpath("src:modifier", namespaces=ns)
                    if len(modifier_num) != 0:
                        first_mod_index=decl[0].index(modifier_num[0])
                        modifier = Element('modifier')
                        modifier.text = ''
                        for modi in decl[0][first_mod_index:]:
                            modifier.text += modi.text
                            decl[0].remove(modi)
                        modifier.text+=' '
                        decl.insert(1, modifier)
                    ls_decl.append(decl)  # Get decl tags
        for decl in ls_decl:   #Add variable types to all decl
            typ = deepcopy(decl.getparent()[0][0])
            if len(decl[0])==0:
                decl.remove(decl.getchildren()[0])
                decl.insert(0,typ)

        for decl in ls_decl:


            var_index=block_con.index(decl.getparent())

            if len(decl.xpath('src:name',namespaces=ns)[0])!=0:
                des_index, f,b_ele = judge_ini(var_index, block_con, decl.xpath('src:name',namespaces=ns)[0][0].text)
            else:
                des_index,f,b_ele=judge_ini(var_index,block_con,decl.xpath('src:name',namespaces=ns)[0].text)



            decl_prev=block_con[des_index].getprevious()

            flag=True

            if f==True:
                instances.append((decl_prev, decl, block_con, des_index,b_ele))
            # elif decl.getparent().index(decl)!=1:
            #     decl.remove(decl[0])
    return instances

def trans_temp_var(e,ignore_list=[],instances=None):
    global flag
    flag=False
    #Only consider the temporary variables in the loop and the condition body Get the <block_content> tag
    decls=[get_instances(e) if instances is None else (instance[0] for instance in instances if len(instance)>0)]
    #Get all initial temporary variables in the statement block

    tree_root = e('/*')[0].getroottree()
    new_ignore_list = []
    block_con=[]
    for item in decls:
        for inst_tuple in item:
            decl_prev = inst_tuple[0]
            decl = inst_tuple[1]
            block_con = inst_tuple[2]
            des_index = inst_tuple[3]
            b_ele=inst_tuple[4]
            decl_prev_path=tree_root.getpath(decl_prev)
            if decl_prev_path in ignore_list:continue
            flag = True
            if decl is None or decl.tail is None:continue
            if decl.tail.replace(' ','').replace('\n','')==';' and len(decl.getparent())!=1:
                decl.getparent()[-2].tail=';'
            if decl.tail!=';':
                decl.tail=''
                decl[-1].tail=';\n'
            block_con.insert(block_con.index(b_ele),decl)

            new_ignore_list.append(decl_prev_path)


    block_s = get_block_cons(e)
    for block_ in block_s:
        ls_dec = []  # Store the variables defined by the header
        for decl_stml in block_:
            if decl_stml.tag != '{http://www.srcML.org/srcML/src}decl_stmt':
                break
            # If the start is decl_stmt, get all the decl that meet the requirements
            for decl in decl_stml:
                if len(decl) == 2 or len(decl) == 3:# and decl[2][0][0].tag == '{http://www.srcML.org/srcML/src}literal'):
                    ls_dec.append(decl)
        for decl in ls_dec:
            if decl.getparent().tag == '{http://www.srcML.org/srcML/src}decl_stmt' and decl.getparent().index(decl) != 0:
                decl.remove(decl.getchildren()[0])

    return flag, tree_root, new_ignore_list


        #Delete redundant variable name types
        #
        #init_temps=get_init_temps(block_con)
        # #Get the name in the expression statement under the current statement block
        # exprs=get_exprs(block_con)
        # #To prevent multiple variable declarations, add the type of each variable name
        # for init_tmp in init_temps:
        #     typ=str(init_tmp.getparent()[0][0][0].text)+' '
        #     if len(init_tmp[0])==0:
        #         init_tmp[0].text=typ
        # for init_tmp in init_temps:
        #     for expr in exprs:
        #         if init_tmp[1].text==expr.text:#当临时变量名与表达式中第一个变量匹配上就删除该临时变量的定义
        #             #先获取变量类型 然后加到表达式前边 之后删除该变量声明
        #             if init_tmp.tail!=';':
        #                 init_tmp.tail=''
        #                 init_tmp[-1].tail=';\n'
        #             expr.getparent().insert(0,init_tmp)
        #             flag=True
        #             break
        # #删除多余的变量名类型
        # re_temps=get_init_temps(block_con)
        # for init_tmp in re_temps[1:]:
        #     init_tmp[0].text=''





def save_file(doc, param):
    with open(param,'w') as f:
        f.write(etree.tostring(doc).decode('utf-8'))

def get_style(xmlfilepath):
    e = init_parse(xmlfilepath)

    num=0
    block_cons = get_block_cons(e)
    for block_con in block_cons:
        ls_decl = []
        for decl_stml in block_con:
            if decl_stml.tag != '{http://www.srcML.org/srcML/src}decl_stmt':
                break
            for decl in decl_stml:
                if len(decl) == 2 or (
                        len(decl) == 3 and len(decl[2])>0 and len(decl[2][0])>0 and decl[2][0][0].tag == '{http://www.srcML.org/srcML/src}literal'):
                    ls_decl.append(decl)  #
        for decl in ls_decl:  #
            typ = deepcopy(decl.getparent()[0][0])
            if len(decl[0]) == 0:
                pass

        for decl in ls_decl:
            var_index = block_con.index(decl.getparent())
            des_index, f, b = judge_ini(var_index, block_con, decl[1].text)
            if f == True:
                num+=1
    return ['5.1', num]
def program_transform(program_path):
    e = init_parse(program_path)
    trans_temp_var(e)
    save_file(doc, './style/style.xml')
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
        trans_temp_var(e)
        if flag == True:
            str = xml_path_elem.split('\\')[-1]
            sub_dir = xml_path_elem.split('\\')[-2]
            if not os.path.exists(os.path.join(save_xml_file, sub_dir)):
                os.mkdir(os.path.join(save_xml_file, sub_dir))
            save_file(doc, os.path.join(save_xml_file, sub_dir, str))
    return save_xml_file, transform_java_file
