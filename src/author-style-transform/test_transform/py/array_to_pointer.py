"""

8：Access to array/pointer elements
8.1  array -> pointer

"""

import sys
import os              
from lxml import etree
ns = {'src': 'http://www.srcML.org/srcML/src',
	'cpp': 'http://www.srcML.org/srcML/cpp',
	'pos': 'http://www.srcML.org/srcML/position'}
doc=None
Flag = False

# parsing XML file into tree
def init_parse(file):
    global doc
    doc = etree.parse(file)
    e = etree.XPathEvaluator(doc)
    for k,v in ns.items():
        e.register_namespace(k, v)
    return e

# get the label 'index'
def get_index(e):
    return e('//src:index')

# save tree to file
def save_tree_to_file(tree, file):
    with open(file, 'w') as f:
        f.write(etree.tostring(tree).decode('utf8'))


# change the array 'a[i]' to the pointer '*(a+i)' or change the array 'a[i][j]' to the pointer '*(*(a+i)+j)'
def transform(e,ignore_list=[], instances=None):
    global flag
    flag = False
    index_elems = [get_index(e) if instances is None else (instance[0] for instance in instances)]
    # get the tree root
    tree_root = e('/*')[0].getroottree()
    new_ignore_list = []
    for item in index_elems:
        for index_elem in item:
            if index_elem.getparent().getparent().tag == '{http://www.srcML.org/srcML/src}expr':         
                if index_elem.text is not None and  index_elem.text.strip() == '[' and index_elem[-1].tail.strip() == ']' \
                        and index_elem[0][0].tag == '{http://www.srcML.org/srcML/src}name':
                    array_grandparent = index_elem.getparent().getparent()
                    array_grandparent_prev = array_grandparent.getprevious()
                    array_prev = array_grandparent if array_grandparent_prev is None else array_grandparent_prev
                    array_prev_path = tree_root.getpath(array_prev)
                    if array_prev_path in ignore_list: continue
                    # the parent node of 'index' is p
                    p = index_elem.getparent()
                    if p.getnext() is not None and p.getprevious() is not None:
                        # If it is a one-dimensional array, the length of the node(p) is 2
                        if len(index_elem.getparent()) == 2 and p.getnext().text != '.' and p.getprevious().text != '.':
                            # change '[' to '+'
                            index_elem.text = '+'
                            # change ']' to '）'
                            index_elem[-1].tail = ')'
                            p.text = '*' + '('
                        # If it is a two-dimensional array, the length of the node(p) is 3
                        elif len(index_elem.getparent()) == 3:
                            p[1].text = '+'
                            p[1][0].tail = ')'
                            p[2].text = '+'
                            p[2][0].tail = ')'
                            p.text = '*(*('
                        flag = True
                        new_ignore_list.append(array_prev_path)
    return flag, tree_root, new_ignore_list
    
def countnum(xml_path):
    count = 0
    xmlfilepath = os.path.abspath(xml_path)
    e = init_parse(xmlfilepath)
    index_elems = get_index(e)
    for index_elem in index_elems:
        if index_elem.getparent().getparent().tag == '{http://www.srcML.org/srcML/src}expr':
            if index_elem.text.strip() == '[' and index_elem[-1].tail.strip() == ']' \
                    and index_elem[0][0].tag == '{http://www.srcML.org/srcML/src}name':
                if len(index_elem.getparent()) == 2 or len(index_elem.getparent()) == 3:
                    count = count + 1
    return count

def xml_file_path(xml_path):
    global flag
    save_xml_file = './transform_xml_file/array_to_pointer'
    transform_java_file = './pre_file/transform_c/array_to_pointer'
    if not os.path.exists(transform_java_file):
        os.mkdir(transform_java_file)
    if not os.path.exists(save_xml_file):
        os.mkdir(save_xml_file)
    for xml_path_elem in xml_path:
        xmlfilepath = os.path.abspath(xml_path_elem)
        e = init_parse(xmlfilepath)
        flag = False
        transform(e)
        if flag == True:
            str = xml_path_elem.split('\\')[-1]
            sub_dir = xml_path_elem.split('\\')[-2]
            path = os.path.join(save_xml_file, sub_dir)
            if not os.path.exists(os.path.join(save_xml_file, sub_dir)):
                os.mkdir(os.path.join(save_xml_file, sub_dir))
            save_tree_to_file(doc, os.path.join(path, str))
    return save_xml_file, transform_java_file

# program's input port
def program_transform(xml_path):
    e = init_parse(xml_path)
    transform(e)
    save_tree_to_file(doc, './style/style.xml')