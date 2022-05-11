import copy
import sys
import os
from lxml import etree
from lxml.etree import Element

ns = {'src': 'http://www.srcML.org/srcML/src',
      'cpp': 'http://www.srcML.org/srcML/cpp'}
doc = None
include_flag = False


def init_parse(file):
    global doc
    doc = etree.parse(file)
    e = etree.XPathEvaluator(doc)
    for k,v in ns.items():
        e.register_namespace(k, v)
    return e


# get all functions
def get_functions(e):
    return e('//src:function/src:block/src:block_content')


# get all the call label
def get_call(elem):
    return elem.xpath('.//src:call', namespaces=ns)


def get_expr(elem):
    return elem.xpath('src:argument_list/src:argument', namespaces=ns)


# printf convert
def transform_printf(call):
    exprs = get_expr(call)
    node = etree.Element('expr')
    node.text = 'cout'

    if len(exprs)==1:
        expr = exprs[0]
        node_elem = etree.Element('expr')
        node_elem.text = '<<'
        node.append(node_elem)
        expr.tail = ''
        node.append(expr)
    else:
        str = exprs[0]
        if len(str) > 0:
            str = exprs[0][0]
        if len(str) > 0:
            str = exprs[0][0][0].text
        node_elem = etree.Element('expr')
        node_elem.text = '<<'
        node_elem.tail = ''
        j = 0
        strlist = []
        if str is None: return
        str = str[1:]
        for i in range(0, len(str)):
            if (str[i] == '%' and j < i) or (i == len(str)-1 and j <= i and str[j] != '%'):
                node_elem2 = copy.deepcopy(node_elem)
                node_elem2.text = '<<"'+str[j:i]+'"'
                strlist.append(node_elem2)
                j = i+2
        i = 0

        for i in range(1, len(exprs)):
            if i <= len(strlist):
                node.append(strlist[i-1])
            node.append(copy.deepcopy(node_elem))
            exprs[i].tail = ''
            node.append(exprs[i])
        for i in range(i,len(strlist)):
            node.append(strlist[i])
    node_elem = etree.Element('expr')
    node_elem.text = '<<endl'
    node.append(node_elem)
    call.getparent().replace(call, node)


# scanf convert
def transform_scanf(call):
    exprs = get_expr(call)
    node = etree.Element('expr')
    node.text = 'cin'
    for expr in exprs[1:]:
        node_elem = etree.Element('expr')
        node_elem.text = '>>'
        node.append(node_elem)
        operater_node = expr.xpath('src:expr/src:operator',namespaces=ns)
        if len(operater_node) > 0 and operater_node[0].text == '&':
            operater_node[0].getparent().remove(operater_node[0])
        expr.tail = ''
        node.append(expr)
    call.getparent().replace(call, node)


# scanf and pirntf convert into cin and cout
def c_lib_to_cpp(e, ignore_list=[], instances=None):
    global flag
    flag = False
    functions = [get_functions(e) if instances is None else (instance[0] for instance in instances)]
    tree_root = e('/*')[0].getroottree()
    new_ignore_list = []
    global include_flag
    for item in functions:
        for func in item:
            calls = get_call(func)
            for call in calls:
                call_prev = call.getprevious()
                call_prev = call_prev if call_prev is not None else call
                call_prev_path = tree_root.getpath(call_prev)
                if call_prev_path in ignore_list:
                    continue
                index = call.getparent().index(call)
                call_parent = call.getparent()
                if len(call_parent) > index+1 and call_parent[index+1].text == '==':continue
                children = call.getchildren()
                ignore_list_added = False
                for child in children:
                    tag = etree.QName(child)
                    if tag.localname == 'name':
                        if child.text == 'printf' or child.text =='scanf':
                            include_flag = True
                            flag = True
                            if not ignore_list_added:
                                new_ignore_list.append(call_prev_path)
                                ignore_list_added = True
                        # printf convert
                        if child.text == 'printf':
                            transform_printf(call)
                        # scanf convert
                        if child.text == 'scanf':
                            transform_scanf(call)
    if include_flag == True:
        add_include(e)
    return flag, tree_root, new_ignore_list


# C use C + + function, need part of the necessary header file
def add_include(e):
    include_elems = e('//cpp:include/cpp:file/text()')
    namespaces_elem = e('//src:using/src:namespace/src:name')
    if len(namespaces_elem)==0:
        node1 = Element('using')
        node1.text = 'using '
        node2 = Element('namespace')
        node2.text = 'namespace '
        node3 = Element('name')
        node3.text = 'std'
        node3.tail = ';\n'
        node1.append(node2)
        node2.append(node3)
        if len(e('//cpp:include')) !=0 :
            include_last = e('//cpp:include')[-1]
            index = include_last.getparent().index(include_last)
            include_last.getparent().insert(index+1, node1)
    if '<iostream>' not in include_elems:
        node1 = Element('directive')
        node1.text = '#include '
        node2 = Element('file')
        node2.text = '<iostream>'
        node2.tail = '\n'
        if len(e('src:function'))>0:
            funcs = e('src:function')[0]
            funcs.getparent().insert(0, node1)
            funcs.getparent().insert(1, node2)


# calculate the number of 'printf' and 'scanf' statement
def get_number(xml_path):
    e = init_parse(xml_path)
    return count(e)


def count(e):
    count_num = 0
    functions = get_functions(e)
    for func in functions:
        calls = get_call(func)
        for call in calls:
            index = call.getparent().index(call)
            call_parent = call.getparent()
            if len(call_parent) > index + 1 and call_parent[index + 1].text == '==': continue
            children = call.getchildren()
            for child in children:
                tag = etree.QName(child)
                if tag.localname == 'name':
                    if child.text == 'printf':
                        count_num += 1
                    if child.text == 'scanf':
                        count_num += 1
    return count_num


def save_tree_to_file(tree, file):
    with open(file, 'w') as f:
        f.write(etree.tostring(tree).decode('utf8'))


# the program's input port
def program_transform(program_path):
    e = init_parse(program_path)
    c_lib_to_cpp(e)
    save_tree_to_file(doc, './style/style.xml')
