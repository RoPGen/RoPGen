"""

16：Loop Structure
16.1  while structure -> for structure

"""
import os
from lxml import etree
ns = {'src': 'http://www.srcML.org/srcML/src'}
doc = None
flag = False


# parsing XML file into tree
def init_parse(file):
    global doc
    doc = etree.parse(file)
    e = etree.XPathEvaluator(doc, namespaces=ns)
    return e


def get_while(e):
    return e('//src:while')


# get the loop condition of while
def get_condition(elem):
    return elem.xpath('src:condition', namespaces=ns)


# start to transform
def trans_tree(e, ignore_list=[], instances=None):
    global flag
    flag = False
    # get all while
    while_elems = [get_while(e) if instances is None else (instance[0] for instance in instances)]
    # Get the root of the tree
    tree_root = e('/*')[0].getroottree()
    new_ignore_list = []
    if len(while_elems) > 0:
        for item in while_elems:
            for while_elem in item:
                '''
                filter out the while transformed from for in the last round.
                Because for to while inserts a statement before while, 
                the part not changed is to count two statements up from the while statement.
                run 'getprevious()' twice,
                if there are less than two statements in front of the while statement (getprevious() returns none), 
                the last one that is not none will be retained.
                '''
                while_prev = while_elem.getprevious()
                while_prev_prev = [while_prev.getprevious() if while_prev is not None else while_elem][0]
                while_prev_prev = [while_prev_prev if while_prev_prev is not None else while_prev][0]
                # take the path to see if it is in the passed in ignore list
                while_prev_path = tree_root.getpath(while_prev_prev)
                if while_prev_path in ignore_list:
                    continue
                flag = True

                # get the loop condition of while
                elem_condition = get_condition(while_elem)[0]
                while_elem.text = 'for'
                while_elem.tag = 'for'
                elem_condition.text = '(;'
                elem_condition[-1].tail = ';)'

                # Record the unchanged position after this transformation,
                # that is the position of the previous while statement

                new_ignore_list.append(while_prev_path)
    return flag, tree_root, new_ignore_list


def count(e):
    # get all while
    while_elems = get_while(e)
    return len(while_elems)


# calculate the number of while in the program
def get_number(xml_path):
    xmlfilepath = os.path.abspath(xml_path)
    e = init_parse(xmlfilepath)
    return count(e)


# save tree to file
def save_tree_to_file(tree, file):
    with open(file, 'w') as f:
        f.write(etree.tostring(tree).decode('utf8'))


# program's input port
def program_transform(program_path):
    e = init_parse(program_path)
    trans_tree(e)
    save_tree_to_file(doc, './style/style.xml')