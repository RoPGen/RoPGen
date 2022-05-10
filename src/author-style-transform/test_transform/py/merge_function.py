import sys
import os
from lxml import etree
from py import split_function

ns = {'src': 'http://www.srcML.org/srcML/src',
    'cpp': 'http://www.srcML.org/srcML/cpp',
    'pos': 'http://www.srcML.org/srcML/position'}
doc = None

def init_parser(file):
    global doc
    doc = etree.parse(file)
    return doc

# get the number of the line where 'elem' starts in source code
def get_elem_line_start(elem):
    pos_start = elem.xpath('@pos:start', namespaces=ns)[0]
    line_start = int(pos_start.split(':')[0])
    return line_start

# get the number of the line where 'elem' ends in source code
def get_elem_line_end(elem):
    pos_end = elem.xpath('@pos:end', namespaces=ns)[0]
    line_end = int(pos_end.split(':')[0])
    return line_end

def get_elem_len(elem):
    return get_elem_line_end(elem) - get_elem_line_start(elem)

# get all variables and their types
def get_vars_and_types(func):
    vars = {}
    decl_stmts = func.xpath('//src:decl_stmt', namespaces=ns)
    for decl_stmt in decl_stmts:
        decl_type = ''.join(decl_stmt.xpath('src:decl/src:type', namespaces=ns)[0].itertext())
        for decl in decl_stmt.xpath('src:decl', namespaces=ns):
            decl_name = ''.join(decl.xpath('src:name', namespaces=ns)[0].itertext())
            vars[decl_name] = decl_type
    return vars

def get_args_and_types(func):
    args = {}
    parameters = func.xpath('src:parameter_list/src:parameter', namespaces=ns)
    if len(parameters) == 0: return args
    for parameter in parameters:
        decl_type = ''.join(parameter.xpath('src:decl/src:type', namespaces=ns)[0].itertext())
        decl_name = ''.join(parameter.xpath('src:decl/src:name', namespaces=ns)[0].itertext())
        args[decl_name] = decl_type
    return args

def get_functions(elem):
    return elem.xpath('//src:function', namespaces=ns)

def get_function(e, name):
    functions = e('//src:function')
    for func in functions:
        func_name = func.xpath('src:name', namespaces=ns)
        if func_name[0].text == name:
            return func
    return None

def get_line(elem, lineno):
    line = elem.xpath('//*[starts-with(@pos:start, \'' + str(lineno) + ':\')]', namespaces=ns)
    return line

def save_tree_to_file(tree, file):
    with open(file, 'w') as f:
        f.write(etree.tostring(tree).decode('utf8'))

def remove_preserve_tail(element):
    parent = element.getparent()
    if element.tail:
        prev = element.getprevious()
        if prev is not None:
            prev.tail = (prev.tail or '') + element.tail
        else:
            parent.text = (parent.text or '') + element.tail
    parent.remove(element)

# see split_function.py
def count_func_avg_len_by_author(author):
    total_sum = 0
    file_cnt = 0
    file_list = os.listdir(author) if os.path.isdir(author) else [author]
    for file in file_list:
        per_file_sum = 0
        if not file.endswith('.xml'): continue
        file_cnt += 1
        filename = os.path.join(author, file)
        print(filename)
        p = init_parser(filename)
        funcs = get_functions(p)
        for func in funcs:
            pos_start = func.xpath('@pos:start', namespaces=ns)[0]
            pos_end = func.xpath('@pos:end', namespaces=ns)[0]
            line_start = int(pos_start.split(':')[0])
            line_end = int(pos_end.split(':')[0])
            per_file_sum += line_end - line_start
        if len(funcs) == 0: per_file_avg = 0
        else: per_file_avg = per_file_sum / len(funcs)
        total_sum += per_file_avg
    total_avg = total_sum / file_cnt
    return total_avg

def is_stmt(elem):
    stmt = ['break', 'case', 'continue', 'default', 'do', 'empty_stmt',
        'expr_stmt', 'for', 'goto', 'if_stmt', 'label', 'return',
        'switch', 'while', 'decl_stmt', 'else']
    tag = etree.QName(elem)
    if tag.localname in stmt:
        return True
    return False

def get_stmt_at_line(func, lineno):
    block = func.xpath('src:block/src:block_content//*', namespaces=ns)
    for child in block:
        if is_stmt(child):
            key = '{http://www.srcML.org/srcML/position}start'
            stmt_lineno = child.get(key).split(':')[0]
            if int(stmt_lineno) == lineno:
                return child
    return None

def get_next_stmt(func, start_from_elem):
    block = func.xpath('src:block/src:block_content', namespaces=ns)[0].getchildren()
    found_start_loc = False
    for child in block:
        if not found_start_loc and child != start_from_elem:
            continue
        if not found_start_loc:
            found_start_loc = True
            continue
        if is_stmt(child):
            return child
    return None

def list_elem(func):
    elems = []
    line_start = get_elem_line_start(func.xpath('src:block/src:block_content/descendant::node()/*', namespaces=ns)[0])
    line_end = get_elem_line_end(func)
    elem = get_stmt_at_line(func, line_start)
    while True:
        elem = get_next_stmt(func, elem)
        if elem == None: break
        elems.append(elem)
    return elems

def replace_call(func, call_to_replace, new_func_name, selector):
    calls = func.xpath('//src:call', namespaces=ns)
    for call in calls:
        call_name = call.xpath('src:name', namespaces=ns)[0]
        if call_name.text == call_to_replace:
            call_name.text = new_func_name
            argument_list = call.xpath('src:argument_list', namespaces=ns)[0]
            if argument_list.text.strip() == '(':
                argument_list.text = '(' + str(selector) + ','
            elif argument_list.text.strip() == '()':
                argument_list.text = '(' + str(selector) + ')'

def merge_func(parser, src_func, dst_func):
    # don't merge main()
    src_func_name = src_func.xpath('src:name', namespaces=ns)[0].text
    dst_func_name = dst_func.xpath('src:name', namespaces=ns)[0].text
    if src_func_name == 'main' or dst_func_name == 'main': return False

    # don't merge functions that have different types of return value
    src_func_type = ''.join(src_func.xpath('src:type', namespaces=ns)[0].itertext())
    dst_func_type = ''.join(dst_func.xpath('src:type', namespaces=ns)[0].itertext())
    if src_func_type != dst_func_type: return False

    #print('merging ', src_func_name, dst_func_name)
    # replace all original calls
    new_func_name = '_'.join(['merged', src_func_name, dst_func_name])
    #replace_call(parser, src_func_name, new_func_name, 0)
    #replace_call(parser, dst_func_name, new_func_name, 1)

    new_func = etree.Element('function')
    if_stmt = etree.Element('if_stmt')
    if_branch = etree.Element('if')
    else_branch = etree.Element('else')
    src_func_block_content = src_func.xpath('src:block/src:block_content', namespaces=ns)[0]
    dst_func_block_content = dst_func.xpath('src:block/src:block_content', namespaces=ns)[0]

    if_branch.text = 'if (selector == 0) {\n'
    if_branch.text += ''.join(src_func_block_content.itertext())
    if_branch.text += '}\n'

    else_branch.text = 'else if (selector == 1) {\n'
    else_branch.text += ''.join(dst_func_block_content.itertext())
    else_branch.text += '}\n'

    if_stmt.append(if_branch)
    if_stmt.append(else_branch)

    src_func_args = get_args_and_types(src_func)
    dst_func_args = get_args_and_types(dst_func)
    print(dst_func_args)

    # check for arguments which have the same name but not the same type
    shorter_list = src_func_args if len(src_func_args) < len(dst_func_args) else dst_func_args
    longer_list = src_func_args if len(src_func_args) >= len(dst_func_args) else dst_func_args
    for arg_name in shorter_list:
        if arg_name in longer_list:
            if longer_list[arg_name] != shorter_list[arg_name]:
                print('Argument conflict! Cannot merge!')
                return False

    # do the merging
    new_func.text = src_func_type + ' '
    new_func.text += new_func_name
    new_func.text += '(int selector,'
    dst_func_args.update(src_func_args)
    for arg_name in dst_func_args:
        new_func.text += dst_func_args[arg_name] + ' ' + arg_name + ','
    new_func.text = new_func.text[:-1]
    new_func.text += ') {\n'
    new_func.text += ''.join(if_stmt.itertext())
    new_func.text += '}\n'

    dst_func.getparent().replace(dst_func, new_func)
    src_func.getparent().remove(src_func)

    return True
    #print(''.join(new_func.itertext()))
    #print('==================')

# see split_function.py
def transform_by_line_cnt(src_author, dst_author, save_to='tmp.xml'):
    #src_avg_func_len = count_func_avg_len_by_author(src_author)
    dst_avg_func_len = count_func_avg_len_by_author(dst_author)
    #print('Src author function avg len:', src_avg_func_len)
    #print('Dst author function avg len:', dst_avg_func_len)
    #if dst_avg_func_len - 5 <= src_avg_func_len:
    #    print('Authors have similar function lengths, or source author has longer functions than the target. No need to split!')
    #    return
    file_list = os.listdir(src_author) if os.path.isdir(src_author) else [src_author]
    for file in file_list:
        if not file.endswith('.xml'): continue
        filename = os.path.join(src_author, file)
        p = init_parser(filename)
        funcs = get_functions(p)
        short_funcs = []
        other_funcs = []
        for func in funcs:
            func_len = get_elem_len(func)
            func_name = func.xpath('src:name', namespaces=ns)[0].text
            if func_name.startswith('split_') or func_name.startswith('merged_'):
                continue
            other_funcs.append((func, func_len))
            if func_len < dst_avg_func_len:
                short_funcs.append((func, func_len))

        merged_funcs = []
        for short_func, short_func_len in short_funcs:
            if short_func in merged_funcs: continue
            merged = False
            len_diff = dst_avg_func_len - short_func_len
            for other_func, other_func_len in other_funcs:
                if other_func in merged_funcs: continue
                if short_func == other_func: continue
                if merged: break
                if other_func != short_func and short_func_len + other_func_len <= dst_avg_func_len:
                    merged = merge_func(p, short_func, other_func)
                    if merged:
                        merged_funcs.append(short_func)
                        merged_funcs.append(other_func)
                    
        save_tree_to_file(doc, save_to)


def xml_file_path(xml_path):
    global flag
    # xml_path 需要转化的xml路径
    # sub_dir_list 每个作者的包名
    # name_list 具体的xml文件名
    save_xml_file = './transform_xml_file/merge_function'
    transform_java_file = './target_author_file/transform_java/merge_function'
    if not os.path.exists(transform_java_file):
        os.mkdir(transform_java_file)
    if not os.path.exists(save_xml_file):
        os.mkdir(save_xml_file)
    for xml_path_elem in xml_path:
            xmlfilepath = os.path.abspath(xml_path_elem)
            # 解析成树
            e = init_parser(xmlfilepath)
            # 转换
            flag = False
            transform(e)
            # 保存文件
            if flag == True:
                str = xml_path_elem.split('/')[-1]
                sub_dir = xml_path_elem.split('/')[-2]
                if not os.path.exists(os.path.join(save_xml_file, sub_dir)):
                    os.mkdir(os.path.join(save_xml_file, sub_dir))
                save_tree_to_file(doc, os.path.join(save_xml_file, sub_dir, str))
    return save_xml_file, transform_java_file

if __name__ == '__main__':
    src_author = sys.argv[1]
    dst_author = sys.argv[2]
    transform_by_line_cnt(src_author, dst_author)