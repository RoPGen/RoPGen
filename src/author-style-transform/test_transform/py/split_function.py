import sys
import os
from lxml import etree

ns = {'src': 'http://www.srcML.org/srcML/src',
    'cpp': 'http://www.srcML.org/srcML/cpp',
    'pos': 'http://www.srcML.org/srcML/position'}
doc = None

def init_parser(file):
    global doc
    doc = etree.parse(file)
    e = etree.XPathEvaluator(doc)
    for k,v in ns.items():
        e.register_namespace(k, v)
    return e

def get_elem_line_start(elem):
    pos_start = elem.xpath('@pos:start', namespaces=ns)[0]
    line_start = int(pos_start.split(':')[0])
    return line_start

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
        if len(decl_stmt.xpath('src:decl/src:type', namespaces=ns))==0:continue
        decl_type = ''.join(decl_stmt.xpath('src:decl/src:type', namespaces=ns)[0].itertext())
        for decl in decl_stmt.xpath('src:decl', namespaces=ns):
            # type modifiers are a headache! we deal with them here
            # they can be a problem when constructing the arguments to the newly split out function
            modifier_text = ''
            type_ref = decl.attrib.get('ref', '')
            if type_ref == 'prev':
                modifiers = decl.xpath('src:type/src:modifier', namespaces=ns)
                if len(modifiers) > 0:
                    for modifier in modifiers:
                        modifier_text += ''.join(modifier.itertext())
            decl_name = ''.join(decl.xpath('src:name', namespaces=ns)[0].itertext())
            if '[' in decl_name: # array
                # srcML includes "[length]" as part of the variable name
                # so we should remove the brackets
                idx_start = decl_name.find('[')
                array_idx = decl_name[idx_start:]
                decl_name = decl_name.split('[')[0].strip()
                # include modifiers and array length in type name
                decl_type_this = decl_type + modifier_text
                decl_type_this += array_idx
                decl_name_type = vars.get(decl_name, '')
                if decl_name_type == '' or decl_name_type == decl_type_this:
                    vars[decl_name] = decl_type_this 
                else:
                    # we have multiple variables with the same name but not the same type
                    vars[decl_name] += ',' + decl_type_this
            else:
                decl_name_type = vars.get(decl_name, '')
                if decl_name_type == '' or decl_name_type == decl_type:
                    vars[decl_name] = decl_type + modifier_text
                else:
                    # we have multiple variables with the same name but not the same type
                    vars[decl_name] += ',' + decl_type + modifier_text
    # also include function parameters
    parameters = func.xpath('.//src:parameter', namespaces=ns)
    for param in parameters:
        if len(param.xpath('src:decl/src:type', namespaces=ns))==0:continue
        decl_type = ''.join(param.xpath('src:decl/src:type', namespaces=ns)[0].itertext())
        for decl in param.xpath('src:decl', namespaces=ns):
            if len(decl.xpath('src:name', namespaces=ns))==0:continue
            decl_name = ''.join(decl.xpath('src:name', namespaces=ns)[0].itertext())
            if '[' in decl_name:
                idx_start = decl_name.find('[')
                array_idx = decl_name[idx_start:]
                decl_name = decl_name.split('[')[0].strip()
                decl_type_this = decl_type
                decl_type_this += array_idx
                decl_name_type = vars.get(decl_name, '')
                if decl_name_type == '' or decl_name_type == decl_type_this:
                    vars[decl_name] = decl_type_this 
                else:
                    vars[decl_name] += ',' + decl_type_this
            else:
                decl_name_type = vars.get(decl_name, '')
                if decl_name_type == '' or decl_name_type == decl_type:
                    vars[decl_name] = decl_type
                else:
                    vars[decl_name] += ',' + decl_type
    return vars

def get_typename(elem):
	if elem is None: return None
	tag = etree.QName(elem)
	if tag.localname != 'decl': return None
	elem_type = elem.xpath('src:type', namespaces=ns)
	if len(elem_type) == 0 or elem[0].get('ref') == 'prev':
		return get_typename(elem.getprevious())
	return elem.xpath('src:type', namespaces=ns)

# get all temporary variables in 'func' and their types
# temporary variables are basically those that are defined between braces inside a function (i.e. excluding the outermost braces of the function itself)
def get_tmp_vars_and_types(func):
    global decl
    vars = {}
    decls = func.xpath('.//src:block//src:block//src:decl[not(ancestor::src:init)]', namespaces=ns)
    for decl in decls:
        if len(decl.xpath('src:name', namespaces=ns))==0:continue
        decl_name = ''.join(decl.xpath('src:name', namespaces=ns)[0].itertext())
        typename = get_typename(decl)
        if typename is None: continue
        decl_type = ''.join(typename[0].itertext())
        if '[' not in decl_name:
            decl_name_type = vars.get(decl_name, '')
            if decl_name_type == '' or decl_name_type == decl_type:
                vars[decl_name] = decl_type
            else:
                vars[decl_name] += ',' + decl_type
        else: # array
            idx_start = decl_name.find('[')
            array_idx = decl_name[idx_start:]
            decl_name = decl_name.split('[')[0].strip()
            decl_name_type = vars.get(decl_name, '')
            if decl_name_type == '' or decl_name_type == (decl_type + array_idx):
                vars[decl_name] = decl_type + array_idx
            else:
                vars[decl_name] += ',' + decl_type + array_idx
    for_inits = func.xpath('.//src:block//src:init/src:decl', namespaces=ns)
    for init in for_inits:
        if len(init.xpath('src:name', namespaces=ns))==0:continue
        decl_name = ''.join(init.xpath('src:name', namespaces=ns)[0].itertext())
        typename = get_typename(init)
        if typename is None: continue
        decl_type = ''.join(typename[0].itertext())
        if '[' not in decl_name:
            decl_name_type = vars.get(decl_name, '')
            if decl_name_type == '' or decl_name_type == decl_type:
                vars[decl_name] = decl_type
            else:
                vars[decl_name] += ',' + decl_type
        else: # array
            idx_start = decl_name.find('[')
            array_idx = decl_name[idx_start:]
            decl_name = decl_name.split('[')[0].strip()
            decl_name_type = vars.get(decl_name, '')
            if decl_name_type == '' or decl_name_type == (decl_type + array_idx):
                vars[decl_name] = decl_type + array_idx
            else:
                vars[decl_name] += ',' + decl_type + array_idx
    for_inits2 = func.xpath('.//src:control/src:init/src:expr/src:call', namespaces=ns)
    for init in for_inits2:
        typename = init.xpath('src:name', namespaces=ns)
        if typename is None or len(typename) == 0: continue
        decl_type = ''.join(typename[0].itertext())
        primitive_types = ['int', 'long']
        if decl_type not in primitive_types: continue
        name = init.xpath('src:argument_list/src:argument/src:expr/src:name', namespaces=ns)
        if len(name) == 0: continue
        decl_name = ''.join(name[0].itertext())
        decl_name = decl_name.strip()
        if '[' not in decl_name:
            decl_name_type = vars.get(decl_name, '')
            if decl_name_type == '' or decl_name_type == decl_type:
                vars[decl_name] = decl_type
            else:
                vars[decl_name] += ',' + decl_type
        else: # array
            idx_start = decl_name.find('[')
            array_idx = decl_name[idx_start:]
            decl_name = decl_name.split('[')[0]
            decl_name_type = vars.get(decl_name, '')
            if decl_name_type == '' or decl_name_type == (decl_type + array_idx):
                vars[decl_name] = decl_type + array_idx
            else:
                vars[decl_name] += ',' + decl_type + array_idx
    return vars

def get_functions(e):
    return e('//src:function')

def get_function(e, name):
    functions = e('//src:function')
    for func in functions:
        if func is None:continue
        func_name = func.xpath('src:name', namespaces=ns)
        if len(func_name)==0:continue
        if func_name[0].text == name:
            return func
    return None

def get_line(elem, lineno):
    line = elem.xpath('//*[starts-with(@pos:start, \'' + str(lineno) + ':\')]', namespaces=ns)
    return line

# copy lines from source code file 'f' and write into 'newf'
def identity_copy_linewise(f, newf, line_start, line_end=None):
    if line_end:
        for curr_line in range(line_start, line_end+1):
            newf.write(f.readline())
    else:
        line = f.readline()
        while line:
            newf.write(line)
            line = f.readline()

# copy characters from source code file 'f', save copied content into 'newbuf' and return it
# you need to seek() to the line where there are characters you want to copy first
def identity_copy_charwise(f, newbuf, col_start, col_end=None):
    if col_end:
        for _ in range(col_start, col_end+1):
            ch = f.read(1)
            newbuf += ch
    elif col_end != 0:
        crlf = [10, 13]
        next_char = f.read(1)
        next_next_char = f.read(1)
        f.seek(-1,1)
        while ord(next_char) not in crlf:
            newbuf += next_char
            next_char = f.read(1)
            next_next_char = f.read(1)
            f.seek(-1,1)
        if ord(next_next_char) in crlf and ord(next_char) != ord(next_next_char):
            f.seek(1,1)
        newbuf += b'\n'
    return newbuf

def save_tree_to_file(tree, file):
    with open(file, 'w') as f:
        f.write(etree.tostring(tree, encoding='utf8').decode('utf8'))

def remove_preserve_tail(element):
    parent = element.getparent()
    if element.tail:
        prev = element.getprevious()
        if prev is not None:
            prev.tail = (prev.tail or '') + element.tail
        else:
            parent.text = (parent.text or '') + element.tail
    parent.remove(element)

def check_member_access(func, line_start, line_end):
    operators = func.xpath('.//src:operator', namespaces=ns)
    for elem in operators:
        if elem.text != '.' and elem.text != '->': continue
        elem_line_start = get_elem_line_start(elem)
        if line_start <= elem_line_start <= line_end:
            return True
    return False

def check_continue_break(func, line_start, line_end):
	continue_breaks = func.xpath('.//src:continue', namespaces=ns) + func.xpath('.//src:break', namespaces=ns) + func.xpath('.//src:goto', namespaces=ns)
	for elem in continue_breaks:
		elem_line_start = get_elem_line_start(elem)
		if line_start <= elem_line_start <= line_end:
			return True
	return False

def read_lines_from_file(file, line_start, line_end):
    text = ''
    with open(file, 'r') as f:
        for i, line in enumerate(f):
            if line_start <= i+1 <= line_end:
                text += line
    return text

def check_return(func, line_start, line_end):
    # text = read_lines_from_file(file, line_start, line_end)
    # return 'return' in text
    returns = func.xpath('.//src:return', namespaces=ns)
    for elem in returns:
        elem_line_start = get_elem_line_start(elem)
        if line_start <= elem_line_start <= line_end:
            return True
    return False

def check_recursive_call(func, func_name, line_start, line_end):
    calls = func.xpath('.//src:call', namespaces=ns)
    for elem in calls:
        elem_line_start = get_elem_line_start(elem)
        if line_start <= elem_line_start <= line_end:
            if func_name in ''.join(elem.itertext()):
                return True
    return False

def check_func_decls(func):
    func_decls = func.xpath('.//src:function_decl', namespaces=ns)
    if len(func_decls) > 0:
            return True
    return False

# do the splitting
# argument 'xml_file' is the srcML XML path
# 'c_file' source code path
# 'func_name' name of the function to split
# 'line_start' start splitting from this line
# 'line_end' end splitting on this line
# 'new_file' path to a (currently non-existent) temporary file, could be arbitrary as long as it's accessible
def split_func(xml_file, c_file, func_name, line_start, line_end, new_file):
    e = init_parser(xml_file)
    f = open(c_file, 'rb')
    newf = open(new_file, 'wb')
    func = get_function(e, func_name)
    # get all variables in 'func'
    vars = get_vars_and_types(func)
    vars.update(get_tmp_vars_and_types(func))
    vars_pos = []
    vars_lines = set()
    new_vars = {}
    vars_to_pop = []
    ignore_vars = []
    keywords = ['size']
    line_start = int(line_start)
    line_end = int(line_end)
    # print(vars)

    # check for statements that affect control flow (e.g. continue or break) in the lines to split out,
    # because if the newly split function includes these statements it will be a syntax error
    # also check for other things that will lead to incorrect results
    if check_continue_break(func, line_start, line_end): return None
    if check_member_access(func, line_start, line_end): return None
    if check_recursive_call(func, func_name, line_start, line_end): return None
    if check_func_decls(func): return None
    identity_copy_linewise(f, newf, 1, line_start-1)

    # basically, this large chunk of code is:
    # looking for all variables used in the lines to split out, and saving them into 'new_vars'
    # also recording their line numbers and column numbers,
    # so that when we copy lines and characters from source code,
    # we can stop at the place where they occur and surround them with '(*...)' in the newly split function
    new_func_body = ''
    func_call_node = etree.Element('call')
    line_elems_dict = {}
    for curr_line in range(line_start, line_end+1):
        global doc
        line_elems = get_line(doc, curr_line)
        line_elems_dict[curr_line] = line_elems
    is_first = True
    lines_to_skip = []
    for curr_line in range(line_start, line_end+1):
        line_elems = line_elems_dict[curr_line]
        for elem in line_elems:
            if curr_line in lines_to_skip: continue
            tag = etree.QName(elem)
            if tag.localname == 'name':
                name = ''.join(elem.itertext())
                if name in ignore_vars: continue
                if name in keywords: return None
                parent = elem.getparent()
                parent_tag = etree.QName(parent)
                if parent_tag.localname == 'decl':
                    if '[' in name: name = name.split('[')[0]
                    if name in set(new_vars):
                        return None
                    ignore_vars.append(name)
                    continue
                type = vars.get(name, '')
                if '[' in type:
                    if '*' in type: return None
                    ary_len = type.split('[')[1]
                    ary_len = ary_len[:-1]
                    if not ary_len.isnumeric(): return None
                if ',' in type: return None
                if type == 'auto': return None
                # print(new_vars)
                if type != '':
                    new_vars[name] = type
                    pos_start = elem.xpath('@pos:start', namespaces=ns)[0]
                    pos_str = pos_start
                    pos_str += ','
                    pos_end = elem.xpath('@pos:end', namespaces=ns)[0]
                    pos_str += pos_end + ',' + name
                    pos_str += ',' + type
                    vars_pos.append(pos_str)
                    vars_lines.add(pos_start.split(':')[0])
            elif tag.localname == 'return':
                # skip return()
                lines_to_skip.append(curr_line)
                continue
            elif tag.localname == 'control':
                # for declarations like 'for (int(xxx)...'
                calls = elem.xpath('src:init/src:expr/src:call', namespaces=ns)
                primitive_types = ['int', 'long']
                for call in calls:
                    typename = call.xpath('src:name', namespaces=ns)
                    if typename is None or len(typename) == 0: continue
                    decl_type = ''.join(typename[0].itertext())
                    if decl_type not in primitive_types: continue
                    init_names = call.xpath('src:argument_list/src:argument/src:expr/src:name', namespaces=ns)
                    for init_name in init_names:
                        name = ''.join(init_name.itertext())
                        vars_to_pop.append(name)
            #pos_end = elem.xpath('@pos:end', namespaces=ns)[0]
            #end = int(pos_end.split(':')[0])
    # print(vars_to_pop)

    # remove lines to split out from the original function
    for curr_line in range(line_start, line_end+1):
        line_elems = line_elems_dict[curr_line]
        for elem in line_elems:
            if curr_line in lines_to_skip: continue
            if is_first:
                func_call_node.tail = elem.tail
                elem.getparent().replace(elem, func_call_node)
                is_first = False
            else:
                remove_preserve_tail(elem)
    for var in vars_to_pop:
        try:
            new_vars.pop(var)
            ignore_vars.append(var)
        except KeyError:
            continue

    # construct new function call statement and the beginning of the new function
    new_func_name = '_'.join(['split', func_name, str(line_start), str(line_end)])
    new_func = b'void ' + new_func_name.encode('utf8') + b'('
    new_func_call = new_func_name + '('
    is_first = True
    for k, v in new_vars.items():
        if not is_first:
            new_func += b','
            new_func_call += ','
        else:
            is_first = False
        if '[' not in v:
            new_func += (v.encode('utf8') + b'* ' + k.encode('utf8'))
            new_func_call += ('&' + k)
        else:
            new_func += (v.split('[')[0].encode('utf8') + b' ' + k.encode('utf8') + b'ary[' + '['.join(v.split('[')[1:]).encode('utf8'))
            new_func_call += k
    new_func += b') {\n'
    new_func_call += ');\n'
    func_call_node.text = new_func_call

    # basically, this chunk of code is:
    # copying lines and characters from source code into the newly split function
    # when a variable in 'vars_pos' is encountered, stop and surround it with '(*...)', and then proceed
    for curr_line in range(line_start, line_end+1):
        if curr_line in lines_to_skip: continue
        if str(curr_line) not in vars_lines:
            new_func += f.readline()
        else:
            pos_end = 0
            line_pos = 1
            last_pos_start = 0
            for item in vars_pos:
                line = int(item.split(',')[0].split(':')[0])
                if line == curr_line:
                    name = item.split(',')[2]
                    type = item.split(',')[3]
                    if name in ignore_vars: continue
                    pos_start = int(item.split(',')[0].split(':')[1])
                    pos_end = int(item.split(',')[1].split(':')[1])
                    new_func = identity_copy_charwise(f, new_func, line_pos, pos_start-1)
                    # print(new_func)
                    line_pos = pos_end + 1
                    if '[' not in type:
                        new_func += b'(*' + name.encode('utf8') + b')'
                    else:
                        new_func += name.encode('utf8') + b'ary'
                    f.read(len(name))
            new_func = identity_copy_charwise(f, new_func, pos_end)

    # end of new function
    new_func += b'}\n'
    newf.write(new_func_call.encode('utf8'))
    identity_copy_linewise(f, newf, line_end)

    new_func_node = etree.Element('function')
    new_func_node.text = new_func.decode('utf8')
    func_index = func.getparent().index(func)
    func.getparent().insert(func_index, new_func_node)

    f.close()
    newf.close()
    tree_root = e('/*')[0].getroottree()
    return tree_root.getpath(new_func_node)

# given an author or a source code file, calcuate the average length of all functions
# when given an author, each file by this author is processed individually and then averaged
def count_func_avg_len_by_author(author):
    total_sum = 0
    file_cnt = 0
    file_list = os.listdir(author) if os.path.isdir(author) else [author]
    for file in file_list:
        per_file_sum = 0
        if not file.endswith('.xml'): continue
        file_cnt += 1
        filename = os.path.join(author if os.path.isdir(author) else '', file)
        p = init_parser(filename)
        funcs = get_functions(p)
        for func in funcs:
            pos_start = func.xpath('@pos:start', namespaces=ns)[0]
            pos_end = func.xpath('@pos:end', namespaces=ns)[0]
            line_start = int(pos_start.split(':')[0])
            line_end = int(pos_end.split(':')[0])
            per_file_sum += line_end - line_start
        if len(funcs) ==0 :continue
        per_file_avg = per_file_sum / len(funcs)
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

# split function based on line number
# i.e. if the average function length of source author exceeds that of target author by too much (say 5), split one function of source author
# arguments 'src_author' path containing source author srcML XMLs
# 'dst_author' path containing target author srcML XMLs
# 'ignore_list' is pretty much legacy code and can be ignored
# 'srccode_path' path to source code (assuming 'src_author' is just path to one source code file), if None, defaults to the XML name, except that '.xml' is changed to '.cpp'
# 'save_to' path where resulting XML should be saved to
def transform_by_line_cnt(src_author, dst_author, ignore_list=[], srccode_path=None, save_to='tmp.xml'):
    global flag
    flag = False
    new_ignore_list = []
    #src_avg_func_len = count_func_avg_len_by_author(src_author)
    dst_avg_func_len = count_func_avg_len_by_author(dst_author)
    #print('Src author function avg len:', src_avg_func_len)
    #print('Dst author function avg len:', dst_avg_func_len)
    #if src_avg_func_len - 5 <= dst_avg_func_len:
    #    print('Authors have similar function lengths, or target author has longer functions than the source. No need to split!')
    #    return
    file_list = os.listdir(src_author) if os.path.isdir(src_author) else [src_author]
    for file in file_list:
        if not file.endswith('.xml'): continue
        filename = os.path.join(src_author if os.path.isdir(src_author) else '', file)
        p = init_parser(filename)
        tree_root = p('/*')[0].getroottree()
        funcs = get_functions(p)
        for func in funcs:
            func_len = get_elem_len(func)
            func_name = func.xpath('src:name', namespaces=ns)[0].text
            if func_name is not None and (func_name.startswith('split_') or func_name.startswith('merged_')):
                #print('变换过，忽略')
                continue
            if func_len - 5 <= dst_avg_func_len: continue
            len_diff = func_len - dst_avg_func_len
            elems = list_elem(func)[::-1]
            if len(elems) == 0: continue
            tag = etree.QName(elems[0])
            if tag.localname == 'return': del elems[0]
            elem_len_sum = 0
            start_elem = None
            for elem in elems:
                elem_len_sum += get_elem_len(elem)
                if elem_len_sum >= len_diff:
                    start_elem = elem
                    break
            if start_elem == None:continue
            flag = True
            start_elem_line_start = get_elem_line_start(start_elem)
            end_elem_line_end = get_elem_line_end(elems[0])
            c_file = srccode_path if srccode_path is not None else filename.replace('.xml', '.cpp')
            if not check_return(func, start_elem_line_start, end_elem_line_end):
                new_func_path = split_func(filename, c_file, func_name, start_elem_line_start, end_elem_line_end, 'new.c')
        save_tree_to_file(doc, save_to)
    return flag, tree_root, new_ignore_list

def count_nesting_level_by_func(func, level):
    children = func.getchildren()
    nested_elems = ['do', 'for', 'if_stmt', 'switch', 'while', 'else']
    max_level = level
    for child in children:
        tag = etree.QName(child)
        if tag.localname in nested_elems:
            if tag.localname == 'if_stmt':
                block_contents = child.xpath('src:if/src:block/src:block_content', namespaces=ns)
                else_branch = child.xpath('src:else/src:block/src:block_content', namespaces=ns)
                if len(else_branch) > 0: block_contents.append(else_branch[0])
            else:
                if len(child.xpath('src:block/src:block_content', namespaces=ns)) == 0:continue
                block_contents = [child.xpath('src:block/src:block_content', namespaces=ns)[0]]
            for block_content in block_contents:
                block_level = count_nesting_level_by_func(block_content, level+1)
                if block_level > max_level: max_level = block_level
    return max_level

# given an author or a source code file, calcuate the average nesting level (i.e. depth of nested branch/loop structures)
# when given an author, each file by this author is processed individually and then averaged
def count_avg_nesting_level_by_author(author):
    total_sum = 0
    file_cnt = 0
    file_list = os.listdir(author) if os.path.isdir(author) else [author]
    for file in file_list:
        per_file_sum = 0
        if not file.endswith('.xml'): continue
        file_cnt += 1
        filename = os.path.join(author if os.path.isdir(author) else '', file)
        p = init_parser(filename)
        funcs = get_functions(p)
        for func in funcs:
            func_name = func.xpath('src:name', namespaces=ns)[0].text
            block_content = func.xpath('src:block/src:block_content', namespaces=ns)[0]
            per_file_sum += count_nesting_level_by_func(block_content, 0)
        if len(funcs) == 0:continue
        per_file_avg = per_file_sum / len(funcs)
        total_sum += per_file_avg
    total_avg = total_sum / file_cnt
    return total_avg

# split function based on nesting level
# i.e. if the nesting level of one branch/loop construct of source author exceeds that of target author, split it
# see transform_by_line_cnt() for meaning of arguments
def transform_by_nesting_level(src_author, dst_author, ignore_list=[], srccode_path=None, save_to='tmp.xml'):
    global flag
    flag = False
    new_func_path = None
    new_ignore_list = []
    #src_avg_nesting_level = count_avg_nesting_level_by_author(src_author)
    dst_avg_nesting_level = count_avg_nesting_level_by_author(dst_author)
    # dst_avg_nesting_level = 0.5
    #print('Src author avg nesting level: ', src_avg_nesting_level)
    #print('Dst author avg nesting level: ', dst_avg_nesting_level)
    #if src_avg_nesting_level - 1 < dst_avg_nesting_level:
    #    print('Authors have similar nesting depths, or target author has deeper nesting than the source. No need to split!')
    #    return
    nested_elems = ['do', 'for', 'if_stmt', 'switch', 'while', 'else']
    file_list = os.listdir(src_author) if os.path.isdir(src_author) else [src_author]
    for file in file_list:
        if not file.endswith('.xml'): continue
        filename = os.path.join(src_author if os.path.isdir(src_author) else '', file)
        p = init_parser(filename)
        tree_root = p('/*')[0].getroottree()
        funcs = get_functions(p)
        for func in funcs:
            func_len = get_elem_len(func)
            func_name = func.xpath('src:name', namespaces=ns)[0].text
            if func_name.startswith('split_') or func_name.startswith('merged_'):
                #print('变换过，忽略')
                continue
            level = 0
            block_content = func.xpath('src:block/src:block_content', namespaces=ns)[0]
            elems_to_explore = [(block_content, level, func)]
            while len(elems_to_explore) > 0:
                next_elem = elems_to_explore[0]
                level = next_elem[1]
                # print(level, dst_avg_nesting_level)
                if level > dst_avg_nesting_level:
                    elem_to_split = next_elem[2]
                    line_start = get_elem_line_start(elem_to_split)
                    line_end = get_elem_line_end(elem_to_split)
                    func_name = func.xpath('src:name', namespaces=ns)[0].text
                    c_file = srccode_path if srccode_path is not None else filename.replace('.xml', '.cpp')
                    if not check_return(func, line_start, line_end): 
                        new_func_path = split_func(filename, c_file, func_name, line_start, line_end, 'new.c')
                else:
                    for child in next_elem[0].getchildren():
                        tag = etree.QName(child)
                        if tag.localname in nested_elems:
                            if tag.localname == 'if_stmt':
                                block_contents = child.xpath('src:if/src:block/src:block_content', namespaces=ns)
                                else_branch = child.xpath('src:else/src:block/src:block_content', namespaces=ns)
                                if len(else_branch) > 0: block_contents.append(else_branch[0])
                            else:
                                if len(child.xpath('src:block/src:block_content', namespaces=ns)) == 0:continue
                                block_contents = [child.xpath('src:block/src:block_content', namespaces=ns)[0]]
                            for block_content in block_contents:
                                elems_to_explore.append((block_content, level+1, child))
                del elems_to_explore[0]
            if new_func_path is not None:
                new_ignore_list.append(new_func_path)
        save_tree_to_file(doc, save_to)
    return flag, tree_root, new_ignore_list

def xml_file_path(xml_path):
    global flag
    # xml_path 需要转化的xml路径
    # sub_dir_list 每个作者的包名
    # name_list 具体的xml文件名
    save_xml_file = './transform_xml_file/static_dyn_mem'
    transform_java_file = './target_author_file/transform_java/static_dyn_mem'
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
            static_to_dyn(e)
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
    srccode_path = sys.argv[3]
    # transform_by_line_cnt(src_author, dst_author, [], srccode_path)
    transform_by_nesting_level(src_author, dst_author, [], srccode_path)