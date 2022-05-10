import sys
import os
from lxml import etree

ns = {'src': 'http://www.srcML.org/srcML/src',
	'cpp': 'http://www.srcML.org/srcML/cpp',
	'pos': 'http://www.srcML.org/srcML/position'}
doc = None
flag = False

def init_parser(file):
	global doc
	doc = etree.parse(file)
	e = etree.XPathEvaluator(doc)
	for k,v in ns.items():
		e.register_namespace(k, v)
	return e

def get_decl_stmts(e):
	return e('//src:decl_stmt')

def get_decl(elem):
	return elem.xpath('src:decl', namespaces=ns)

def get_oprs(elem):
	return elem.xpath('src:operator', namespaces=ns)

def get_typename(elem):
	return elem.xpath('src:type/src:name', namespaces=ns)

def get_calls(elem):
	return elem.xpath('.//src:call', namespaces=ns)

def get_init(elem):
	return elem.xpath('.//src:init', namespaces=ns)

def get_arg_expr(elem):
	return elem.xpath('.//src:argument/src:expr', namespaces=ns)

def get_name(elem):
	return elem.xpath('src:name', namespaces=ns)

def get_malloc_calls(elem):
	malloc_calls = []
	calls = get_calls(elem)
	for call in calls:
		name = get_name(call)[0]
		if name.text == 'malloc':
			malloc_calls.append(call)
	return malloc_calls

# get all variable declarations where the variable is initialized using return value of malloc()
# also make sure malloc call is of the form "malloc(sizeof(some type) * elem num)"
def get_malloc_in_decls(e):
	malloc_in_decls = []
	for decl_stmt in get_decl_stmts(e):
		malloc_calls = get_malloc_calls(decl_stmt)
		if len(malloc_calls) < 1: continue
		decl = get_decl(decl_stmt)[0]
		type_node = get_typename(decl)
		if len(type_node) == 0: continue
		type_text = ''.join(type_node[0].itertext())
		if type_text is None: continue
		expr = get_arg_expr(decl_stmt)[0]
		oprs = get_oprs(expr)
		has_bad_opr = False
		for opr in oprs:
			if opr.text not in ['(', ')', '*']:
				has_bad_opr = True
				break
		if has_bad_opr: continue
		malloc_len = ''.join(expr.itertext())
		if 'sizeof' not in malloc_len: continue
		if type_text not in malloc_len: continue
		if '*' not in malloc_len: continue
		name = get_name(decl)[0]
		static_ary_name = name.text
		if static_ary_name is None: continue
		malloc_in_decls.append(decl_stmt)
	return malloc_in_decls

def save_tree_to_file(tree, file):
	with open(file, 'w') as f:
		f.write(etree.tostring(tree).decode('utf8'))

# transform arrays allocated on the heap into stack-based ones
# arguments 'ignore_list' and 'instances' are pretty much legacy code and can be ignored
# int *a = (int *)malloc(sizeof(int) * 8);
# int a[8];
# Step 1: find all declarations involving malloc()
# Step 2: for each one of these declarations, get the declared variable name and its type
# Step 3: get the argument to this malloc() call, which is an expression specifying how much space malloc() should get
# Step 4: get all operators in this expression
# Step 5: check that the operators are either '(', ')', or '*', if not, skip this declaration
# Step 6: check that 'sizeof' is present in the expression, if not, skip
# Step 7: check that the type name retrieved in Step 2 is present in the expression, if not, skip
# Step 8: check that operator '*' is present in the expression, if not, skip
# Step 9: split the expression by '*'
# Step 10: concatenate all parts that don't have 'sizeof' in them, this will be the length of the new stack-based array
# Step 11: construct a declaration statement for the new array: type_name + variable_name[length]
# Step 12: replace the heap-based array declaration by the new stack-based one
def dyn_to_static(e, ignore_list=[], instances=None):
	global flag
	flag = False
	decl_stmts = [get_malloc_in_decls(e) if instances is None else (instance[0] for instance in instances)]
	tree_root = e('/*')[0].getroottree()
	new_ignore_list = []
	for item in decl_stmts:
		for decl_stmt in item:
			decl_stmt_prev = decl_stmt.getprevious()
			decl_stmt_prev = decl_stmt_prev if decl_stmt_prev is not None else decl_stmt
			decl_stmt_prev_path = tree_root.getpath(decl_stmt_prev)
			if decl_stmt_prev_path in ignore_list:
				print('变换过，忽略')
				continue

			decl = get_decl(decl_stmt)
			if len(decl) > 1:
				continue
			else:
				decl = decl[0]
			type_node = get_typename(decl)
			if len(type_node) == 0: continue
			type_text = ''.join(type_node[0].itertext())
			if type_text is None: continue
			expr = get_arg_expr(decl_stmt)[0]
			oprs = get_oprs(expr)
			malloc_len = ''.join(expr.itertext())
			if type_text not in malloc_len: continue
			if type_text + '*' in malloc_len.strip(): continue
			factors = malloc_len.split('*')
			static_ary_len = ''
			for factor in factors:
				if not factor.isnumeric(): continue
				if 'sizeof' not in factor:
					static_ary_len += factor + '*'
			static_ary_len = static_ary_len[:-1].strip()
			if len(static_ary_len) == 0: continue

			init = get_init(decl)
			if len(init) == 0: continue
			init = init[0]
			init.getparent().remove(init)

			name = get_name(decl)[0]
			static_ary_name = name.text
			if static_ary_name is None: continue
			flag = True

			ary_decl_stmt = type_text + ' ' + static_ary_name + '[' + static_ary_len + '];'
			#print(ary_decl_stmt)

			new_elem = etree.Element('decl_stmt')
			new_elem.text = ary_decl_stmt
			new_elem.tail = decl_stmt.tail
			decl_stmt.getparent().replace(decl_stmt, new_elem)

			new_ignore_list.append(decl_stmt_prev_path)
	return flag, tree_root, new_ignore_list

def xml_file_path(xml_path):
	global flag
	# xml_path 需要转化的xml路径
	# sub_dir_list 每个作者的包名
	# name_list 具体的xml文件名
	save_xml_file = './transform_xml_file/dyn_static_mem'
	transform_java_file = './target_author_file/transform_java/dyn_static_mem'
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
			dyn_to_static(e)
			# 保存文件
			if flag == True:
				str = xml_path_elem.split('/')[-1]
				sub_dir = xml_path_elem.split('/')[-2]
				if not os.path.exists(os.path.join(save_xml_file, sub_dir)):
					os.mkdir(os.path.join(save_xml_file, sub_dir))
				save_tree_to_file(doc, os.path.join(save_xml_file, sub_dir, str))
	return save_xml_file, transform_java_file

def program_transform(program_path):
	e = init_parser(program_path)
	dyn_to_static(e)
	save_tree_to_file(doc, './style/style.xml')
if __name__ == '__main__':
	import sys
	e = init_parser(sys.argv[1])
	dyn_to_static(e)
	save_tree_to_file(doc, './dyn_static_mem.xml')
