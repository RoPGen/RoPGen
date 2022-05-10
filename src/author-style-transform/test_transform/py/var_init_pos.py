import sys
import os
from lxml import etree

ns = {'src': 'http://www.srcML.org/srcML/src',
	'cpp': 'http://www.srcML.org/srcML/cpp',
	'pos': 'http://www.srcML.org/srcML/position'}
doc = None
flag = True

def init_parser(file):
	global doc
	doc = etree.parse(file)
	e = etree.XPathEvaluator(doc)
	for k,v in ns.items():
		e.register_namespace(k, v)
	return e

def get_decl_stmts(e):
	return e('//src:function//src:decl_stmt')

def get_decl(elem):
	return elem.xpath('src:decl', namespaces=ns)

def get_typespec(elem):
	if elem is None: return []
	if len(elem.xpath('src:type', namespaces=ns))!=0:
		if elem.xpath('src:type', namespaces=ns)[0].get('ref') == 'prev':
			return get_typespec(elem.getprevious())
		return elem.xpath('src:type/src:specifier', namespaces=ns)

def get_init(elem):
	return elem.xpath('src:init', namespaces=ns)

def save_tree_to_file(tree, file):
	with open(file, 'w') as f:
		f.write(etree.tostring(tree).decode('utf8'))

def get_decl_init_stmts(e):
	decl_init_stmts = []
	decl_stmts = get_decl_stmts(e)
	for decl_stmt in decl_stmts:
		decls = get_decl(decl_stmt)
		for decl in decls:
			init = get_init(decl)
			type_node = get_typespec(decl)
			if type_node is not None and len(type_node) > 0:
				type_text = ''.join(type_node[0].itertext())
				#print(type_text)
				if 'const' in type_text: continue
			if len(init) == 0: continue
			var_name = init[0].getprevious().text
			#跳过数组的初始化
			if var_name == None: continue
			decl_init_stmts.append((decl_stmt, decl, 0, 0))
	return decl_init_stmts

# transform variable initialization to afterwards-assignment
#int a = 1;
#int a; a = 1;
# Step 1: get all declaration statements
# Step 2: for each of these statements, get its initializer part
# Step 3: if no initializer exists, skip
# Step 4: get the name of the variable declared
# Step 5: if it is an array, skip
# Step 6: construct new assignment statement: variable name + initializer
# Step 7: append the constructed assignment statement to original declaration
# Step 8: remove the initializer in the original declaration
def transform_standalone_stmts(e, instances=None, ignore_list=[]):
	global flag
	flag = False
	tree_root = e('/*')[0].getroottree()
	new_ignore_list = []
	decl_init_stmts = [get_decl_init_stmts(e) if instances is None else (instance[0] for instance in instances if len(instance)>0)]
	for decl_init_stmt in decl_init_stmts:
		for item in decl_init_stmt:
			decl_stmt = item[0]
			decl = item[1]
			stmt_prev = decl_stmt.getprevious()
			stmt_prev = stmt_prev if stmt_prev is not None else decl_stmt
			#取路径，看路径是否在传进来的忽略列表里
			stmt_prev_path = tree_root.getpath(stmt_prev)
			if stmt_prev_path in ignore_list:
				#print('变换过，忽略')
				continue
			#变换
			init = get_init(decl)
			if '{' in ''.join(init[0].itertext()): continue
			flag = True
			var_name = init[0].getprevious().text
			new_decl_stmt_node = etree.Element('decl_stmt')
			new_decl_stmt_node.text = var_name + ' ' + ''.join(init[0].itertext()) + ';'
			new_decl_stmt_node.tail = '\n'
			decl_stmt_index = decl_stmt.getparent().index(decl_stmt) + 1
			decl_stmt.getparent().insert(decl_stmt_index, new_decl_stmt_node)
			init[0].getparent().remove(init[0])
			stmt_prev = decl_stmt.getprevious()
			stmt_prev = stmt_prev if stmt_prev is not None else decl_stmt
			stmt_prev_path = tree_root.getpath(stmt_prev)
			new_ignore_list.append(stmt_prev_path)
	return flag, tree_root, new_ignore_list


def xml_file_path(xml_path):
	global flag
	# xml_path 需要转化的xml路径
	# sub_dir_list 每个作者的包名
	# name_list 具体的xml文件名
	save_xml_file = './transform_xml_file/var_init_pos'
	transform_java_file = './target_author_file/transform_java/var_init_pos'
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
			transform_standalone_stmts(e)
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
	transform_standalone_stmts(e)
	save_tree_to_file(doc, './style/style.xml')
if __name__ == '__main__':
	e = init_parser(sys.argv[1])
	transform_standalone_stmts(e)
	save_tree_to_file(doc, './var_init_pos.xml')
