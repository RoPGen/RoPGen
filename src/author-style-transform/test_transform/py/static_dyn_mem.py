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

def get_functions(e):
	return e('//src:function/src:block/src:block_content')

def get_decl_stmts(elem):
	return elem.xpath('src:decl_stmt', namespaces=ns)

def get_index(decl):
	return decl.xpath('src:name/src:index', namespaces=ns)

def get_array_size(elem):
	expr = elem.xpath('src:expr', namespaces=ns)[0]
	return ''.join(expr.itertext())

def get_array_decls(elem):
	array_decls = []
	decls = elem.xpath('src:decl', namespaces=ns)
	if len(decls) > 1: return array_decls
	for decl in decls:
		index = get_index(decl)
		init = decl.xpath('src:init', namespaces=ns)
		if len(index) >= 1 and len(init) < 1:
			array_decls.append(decl)
	return array_decls

def get_static_mem_allocs(e):
	static_mem_allocs = []
	functions = get_functions(e)
	for func in functions:
		decl_stmts = get_decl_stmts(func)
		if len(decl_stmts) < 1: continue
		for decl_stmt in decl_stmts:
			decls = get_array_decls(decl_stmt)
			if len(decls) < 1: continue
			for decl in decls:
				static_mem_allocs.append(decl)
	return static_mem_allocs

def get_typename(elem):
	if elem.xpath('src:type', namespaces=ns)[0].get('ref') == 'prev':
		return get_typename(elem.getprevious())
	return elem.xpath('src:type/src:name', namespaces=ns)

def save_tree_to_file(tree, file):
	with open(file, 'w') as f:
		f.write(etree.tostring(tree).decode('utf8'))

#静态内存分配转动态
#int a[8];
#int *a = (int *)malloc(sizeof(int) * 8);
#第一步：获取所有函数
#第二步：对每个函数，获取其中所有声明语句
#第三步：对每个声明语句，获取其中的数组下标方括号和初始化部分
#第四步：判断获取到有方括号且没有初始化部分，否则不合要求
#第五步：对每个合要求的声明，获取类型名和方括号内的数组长度
#第六步：构造动态分配声明语句的初始化部分：= malloc(sizeof(类型名) * 数组长度)
#第七步：把构造的初始化部分追加到方括号后面
#第八步：在类型名后面加上*
#第九步：删除方括号
def static_to_dyn(e, ignore_list=[], instances=None):
	global flag
	flag = False
	decls = [get_static_mem_allocs(e) if instances is None else (instance[0] for instance in instances)]
	tree_root = e('/*')[0].getroottree()
	new_ignore_list = []
	for item in decls:
		for decl in item:
			decl_stmt = decl.getparent()
			decl_stmt_prev = decl_stmt.getprevious()
			decl_stmt_prev = decl_stmt_prev if decl_stmt_prev is not None else decl_stmt
			decl_stmt_prev_path = tree_root.getpath(decl_stmt_prev)
			if decl_stmt_prev_path in ignore_list:
				continue
			type_name = get_typename(decl)
			if len(type_name) < 1: continue

			index = get_index(decl)
			if len(index) != 1: continue
			flag = True
			array_size = get_array_size(index[0])

			init_node = etree.Element('init')
			type_name_str = ''.join(type_name[0].itertext())
			init_node.text = ' = (' + type_name_str + '*)malloc(sizeof(' + type_name_str + ')*(' + array_size + '))'
			decl.append(init_node)

			type_name[0].tail = '*'
			index[0].getparent().remove(index[0])

			new_ignore_list.append(decl_stmt_prev_path)
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

def program_transform(program_path):
	e = init_parser(program_path)
	static_to_dyn(e)
	save_tree_to_file(doc, './style/style.xml')
if __name__ == '__main__':
	import sys
	e = init_parser(sys.argv[1])
	static_to_dyn(e)
	save_tree_to_file(doc, './static_to_dyn.xml')