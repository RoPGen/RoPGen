import sys
import os
from lxml import etree
from . import incr_opr_usage

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

def get_expr_stmts(e):
	return e('//src:expr_stmt')

def get_expr(elem):
	return elem.xpath('src:expr', namespaces=ns)

def get_operator(elem):
	return elem.xpath('src:operator', namespaces=ns)

def get_for_incrs(e):
	return e('//src:for/src:control/src:incr/src:expr')

def save_tree_to_file(tree, file):
	with open(file, 'w') as f:
		f.write(etree.tostring(tree).decode('utf8'))

def get_standalone_exprs(e):
	standalone_exprs = []
	#get all expression statements
	expr_stmts = get_expr_stmts(e)
	for expr_stmt in expr_stmts:
		expr = get_expr(expr_stmt)
		#there should be exactly one expression in a statement
		if len(expr) != 1: continue
		standalone_exprs.append(expr[0])
	return standalone_exprs

# not used
def transform_standalone_stmts(e):
	global flag
	exprs = get_standalone_exprs(e)
	for expr in exprs:
		opr = get_operator(expr)
		#and exactly one operator, which should be ++ or --
		if len(opr) == 1:
			if opr[0].text == '++':
				flag = True
				if opr[0].getparent().index(opr[0]) == 0:
					opr[0].getparent().remove(opr[0])
					expr.tail = '++;'
				else:
					opr[0].getparent().remove(opr[0])
					expr.text = '++'
			elif opr[0].text == '--':
				flag = True
				if opr[0].getparent().index(opr[0]) == 0:
					opr[0].getparent().remove(opr[0])
					expr.tail = '--;'
				else:
					opr[0].getparent().remove(opr[0])
					expr.text = '--'

# not used
def transform_for_loops(e):
	for incr in get_for_incrs(e):
		opr = get_operator(incr)
		if len(opr) == 1:
			if opr[0].text == '++':
				flag = True
				if opr[0].getparent().index(opr[0]) == 0:
					opr[0].getparent().remove(opr[0])
					incr.tail = '++;'
				else:
					opr[0].getparent().remove(opr[0])
					incr.text = '++'
			elif opr[0].text == '--':
				flag = True
				if opr[0].getparent().index(opr[0]) == 0:
					opr[0].getparent().remove(opr[0])
					incr.tail = '--;'
				else:
					opr[0].getparent().remove(opr[0])
					incr.text = '--'

# entry and dispatcher function
# actual code that does the transformation is in incr_opr_usage.py (except for style 9.1 to/from 9.2, which are in this function)
# arguments 'ignore_list' and 'instances' are pretty much legacy code and can be ignored
# argument 'e' is obtained by calling init_parser(srcML XML path)
# 'src_style' the style of source author
# 'dst_style' the style of target author 
def transform(e, src_style, dst_style, ignore_list=[], instances=None):
	global flag
	flag = False
	incr_exprs = [get_standalone_exprs(e) if instances is None else (instance[0] for instance in instances)]
	tree_root = e('/*')[0].getroottree()
	new_ignore_list = []
	src_dst_tuple = (src_style, dst_style)
	for item in incr_exprs:
		for incr_expr in item:
			incr_expr_grandparent = incr_expr.getparent().getparent()
			if incr_expr_grandparent is None:
				#print('判断失败')
				return flag, tree_root, new_ignore_list
			incr_expr_grandparent_path = tree_root.getpath(incr_expr_grandparent)
			if incr_expr_grandparent_path in ignore_list:
				#print('变换过，忽略')
				continue

			opr = get_operator(incr_expr)
			if len(opr) == 1:
				if opr[0].text == '++':
					flag = True
					if src_dst_tuple == ('9.2', '9.1'):
						opr[0].getparent().remove(opr[0])
						new_opr = etree.Element('operator')
						new_opr.text = '++'
						incr_expr.append(new_opr)
					elif src_dst_tuple == ('9.1', '9.2'):
						opr[0].getparent().remove(opr[0])
						incr_expr.text = '++'
					elif src_dst_tuple == ('9.1', '9.4'):
						incr_opr_usage.incr_to_separate_incr(opr, incr_expr)
					elif src_dst_tuple == ('9.2', '9.4'):
						incr_opr_usage.incr_to_separate_incr(opr, incr_expr)
					elif src_dst_tuple == ('9.4', '9.1'):
						incr_opr_usage.separate_incr_to_incr_postfix(opr)
					elif src_dst_tuple == ('9.4', '9.2'):
						incr_opr_usage.separate_incr_to_incr_prefix(opr)
					elif src_dst_tuple == ('9.1', '9.3'):
						incr_opr_usage.incr_to_full_incr(opr, incr_expr, 1)
					elif src_dst_tuple == ('9.2', '9.3'):
						incr_opr_usage.incr_to_full_incr(opr, incr_expr, 0)
					elif src_dst_tuple == ('9.4', '9.3'):
						incr_opr_usage.separate_incr_to_full_incr(opr, incr_expr)
				elif opr[0].text == '--':
					flag = True
					if src_dst_tuple == ('9.2', '9.1'):
						opr[0].getparent().remove(opr[0])
						new_opr = etree.Element('operator')
						new_opr.text = '--'
						incr_expr.append(new_opr)
					elif src_dst_tuple == ('9.1', '9.2'):
						opr[0].getparent().remove(opr[0])
						incr_expr.text = '--'
					elif src_dst_tuple == ('9.1', '9.4'):
						incr_opr_usage.incr_to_separate_incr(opr, incr_expr)
					elif src_dst_tuple == ('9.2', '9.4'):
						incr_opr_usage.incr_to_separate_incr(opr, incr_expr)
					elif src_dst_tuple == ('9.4', '9.1'):
						incr_opr_usage.separate_incr_to_incr_postfix(opr)
					elif src_dst_tuple == ('9.4', '9.2'):
						incr_opr_usage.separate_incr_to_incr_prefix(opr)
			elif len(opr) == 2:
				if src_dst_tuple == ('9.3', '9.1'):
					incr_opr_usage.full_incr_to_incr(opr, incr_expr, 1)
				elif src_dst_tuple == ('9.3', '9.2'):
					incr_opr_usage.full_incr_to_incr(opr, incr_expr, 0)
				elif src_dst_tuple == ('9.3', '9.4'):
					incr_opr_usage.full_incr_to_separate_incr(opr, incr_expr)
			if flag:
				new_ignore_list.append(incr_expr_grandparent_path)
	return flag, tree_root, new_ignore_list

def xml_file_path(xml_path):
	global flag
	# xml_path 需要转化的xml路径
	# sub_dir_list 每个作者的包名
	# name_list 具体的xml文件名
	save_xml_file = './transform_xml_file/incr_opr_prepost'
	transform_java_file = './target_author_file/transform_java/incr_opr_prepost'
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
def program_transform(program_path, style1, style2):
	list1 = []
	instances = None
	e = init_parser(program_path)
	transform(e, style1, style2, list1, instances)
	save_tree_to_file(doc, './style/style.xml')
if __name__ == '__main__':
	e = init_parser(sys.argv[1])
	#transform_standalone_stmts(e)
	transform(e,'9.4', '9.2')
	save_tree_to_file(doc, './incr_opr.xml')
