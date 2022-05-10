import sys
import os
from lxml import etree
from collections import Counter
from timeit import default_timer as timer

ns = {'src': 'http://www.srcML.org/srcML/src',
	'cpp': 'http://www.srcML.org/srcML/cpp',
	'pos': 'http://www.srcML.org/srcML/position'}
doc = None

def init_parser(file):
	global doc
	#print(file)
	doc = etree.parse(file)
	return doc

def get_functions(elem):
	return elem.xpath('//src:function/src:block/src:block_content', namespaces=ns)

def get_names(elem):
	return elem.xpath('//src:name', namespaces=ns)

def get_typename(elem):
	if elem is None: return None
	tag = etree.QName(elem)
	if tag.localname != 'decl': return None
	elem_type = elem.xpath('src:type', namespaces=ns)
	if len(elem_type) == 0 or elem[0].get('ref') == 'prev':
		return get_typename(elem.getprevious())
	return elem.xpath('src:type', namespaces=ns)

def is_incr_decr(name_elem):
	cond1 = name_elem.xpath('following-sibling::src:operator[text()="++" or text()="--"] | preceding-sibling::src:operator[text()="++" or text()="--"]', namespaces=ns)
	return cond1, ['incr_decr']

def judge_ancestor(name_elem):
	#loops = ['for', 'while', 'do', 'if_stmt', 'switch']
	loops = ['for']
	printfs_scanfs = ['printf', 'scanf']
	couts_cins = ['cout', 'cin']
	labels = set()
	ancestors = name_elem.xpath('ancestor::*', namespaces=ns)
	for ancestor in ancestors:
		tag = etree.QName(ancestor)
		if tag.localname == 'block':
			parent = ancestor.getparent()
			if parent is not None:
				tag = etree.QName(parent)
				if tag.localname in loops:
					labels.add(tag.localname)
		elif tag.localname == 'init':
			parent = ancestor.getparent()
			if parent is not None:
				tag = etree.QName(parent)
				if tag.localname == 'control':
					labels.add('for_init')
		elif tag.localname == 'condition':
			relops = ['<', '>', '<=', '>=', '==', '!=', '!', '~', '&lt;', '&gt;', '&lt;=', '&gt;=']
			parens = ['(', ')']
			parent = ancestor.getparent()
			pre_sibling = name_elem.getprevious()
			while pre_sibling is not None and pre_sibling.text in parens:
				pre_sibling = pre_sibling.getprevious()
			if pre_sibling is None: pre_sibling = name_elem
			next_sibling = name_elem.getnext()
			while next_sibling is not None and next_sibling.text in parens:
				next_sibling = next_sibling.getnext()
			if next_sibling is None: next_sibling = name_elem
			if parent is not None and pre_sibling is not None and next_sibling is not None:
				parent_tag = etree.QName(parent)
				pre_sibling_tag = etree.QName(pre_sibling)
				next_sibling_tag = etree.QName(next_sibling)
				if parent_tag.localname == 'control':
					if pre_sibling_tag.localname == 'operator' and pre_sibling.text in relops:
						labels.add('for_condition_right')
					elif next_sibling_tag.localname == 'operator' and next_sibling.text in relops:
						labels.add('for_condition_left')
		'''
		elif tag.localname == 'index':
			labels.add('index')
		elif tag.localname == 'call':
			names = ancestor.xpath('./src:name[text()="printf" or text()="scanf"]', namespaces=ns)
			for name in names:
				name_text = name.text
				if name_text in printfs_scanfs:
					labels.add(name_text)
		elif tag.localname == 'expr_stmt':
			names = ancestor.xpath('./src:expr/src:name[text()="cin"]', namespaces=ns)
			for name in names:
				name_text = name.text
				if name_text in couts_cins:
					labels.add(name_text)
		'''
	return len(labels)>0, labels

def get_all_vars_info(func, vars_and_types):
	vars_info = {}
	for k, v in vars_and_types.items():
		vars_info[k] = {'type': v, 'attribs': set()}
	#infos = [judge_ancestor, is_incr_decr]
	infos = [judge_ancestor]
	for name_elem in get_names(func):
		name = ''.join(name_elem.itertext())
		if '[' in name:
			name = name.split('[')[0]
		vars_info_by_name = vars_info.get(name, None)
		if vars_info_by_name is None: continue
		for judge_func in infos:
			ret = judge_func(name_elem)
			if ret[0]:
				vars_info_by_name['attribs'].update(ret[1])
	return vars_info

# get all temporary variables in 'func' and their types
# temporary variables are basically those that are defined between braces inside a function (i.e. excluding the outermost braces of the function itself)
def get_tmp_vars_and_types(func):
	global decl
	vars = {}
	vars_type = {}
	decls = func.xpath('.//src:function/src:block//src:block//src:decl[not(ancestor::src:init)]', namespaces=ns)
	for decl in decls:
		name = decl.xpath('src:name', namespaces=ns)
		if len(name)==0:continue
		decl_name = ''.join(name[0].itertext())
		typename = get_typename(decl)
		if typename is None or len(typename) == 0: continue
		decl_type = ''.join(typename[0].itertext())
		if '[' in decl_name: #array
			decl_name = decl_name.split('[')[0].strip()
			decl_type += '[]'
		vars[decl_name] = vars.get(decl_name, 0) + 1
		vars_type[decl_name] = decl_type
	for_inits = func.xpath('.//src:function/src:block//src:control/src:init/src:decl', namespaces=ns)
	for init in for_inits:
		if len(init.xpath('src:name', namespaces=ns))==0:continue
		decl_name = ''.join(init.xpath('src:name', namespaces=ns)[0].itertext())
		typename = get_typename(init)
		if typename is None or len(typename) == 0: continue
		decl_type = ''.join(typename[0].itertext())
		if '[' in decl_name: #array
			decl_name = decl_name.split('[')[0].strip()
			decl_type += '[]'
		vars[decl_name] = vars.get(decl_name, 0) + 1
		vars_type[decl_name] = decl_type
	for_inits2 = func.xpath('.//src:function/src:block//src:control/src:init/src:expr/src:call', namespaces=ns)
	for init in for_inits2:
		typename = init.xpath('src:name', namespaces=ns)
		if typename is None or len(typename) == 0: continue
		decl_type = ''.join(typename[0].itertext())
		primitive_types = ['int', 'long']
		if decl_type not in primitive_types: continue
		name = init.xpath('src:argument_list/src:argument/src:expr/src:name', namespaces=ns)
		if len(name) == 0: continue
		decl_name = ''.join(name[0].itertext())
		if '[' in decl_name: #array
			decl_name = decl_name.split('[')[0].strip()
			decl_type += '[]'
		vars[decl_name] = vars.get(decl_name, 0) + 1
		vars_type[decl_name] = decl_type
	return vars, vars_type

# get all variables and their types
def get_vars_and_types(func):
	vars = {}
	vars_type = {}
	decls = func.xpath('.//src:decl', namespaces=ns)
	for decl in decls:
		if len(decl.xpath('src:name', namespaces=ns)) == 0: continue
		decl_name = ''.join(decl.xpath('src:name', namespaces=ns)[0].itertext())
		typename = get_typename(decl)
		if typename is None or len(typename)==0: continue
		decl_type = ''.join(typename[0].itertext())
		if '[' not in decl_name:
			vars[decl_name] = vars.get(decl_name, 0) + 1
			vars_type[decl_name] = decl_type
		else: #array
			decl_name = decl_name.split('[')[0].strip()
			decl_type += '[]'
			vars[decl_name] = vars.get(decl_name, 0) + 1
			vars_type[decl_name] = decl_type
	for_inits2 = func.xpath('.//src:function/src:block//src:control/src:init/src:expr/src:call', namespaces=ns)
	for init in for_inits2:
		typename = init.xpath('src:name', namespaces=ns)
		if typename is None or len(typename) == 0: continue
		decl_type = ''.join(typename[0].itertext())
		name = init.xpath('src:argument_list/src:argument/src:expr/src:name', namespaces=ns)
		if len(name) == 0: continue
		decl_name = ''.join(name[0].itertext())
		if '[' in decl_name: #array
			decl_name = decl_name.split('[')[0].strip()
			decl_type += '[]'
		vars[decl_name] = vars.get(decl_name, 0) + 1
		vars_type[decl_name] = decl_type
	classnames = func.xpath('.//src:class', namespaces=ns)
	for classname in classnames:
		decl_type = 'class'
		name = classname.xpath('src:name', namespaces=ns)
		if len(name) == 0: continue
		decl_name = ''.join(name[0].itertext())
		vars[decl_name] = vars.get(decl_name, 0) + 1
		vars_type[decl_name] = decl_type
	structs = func.xpath('.//src:struct/src:decl', namespaces=ns)
	for struct in structs:
		decl_type = 'struct'
		name = struct.xpath('src:name', namespaces=ns)
		if len(name) == 0: continue
		decl_name = ''.join(name[0].itertext())
		if '[' in decl_name: #array
			decl_name = decl_name.split('[')[0].strip()
			decl_type += '[]'
		vars[decl_name] = vars.get(decl_name, 0) + 1
		vars_type[decl_name] = decl_type
	return vars, vars_type

def get_template_names(func):
	vars = {}
	vars_type = {}
	templates = func.xpath('.//src:template/src:parameter_list/src:parameter', namespaces=ns) + func.xpath('.//src:typedef', namespaces=ns) + \
	func.xpath('.//src:struct//src:decl', namespaces=ns) + func.xpath('.//src:union//src:decl', namespaces=ns)
	for template in templates:
		typename = template.xpath('src:type', namespaces=ns)
		if typename is None or len(typename) == 0: continue
		decl_type = ''.join(typename[0].itertext())
		name = template.xpath('src:name', namespaces=ns)
		if len(name) == 0: continue
		decl_name = ''.join(name[0].itertext())
		if '[' in decl_name: #array
			decl_name = decl_name.split('[')[0]
			decl_type += '[]'
		vars[decl_name] = vars.get(decl_name, 0) + 1
		vars_type[decl_name] = decl_type
	macros = func.xpath('.//cpp:define', namespaces=ns)
	for macro in macros:
		decl_type = 'macro'
		name = macro.xpath('.//src:name', namespaces=ns)
		if len(name) == 0: continue
		decl_name = ''.join(name[0].itertext())
		vars[decl_name] = vars.get(decl_name, 0) + 1
		vars_type[decl_name] = decl_type
	return vars

def get_func_class_names(elem):
	func_names = {}
	funcs = elem.xpath('.//src:function', namespaces=ns) + elem.xpath('.//src:class', namespaces=ns)
	for func in funcs:
		if len(func.xpath('src:name', namespaces=ns))==0:continue
		func_name = func.xpath('src:name', namespaces=ns)[0].text
		func_names[func_name] = func_names.get(func_name, 0) + 1
	return func_names

def save_tree_to_file(tree, file):
	with open(file, 'w') as f:
		f.write(etree.tostring(tree).decode('utf8'))

def check_ancestor_define(name):
	ancestor_defines = name.xpath('ancestor::cpp:define', namespaces=ns)
	return len(ancestor_defines) > 0

def replace_names(src_author_file, src_vars_cnt, src_vars_info, dst_vars_cnt, dst_vars_info, is_tmp, save_to, ignore_list=[], keep_log=False):
	i = 0
	is_first = True
	broken = False
	new_ignore_list = []
	var_replace_log = []
	# print(src_vars_cnt)
	import shutil

	dst_vars_cnt = {k: v for k, v in dst_vars_cnt}
	dst_vars_info = sorted(dst_vars_info.items(), key=lambda pair: dst_vars_cnt[pair[0]], reverse=True)
	dst_vars_info = {k: v for k, v in dst_vars_info}
	# print(dst_vars_cnt)

	dst_vars_list = [k for k in dst_vars_cnt]
	for src_name, src_cnt in src_vars_cnt:
		dst_name = ''
		if src_name in ignore_list: #不能变换已变换过的
			#print('变换过，忽略')
			continue
		# src_name_info = src_vars_info.get(src_name, None)
		# if src_name_info is None:
		# 	continue
		# if src_name_info['type'] == 'typename': continue

		# for each identifier name used by the source author, pick a name from the target author to replace it with
		for kd, vd in dst_vars_info.items():
			#if is_tmp and kd not in dst_vars_list: continue
			#vd = dst_vars_info.get(kd, None)
			#if vd is None: continue
			#if src_name_info['type'] == vd['type'] and src_name_info['attribs'] == vd['attribs']:
			#if src_name_info['type'] == vd['type']:
			dst_name = kd
			break
		if dst_name == '':
			continue
		#if i >= len(src_vars_cnt): break

		# some names with special significance, don't replace them
		whitelist = ['main', 'size', 'operator', 'map', 'count', 'left', 'end',
		 'index', 'length', 'test', 'swap', 'time', 'min', 'max', 'exp', 'log',
		 'less', 'j1', 'y0', 'y1', 'free', 'right', 'pow', 'div', '__attribute__',
		 'remove', 'int', 'exit', 'remainder', 'time', 'read']
		if src_name in whitelist:
			continue
		if dst_name in whitelist:
			dst_vars_info.pop(dst_name)
			continue
		# print(src_name, dst_name)

		# now a (src_name, dst_name) pair has been determined
		# start looking for src_name in source author's code
		# and replace it with dst_name
		file = src_author_file if is_first else file
		try:
			p = init_parser(file)
		except:
			backup = os.path.join('/'.join(save_to.split('/')[:-1]), 'transform.bak')
			if not is_first: shutil.copy(backup, save_to)
			else: save_tree_to_file(doc, save_to)
			broken = True
			break

		if keep_log:
			var_replace_log.append((src_name, dst_name))

		names = get_names(p)
		tmp_vars, tmp_vars_type = get_tmp_vars_and_types(p)

		for name in names:
			if check_ancestor_define(name): continue
			if is_tmp and tmp_vars.get(name.text, None) == None: continue

			if name.text == src_name:
				name.text = dst_name
				new_ignore_list.append(dst_name)
		dst_vars_info.pop(dst_name)
		
		file = save_to
		backup = os.path.join('/'.join(file.split('/')[:-1]), 'transform.bak')
		if not is_first: shutil.copy(file, backup)
		save_tree_to_file(doc, file)
		is_first = False
		i += 1
	try:
		if not is_first:
			p = init_parser(file)
		else:
			p = init_parser(src_author_file)
	except:
		backup = os.path.join('/'.join(save_to.split('/')[:-1]), 'transform.bak')
		shutil.copy(backup, save_to)
		broken = True
	return new_ignore_list, broken, p, var_replace_log

def get_vars_cnt_by_author(author, tmp_only=True, need_extra_info=False):
	dst_vars_cnt = {}
	dst_vars_type = {}
	dst_vars_info = {}
	file_list = os.listdir(author) if os.path.isdir(author) else [author]
	#print(file_list)
	for dst_filename in file_list:
		if not dst_filename.endswith('.xml'): continue
		dst_file = os.path.join(author if os.path.isdir(author) else '', dst_filename)
		p = init_parser(dst_file)
		vars, vars_type = get_tmp_vars_and_types(p) if tmp_only else get_vars_and_types(p)
		dst_vars_cnt = Counter(vars) + Counter(dst_vars_cnt)
		dst_vars_type.update(vars_type)
		if need_extra_info:
			vars_info = get_all_vars_info(p, vars_type)
			dst_vars_info.update(vars_info)
	return dst_vars_cnt, dst_vars_info

def get_template_names_by_author(author):
	templates = {}
	file_list = os.listdir(author) if os.path.isdir(author) else [author]
	for dst_filename in file_list:
		if not dst_filename.endswith('.xml'): continue
		dst_file = os.path.join(author if os.path.isdir(author) else '', dst_filename)
		p = init_parser(dst_file)
		templates = get_template_names(p)
	return templates

def get_func_name_cnt_by_author(author):
	func_name_cnt = {}
	file_list = os.listdir(author) if os.path.isdir(author) else [author]
	for dst_filename in file_list:
		if not dst_filename.endswith('.xml'): continue
		dst_file = os.path.join(author if os.path.isdir(author) else '', dst_filename)
		p = init_parser(dst_file)
		funcs = get_func_class_names(p)
		func_name_cnt = Counter(funcs) + Counter(func_name_cnt)
	# prioritize function names
	func_name_cnt = {k: func_name_cnt[k]*100 for k in set(func_name_cnt)}
	return func_name_cnt

# transform temporary identifier names
# basically collecting identifier names of source author and target author here
# arguments 'src_author' path containing source author srcML XMLs
# 'dst_author' path containing target author srcML XMLs
# 'ignore_list' is pretty much legacy code and can be ignored
# 'save_to' path where resulting XML should be saved to
def transform_tmp_id_names(src_author, dst_author, ignore_list=[], save_to='tmp.xml', keep_log=False):
	broken = False
	dst_vars_cnt, dst_vars_info = get_vars_cnt_by_author(dst_author, need_extra_info=True)
	src_vars_cnt, src_vars_info = get_vars_cnt_by_author(src_author, need_extra_info=True)
	src_all, src_all_vars_info = get_vars_cnt_by_author(src_author, tmp_only=False)
	src_funcs = get_func_name_cnt_by_author(src_author)
	src_templates = get_template_names_by_author(src_author)
	dst_templates = get_template_names_by_author(dst_author)
	src_all += src_funcs
	intersect = set(dst_vars_cnt).intersection(set(src_all))
	#print(src_vars_info)
	#print(intersect)
	#print(dst_vars_info)
	diff = {k : dst_vars_cnt[k] for k in set(dst_vars_cnt) - intersect - set(src_templates) - set(dst_templates)} # exclude identifier names that source author already uses
	dst_vars_info = {k : dst_vars_info[k] for k in set(dst_vars_info) - intersect - set(src_templates) - set(dst_templates)} # exclude identifier names that source author already uses
	src_vars_cnt = {k : src_vars_cnt[k] for k in set(src_vars_cnt) - set(src_templates)}
	src_vars_cnt = sorted(src_vars_cnt.items(), key = lambda d: d[1], reverse=True)
	diff = sorted(diff.items(), key = lambda d: d[1], reverse=True)
	#print(diff)
	file_list = os.listdir(src_author) if os.path.isdir(src_author) else [src_author]
	for src_filename in file_list:
		if not src_filename.endswith('.xml'): continue
		src_file = os.path.join(src_author if os.path.isdir(src_author) else '', src_filename)
		new_ignore_list, this_broken, doc, var_replace_log = replace_names(src_file, src_vars_cnt, src_vars_info, diff, dst_vars_info, True, save_to, ignore_list, keep_log)
		if this_broken: broken = True
	if not broken:
		save_tree_to_file(doc, save_to)
	if keep_log:
		return new_ignore_list, var_replace_log
	else:
		return new_ignore_list

def is_transformable(src_author, dst_author):
	dst_vars_cnt, dst_vars_info = get_vars_cnt_by_author(dst_author)
	src_vars_cnt, src_vars_info = get_vars_cnt_by_author(src_author)
	src_all, src_all_vars_info = get_vars_cnt_by_author(src_author, tmp_only=False)
	diff = {k : dst_vars_cnt[k] for k in set(dst_vars_cnt) - set(src_vars_cnt) - set(src_all)} #不能换成原作者已经有的变量名
	if len(src_vars_cnt) * len(diff) > 0: return True
	return False

def get_target_author_id_names(dst_author):
	dst_vars_cnt, dst_vars_info = get_vars_cnt_by_author(dst_author, tmp_only=False)
	return dst_vars_cnt

def select_both_id_names(src_author, dst_author, ignore_list=[], save_to='tmp.xml'):
	dst_all = get_vars_cnt_by_author(dst_author, False)
	src_all = get_vars_cnt_by_author(src_author, False)
	diff = {k : dst_all[k] for k in set(dst_all) - set(src_all)}
	src_all = sorted(src_all.items(), key = lambda d: d[1], reverse=True)
	diff = sorted(diff.items(), key = lambda d: d[1], reverse=False)
	#print(src_all, diff)
	file_list = os.listdir(src_author) if os.path.isdir(src_author) else [src_author]
	for src_filename in file_list:
		if not src_filename.endswith('.xml'): continue
		src_file = os.path.join(src_author if os.path.isdir(src_author) else '', src_filename)
		new_ignore_list = replace_names(src_file, src_all, diff, False, save_to, ignore_list)
	return new_ignore_list

if __name__ == '__main__':
	src_author = sys.argv[1]
	dst_author = sys.argv[2]
	transform_tmp_id_names(src_author, dst_author)
