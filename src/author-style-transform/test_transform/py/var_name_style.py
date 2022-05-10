import sys
from lxml import etree
import inflection

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

def get_names(e):
	return e('//src:name')

def get_decl(elem):
	return elem.xpath('src:decl', namespaces=ns)

def get_declname(elem):
	return elem.xpath('src:name', namespaces=ns)

def save_tree_to_file(tree, file):
	with open(file, 'w') as f:
		f.write(etree.tostring(tree).decode('utf8'))

def is_all_lowercase(name):
	return name.lower() == name

def is_all_uppercase(name):
	return name.upper() == name

def is_camel_case(name):
	if is_all_lowercase(name): return False
	if not name[0].isalpha(): return False
	return inflection.camelize(name, uppercase_first_letter=False) == name

def is_initcap(name):
	if is_all_uppercase(name): return False
	if not name[0].isalpha(): return False
	return inflection.camelize(name, uppercase_first_letter=True) == name

def is_underscore(name):
	return '_' in name.strip('_')

def is_init_underscore(name):
	return name[0] == '_' and name[1:].strip('_') != ''

def is_init_dollar(name):
	return name[0] == '$' and name[1:].strip('$') != ''

def underscore_to_initcap(name):
	if not is_underscore(name): return name
	new_name = ''.join(name[0].upper())
	is_prev_underscore = False
	for ch in name[1:]:
		if ch == '_':
			is_prev_underscore = True
		else:
			if is_prev_underscore:
				new_name += ch.upper()
				is_prev_underscore = False
			else:
				new_name += ch
	return new_name

def underscore_to_camel(name):
	if not is_underscore(name): return name
	new_name = ''
	is_prev_underscore = False
	for ch in name:
		if ch == '_':
			is_prev_underscore = True
		else:
			if is_prev_underscore:
				new_name += ch.upper()
				is_prev_underscore = False
			else:
				new_name += ch
	return new_name

def underscore_to_init_symbol(name, symbol):
	if not is_underscore(name): return name
	return symbol + name

def init_symbol_to_underscore(name):
	if not is_init_underscore(name) and not is_init_dollar(name): return name
	new_name = name[1:]
	if is_camel_case(new_name):
		return camel_to_underscore(new_name)
	elif is_initcap(new_name):
		return initcap_to_underscore(new_name)
	return new_name

def camel_to_underscore(name):
	if not is_camel_case(name): return name
	new_name = ''
	for ch in name:
		if ch.isupper():
			new_name += '_' + ch.lower()
		else:
			new_name += ch
	return new_name

def initcap_to_underscore(name):
	if not is_initcap(name): return name
	new_name = ''.join(name[0].lower())
	for ch in name[1:]:
		if ch.isupper():
			new_name += '_' + ch.lower()
		else:
			new_name += ch
	return new_name

def to_upper(name):
	return name.upper()

def get_decls(e):
	decls = []
	decl_stmts = get_decl_stmts(e)
	for decl_stmt in decl_stmts:
		decl_list = get_decl(decl_stmt)
		for decl in decl_list:
			decls.append(decl)
	return decls

# entry and dispatcher function
# arguments 'ignore_list' and 'instances' are pretty much legacy code and can be ignored
# argument 'e' is obtained by calling init_parser(srcML XML path)
# 'src_style' the style of source author
# 'dst_style' the style of target author 
def transform(e, src_style, dst_style, ignore_list=[], instances=None):
	global flag
	flag = False
	decls = [get_decls(e) if instances is None else (instance[0] for instance in instances)]
	tree_root = e('/*')[0].getroottree()
	new_ignore_list = []
	for item in decls:
		for decl in item:
			decl_stmt = decl.getparent()
			decl_stmt_prev = decl_stmt.getprevious()
			decl_stmt_prev = decl_stmt_prev if decl_stmt_prev is not None else decl_stmt
			decl_stmt_prev_path = tree_root.getpath(decl_stmt_prev)
			if decl_stmt_prev_path in ignore_list:
				#print('变换过，忽略')
				continue

			flag = True
			if len(get_declname(decl)) == 0:continue
			name_node = get_declname(decl)[0]
			name_text = name_node.text
			if name_text == None:
				name_node = get_declname(name_node)[0]
				name_text = name_node.text
			src_dst_tuple = (src_style, dst_style)
			if src_dst_tuple == ('1.1', '1.2'):
				new_name = underscore_to_initcap(camel_to_underscore(name_text))
			elif src_dst_tuple == ('1.1', '1.3'):
				new_name = camel_to_underscore(name_text)
			elif src_dst_tuple == ('1.1', '1.4'):
				new_name = underscore_to_init_symbol(camel_to_underscore(name_text), '_')
			elif src_dst_tuple == ('1.1', '1.5'):
				new_name = underscore_to_init_symbol(camel_to_underscore(name_text), '$')
			elif src_dst_tuple == ('1.2', '1.1'):
				new_name = underscore_to_camel(initcap_to_underscore(name_text))
			elif src_dst_tuple == ('1.2', '1.3'):
				new_name = initcap_to_underscore(name_text)
			elif src_dst_tuple == ('1.2', '1.4'):
				new_name = underscore_to_init_symbol(initcap_to_underscore(name_text), '_')
			elif src_dst_tuple == ('1.2', '1.5'):
				new_name = underscore_to_init_symbol(initcap_to_underscore(name_text), '$')
			elif src_dst_tuple == ('1.3', '1.1'):
				new_name = underscore_to_camel(name_text)
			elif src_dst_tuple == ('1.3', '1.2'):
				new_name = underscore_to_initcap(name_text)
			elif src_dst_tuple == ('1.3', '1.4'):
				new_name = underscore_to_init_symbol(name_text, '_')
			elif src_dst_tuple == ('1.3', '1.4'):
				new_name = underscore_to_init_symbol(name_text, '$')
			elif src_dst_tuple == ('1.4', '1.1'):
				new_name = underscore_to_camel(init_symbol_to_underscore(name_text))
			elif src_dst_tuple == ('1.4', '1.2'):
				new_name = underscore_to_initcap(init_symbol_to_underscore(name_text))
			elif src_dst_tuple == ('1.4', '1.3'):
				new_name = init_symbol_to_underscore(name_text)
			elif src_dst_tuple == ('1.4', '1.5'):
				new_name = underscore_to_init_symbol(init_symbol_to_underscore(name_text), '$')
			elif src_dst_tuple == ('1.5', '1.1'):
				new_name = underscore_to_camel(init_symbol_to_underscore(name_text))
			elif src_dst_tuple == ('1.5', '1.2'):
				new_name = underscore_to_initcap(init_symbol_to_underscore(name_text))
			elif src_dst_tuple == ('1.5', '1.3'):
				new_name = init_symbol_to_underscore(name_text)
			elif src_dst_tuple == ('1.5', '1.4'):
				new_name = underscore_to_init_symbol(init_symbol_to_underscore(name_text), '_')

			whitelist = ['main', 'size', 'operator', 'case']
			names = get_names(e)
			name_list = [name.text for name in names]
			if new_name in whitelist or new_name in name_list:
				continue
			name_node.text = new_name

			for name in names:
				if name.text == name_text:
					name.text = new_name

			new_ignore_list.append(decl_stmt_prev_path)
	return flag, tree_root, new_ignore_list

def transform_standalone_stmts(e):
	for decl in get_decls(e):
		name_node = get_declname(decl)[0]
		name_text = name_node.text
		if name_text == None:
			name_node = get_declname(name_node)[0]
			name_text = name_node.text
		print(initcap_to_underscore(name_text))

if __name__ == '__main__':
	e = init_parser(sys.argv[1])
	transform(e,'1.3','1.1')
	#print(get_program_style(e))
	save_tree_to_file(doc, './var_name_style.xml')

def program_transform(program_path, style1, style2):
	ignore_list = []
	instances = None
	e = init_parser(program_path)
	transform(e, style1, style2, ignore_list, instances)
	save_tree_to_file(doc, './style/style.xml')