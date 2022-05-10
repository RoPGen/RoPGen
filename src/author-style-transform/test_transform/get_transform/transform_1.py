from test_transform.py import var_name_style
def get_possible_styles():
	return ['1.1', '1.2', '1.3', '1.4']

def get_instances_and_styles(e, root):
	instances_and_styles = []
	camel_cases = []
	initcaps = []
	underscores = []
	init_underscores = []
	total_len = 0
	for decl in var_name_style.get_decls(e):
		name_node = var_name_style.get_declname(decl)[0]
		name_text = name_node.text
		if name_text == None:
			name_node = var_name_style.get_declname(name_node)[0]
			name_text = name_node.text
		if var_name_style.is_camel_case(name_text):
			camel_cases.append(decl)
		elif var_name_style.is_initcap(name_text):
			initcaps.append(decl)
		elif var_name_style.is_init_underscore(name_text):
			init_underscores.append(decl)
		elif var_name_style.is_underscore(name_text):
			underscores.append(decl)
	for instance in camel_cases:
		path_list = []
		path_list.append(root.getpath(instance))
		instances_and_styles.append((path_list, 1, '1.1'))
	for instance in initcaps:
		path_list = []
		path_list.append(root.getpath(instance))
		instances_and_styles.append((path_list, 1, '1.2'))
	for instance in underscores:
		path_list = []
		path_list.append(root.getpath(instance))
		instances_and_styles.append((path_list, 1, '1.3'))
	for instance in init_underscores:
		path_list = []
		path_list.append(root.getpath(instance))
		instances_and_styles.append((path_list, 1, '1.4'))
	return instances_and_styles

def get_program_style(xml_path):
	e = var_name_style.init_parser(xml_path)
	camel_cases = []
	initcaps = []
	underscores = []
	init_underscores = []
	init_dollars = []
	total_len = 0
	for decl in var_name_style.get_decls(e):
		if len(var_name_style.get_declname(decl))==0:continue
		name_node = var_name_style.get_declname(decl)[0]
		name_text = name_node.text
		if name_text == None:
			name_node = var_name_style.get_declname(name_node)[0]
			name_text = name_node.text
		if var_name_style.is_camel_case(name_text):
			camel_cases.append(decl)
		elif var_name_style.is_initcap(name_text):
			initcaps.append(decl)
		elif var_name_style.is_init_underscore(name_text):
			init_underscores.append(decl)
		elif var_name_style.is_underscore(name_text):
			underscores.append(decl)
		elif var_name_style.is_init_dollar(name_text):
			init_dollars.append(decl)
		if not var_name_style.is_all_lowercase(name_text) or '_' in name_text:
			total_len += 1
	print('1.1:', len(camel_cases), '1.2:', len(initcaps), '1.3:', len(underscores), '1.4:', len(init_underscores),
		  '1.5:', len(init_dollars))
	return {'1.1': len(camel_cases), '1.2': len(initcaps), '1.3': len(underscores),
			'1.4': len(init_underscores), '1.5': len(init_dollars)}

def transform(e, path_list, src_style, dst_style, save_to, ignore_list):
	#print(path_list)
	instances = [[e(path)[0] for path in path_list if len(e(path)) > 0]]
	#print(instances)
	per_tf_ignore_list = []
	flag, doc, new_ignore_list = var_name_style.transform(e, src_style, dst_style, ignore_list, instances)
	if flag:
		per_tf_ignore_list = new_ignore_list
		var_name_style.save_tree_to_file(doc, save_to)
	return per_tf_ignore_list
