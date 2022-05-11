from transform.py import init_declaration, var_init_split

def get_possible_styles():
	return [1,2]

def get_instances_and_styles(e, root):
	instances_and_styles = []
	style1 = var_init_split.get_multi_decl_stmts(e)
	style2 = init_declaration.get_b_cont(e)
	for instance in style1:
		path_list = []
		path_list.append(root.getpath(instance))
		instances_and_styles.append((path_list, 8, 1))
	for instance in style2:
		path_list = []
		path_list.append(root.getpath(instance))
		instances_and_styles.append((path_list, 8, 2))
	return instances_and_styles

def get_program_style(file_path, file_type):
    # 7.1
    e2 = var_init_split.init_parser(file_path)
    l2 = len(var_init_split.get_multi_decl_stmts(e2))
    # 7.2
    e1 = init_declaration.init_parser(file_path)
    l1 = init_declaration.get_style(e1)
    re = {'8.1': l2, '8.2': l1[1]}
    print('8.1:', l2, '8.2:', l1[1])
    return re

def check_transform(auth_style, program_style, path_program, path_author, converted_styles):
	if auth_style == '8.1' and program_style['8.2'] > 0:
		converted_styles.append('8')
		init_declaration.program_transform(path_program)
	elif auth_style == '8.2' and program_style['8.1'] > 0:
		converted_styles.append('8')
		var_init_split.program_transform(path_program)


def transform(e, path_list, src_style, dst_style, save_to, ignore_list):
	#print(path_list)
	instances = [[e(path)[0] for path in path_list if len(e(path)) > 0]]
	#print(instances)
	per_tf_ignore_list = []
	if dst_style == 1:
		flag, doc, new_ignore_list = init_declaration.tramsform(e, ignore_list, instances)
		if flag:
			per_tf_ignore_list = new_ignore_list
			init_declaration.save_file(doc, save_to)
	elif dst_style == 2:
		flag, doc, new_ignore_list = var_init_split.transform_standalone_stmts(e, ignore_list, instances)
		if flag:
			per_tf_ignore_list = new_ignore_list
			var_init_split.save_tree_to_file(doc, save_to)
	return per_tf_ignore_list
