from transform.py import incr_opr_prepost, incr_opr_usage

def get_possible_styles():
	return [1, 2, 4]

def get_instances_and_styles(e, root):
	instances_and_styles = []
	postfix_incrs = []
	prefix_incrs = []
	incr_full = []
	incr_plus_literals = []
	for expr, style in incr_opr_usage.get_incr_exprs(e, 3):
		if style == 1:
			postfix_incrs.append(expr)
		elif style == 2:
			prefix_incrs.append(expr)
		elif style == 3:
			incr_full.append(expr)
		elif style == 4:
			incr_plus_literals.append(expr)
	for instance in postfix_incrs:
		path_list = []
		path_list.append(root.getpath(instance))
		instances_and_styles.append((path_list, 10, 1))
	for instance in prefix_incrs:
		path_list = []
		path_list.append(root.getpath(instance))
		instances_and_styles.append((path_list, 10, 2))
	for instance in incr_plus_literals:
		path_list = []
		path_list.append(root.getpath(instance))
		instances_and_styles.append((path_list, 10, 4))
	return instances_and_styles

def get_program_style(xml_path, file_type):

	e = incr_opr_usage.init_parser(xml_path)
	prefix_incrs_len = 0
	postfix_incrs_len = 0
	incr_plus_literals_len = 0
	incr_full_len = 0
	for expr, style in incr_opr_usage.get_incr_exprs(e, 3):
		if style == 1:
			postfix_incrs_len += 1
		elif style == 2:
			prefix_incrs_len += 1
		elif style == 3:
			incr_full_len += 1
		elif style == 4:
			incr_plus_literals_len += 1
	print('10.1:', postfix_incrs_len, '10.2:', prefix_incrs_len, '10.3:', incr_full_len, '10.4:', incr_plus_literals_len)
	return {'10.1': postfix_incrs_len, '10.2': prefix_incrs_len, '10.3': incr_full_len, '10.4': incr_plus_literals_len}


def check_transform(auth_style, program_style, path_program, path_author, converted_styles):
	for key,value in program_style.items():
		if max(program_style.values())>0 and value == max(program_style.values()):
			if auth_style != key:
				converted_styles.append('10')
				# meets the conditions and starts the conversion
				incr_opr_prepost.program_transform(path_program, key, auth_style)
				break


def transform(e, path_list, src_style, dst_style, save_to, ignore_list):
	#print(path_list)
	instances = [[e(path)[0] for path in path_list if len(e(path)) > 0]]
	#print(instances)
	per_tf_ignore_list = []
	flag, doc, new_ignore_list = incr_opr_prepost.transform(e, src_style, dst_style, ignore_list, instances)
	if flag:
		per_tf_ignore_list = new_ignore_list
		incr_opr_prepost.save_tree_to_file(doc, save_to)
	return per_tf_ignore_list