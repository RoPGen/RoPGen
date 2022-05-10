from test_transform.py import include

def get_program_style(program_path, author_path):
    a, b, c = include.transform_include(program_path, author_path)
    # print("12:", c)
    return {'12': c}

def transform(prog_fullpath, target_author, orig_prog_path, save_to, ignore_list):
	per_tf_ignore_list = []
	flag, doc, new_ignore_list = include.program_transform(prog_fullpath, target_author, ignore_list)
	if flag:
		per_tf_ignore_list = new_ignore_list
		include.save_tree_to_file(doc, save_to)
	return per_tf_ignore_list
