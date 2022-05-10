from test_transform.py import select_tmp_id_names

def transform(prog_fullpath, target_author, orig_prog_path, save_to, ignore_list):
	per_tf_ignore_list = []
	new_ignore_list = select_tmp_id_names.transform_tmp_id_names(prog_fullpath, target_author, ignore_list, save_to)
	if len(new_ignore_list) > 0:
		per_tf_ignore_list = new_ignore_list
	return per_tf_ignore_list