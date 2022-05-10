from test_transform.py import const_vars

def get_program_style(file_path):
    if(const_vars.get_style(file_path)):
        print('4.1: 1', '4.2: 0')
        return {"4.1": 1, "4.2": 0}
    else:
        print('4.1: 0', '4.2: 1')
        return {"4.1": 0, "4.2": 1}

def transform(prog_fullpath, target_author, orig_prog_path, save_to, ignore_list):
	per_tf_ignore_list = []
	flag, doc, new_ignore_list = const_vars.program_transform(prog_fullpath, target_author, ignore_list)
	if flag:
		per_tf_ignore_list = new_ignore_list
		const_vars.save_file(doc, save_to)
	return per_tf_ignore_list

if __name__=='__main__':
    get_program_style('../xml_file/flym/za.xml')