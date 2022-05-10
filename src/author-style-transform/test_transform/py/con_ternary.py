import os
from lxml import etree
from lxml.etree import Element
flag = True
ns = {'src': 'http://www.srcML.org/srcML/src'}
doc = None
str = '{http://www.srcML.org/srcML/src}'
def init_parse(file):
    global doc
    doc = etree.parse(file)
    return etree.XPathEvaluator(doc, namespaces={'src': 'http://www.srcML.org/srcML/src'})
def get_if(e):
    return e('//src:if_stmt')
def get_condition(elem):
    return elem.xpath('src:if/src:condition/src:expr', namespaces=ns)
def get_if_block(elem):
    return elem.xpath('src:if/src:block/src:block_content/src:expr_stmt/src:expr', namespaces=ns)
def get_else_block(elem):
    return elem.xpath('src:else/src:block/src:block_content/src:expr_stmt/src:expr', namespaces=ns)
def save_tree_to_file(tree, file):
    with open(file, 'w') as f:
        f.write(etree.tostring(tree).decode('utf8'))

def trans_tree(e):
    global flag
    # 得到所有的if
    if_stmts = get_if(e)
    for if_stmt in if_stmts:
        # 语句必须有if 和 else
        if len(if_stmt) > 2:
            if if_stmt[0].tag == str + 'if' and if_stmt[1].tag == str + 'else':
                # 得到if的条件
                if_condition = get_condition(if_stmt)[0]
                # 得到if的执行语句
                if_expr = get_if_block(if_stmt)
                # 得到else的执行语句
                else_expr = get_else_block(if_stmt)
                # if和else的都只能有一个执行语句
                if len(if_expr)==1 and len(else_expr)==1:
                    # 执行语句的标签数量一定>=3
                    if len(if_expr[0]) and len(else_expr[0]) >=3:
                        # 判断if else的执行语句符合合成三目运算的规则？
                        if if_expr[0][0].text == else_expr[0][0].text and if_expr[0][1].text == '=' and else_expr[0][1].text == '=':
                            del if_expr[0][0], if_expr[0][0]
                            # 构造三目运算结构
                            node_1 = Element('ternary')
                            node_1.append(else_expr[0][0])
                            node_1.append(else_expr[0][0])
                            # 条件后面需要加'?'
                            if_condition.tail = '?'
                            # 将条件增加到结构里
                            node_1.append(if_condition)
                            # 增加then结构
                            node_2 = Element('then')
                            # 去掉结尾的‘;’
                            if_expr[0].tail = ''
                            node_2.append(if_expr[0])
                            # 将then标签增加到ternary结构里
                            node_1.append(node_2)
                            # 最后增加else标签
                            node_3 = Element('else')
                            node_3.text = ':'
                            node_3.append(else_expr[0])
                            # 将else标签增加到ternary结构里
                            node_1.append(node_3)
                            # 删除if语句和else语句
                            del if_stmt[0], if_stmt[0]
                            # 将三目运算插入到if位置
                            if_stmt.append(node_1)
                            # 转化成功
                            flag = True


def count(e):
    count_num = 0
    return count_num
def get_number(xml_path):
    xmlfilepath = os.path.abspath(xml_path)
    # 解析成树
    e = init_parse(xmlfilepath)
    return count(e)
def xml_file_path(xml_path):
    global flag
    # xml_path 需要转化的xml路径
    # sub_dir_list 每个作者的包名
    # name_list 具体的xml文件名
    save_xml_file = './transform_xml_file/con_ternary'
    transform_java_file = './target_author_file/transform_java/con_ternary'
    if not os.path.exists(transform_java_file):
        os.mkdir(transform_java_file)
    if not os.path.exists(save_xml_file):
        os.mkdir(save_xml_file)
    for xml_path_elem in xml_path:
        xmlfilepath = os.path.abspath(xml_path_elem)
        # 解析成树
        e = init_parse(xmlfilepath)
        # 转换
        flag = False
        trans_tree(e)
        # 保存文件
        if flag == True:
            str = xml_path_elem.split('/')[-1]
            sub_dir = xml_path_elem.split('/')[-2]
            if not os.path.exists(os.path.join(save_xml_file, sub_dir)):
                os.mkdir(os.path.join(save_xml_file, sub_dir))
            save_tree_to_file(doc, os.path.join(save_xml_file, sub_dir, str))
    return save_xml_file, transform_java_file
