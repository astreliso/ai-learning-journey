from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from math import log
import operator



def create_dataset():
    dataset = [[0, 0, 0, 0, 'no'],
            [0, 0, 0, 1, 'no'],
            [0, 1, 0, 1, 'yes'],
            [0, 1, 1, 0, 'yes'],
            [0, 0, 0, 0, 'no'],
            [1, 0, 0, 0, 'no'],
            [1, 0, 0, 1, 'no'],
            [1, 1, 1, 1, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [2, 0, 1, 2, 'yes'],
            [2, 0, 1, 1, 'yes'],
            [2, 1, 0, 1, 'yes'],
            [2, 1, 0, 2, 'yes'],
            [2, 0, 0, 0, 'no']]
    feature_names = ['F1-AGE', 'F2-WORK', 'F3-HOME', 'F4-LOAN']
    return dataset, feature_names

def create_tree(dataset, feature_names, used_feature_names):
    # 两个停止条件
    class_list = [example[-1] for example in dataset]
    if class_list.count(class_list[0]) == len(class_list):  # 全是同一类
        return class_list[0]
    if len(dataset[0]) == 1:    # 无特征可分
        return majority_cnt(class_list)
    # 取最佳特征并建树
    best_feat = choose_best_feature_to_split(dataset)
    best_feature_name = feature_names[best_feat]
    used_feature_names.append(best_feature_name)
    my_tree = {best_feature_name:{}}    # 嵌套字典建树
    del feature_names[best_feat]
    # 取值并开始递归
    feat_value = [example[best_feat] for example in dataset]
    unique_values = set(feat_value)
    for value in unique_values:
        sub_feature_names = feature_names[:]
        sub_data_set = split_data_set(dataset,best_feat,value)
        sub_tree = create_tree(sub_data_set, sub_feature_names, used_feature_names)
        my_tree[best_feature_name][value] = sub_tree
    return my_tree


def majority_cnt(class_list):
    class_count = {}                                    # 统计字典
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1                          # 计数
    sorted_class_count = sorted(class_count.items(),
                              key=operator.itemgetter(1),   # 按次数排序
                              reverse=True)                 # 降序
    return sorted_class_count[0][0]                      # 取次数最多的类别


def choose_best_feature_to_split(dataset):
    num_features = len(dataset[0])-1
    base_entropy = clac_shannon_ent(dataset)
    best_feature = -1
    best_info_gain = 0
    for i in range(num_features):
        feature_list =[example[i] for example in dataset]
        unique_values = set(feature_list)
        new_entropy = 0
        for value in unique_values:
            sub_dataset = split_data_set(dataset,i,value)
            prob = len(sub_dataset)/float(len(dataset))
            new_entropy += prob * clac_shannon_ent(sub_dataset)
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def split_data_set(dataset, axis, value):
    ret_dataset = []
    for feat_vec in dataset:
        if feat_vec[axis] == value:
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis+1:])
            ret_dataset.append(reduced_feat_vec)
    return ret_dataset


def clac_shannon_ent(dataset):
    num_examples = len(dataset)
    label_counts = {}
    for feat_vec in dataset:
        current_label = feat_vec[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    shannon_ent = 0
    for key in label_counts:
        prop =float(label_counts[key]) / num_examples
        shannon_ent -= prop * log(prop, 2)
    return shannon_ent

def get_num_leafs(my_tree):
    num_leafs = 0
    first_str = next(iter(my_tree))
    second_dict = my_tree[first_str]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__=='dict':
            num_leafs += get_num_leafs(second_dict[key])
        else:   num_leafs +=1
    return num_leafs


def get_tree_depth(my_tree):
    max_depth = 0
    first_str = next(iter(my_tree))
    second_dict = my_tree[first_str]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__=='dict':
            this_depth = 1 + get_tree_depth(second_dict[key])
        else:   this_depth = 1
        if this_depth > max_depth: max_depth = this_depth
    return max_depth

def plot_node(node_nxt, center_pt, parent_pt, node_type):
    arrow_args = dict(arrowstyle="<-")
    font = FontProperties(family='Arial Unicode MS', size=14)
    create_plot.ax1.annotate(node_nxt, xy=parent_pt,  xycoords='axes fraction',
        xytext=center_pt, textcoords='axes fraction',
        va="center", ha="center", bbox=node_type, arrowprops=arrow_args, fontproperties=font)


def plot_mid_text(cntr_pt, parent_pt, txt_string):
    x_mid = (parent_pt[0]-cntr_pt[0])/2.0 + cntr_pt[0]
    y_mid = (parent_pt[1]-cntr_pt[1])/2.0 + cntr_pt[1]
    create_plot.ax1.text(x_mid, y_mid, txt_string, va="center", ha="center", rotation=30)


def plot_tree(my_tree, parent_pt, node_txt):
    decision_node = dict(boxstyle="sawtooth", fc="0.8")
    leaf_node = dict(boxstyle="round4", fc="0.8")
    num_leafs = get_num_leafs(my_tree)
    depth = get_tree_depth(my_tree)
    first_str = next(iter(my_tree))
    cntr_pt = (plot_tree.xOff + (1.0 + float(num_leafs))/2.0/plot_tree.totalW, plot_tree.yOff)
    plot_mid_text(cntr_pt, parent_pt, node_txt)
    plot_node(first_str, cntr_pt, parent_pt, decision_node)
    second_dict = my_tree[first_str]
    plot_tree.yOff = plot_tree.yOff - 1.0/plot_tree.totalD
    for key in second_dict.keys():
        if type(second_dict[key]).__name__=='dict':
            plot_tree(second_dict[key],cntr_pt,str(key))
        else:
            plot_tree.xOff = plot_tree.xOff + 1.0/plot_tree.totalW
            plot_node(second_dict[key], (plot_tree.xOff, plot_tree.yOff), cntr_pt, leaf_node)
            plot_mid_text((plot_tree.xOff, plot_tree.yOff), cntr_pt, str(key))
    plot_tree.yOff = plot_tree.yOff + 1.0/plot_tree.totalD


def create_plot(inTree):
    fig = plt.figure(1, facecolor='white')													#创建fig
    fig.clf()																				#清空fig
    axprops = dict(xticks=[], yticks=[])
    create_plot.ax1 = plt.subplot(111, frameon=False, **axprops)    							#去掉x、y轴
    plot_tree.totalW = float(get_num_leafs(inTree))											#获取决策树叶结点数目
    plot_tree.totalD = float(get_tree_depth(inTree))											#获取决策树层数
    plot_tree.xOff = -0.5/plot_tree.totalW; plot_tree.yOff = 1.0;								#x偏移
    plot_tree(inTree, (0.5,1.0), '')															#绘制决策树
    plt.show()


if __name__ == '__main__':
    dataset, feature_names = create_dataset()
    used_feature_names = []
    my_tree = create_tree(dataset, feature_names, used_feature_names)
    create_plot(my_tree)







