
import numpy as np
import matplotlib.pyplot as plt
import math

# Will split the given data for a specific feature (The index)

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


def load_data():
    data = pd.read_csv('hw2_question1.csv', names=[
                       'Clump', 'CellSize', 'CellShape', 'Adhesion',
                       'Epithelial', 'Nuclei', 'Chromatin', 'Nucleoli',
                       'Mitoses', 'Class'])
    return data


def split_d(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                                        random_state=42, stratify=y)

    return X_train, y_train, X_test, y_test


def split_data(index, data):

	groups = [[] for _ in range(10)]

	for s in data:
		groups[s[index]-1].append(s)

	return groups

def compute_entropy(group, data):

	e = 0

	data_in_group = 0
	for g in group:
		data_in_group += len(g)

	# For each group we need to compute the entropy and
	# and then sum it all up.
	for g in group:

		b, m = 0, 0  # Keep track of benign/malignant frequency

		for d in g:
			if d[9] == 2:
				b += 1
			else:
				m += 1

		if b == 0 or m == 0:
			continue
		else:
			p1 = float(b)/(m+b) * math.log(float(b)/(m+b), 2)
			p2 = float(m)/(m+b) * math.log(float(m)/(m+b), 2)

		e += (-(p1+p2))*(float(m+b)/data_in_group)

	return e

# I utilized the same method from the following website:
# https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/


def compute_gini_index(group, data):
	n_instances = float(sum([len(g) for g in group]))
	classes = [2, 4]

	gini = 0.0
	for g in group:
		size = float(len(g))
		# avoid divide by zero
		if size == 0:
			continue
		score = 0.0
		# score the group based on the score for each class
		for c in classes:
			p = [row[-1] for row in g].count(c) / size
			score += p * p 
		# weight the group score by its relative size
		gini += (1-score)*score * (size / n_instances)

	return gini


def determine_split(data, e):
	best_index, best_score, best_group = 1000000, 100000, None

	# For each feature in our data, we will test the split
	# and choose the one with the best entropy
	for f in range(9):
		# Split the group for feature f
		group = split_data(f, data)
		# Compute entropy for the given group
		if e:
			entropy = compute_entropy(group, data)
		else:
			entropy = compute_gini_index(group, data)

		if entropy <= best_score:
			best_index, best_score, best_group = f, entropy, group

	return {'index': best_index, 'score': best_score, 'group': best_group}

# This function was influenced by:
# https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/


def terminal(g):
	o = [row[9] for row in g]
	return max(set(o), key=o.count)


def split(node, max_depth, depth, e, test):

	for g in range(10):
		# If a group is empty, we can just take the majority
		# class for all the nodes on the level.
		if not node[g]:
			for i in range(10):
				if node[i] == 2:
					node[g] = 2
				elif node[i] == 4:
					node[g] = 4
				else:
					node[g] = 4
			continue

		if depth >= max_depth:
			node[g] = terminal(node[g])
			continue

		node[g] = determine_split(node[g], e)
		# for i in node[g]['group']:
		# 	print len(i)
		split(node[g]['group'], max_depth, depth+1, e, test)


def entropy_tree(data, test):

	root = determine_split(data, True)
	split(root['group'], 5, 1, True, test)
	return root


def gini_tree(data, test):
	root = determine_split(data, False)
	split(root['group'], 5, 1, False, test)
	return root

# Iterate through tree for sample s


def test_tree(node, s):
	if isinstance(node, dict):
		idx = node['index']
		return test_tree(node['group'][s[idx]-1], s)
	else:
		return node


def compute_accuracy(test, cls):

	num_correct = 0

	for i in range(len(cls)):
		if cls[i] == test[i][-1]:
			num_correct += 1

	return (float(num_correct)/len(test))


def print_tree(node, depth=0):
	if isinstance(node, dict):
		print('%s[X%d]' % ((depth*' ', (node['index']+1))))
		for g in node['group']:
			print_tree(g, depth+1)
	else:
		print('%s[%s]' % ((depth*' ', node)))


def main():
    df = load_data()
    X, y = df.values[:,:-1] ,df['Class'].values
    X_train, y_train, X_test, y_test = split_d(X, y)
    train = np.hstack((X_train, y_train[:,None]))
    test = np.hstack((X_test, y_test[:, None]))
    e_tree = entropy_tree(train, test)
    g_tree = gini_tree(train, test)

    e_classes = []
    g_classes = []
    for s in test:
        p = test_tree(e_tree, s)
        e_classes.append(p)
        p = test_tree(g_tree, s)
        g_classes.append(p)

    a = compute_accuracy(test, e_classes)
    print('Entropy Accuracy: %s ' % a)
    a = compute_accuracy(test, g_classes)
    print('Gini Index Accuracy: %s' % a)


if __name__ == '__main__':
    main()
