# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 12:37:26 2021

@author: Joey
"""

## from scratch implementation of various tree-based methods

import numpy as np
import pandas as pd

## load premade datasets to use

from sklearn.datasets import load_boston, load_iris

## boston housing data for tree regression
boston = load_boston()
boston_df = pd.DataFrame(boston['data'],columns = boston['feature_names'])
feature_array = boston_df.to_numpy()
boston_df['MEDV'] = boston.target
boston_array = boston_df.to_numpy()


## iris data for classification tree
iris = load_iris()
iris_df = pd.DataFrame(iris['data'], columns = iris['feature_names'])
iris_df['target'] = iris.target
iris_array = iris_df.to_numpy()


# 1: Tree regression

## want an efficient way to find optimal cutting points, so will order data
## by one of the predictors first
## pass in data as 2D numpy array with response variable last
def sort_by_column(data, i):
    order = data[:,i].argsort()
    ordered_x = data[order, i]
    ordered_y = data[order, -1]
    return [ordered_x, ordered_y]

## on each sorted pair of x's and y's, we want to find the optimal
## place to bisect the two to minimize errors from the prediction 
## of the mean in each region
## plan to just iterate through all possible split locations 
## and keep the best one
def bisect_finder(x,y):
    best_error = np.inf
    best_split = 0
    for i in range(1, len(x)):
        #print(len(x))
        split_value = (x[i]+x[i-1])/2
        left_cut = []
        right_cut = []
        for j in range(0,len(x)):
            if x[j] < split_value:
                left_cut.append(y[j])
            else:
                right_cut.append(y[j])
        sse = np.var(left_cut)*len(left_cut) + np.var(right_cut) * len(right_cut)
        #print(sse)
        #print(i)
        if sse < best_error:
            best_error = sse
            best_split = split_value
    return [best_split, best_error]

## now we do this for all predictors and keep the best split for the best one
## data again passed in as 2D numpy array
def tree_split_finder(data):
    best_error = np.inf
    best_predictor = 0
    best_split_for_predictor = 0
    for i in range(0, len(data[0]) - 1):
        pair = sort_by_column(data, i)
        #print(len(pair[0]))
        best_here_split, best_here_error = bisect_finder(pair[0],pair[1])
        if best_here_error < best_error:
            best_error = best_here_error
            best_predictor = i
            best_split_for_predictor = best_here_split
    return [best_predictor, best_split_for_predictor, best_error]

## now we want to construct the tree with the above splitting rule
## first we'll need a data structure/object for the tree
## data - all the datapoints from the original set in this node
## split - tuple with predictor and which value of predictor to split on
## leaf1, leaf2 - TRNode or None
class TRNode:
    def __init__(self, data, split, leaf1, leaf2):
        self.data = data
        self.split = split
        self.leaf1 = leaf1
        self.leaf2 = leaf2
    
    def node_size(self):
        return len(self.data)
    
    def to_string(self, pre):
        s = "Split: " + str(self.split) + "\n"
        if self.leaf1 is None and self.leaf2 is None:
            tot = 0
            for i in range(0,len(self.data)):
                tot = tot + self.data[i][-1]
            
            s = s + pre + "\t " + str(tot/len(self.data)) + "\n"
        else:
            s = s + pre + "\tLeft: " + self.leaf1.to_string("\t" + pre)
            s = s + pre + "\tRight: " + self.leaf2.to_string("\t" + pre)
        return s
    def print_tree(self):
        print(self.to_string(''))
    def tree_prediction(self,  x_data):
        if self.leaf1 is None and self.leaf2 is None:
            s = 0
            for i in range(0,len(self.data)):
                s  =s + self.data[i][-1]
            return s/len(self.data)
        else:
            best_predictor, best_split = self.split[0], self.split[1]
            if x_data[best_predictor] < best_split:
                return self.leaf1.tree_prediction(x_data)
            else:
                return self.leaf2.tree_prediction(x_data)
        
        
    ## function to get number of terminal nodes starting from current node
    def tree_leaves(self):
        if self.leaf1 is None and self.leaf2 is None:
            return 1
        else:
            return self.leaf1.tree_leaves() + self.leaf2.tree_leaves()
    
    ## function to get sum of square errors for data that comes down this node
    def Node_error(self):
        data = self.data
        sse = 0
        for i in range(0, len(data)):
            sse = sse + (data[i][-1] - self.tree_prediction(data[i][0:-1]))**2
        return sse
    
    # calculate effective alpha: divide the added SSE by the reduction in 
    # terminal nodes if this node is pruned
    def Node_eff_alpha(self):
        if self.leaf1 is None and self.leaf2 is None:
            return np.inf
        y_vals = np.zeros(len(self.data))
        for i in range(0,len(self.data)):
            y_vals[i] = self.data[i][-1]
        if (np.var(y_vals)*len(self.data)) < self.Node_error():
            print("THIS IS TERRIBLE...")
            print(np.var(y_vals)*len(self.data))
            print(self.Node_error())
        return ((np.var(y_vals)*len(self.data)) - self.Node_error())/(self.tree_leaves() - 1)
    
## next we want a recursive way to create a tree on our data
## stop when a node has less than k elements
def Tree_Regression(data, k):
    if len(data) < k:
        return TRNode(data, [-1,-1,-1], None, None)
    
    best_predictor, best_split_for_predictor, best_error = tree_split_finder(data)
    
    y_vals = np.zeros(len(data))
    
    for i in range(0, len(data)):
        y_vals[i] = data[i][-1]
    
    if best_error >= np.var(y_vals)*len(data):
        return TRNode(data, [-1,-1,-1], None, None)
    else:
        sorted_data = data[np.argsort(data[:, best_predictor])]
        split_index = np.argmax(sorted_data[:,best_predictor] > best_split_for_predictor)
        tree_root = TRNode(data, [best_predictor, best_split_for_predictor, best_error], 
                           Tree_Regression(sorted_data[:split_index], k),
                           Tree_Regression(sorted_data[split_index:], k))
        return tree_root

## quick check of errors on the test data
tree = Tree_Regression(boston_array, 10)
sse = 0

for i in range(0, len(feature_array)):
    pred = tree.tree_prediction(feature_array[i])
    data = boston_array[i][-1]
    sse = sse + (pred - data)**2
print(sse/len(feature_array))
print(sse)

## next we want to prune the tree to avoid overfitting
## do this by assigning a cost penalty to the number of terminal nodes
## 
def cost_complexity(tree, alpha, data):
    sse = 0
    for i in range(0,len(data)):
        sse = sse + (data[i][-1] - tree.tree_prediction(data[i][0:-1]))**2
    return sse + alpha * tree.tree_leaves()
## then we want to calculate how much further nodes on a branch really contribute
## we can do this by comparing the increase in SSE vs the reduction in the 
## alpha * |T| term,or quantify it by an effective alpha 

## find the effective alpha of every non-terminal node
## use this list to find the min alpha
def effective_alpha_list(tree):
    if tree.leaf1 is None and tree.leaf2 is None:
        return [np.inf]
    return [tree.Node_eff_alpha()] + effective_alpha_list(tree.leaf1) + effective_alpha_list(tree.leaf2)

# use previously found min alpha to prune that node
def weakest_link_removal(tree, min_alpha):
    if tree.leaf1 is None and tree.leaf2 is None:
            #print('end of branch')
            return  
    elif tree.leaf1.Node_eff_alpha() <= min_alpha:
        tree.leaf1 = TRNode(tree.leaf1.data, [-1,-1,-1], None, None)
    elif tree.leaf2.Node_eff_alpha() <= min_alpha:
        tree.leaf2 = TRNode(tree.leaf2.data, [-1,-1,-1], None, None)
    else:
        weakest_link_removal(tree.leaf1, min_alpha)
        weakest_link_removal(tree.leaf2, min_alpha)
        
## now we set an alpha threshold to keep pruning until the min alpha in the tree
## is above the threshold
def alpha_threshold_pruning(tree, threshold):
    min_alpha = min(effective_alpha_list(tree))
    while min_alpha < threshold:
        weakest_link_removal(tree, min_alpha)
        min_alpha = min(effective_alpha_list(tree))
        print('Current min_eff_alpha: ' + str(min_alpha))
    print('Final Tree Size: ' + str(tree.tree_leaves()))

## choose appropriate alpha threshold by cross validation

def test_errors(tree, test):
    sse = 0
    for i in range(0, len(test)):
        sse = sse + (test[i][-1] - tree.tree_prediction(test[i][0:-1]))**2
    return sse

## make the data splits
def k_folds_split(data, k):
    splits =[]
    remain = data
    for i in range(0,k):
        current_split = remain.sample(frac = 1/(k-i))
        remain = remain.drop(current_split.index)
        splits.append(current_split)
    return splits

## pass in a list of alpha thresholds to try, alongside our data and 
## desired number of folds. Then grow a tree with < n data points
## in each terminal node on the test data, prune it with the alphas

## want to return a list of cross-validation scores against the alphas
## for comparison and plotting
def k_fold_cross_validation(data, k, n, alphas):
    folds = k_folds_split(data, k)
    alphas_len = len(alphas)
    cv_scores = []
    for i in range(0, alphas_len):
        squared_errors = []
        for j in range(0,k):
            test_set = folds[j]
            train_set = data.drop(folds[j].index)
            temp_tree = Tree_Regression(train_set.to_numpy(), n)
            alpha_threshold_pruning(temp_tree, alphas[i])
            squared_errors.append(test_errors(temp_tree, test_set.to_numpy()))
        cv_scores.append(np.mean(squared_errors))
    return (alphas, cv_scores)



## 2: Decision Tree

## somewhat similar to the tree regression implementation
## difference in how we evaluate where to make cuts and what kind of 
## result to output

## instead of predicting the mean of some values, now we want to predict the 
## most common class in a given split region

## helper function for helping to count numbers of each of k classes
## then return the GINI index for this region sum of (p_k*(1-p_k)) over
## each of the k classes
def region_counter(y_data, k):
    counters = np.zeros(k)
    for i in range(0, len(y_data)):
        counters[y_data[i]] = counters[y_data[i]] + 1
    p_ks = counters/sum(counters)
    GINI = sum(p_ks*(1-p_ks))
    return [np.argmax(counters), GINI] 


## on each sorted pair of x's and y's, we want to find the optimal
## place to bisect the two to maximize Gini Index
## plan to just iterate through all possible split locations 
## and keep the best one
def decision_bisect_finder(x, y, k):
    best_GINI = np.inf
    best_split = 0
    for i in range(1, len(x)):
        
        split_value = (x[i]+x[i-1])/2
        left_cut = []
        right_cut = []
        for j in range(0,len(x)):
            if x[j] < split_value:
                left_cut.append(int(y[j]))
            else:
                right_cut.append(int(y[j]))
        # replace old mean prediction and sse scoring with
        # most common class prediction and Gini Index
        
        
        left_GINI = region_counter(np.array(left_cut), k)[1]
        right_GINI = region_counter(np.array(right_cut), k)[1]
        GINI = left_GINI + right_GINI
        
        if GINI < best_GINI:
            best_GINI = GINI
            best_split = split_value
    return [best_split, best_GINI]

## now we do this for all predictors and keep the best split for the best one
## data again passed in as 2D numpy array
def decision_tree_split_finder(data, k):
    best_GINI = np.inf
    best_predictor = 0
    best_split_for_predictor = 0
    for i in range(0, len(data[0]) - 1):
        pair = sort_by_column(data, i)
        #print(len(pair[0]))
        best_here_split, best_here_GINI = decision_bisect_finder(pair[0], pair[1], k)
        if best_here_GINI < best_GINI:
            best_GINI = best_here_GINI
            best_predictor = i
            best_split_for_predictor = best_here_split
    return [best_predictor, best_split_for_predictor, best_GINI]

## again we make a class similar to the previous one for making 
## a tree according to the splitting rule
## data - all the datapoints from the original set in this node
## split - tuple with predictor and which value of predictor to split on
## k - number of different classes that the tree can end in
## leaf1, leaf2 - TRNode or None
class Decision_Tree_Node:
    def __init__(self, data, split, k, leaf1, leaf2):
        self.data = data
        self.split = split
        self.k = k
        self.leaf1 = leaf1
        self.leaf2 = leaf2
    
    def node_size(self):
        return len(self.data)
    
    def to_string(self, pre):
        s = "Split: " + str(self.split) + "\n"
        if self.leaf1 is None and self.leaf2 is None:
            y_vals = np.empty(len(self.data), dtype= int)
            for i in range(0, len(self.data)):
                y_vals[i] = self.data[i][-1]
            node_prediction = region_counter(y_vals, self.k)[0]
            
            s = s + pre + "\t " + str(node_prediction) + "\n"
        else:
            s = s + pre + "\tLeft: " + self.leaf1.to_string("\t" + pre)
            s = s + pre + "\tRight: " + self.leaf2.to_string("\t" + pre)
        return s
    
    def print_tree(self):
        print(self.to_string(''))
    
    def tree_prediction(self,  x_data):
        if self.leaf1 is None and self.leaf2 is None:
            y_vals = np.empty(len(self.data),dtype = int)
            for i in range(0, len(self.data)):
                y_vals[i] = self.data[i][-1]
            return region_counter(y_vals, self.k)[0]
        else:
            best_predictor, best_split = self.split[0], self.split[1]
            if x_data[best_predictor] < best_split:
                return self.leaf1.tree_prediction(x_data)
            else:
                return self.leaf2.tree_prediction(x_data)
        
        
    ## function to get number of terminal nodes starting from current node
    def tree_leaves(self):
        if self.leaf1 is None and self.leaf2 is None:
            return 1
        else:
            return self.leaf1.tree_leaves() + self.leaf2.tree_leaves()
    
    ## function to get classification error rate down the current node
    ## on the training data
    def Node_error(self):
        data = self.data
        errors = 0
        for i in range(0, len(data)):
            if self.tree_prediction(self.data[i][0:-1]) != self.data[i][-1]:
                errors = errors + 1        
        return errors/len(self.data)
    
    # calculate effective alpha: divide the added error rate by the reduction in 
    # terminal nodes if this node is pruned
    def Node_eff_alpha(self):
        if self.leaf1 is None and self.leaf2 is None:
            return np.inf
        y_vals = np.empty(len(self.data), dtype = int)
        for i in range(0,len(self.data)):
            y_vals[i] = int(self.data[i][-1])
        prediction_here = region_counter(y_vals, self.k)[0]
        error = 1 - sum(y_vals == prediction_here)/len(self.data)
        if error < self.Node_error():
            print("THIS IS TERRIBLE...")
            print(error)
            print(self.Node_error())
        return (error - self.Node_error())/(self.tree_leaves() - 1)

## make a decision tree based on some data with k classes and
## end when a node has < n data points

def Decision_Tree_Maker(data, k, n):
    if len(data) < n:
        return Decision_Tree_Node(data, [-1,-1,-1], k, None, None)
    
    best_predictor, best_split_for_predictor, best_GINI = \
        decision_tree_split_finder(data, k)
    print(best_GINI)
    
    y_vals = np.empty(len(data), dtype=int)
    
    for i in range(0, len(data)):
        #print(type(int(data[i][-1])))
        y_vals[i] = int(data[i][-1])
        
    current_GINI = region_counter(y_vals, k)[1]
    
    if best_GINI >= current_GINI:
        return Decision_Tree_Node(data, [-1,-1,-1], k, None, None)
    else:
        sorted_data = data[np.argsort(data[:, best_predictor])]
        split_index = np.argmax(sorted_data[:,best_predictor] > best_split_for_predictor)
        tree_root = Decision_Tree_Node(data, [best_predictor, best_split_for_predictor, best_GINI], n, 
                           Decision_Tree_Maker(sorted_data[:split_index], k, n),
                           Decision_Tree_Maker(sorted_data[split_index:], k, n))
        return tree_root
test_tree = Decision_Tree_Maker(iris_array, 3, 5)

## very similar pruning process as before, replacing sse with classification error rate
# use previously found min alpha to prune that node
def Decision_weakest_link_removal(tree, min_alpha):
    if tree.leaf1 is None and tree.leaf2 is None:
            return  
    elif tree.leaf1.Node_eff_alpha() <= min_alpha:
        tree.leaf1 = Decision_Tree_Node(tree.leaf1.data, [-1,-1,-1], tree.k, None, None)
    elif tree.leaf2.Node_eff_alpha() <= min_alpha:
        tree.leaf2 = Decision_Tree_Node(tree.leaf2.data, [-1,-1,-1], tree.k, None, None)
    else:
        Decision_weakest_link_removal(tree.leaf1, min_alpha)
        Decision_weakest_link_removal(tree.leaf2, min_alpha)
        
## now we set an alpha threshold to keep pruning until the min alpha in the tree
## is above the threshold
def Decision_alpha_threshold_pruning(tree, threshold):
    min_alpha = min(effective_alpha_list(tree))
    while min_alpha < threshold:
        Decision_weakest_link_removal(tree, min_alpha)
        min_alpha = min(effective_alpha_list(tree))
        print('Current min_eff_alpha: ' + str(min_alpha))
    print('Final Tree Size: ' + str(tree.tree_leaves()))

## test_errors function for classification errors on a decision tree:
def test_classification_errors(tree, test):
    errors = 0
    for i in range(0, len(test)):
        if tree.tree_prediction(test[i][0:-1]) != test[-1]:
            errors = errors + 1
    return errors/len(test)

## pass in a list of alpha thresholds to try, alongside our data and 
## desired number of folds. Then grow a tree with < n data points
## in each terminal node on the test data, prune it with the alphas

## want to return a list of cross-validation scores against the alphas
## for comparison and plotting
def k_fold_cross_validation_decision(data, fold_number, k, n, alphas):
    folds = k_folds_split(data, fold_number)
    alphas_len = len(alphas)
    cv_scores = []
    for i in range(0, alphas_len):
        class_errors = []
        for j in range(0,fold_number):
            test_set = folds[j]
            train_set = data.drop(folds[j].index)
            temp_tree = Decision_Tree_Maker(train_set.to_numpy(), k, n)
            Decision_alpha_threshold_pruning(temp_tree, alphas[i])
            class_errors.append(test_classification_errors(temp_tree, test_set.to_numpy()))
        cv_scores.append(np.mean(class_errors))
    return (alphas, cv_scores)



## now to improve the robustness of our decision trees, we can 
## bag, boost, or make a random forest 

## Bagging: we bootstrap the data and make a tree for each bootstrap
## then use majority vote from all of the trees to make a classification

## make B bags, for k classes
## return B trees made on each of the bags with nodes ending with < n datapoints
def Decision_Tree_bagging(data, B, k, n):
    trees = []
    for i in range(0,B):
        data_sample = data.sample(len(data), replace = True).to_numpy()
        trees.append(Decision_Tree_Maker(data_sample, k, n))
    return trees

## now return a prediction for some predictors
## using the bagged trees and majority vote
def Bagging_prediction(trees, predictors):
    B = len(trees)
    counters = np.zeros(B)
    for i in range(0, B):
        counters[trees[i].tree_prediction(predictors)] += 1
    return counters[np.argmax(counters)]


## Random Forest: we make a tree using only a subset of predictors
## at each split
## then make a final classification based on majority vote of
## resulting trees

## make some slight modifications to tree making functions
## to consider subset of predictors at each split

## split_finder gets some changes
def random_forest_tree_split_finder(data, k, m):
    best_GINI = np.inf
    best_predictor = 0
    best_split_for_predictor = 0
    predictor_subset = random.sample(list(range(0, len(data[0])-1)), m)
    for i in range(0, len(predictor_subset)):
        pair = sort_by_column(data, predictor_subset[i])
        #print(len(pair[0]))
        best_here_split, best_here_GINI = decision_bisect_finder(pair[0], pair[1], k)
        if best_here_GINI < best_GINI:
            best_GINI = best_here_GINI
            best_predictor = predictor_subset[i]
            best_split_for_predictor = best_here_split
    return [best_predictor, best_split_for_predictor, best_GINI]

## using the modified tree_split_finder, we pass it back into the same
## logic as used to make the other trees
## this time with extra variable m to denote how many predictors
## to consider at each split

def Random_forest_tree_maker(data, k, n, m):
    if len(data) < n:
        return Decision_Tree_Node(data, [-1,-1,-1], k, None, None)
    
    best_predictor, best_split_for_predictor, best_GINI = \
        random_forest_tree_split_finder(data, k, m)
    print(best_GINI)
    
    y_vals = np.empty(len(data), dtype=int)
    
    for i in range(0, len(data)):
        #print(type(int(data[i][-1])))
        y_vals[i] = int(data[i][-1])
        
    current_GINI = region_counter(y_vals, k)[1]
    
    if best_GINI >= current_GINI:
        return Decision_Tree_Node(data, [-1,-1,-1], k, None, None)
    else:
        sorted_data = data[np.argsort(data[:, best_predictor])]
        split_index = np.argmax(sorted_data[:,best_predictor] > best_split_for_predictor)
        tree_root = Decision_Tree_Node(data, [best_predictor, best_split_for_predictor, best_GINI], n, 
                           Random_forest_tree_maker(sorted_data[:split_index], k, n),
                           Random_forest_tree_maker(sorted_data[split_index:], k, n))
        return tree_root

## now make a forest of F trees, considering m predictors at each step
def forest(data, k, n, m, F):
    trees = []
    for i in range(0,F):
        trees.append(Random_forest_tree_maker(data, k, n, m))
    return trees
