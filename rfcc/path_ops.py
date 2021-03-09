# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 11:56:33 2021

@author: Ingo Marquart, ingo.marquart@esmt.org
"""
import numpy as np


def recurse_path(
    current_node, leaf_path, leaf_feature, leaf_threshold, leaf_direction, clf
):
    """
    Recursive function to traverse decision tree starting from a node going towards the root.
    Saves path, features along the path, decision thresholds and the direction (<= or >) in a list.
    
    A bit complicated since sklearn encodes the tree starting from root,
    however, our clusters are the leafs nodes. So we need to construct "parents" from the list of child nodes.
    
    Logic as follows:
    
    For any current node (for example the leaf starting node defining the cluster),
    find the parent. 
    The current node might be a left child or a right child of the parent.
    Parent selects current node according to some variable X and a threshold Y.
    If current node is a right child, then it is selected if X>Y
    If current node is a left child, then it is selected if X<=Y
    Once we determine this, we save the parent, the variable X, threshold Y and direction of choice D
    D=-1 if X<=Y and D=1 if X>Y was the decision rule
    
    Next, run the function on the parent node. Parent node will then recursively do the above,
    and therefore return every node up until the root node.
    We know root node is reached because it has no parent. 
    Root is the only node without parent.
    This ends the recursion and passes the constructed path variables back to the beginning
    
    current_node (1)->
    find parent->
    get X,Y and direction of parent->
    add to lists->
    call function with parent as current node->go to (1)
    repeat until no parent left-> path fully discovered
    """
    # Get variables from tree
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    # Leaf might be referenced from the right or from the left
    right_parent = np.where(children_right == current_node)[0]
    left_parent = np.where(children_left == current_node)[0]
    if len(right_parent) > 0:  # Is a right child
        # Make parent node id to int
        parent = int(right_parent)
        # Right Parent: Direction was "greater than"
        direction = 1
        # Append this to our lists
        leaf_path.append(parent)
        leaf_feature.append(feature[parent])
        leaf_threshold.append(threshold[parent])
        leaf_direction.append(direction)
        # Call function recursively for parent
        leaf_path, leaf_feature, leaf_threshold, leaf_direction = recurse_path(
            parent, leaf_path, leaf_feature, leaf_threshold, leaf_direction, clf
        )
    elif len(left_parent) > 0:  # Is a left child
        parent = int(left_parent)
        direction = -1
        leaf_path.append(parent)
        leaf_feature.append(feature[parent])
        leaf_threshold.append(threshold[parent])
        leaf_direction.append(direction)
        leaf_path, leaf_feature, leaf_threshold, leaf_direction = recurse_path(
            parent, leaf_path, leaf_feature, leaf_threshold, leaf_direction, clf
        )
    else:  # Root reached
        pass

    return leaf_path, leaf_feature, leaf_threshold, leaf_direction