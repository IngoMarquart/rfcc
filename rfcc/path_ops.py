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


def encode_path_ordinal(
    column_names, leaf_path, leaf_feature, leaf_threshold, leaf_direction, encoding_dict
):
    """
    This function builds a name from the path variables
    Note: This function gives the correct variable levels for the observations in the cluster
    It needs the encoding dictionary from the ordinal encoding (see below)
    This is the best option, because it gives interpretable clusters!
    """
    name_list = dict()
    # Reverse direction
    leaf_feature.reverse()
    leaf_path.reverse()
    leaf_threshold.reverse()
    leaf_direction.reverse()
    # Need numpy arrays
    np_leaf_feature = np.array(leaf_feature)
    np_leaf_threshold = np.array(leaf_threshold)
    np_leaf_direction = np.array(leaf_direction)    
    # Get unique leaf_features, preserving order
    _, idx = np.unique(np_leaf_feature, return_index=True)
    unique_features = np_leaf_feature[np.sort(idx)]

    # Loop over each unique feature to be considered
    for feature in unique_features:
        # Get the description (name) of the feature
        description = column_names[feature]
        # Get the encoding levels (threshold, level_name)
        if description in encoding_dict: # This is a categorical feature
            levels = encoding_dict[description]
            # Get the values/names of the levels
            level_thresholds = np.array([x[0] for x in levels])
            level_names = np.array([x[1] for x in levels])
        else: # It is a continuous feature for which we have no encoding information
            levels = None

        # Get all nodes that have this feature
        path_ids = np.where(np_leaf_feature == feature)[0]
        # Get all directions for this feature
        selected_directions = np_leaf_direction[path_ids]
        selected_features = np_leaf_feature[path_ids]
        # Get all thresholds for this feature
        selected_thresholds = np_leaf_threshold[path_ids]
        # Get all ids from nodes where direction ins positive or negative
        # This means that the feature should be larger than or smaller than
        larger_than_ids = np.where(selected_directions > 0)[0]
        smaller_than_ids = np.where(selected_directions < 0)[0]
        # Three options: The path says larger, smaller, or between two levels / values
        if len(larger_than_ids) > 0 and len(smaller_than_ids) > 0:  # Between
            larger_than_threshold = np.max(selected_thresholds[larger_than_ids])
            lower_than_threshold = -np.max(-selected_thresholds[smaller_than_ids])
            if levels is not None: # Categorical feature
                selected_level_ids = np.where(
                    np.logical_and(
                        level_thresholds >= larger_than_threshold,
                        level_thresholds <= lower_than_threshold,
                    )
                )[0]
                selected_level_ids_NOT = np.where(
                    np.logical_or(
                        level_thresholds < larger_than_threshold,
                        level_thresholds > lower_than_threshold,
                    )
                )[0]
                
                if (len(selected_level_ids) <= len(selected_level_ids_NOT)):
                    selected_levels = level_names[selected_level_ids]
                    name = "{}: {}".format(description, ", ".join(selected_levels))
                    if len(selected_level_ids) == 0:
                        name = " "
                else:
                    selected_levels = level_names[selected_level_ids_NOT]
                    name = "{} is not: {}".format(description, ", ".join(selected_levels))
                    if len(selected_level_ids_NOT) == 0:
                        name = "{} all levels".format(description)
            else: # Continous feature
                name = "{} between {} and {}".format(description, np.round(lower_than_threshold,2), np.round(larger_than_threshold,2))
        elif len(larger_than_ids) > 0:  # Only larger than splits
            larger_than_threshold = np.max(selected_thresholds[larger_than_ids])
            if levels is not None: # Categorical feature
                selected_level_ids = np.where(level_thresholds >= larger_than_threshold)[0]
                selected_level_ids_NOT = np.where(level_thresholds < larger_than_threshold)[0]
                # We can display the cluster either what is IS, or what it is NOT
                # Whatever is shorter, we pick here
                if (len(selected_level_ids) <= len(selected_level_ids_NOT)):
                    selected_levels = level_names[selected_level_ids]
                    name = "{}: {}".format(description, ", ".join(selected_levels))
                    if len(selected_level_ids) == 0:
                        name = "{} all levels".format(description)
                else:
                    selected_levels = level_names[selected_level_ids_NOT]
                    name = "{} is not: {}".format(description, ", ".join(selected_levels))
                    if len(selected_level_ids_NOT) == 0:
                        name = "{} all levels".format(description)
            else: # Continous feature
                name = "{} larger than: {}".format(description, np.round(larger_than_threshold,2))
        else:  # Smaller only
            lower_than_threshold = -np.max(-selected_thresholds[smaller_than_ids])
            if levels is not None: # Categorical feature
                selected_level_ids = np.where(level_thresholds <= lower_than_threshold)[0]
                selected_level_ids_NOT = np.where(level_thresholds > lower_than_threshold)[0]
                if (len(selected_level_ids) <= len(selected_level_ids_NOT)):
                    selected_levels = level_names[selected_level_ids]
                    name = "{}: {}".format(description, ", ".join(selected_levels))
                    if len(selected_level_ids) == 0:
                        name = " "
                else:
                    selected_levels = level_names[selected_level_ids_NOT]
                    name = "{} is not: {}".format(description, ", ".join(selected_levels))
                    if len(selected_level_ids_NOT) == 0:
                        name = "{} all levels".format(description)
            else: # Continous feature
                name = "{} smaller than: {}".format(description, np.round(lower_than_threshold,2))
        # Append the constructed name or decision rule for this feature
        name_list[description]=name
    return name_list

