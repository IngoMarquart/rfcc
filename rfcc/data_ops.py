# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 13:09:07 2021

@author: Ingo Marquart, ingo.marquart@esmt.org
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from typing import Optional


def ordinal_encode(X: pd.DataFrame, categoricals: list, return_enc: Optional[bool] = False) -> tuple:
    """
    
    Encode categorical variables as ordinal variables.
    Create a dictionary that gives descriptions of these ordinal levels.

    Parameters
    ----------
    X : pd.DataFrame
        Data.
    categoricals : list
        List of columns giving the categorical variables
    return_enc : bool, Optional
        Return the encoder. Mostly used for testing.

    Returns
    -------
    tuple (pd.DataFrame,dict,Optional[OrdinalEncoder])
        Data with ordinal variables, dictionary with encodings, optinally the encoder

    """
    Xcat = X[categoricals]
    enc = OrdinalEncoder()
    enc.fit(Xcat)
    newXcat = pd.DataFrame(
        enc.transform(Xcat), columns=Xcat.columns, index=X.index
    )
    # Create a dict of encodings which we can use to make descriptions better
    encoding_dict = dict()
    for i, col in enumerate(newXcat.columns):
        values = np.unique(newXcat[col])
        encodings = enc.categories_[i]
        encodings = [str(i).replace("(", "").replace("]", "")
                     if not type(i) == str else i for i in encodings]
        encoding_dict[col] = list(zip(values, encodings))

    X = X.drop(columns=categoricals)
    newXcat.index = newXcat.index
    X = pd.concat([X, newXcat], axis=1)
    if return_enc:
        return X, encoding_dict, enc
    else:
        return X, encoding_dict


def get_cat_desc_from_data(subset: pd.DataFrame, column_names: list) -> list:
    """
    
    Get for each column in a given subset the elements that are included.
    Then generate a description for the entire subset for output purposes


    Parameters
    ----------
    subset : pd.DataFrame
        Data
    column_names : list
        Columns

    Returns
    -------
    list
        Description of each column
    """
    desc_list = list()
    for col in column_names:
        vals = np.unique(subset[col]).tolist()
        vals = [str(x) for x in vals]
        if len(vals) > 1:
            vals = ", ".join(vals)
        else:
            vals = vals[0]
        desc_list.append("Col {}: {}; ".format(col, vals))
    return desc_list
