# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 18:22:23 2021

@author: marquart
"""
from sklearn.datasets import load_wine
import pandas as pd
from pydataset import data
import numpy as np
from rfcc.data_ops import ordinal_encode
from rfcc.rfcc import rfcc
import pytest


@pytest.fixture
def get_mpg_dataset_cat():
    dataset=data("mpg")
    y_col=["manufacturer"]
    x_col=np.setdiff1d(dataset.columns,y_col)
    categoricals=list(np.setdiff1d(x_col, ['cty','hwy','year', 'displ', 'cyl']))
    Y=dataset[y_col]
    X=dataset[x_col]
    return Y,X,categoricals


@pytest.mark.rfcc
def test_fit(get_mpg_dataset):
    
    Y,X,categoricals=get_mpg_dataset
    
    model=rfcc(max_clusters=15 )
    
    model.fit(X,Y,categoricals,encode_y=True)
    
    print(model.cluster_descriptions(continuous_measures=['mean']))
    
    assert all(X[categoricals]==X_cat2)