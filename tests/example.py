# -*- coding: utf-8 -*-
from sklearn.datasets import load_wine
import pandas as pd
import numpy as np
from rfcc.data_ops import ordinal_encode
from rfcc.rfcc import rfcc
import pytest
import numpy as np
import pandas as pd
import scipy as sp
from itertools import product
from tqdm import tqdm, tnrange,trange
import tqdm.notebook as tq


from pydataset import data
dataset=data("mpg")
y_col=["cty"]
x_col=['displ', 'class' , 'cyl','manufacturer']
categoricals=['class', 'cyl','manufacturer']
Y=dataset[y_col]
X=dataset[x_col]

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from rfcc.rfcc import rfcc
model=rfcc(model=RandomForestRegressor,max_clusters=20,random_state=5 )


model.fit(X,Y,categoricals,encode_y=False, linkage_method="complete")


clusters=model.cluster_descriptions(variables_to_consider=['class','manufacturer'], continuous_measures="mean")


clusters=clusters.sort_values(by="cty-mean")

display(clusters.head(1))


a,b=model.path_analysis(0)

