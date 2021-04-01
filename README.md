# Python RFCC - Data understanding, clustering and outlier detection for regression and classification tasks

Random forests are invariant and robust estimators that can fit complex interactions between input data of different types and binary, categorical, or continuous outcome variables, including those with multiple dimensions. In addition to these desirable properties, random forests impose a structure on the observations from which researchers and data analysts can infer clusters or groups of interest. 

You can use these clusters to:

- structure your data,

- elucidate new patterns of how features influence outcomes,

- define subgroups for further analysis, 

- derive prototypical observations, 

- identify outlier observations, 

- catch mislabeled data, 

- evaluate the performance of the estimation model in more detail.

Random Forest Consensus Clustering and implement is implemented in the Scikit-Learn / SciPy data science ecosystem. This algorithm differs from prior approaches by making use of the entire tree structure. Observations become proximate if they follow similar decision paths across trees of a random forest.

More info in here:

```
Marquart, Ingo and Koca Marquart, Ebru, RFCC: Random Forest Consensus Clustering for Regression and Classification (March 19, 2021). Available at SSRN: https://ssrn.com/abstract=3807828 or http://dx.doi.org/10.2139/ssrn.3807828
```

# Installation

Install via pip!
```python
pip install rfcc
```

# Usage

Let's illustrate the approach with a simple example. We will be regression the miles-per-gallon in the city (__cty__) performance of a set of
cars on the class (compact, pick-up etc.), the number of cylinders and the engine displacement.

The data is available in the pydataset package
```python
dataset=data("mpg")
y_col=["cty"]
x_col=['displ', 'class' , 'cyl']
Y=dataset[y_col]
X=dataset[x_col]
print(X.head(5))
```

```python
   displ    class  cyl
1    1.8  compact    4
2    1.8  compact    4
3    2.0  compact    4
4    2.0  compact    4
5    2.8  compact    6
```

We want __class__ and __cyl__ to be treated as categorical variable, so we'll keep track of these columns.

## Initialization and model choice

The first step is to initialize the model, much like one would initialize an scikit-learn model.
The main class is __cluster_model__ from the rfcc package.
We only need to pass an appropriate ensemble model (RandomForestClassifier, RandomForestRegressor) and specify the options we'd like to use.


Since miles-per-gallon is a continuous measure, we'll be using a random forest regression.

```python
from sklearn.ensemble import RandomForestRegressor
from rfcc.model import cluster_model
model=cluster_model(model=RandomForestRegressor,max_clusters=20,random_state=1)
```

We have two options to specify the size and number of clusters to be returned.

The parameter __max_clusters__ sets the maximum amount of leafs in each decision tree. It ensures that the model does not return too many or too few clusters, but it does change the estimation of the random forest. 

Perhaps you already have the best parameters for your random forest on hand, or want to use the default values?
Another option to get clusters of the desired size is to not set __max_clusters__ here, but use the hierarchical clustering algorithm to extract clusters of the desired size. See below for __t_param__ in the fit method.


## Fitting and optional parameters

Now we need to fit our model to the data.

```python
model.fit(X,Y)
```



The following optional parameters can be passed

- **encode** (list): A list of columns that we'd like to encode before fitting the model. Note that all non-numerical columns will be encoded automatically. However, you can also encode numerical data by passing it in the __encode__ parameter.

- **encode_y** (bool): You can choose to ordinally encode the outcome variables. If you do a classification, scikit learn will choose how to encode the outcome variables. If the variable is continuous, this will usually lead to a rather bad fit, in which case you may want to encode.

- **linkage_method** (str): Linkage method used in the clustering algorithm (average, single, complete, ward)

- **clustering_type** (str): "rfcc" (default) our path based clustering, or "binary" as in prior approaches

- **t_param** (float): If None, number of clusters corresponds to average number of leafs. If __t_param__ is specified,
pick that level of clustering hierarchy where distance between members of the group is less than __t_param__. The higher the value, the larger average size of a cluster. 

Let's check how well our model does on our training set

```python
model.score(X,Y)
```

```python
0.9231321010907177
```

## Cluster compositions

Once the model is fit, we can extract the composition of clusters.
Let's see which car types and manufacturers have the best and worst miles-per-gallon performance.

First, we use the cluster_descriptions method to return the compositions for each cluster.

```python
clusters=model.cluster_descriptions(variables_to_consider=['class','manufacturer'], continuous_measures="mean")
```

The optional parameters are:

- **variables_to_consider** (list): List of columns in X to take into account.

- **continuous_measures** (str, list): Measures to compute for each continuous feature (mean, std, median, max, min, skew)

We will sort our clusters by the average mpg and return the clusters with the two highest and two lowest mpg performances.

```python
clusters=clusters.sort_values(by="cty-mean")
print(clusters.head(2))
print(clusters.tail(2))
```

```python
Nr_Obs	cty-mean	class	                    manufacturer
7	    11.85	    suv: 1.0%	                ford: 0.29%, land rover: 0.57%, mercury: 0.14%
49	    12.02	    pickup: 0.35%, suv: 0.63%	chevrolet: 0.18%, dodge: 0.43%, ford: 0.12%, jeep: 0.1%, lincoln: 0.06%, mercury: 0.02%, nissan: 0.02%, toyota: 0.06%
```

```python
Nr_Obs	cty-mean	class	                                            manufacturer
15	    24.4	    compact: 0.33%, midsize: 0.13%, subcompact: 0.53%	honda: 0.53%, toyota: 0.33%, volkswagen: 0.13%
3	    32.3	    compact: 0.33%, subcompact: 0.67%	                volkswagen: 1.0%
```


## Decision Path Analysis

Cluster descriptions return the proportions of values for any feature we are interested in. However, we also may want to know how a decision tree classifies an observation. For example, it may be that the feature __manufacturer__  has
no predictive value, whereas the number of cylinders or the displacement does.

Currently, path analyses are queried for each estimator in the random forest. 
In the future patch, the path analysis will be available for the entire random forest.

Let's see how the first decision tree (index 0) classifies the observations with the lowest miles-per-gallon performance

```python
paths=model.path_analysis(estimator_id=0)
paths.sort_values(by="Output_cty")
print(paths.head(5))
```

```python
Nr_Obs	Output_cty	class	                        displ	                    manufacturer
17	    [11.4]	    class is not: 2seater, compact	displ between 5.25 and 4.4	manufacturer: audi, chevrolet, dodge
21	    [12.4]	    class: suv	                    displ larger than: 4.4	    manufacturer is not: audi, chevrolet, dodge
5	    [12.6]	    class: midsize, minivan, pickup	displ larger than: 4.4	    manufacturer is not: audi, chevrolet, dodge
13	    [12.6]	    class is not: 2seater, compact	displ larger than: 5.25	    manufacturer: audi, chevrolet, dodge
5	    [13.4]	    class: minivan	                displ between 3.75 and 3.15	-
22	    [14.1]	-	                                displ between 4.4 and 3.85	-
```

Another reason to do a decision path analysis is to help to diagnose outliers or mislabeled data. 

## Detection of outliers and mislabelled data

Outliers are observations that are unusual - not necessarily because their features differ, but rather because their implications for the outcome variable are different from other comparable observations. Mislabelled data may appear as
outlier, since the relationships between outcome and feature values may not make much sense.

Since outliers follow distinct decision paths in the random forest, RFCC does not cluster them with other observations.
We can therefore find outliers by analyzing clusters that have very few observations.

Let's see what outliers exist in the mpg data.
```python
clusters=model.cluster_descriptions(continuous_measures="mean")
clusters=clusters.sort_values(by="Nr_Obs")
outliers=clusters.head(2)
print(outliers)
```

```python
Cluster_ID	Nr_Obs	cty-mean	class	        cyl	        manufacturer	    displ-mean
16	        1	    16.0	    minivan: 1.0%	6: 1.0%	    dodge: 1.0%	        4.0
3	        2	    18.0	    midsize: 1.0%	6: 1.0%	    hyundai: 1.0%	    2.5
```

It seems we have one cluster (id=16) with a dodge minivan, and a cluster (id=3) with two observations.
We can get the constituent observations directly from our model.


```python
ids=model.get_observations(cluster_id=16)
print(dataset.iloc[ids,:])
ids=model.get_observations(cluster_id=3)
print(dataset.iloc[ids,:])
```

```python
	manufacturer	model	       displ	year	cyl	    trans
48  dodge           caravan 2wd    4.0      2008    6       auto(l6)

	manufacturer	model	displ	year	cyl	    trans
113	hyundai	        sonata	2.5	    1999	6	    auto(l4)
114	hyundai	        sonata	2.5	    1999	6	    manual(m5)
```

