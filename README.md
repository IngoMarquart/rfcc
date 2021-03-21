# rfcc
Python RFCC - Data understanding, clustering and outlier detection for regression and classifcation tasks

Description

Example

# Installation


# Usage

Let's illustrate the approach with a simple example. We will be regression the miles-per-gallon in the city (__cty__) performance of a set of
cars on the class (compact, pick-up etc.), the number of cylinders and the engine displacement.

The data is available in the pydataset package
```python
dataset=data("mpg")
y_col=["cty"]
x_col=['displ', 'class' , 'cyl']
categoricals=['class', 'cyl']
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

The first step is to initialize the model, much like one would initialize an scikit-learn model. We only need to pass an appropriate ensemble model (RandomForestClassifier, RandomForestRegressor) and specify the options we'd like to use.

Since miles-per-gallon is a continuous measure, we'll be using a random forest regression.

```python

from sklearn.ensemble import RandomForestRegressor
from rfcc.rfcc import rfcc
model=rfcc(model=RandomForestRegressor,max_clusters=20,random_state=1)
```

We have two options to specify the size and number of clusters to be returned.

The parameter __max_clusters__ sets the maximum amount of leafs in each decision tree. It ensures that the model does not return too many or too few clusters, but it does change the estimation of the random forest.

Another option is to set __max_clusters__ to a high value, or leave it unspecified, and use the hierarchical clustering algorithm to extract clusters of the desired size. See below for __t_param__ in the fit method.


## Fitting and optional parameters

Now we need to fit our model to the data.

```python
model.fit(X,Y,categoricals)
```

Categoricals is a list of columns that we'd like to encode before fitting the model.

The following optional parameters can be passed

- **encode_y** (bool): Also encode the outcome variable as categorical

- **linkage_method** (str): Linkage method used in the clustering algorithm (average, single, complete, ward)

- **clustering_type** (str): "rfcc" (default) our path based clustering, or "binary" as in prior approaches

- **t_param** (float): If None, number of clusters corresponds to average number of leafs. If __t_param__ is specified,
pick that level of clustering hierarchy where distance between members of the group is less than __t_param__. The higher the value, the larger average size of a cluster. 

## Cluster compositions

Once the model is fit, we can extract the composition of clusters.
Let's see which car types and cylinders have the best and worst miles-per-gallon performance.

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

```
Nr_Obs	Output_cty	class	                        displ	                    manufacturer
17	    [11.4]	    class is not: 2seater, compact	displ between 5.25 and 4.4	manufacturer: audi, chevrolet, dodge
21	    [12.4]	    class: suv	                    displ larger than: 4.4	    manufacturer is not: audi, chevrolet, dodge
5	    [12.6]	    class: midsize, minivan, pickup	displ larger than: 4.4	    manufacturer is not: audi, chevrolet, dodge
13	    [12.6]	    class is not: 2seater, compact	displ larger than: 5.25	    manufacturer: audi, chevrolet, dodge
5	    [13.4]	    class: minivan	                displ between 3.75 and 3.15	-
22	    [14.1]	-	                                displ between 4.4 and 3.85	-
```
