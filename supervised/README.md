# Supervided algorithms

**!!!DISCLAIMER!!!**

> This code has been written in the 2018, once the associated paper was approved by the editor. 
> The python version used to run this folder scripts is 3.6. Check the [requirements file](./requirements.txt) for the libraries versions.
> Most of the libraries used are outdated, but it should be easy to update this code to newer python and libraries versions.

In [this folder](.) there are the codes used for the supervised algorithms, while in the [utils](./utils) folder there are some functions or class shared among the scripts.

Beware that the hyperparameters of the algorithms must be chosen carefully, using a random grid search or other optitimisation methods based on the data and the phenomenon to model.

**!!!DISCLAIMER!!!**
> Even though scripts are working with these hyper-parameters, you might not get meaningful results due to the dataset, which is a subsample and also due to the hyper-parameters which have been sampled as well.

## Testing

1. Create a conda evnvironment

```{bash}
conda create --name recsys python=3.6
```

2. Activate the env and installing the libraries

```{bash}
conda activate recsys
cd supervised
pip install requirements.txt
```

3. Run the algorithms
```{bash}
python xgboost_has.py # for xgboost
python lightgbm_has.py # for lightgbm
python catboost_has.py # for catboost
python ann_has.py # for ann
```
