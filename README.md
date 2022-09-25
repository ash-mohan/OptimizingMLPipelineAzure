# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Useful Resources
- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)


## Summary
The dataset we will be working with in this exercise is a bank marketing dataset, specifically data related with 
phone call marketing campaigns of a Portuguese banking institution. This is a classification task, with the goal being to 
predict if the client will subscribe (yes/no) a term deposit.

After training multiple models, the best performing model was a VotingEnsemble created by azure AutoML. Ensemble models are a common approach
for ML tasks and have many benefits over a single estimator. For example, using a single estimator can result in a model that 
relies heavily on one or a few features while making a prediction. Using this methodology, the highest accuracy 
achieved was 91.785%. 


## Scikit-learn Pipeline

Before training the model using our sklearn pipeline, we first had to clean the data. This preprocessing step included converting categorical variables
into indicator variables, removing null values, and separating the predictor variable. All of this was achieved using the clean_data function defined in 
train.py file. The model that we used for training in this pipeline is sklearn.linear_model.LogisticRegression. In this exercise, we only chose to do
hyperparameter tuning on the C and max_iter variables. C represents the inverse of regularization strength, where smaller values cause stronger regularization.
The max_iter variable controls the maximum number of iterations to converge. Hyperparameter tuning was done using azure's HyperDrive package.

For parameter sampling, I chose to use RandomParameterSampling. This type of sampling is much faster than an exhaustive GridSearch, and often provides
just as good results. Another option would be to use Bayesian sampling to intelligently pick the next sample of hyperparameters, based on how the previous samples performed.
This would ensure such new samples improve the reported primary metric. Ultimately, I decided to use random parameter samplinh for the reasons mentioned above. 

For early termination policy, I chose to use BanditPolicy. Based on the provided slack criteria, runs that don't fall within this criteria will be terminated. 
Using this policy will allow us to focus on the best performing runs, and eliminate those that would report poorly if they were allowed to complete.

## AutoML
The AutoML process takes care of data preprocessing, feature engineering, scaling techniques and choice of the machine learning algorithm 
used to generate predictions. The entire process is illustrated in the image below.

![Alt text](autoMLpipeline.png?raw=true "AutoML Pipeline")

The AutoML model was defined using the AutoMLConfig class.
```python
automl_config = AutoMLConfig(
    experiment_timeout_minutes=30,
    compute_target=compute_target,
    task="classification",
    primary_metric="accuracy",
    training_data=clean_ds,
    label_column_name="y",
    n_cross_validations=5,
    enable_early_stopping=True,
    blocked_models=["KNN", "LinearSVM"])
```
The hyperparameters defined above can be explained as followed (information retrieved from [AutoMLConfig documentation](https://learn.microsoft.com/en-us/python/api/azureml-train-automl-client/azureml.train.automl.automlconfig.automlconfig?view=azure-ml-py)):
- experiment_timeout_minutes: Maximum amount of time in minutes that all iterations combined can take before the experiment terminates. In this case, 30 minutes is the maximum to avoid incurring high costs. 
- compute_target: The Azure Machine Learning compute target to run the Automated Machine Learning experiment on. In this experiment, I chose to use a STANDARD_DS3_V2 compute cluster
- task: The type of task to run. Values can be 'classification', 'regression', or 'forecasting' depending on the type of automated ML problem to solve.
- primary_metric: The metric that Automated Machine Learning will optimize for model selection. In this experiment, I chose accuracy but an alternative could
be AUC weighted. For a full list of valid metrics for the given task, you can use get_primary_metrics function. 
- n_cross_validations: How many cross validations to perform when user validation data is not specified. Since we did not provide any validation data,
we will run 5 cross validations on the training data. 
- enable_early_stopping: Whether to enable early termination if the score is not improving in the short term.
- blocked_models: A list of algorithms to ignore for an experiment. For this experiment, I chose to block KNN and LinearSVM models.

## Pipeline comparison

| Model                | Highest Accuracy | 
| -------------------- | ---------------- | 
| Sklearn + HyperDrive | 90.364%          |
| AutoML               | 91.785%          |

The difference in accuracy between the two models can be seen above. AutoML ended up performing slightly better in terms of accuracy and also reported an 
AUC_weighted metric of 0.9486412. The biggest architectural difference between the models is that one is a single estimator and the other is an ensemble of multiple
estimators. This definitely contributed to the difference in accuracy and high AUC of the AutoML model.

Given the results, I would choose AutoML for this task as it eliminates the need for manual trial and error and conducts the ML lifecycle in an efficient manner. 
I also really like the model explainability features that come with it.  

## Future work

The first thing I would do to improve this experiment is fix the class imbalance that is present in this data. This could be achieved by gathering more data, or applying
data imbalance techniques such as SMOTE. Additionally, I would change the primary metric to something more representative of the data. Given the imbalance, accuracy is not
the best metric as we could have a model that is very good at predicting one class. If we were to expose this model to new data that is more balanced, it may not perform
as well. A metric that might be more suitable here is ROC AUC. 

## Proof of cluster clean up
In the image below you can see the cluster used for this experiment processing for deletion. 
![Alt text](deleting-compute.png?raw=true "AutoML Pipeline")
