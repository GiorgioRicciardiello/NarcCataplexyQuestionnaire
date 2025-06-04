import pathlib

import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Optional, Dict, Tuple, List
import joblib
import os
from pandas import DataFrame
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from scipy.optimize import minimize_scalar
from library.metrics_functions import find_best_threshold_for_predictions,compute_metrics, compute_confidence_interval
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer  # needed to use IterativeImputer
from tabulate import tabulate
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

def make_veto_dataset(
        df_data: pd.DataFrame,
        use_dqb10602: bool
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Prepare the modeling data by selecting columns and handling missing values.

    Args:
        df_data (pd.DataFrame): Input DataFrame.
        target (str): Target column name.
        categorical_var (List[str]): List of categorical variable names.
        continuous_var (List[str]): List of continuous variable names.
        use_dqb10602 (bool): Whether to use the DQB10602 column.

    Returns:
        Tuple[pd.DataFrame, Optional[pd.Series]]: Processed DataFrame and veto column if applicable.
    """
    df = df_data.copy()
    if not use_dqb10602:
        # if not use the DQB10602, we then drop it
        df_dqb_veto = df['DQB10602'].copy()
        df.drop(columns='DQB10602', inplace=True)
        print(f'Column DQB10602 dropped from analysis')
    else:
        df_dqb_veto = None

    return df, df_dqb_veto


def train_xgboost(train_data: pd.DataFrame,
                  train_labels: np.ndarray,
                  val_data: pd.DataFrame,
                  val_labels: np.ndarray):

    def specificity_loss(preds, dtrain):
        """
        Custom loss function to approximate maximizing specificity.

        Args:
        - preds: Predicted probabilities.
        - dtrain: DMatrix with labels.

        Returns:
        - grad: Gradient of the loss function.
        - hess: Hessian of the loss function.
        """
        labels = dtrain.get_label()
        preds = 1 / (1 + np.exp(-preds))  # Convert logits to probabilities

        # Calculate gradients and hessians
        grad = -labels * (1 - preds) + (1 - labels) * preds * 3.2  # Penalize FP more
        hess = preds * (1 - preds) * (1 + (1 - labels))

        return grad, hess

    def specificity_eval_metric(preds, dtrain):
        """
        Custom evaluation metric to calculate specificity.

        Args:
        - preds: Predicted probabilities.
        - dtrain: DMatrix with labels.

        Returns:
        - name: Name of the metric.
        - result: Computed specificity.
        """
        labels = dtrain.get_label()
        preds = (preds > 0.5).astype(int)  # Threshold at 0.5
        tn = np.sum((labels == 0) & (preds == 0))
        fp = np.sum((labels == 0) & (preds == 1))
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        return 'specificity', specificity


    def find_best_threshold_for_predictions(y_true_train: np.ndarray,
                                            y_pred_train: np.ndarray,
                                            metric: str = 'specificity') -> float:
        """
        Find the best threshold for binary classification predictions based on a specific metric.
        Uses optimization for fine-tuned threshold selection.

        :param y_true_train: Ground truth binary labels.
        :param y_pred_train: Predicted probabilities (or scores).
        :param metric: Metric to optimize. Options: 'f1', 'accuracy', 'precision', 'recall', 'auc'.
        :return: Best threshold based on the metric.
        """

        def metric_for_threshold(threshold):
            y_pred_thresh = (y_pred_train >= threshold).astype(int)
            if metric == 'f1':
                return -f1_score(y_true_train, y_pred_thresh)
            elif metric == 'accuracy':
                return -accuracy_score(y_true_train, y_pred_thresh)
            elif metric == 'sensitivity':  # Sensitivity (Recall)
                return -recall_score(y_true_train, y_pred_thresh)
            elif metric == 'specificity':  # Specificity (True Negative Rate)
                tn, fp, fn, tp = confusion_matrix(y_true_train, y_pred_thresh).ravel()
                specificity = tn / (tn + fp)
                return -specificity
            elif metric == 'auc':
                return -roc_auc_score(y_true_train, y_pred_thresh)
            else:
                raise ValueError("Unsupported metric. Choose from 'f1', 'accuracy', 'sensitivity', 'specificity', 'auc'.")

        # Use scalar minimization for the threshold search
        result = minimize_scalar(metric_for_threshold, bounds=(0.0, 1.0), method='bounded')
        best_threshold = result.x
        best_metric_value = -result.fun

        # print(f"Best threshold based on {metric}: {best_threshold:.4f} with score: {best_metric_value:.4f}")
        return best_threshold


    params = {
        'scale_pos_weight': 16,
        'max_depth': 12,
        'reg_lambda': 0.001,
        'gamma': 0.2,
        'reg_alpha': 0.1,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss'
    }
    dtrain = xgb.DMatrix(train_data, label=train_labels)
    dval = xgb.DMatrix(val_data, label=val_labels)

    model = xgb.train(params,
                      dtrain,
                      num_boost_round=2,
                      custom_metric=specificity_eval_metric,
                      obj=specificity_loss,
                      evals=[(dtrain, 'train'), (dval, 'valid')],
                      verbose_eval=False, # 200,

                      )

    best_threshold = find_best_threshold_for_predictions(
        y_true_train=train_labels,
        y_pred_train=model.predict(dtrain),
        metric='specificity'
    )
    # print(f'Best Threshold: {best_threshold}')

    y_pred_val = (model.predict(dval) > best_threshold).astype(int)
    y_pred_prob_val = model.predict(dval)

    y_pred_train = (model.predict(dtrain) > best_threshold).astype(int)
    y_pred_prob_train = model.predict(dtrain)

    return y_pred_val, y_pred_prob_val, y_pred_train, y_pred_prob_train


def okun_decision(fold_split: pd.DataFrame,
                  columns_tree: Optional[list[str]] = None) :
    """
    Replicate the Okun paper decision tree with classification metrics.
    :param fold_split: Input DataFrame
    :param columns_tree: List of columns for the decision tree
    :return: Counts, percentages, and split metrics for the decision tree
    """

    if columns_tree is None:
        columns_tree = ['JOKING', 'LAUGHING', 'ANGER']
    df_tree = fold_split[columns_tree]

    # Initialize predictions (default to 0)
    predictions = np.zeros(len(df_tree))

    # Decision tree logic
    mask_joke_yes = df_tree['JOKING'] == 1
    mask_joke_no = df_tree['JOKING'] == 0
    mask_angry_yes = df_tree['ANGER'] == 1
    mask_laugh_yes = df_tree['LAUGHING'] == 1

    # Assign class 1 based on Okun's rules
    predictions[mask_joke_yes & mask_angry_yes] = 1
    predictions[mask_joke_no & mask_laugh_yes] = 1

    return predictions


# def model_find_best_ess_threshold(train_data:np.ndarray,
#                                   train_labels:np.ndarray) -> float:
#     """
#     Finds the best threshold for classifying based on ESS, using sensitivity and specificity.
#     The threshold is computed on train_data and evaluated on val_data.
#
#     Parameters:
#     - train_data: Pandas DataFrame containing the training ESS scores.
#     - train_labels: True labels for training data.
#
#     Returns:
#     - best_threshold: The optimal threshold value.
#     """
#
#     def calculate_metrics(y_true, y_pred, output_dict: Optional[str] = False):
#         """Calculate sensitivity, specificity, PPV, and NPV."""
#         tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
#         sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
#         specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
#         ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
#         npv = tn / (tn + fn) if (tn + fn) > 0 else 0
#         if output_dict:
#             return {'sensitivity': sensitivity, 'specificity': specificity, 'ppv': ppv, 'npv': npv}
#         return sensitivity, specificity, ppv, npv
#
#     # Define possible threshold values from max to min ESS
#     thresholds = np.arange(train_data['ESS'].max(), train_data['ESS'].min() - 1, -1)
#     thresholds = np.sort(thresholds)
#
#     # Dictionary to store performance at each threshold
#     predictions = {}
#
#     # Iterate over all possible thresholds and compute sensitivity & specificity
#     for thresh in thresholds:
#         train_predicted = train_data['ESS'].apply(lambda x: 1 if x >= thresh else 0)
#         sensitivity, specificity, ppv, npv = calculate_metrics(train_labels, train_predicted)
#         predictions[thresh] = {'sensitivity': sensitivity, 'specificity': specificity, 'ppv': ppv, 'npv': npv}
#
#     # Convert to DataFrame for analysis
#     predictions_df = pd.DataFrame.from_dict(predictions, orient='index')
#
#     # Find the threshold where sensitivity and specificity are closest
#     predictions_df['difference'] = np.abs(predictions_df['sensitivity'] - predictions_df['specificity'])
#     best_threshold = predictions_df['difference'].idxmin()
#     return best_threshold
#


def model_find_best_ess_threshold(train_data: np.ndarray,
                                  train_labels: np.ndarray,
                                  plot:Optional[bool]=False) -> float:
    """
    Finds the best threshold for classifying based on ESS, using sensitivity and specificity.
    The threshold is chosen by maximizing Youden's index (sensitivity + specificity - 1).

    Parameters:
    - train_data: A numpy array containing the continuous ESS scores.
    - train_labels: A numpy array of true binary labels (0 or 1) corresponding to train_data.

    Returns:
    - best_threshold: The optimal threshold value that provides the best balance
                      between sensitivity and specificity.
    """
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)

    best_threshold = None
    best_youden_index = -np.inf

    # Use unique ESS values as candidate thresholds
    candidate_thresholds = np.unique(train_data)
    sensitivities = []
    specificities = []
    youden_indices = []

    for threshold in candidate_thresholds:
        # Generate predictions based on the threshold
        predictions = (train_data >= threshold).astype(int)

        # Calculate confusion matrix components
        tp = np.sum((predictions == 1) & (train_labels == 1))
        tn = np.sum((predictions == 0) & (train_labels == 0))
        fp = np.sum((predictions == 1) & (train_labels == 0))
        fn = np.sum((predictions == 0) & (train_labels == 1))

        # Compute sensitivity and specificity, ensuring no division by zero
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        # Calculate Youden's index
        youden_index = sensitivity + specificity - 1

        sensitivities.append(sensitivity)
        specificities.append(specificity)
        youden_indices.append(youden_index)

        # Update the best threshold if the current one is better
        if youden_index > best_youden_index:
            best_youden_index = youden_index
            best_threshold = threshold

    if plot:
        # Plot the metrics
        sensitivities = np.array(sensitivities)
        specificities = np.array(specificities)
        candidate_thresholds = np.array(candidate_thresholds)

        # Find the threshold where sensitivity and specificity are closest
        diff = np.abs(sensitivities - specificities)
        idx_intersect = np.argmin(diff)
        threshold_intersect = candidate_thresholds[idx_intersect]
        # Compute the approximate intersection value (average)
        intersection_value = (sensitivities[idx_intersect] + specificities[idx_intersect]) / 2

        # Plot the metrics
        plt.figure(figsize=(10, 6))
        plt.plot(candidate_thresholds, sensitivities, label='Sensitivity', color='blue')
        plt.plot(candidate_thresholds, specificities, label='Specificity', color='green')
        plt.plot(candidate_thresholds, youden_indices, label="Youden's Index", color='red', linestyle='--')

        # Mark the best threshold (max Youden's index)
        plt.axvline(best_threshold, color='black', linestyle=':', label=f'Best Threshold = {best_threshold:.2f}')

        # Plot a red dot at the intersection
        plt.plot(threshold_intersect, intersection_value, 'ro', markersize=8,
                 label=f'Intersection (Sens/Spec ~ {intersection_value:.2f})')

        plt.xlabel("ESS Threshold")
        plt.ylabel("Metric Value")
        plt.title("ESS Threshold vs. Sensitivity, Specificity, and Youden's Index")
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        plt.show()

    return best_threshold


def classify_predictions(predictions: np.ndarray,
                         labels: np.ndarray,
                         indices: np.ndarray,
                         fold: int,
                         model_name:str,
                         dataset_type: str) -> pd.DataFrame:
    """
    Classify each observation as True Positive (TP), True Negative (TN),
    False Positive (FP), or False Negative (FN).

    Parameters:
    - predictions (np.ndarray): Predicted binary labels.
    - labels (np.ndarray): True binary labels.
    - indices (np.ndarray): Original indices of the dataset.
    - fold (int): Fold number.
    - dataset_type (str): Either 'train' or 'validation' to indicate data type.

    Returns:
    - pd.DataFrame: DataFrame with original index and classification labels.
    """
    classification = np.where((labels == 1) & (predictions == 1), 'TP',
                              np.where((labels == 0) & (predictions == 0), 'TN',
                                       np.where((labels == 0) & (predictions == 1), 'FP', 'FN')))

    return pd.DataFrame({
        'model_name': model_name,
        'index': indices,  # Preserve original index
        'true_label': labels,
        'predicted_label': predictions,
        'classification': classification,
        'fold': fold,
        'dataset_type': dataset_type  # 'train' or 'validation'
    })


def compute_cross_validation(models:Dict[str, object],
                     df_model:pd.DataFrame,
                     features:list[str],
                     target:str,
                     k:int=5) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    """
    Run the stratified k-fold cross validation of each of the expected models in the dictionary of objects
    present in this library file. Compute the metrics and records which observation is a TP, TN, FN, and FP.
    :param models:
    :return:
        - Average metrics of each model across the folds with confidence intervals for specificity and sensitivity
        - classification results of each observation (TP, TN, FN, FP) with the labels and indexes from the source
    """

    def save_imputed_folds(imputed_folds: list,
                           save_path: str) -> None:
        """
        Saves each fold's imputed data to disk using joblib.

        :param imputed_folds: List of dicts with train/val data and labels.
        :param save_path: Directory where the folds will be saved.
        """
        save_path = pathlib.Path(save_path)
        if not save_path.exists():
            save_path.mkdir(parents=True, exist_ok=True)

        for i, fold in enumerate(imputed_folds):
            joblib.dump(fold, os.path.join(str(save_path), f"fold_{i}.joblib"))
        print(f"Saved {len(imputed_folds)} folds to {save_path}")

    def data_imputation(data_split: pd.DataFrame,
                        verbose: bool = True) -> pd.DataFrame:
        """
        Compute data imputation on each fold separately to avoid data leakage.
        This version imputes missing values in the specified continuous and categorical columns.
        It also prints descriptive statistics for the imputed columns before and after imputation if verbose is True.

        :param data_split: Input DataFrame with missing values.
        :param verbose: Whether to print detailed descriptions of the imputed columns.
        :return: DataFrame with imputed values.
        """
        # Define the categorical and continuous variables for imputation
        categorical_var = ['sex', 'Ethnicity', 'LAUGHING', 'ANGER', 'EXCITED', 'SURPRISED', 'HAPPY',
                           'EMOTIONAL', 'QUICKVERBAL', 'EMBARRAS', 'DISCIPLINE', 'SEX', 'DURATHLETIC',
                           'AFTATHLETIC', 'ELATED', 'STRESSED', 'STARTLED', 'TENSE', 'PLAYGAME',
                           'ROMANTIC', 'JOKING', 'MOVEDEMOT', 'KNEES', 'JAW', 'HEAD', 'HAND', 'SPEECH']
        continuous_var = ['Age', 'BMI', 'ESS']

        # Filter the list to only include columns present in the dataset
        covariates_list = [col for col in (continuous_var + categorical_var) if col in data_split.columns]
        cols_with_many_nans = data_split.columns[data_split.isna().sum() > 0]
        if len(cols_with_many_nans) == 0:
            print(f'Imputation not required: There are {len(cols_with_many_nans)} columns with missing values.')
            return data_split
        # Display summary of columns before imputation
        if verbose:
            print(f'Imputation for columns ({len(cols_with_many_nans)}): \n{cols_with_many_nans}')
            print("=== Before Imputation ===")
            print("Missing value counts:")
            print(data_split[cols_with_many_nans].isna().sum())
            print("\nDescriptive statistics:")
            print(data_split[cols_with_many_nans].describe())

        # Work on a copy of the full dataset
        df_impute = data_split.copy()

        # Create a dictionary to keep track of column types for post-processing
        covariates = {col: 'continuous' for col in continuous_var if col in covariates_list}
        covariates.update({col: 'ordinal' for col in categorical_var if col in covariates_list})

        # Initialize the imputer (IterativeImputer is from sklearn.impute)
        imputer = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=100,
            tol=1e-3,
            n_nearest_features=2,
            initial_strategy="mean",
            imputation_order="ascending"
        )

        # Fit the imputer on the selected columns and transform the data
        imputed_data = imputer.fit_transform(df_impute[covariates_list])
        df_imputed = pd.DataFrame(imputed_data, columns=covariates_list, index=df_impute.index)

        # For categorical (ordinal) columns, round the imputed values to the nearest integer
        for col, col_type in covariates.items():
            if col_type == 'ordinal':
                df_imputed[col] = np.round(df_imputed[col]).astype(int)

        # Replace the original columns with the imputed values
        df_impute[covariates_list] = df_imputed[covariates_list]

        # Display summary of columns after imputation
        if verbose:
            print("\n=== After Imputation ===")
            print("Missing value counts:")
            print(df_impute[cols_with_many_nans].isna().sum())
            print("\nDescriptive statistics:")
            print(df_impute[cols_with_many_nans].describe())

        return df_impute

    data = df_model[features]
    labels = df_model[target]

    skf = StratifiedKFold(n_splits=k,
                          shuffle=True,
                          random_state=42)
    fold_indices = list(skf.split(data, labels))
    # Precompute the imputed folds
    imputed_folds = []
    for fold, (train_indices, val_indices) in enumerate(fold_indices):
        print(f'|----Creating Fold {fold}---|')
        # Get the raw train/val splits
        train_data = data.iloc[train_indices].copy()
        val_data = data.iloc[val_indices].copy()
        train_labels = labels.iloc[train_indices].values.ravel()
        val_labels = labels.iloc[val_indices].values.ravel()

        # Compute imputation for each fold once
        imputed_train_data = data_imputation(data_split=train_data, verbose=True)
        imputed_val_data = data_imputation(data_split=val_data, verbose=True)

        # Store imputed data and labels for later reuse
        imputed_folds.append({
            'train_data': imputed_train_data,
            'val_data': imputed_val_data,
            'train_labels': train_labels,
            'val_labels': val_labels
        })
    # save_imputed_folds(imputed_folds=imputed_folds, save_path=r'\NarcCataplexyQuestionnaire\data')

    # Aggregate metrics DataFrame
    metrics_records = []
    classification_records = []  # Store all classifications
    elastic_net_coefs_list = []
    elastic_net_predictions_list = []  # store elastic net predicted prob for net benefit curves
    for model_name, model in models.items():
        model_name = "Elastic Net"
        model = models.get(model_name)
        for fold, fold_data in enumerate(imputed_folds):
            train_data = fold_data.get('train_data').copy()
            val_data = fold_data.get('val_data').copy()
            train_labels = fold_data.get('train_labels')
            val_labels = fold_data.get('val_labels')
            # df_visualize = val_data.copy()
            # df_visualize['NT1'] = val_labels
            # print(f'\nfold {fold}')
            # visualize_table(df=df_visualize, group_by=['NT1', 'DQB10602'])

            if not model_name in ["Threshold on ESS", "Okun Tree"]:
                # normalize continuous features, the rest of the data is sparse
                col_cont = [col for col in train_data.columns if
                            not set(train_data[col].dropna().unique()).issubset({0, 1})]

                scaler = StandardScaler()
                train_data[col_cont] = scaler.fit_transform(train_data[col_cont])
                val_data[col_cont] = scaler.transform(val_data[col_cont])

            # Selecting features for ESS-only and ESS + Age + Gender models
            if model_name == "LogReg (ESS Only)":
                train_data = train_data[['ESS']]
                val_data = val_data[['ESS']]

            elif model_name == "LogReg (ESS + Age + Gender)":
                if not all(col in train_data.columns for col in ['Age', 'sex']):
                    continue
                train_data = train_data[['ESS', 'Age', 'sex']]
                val_data = val_data[['ESS', 'Age', 'sex']]

            elif model_name == "Threshold on ESS":
                # Compute the best threshold from training data
                best_threshold = models.get("Threshold on ESS")(train_data['ESS'], train_labels)
                # Apply threshold to validation data
                predictions = (val_data['ESS'] >= best_threshold).astype(int)

                # Compute metrics
                metrics = compute_metrics(predictions, val_labels)
                metrics.update({'fold': fold + 1, 'model': model_name, 'best_threshold': best_threshold})
                metrics_records.append(metrics)

                val_classification = classify_predictions(predictions=predictions,
                                                          labels=val_labels,
                                                          indices=val_data.index,
                                                          fold=fold + 1,
                                                          dataset_type='validation',
                                                          model_name=model_name)

                classification_records.append(val_classification)
                continue  # Skip next model

            elif model_name == "Okun Tree":
                predictions = models.get("Okun Tree")(fold_split=val_data)
                # Compute metrics
                metrics = compute_metrics(predictions, val_labels)
                metrics.update({'fold': fold + 1, 'model': model_name})
                metrics_records.append(metrics)

                val_classification = classify_predictions(predictions=predictions,
                                                          labels=val_labels,
                                                          indices=val_data.index,
                                                          fold=fold + 1,
                                                          dataset_type='validation',
                                                          model_name=model_name)

                classification_records.append(val_classification)
                continue


            elif model_name == "XGBoost":
                y_pred_val, y_pred_prob_val, y_pred_train, y_pred_prob_train = models.get("XGBoost")(
                    train_data=train_data,
                    train_labels=train_labels,
                    val_data=val_data,
                    val_labels=val_labels)

                metrics = compute_metrics(y_pred_val, val_labels)
                metrics.update({'fold': fold + 1, 'model': model_name})
                metrics_records.append(metrics)

                # Classify train and validation sets

                val_classification = classify_predictions(predictions=y_pred_val,
                                                          labels=val_labels,
                                                          indices=val_data.index,
                                                          fold=fold + 1,
                                                          dataset_type='validation',
                                                          model_name=model_name)

                classification_records.append(val_classification)

                continue

            # Train model and predict
            model.fit(train_data, train_labels)
            predictions = model.predict(val_data)
            # Extract only the probability of the positive class (1)
            # y_pred_prob = model.predict_proba(val_data)
            #
            # best_threshold = find_best_threshold_for_predictions(
            #     y_true_train=train_labels,
            #     y_pred_train=model.predict_proba(train_data)[:, 1],
            #     metric='specificity'
            # )

            # Compute metrics
            metrics = compute_metrics(predictions, val_labels)
            metrics.update({'fold': fold + 1, 'model': model_name})
            metrics_records.append(metrics)

            val_classification = classify_predictions(predictions=predictions,
                                                      labels=val_labels,
                                                      indices=val_data.index,
                                                      fold=fold + 1,
                                                      dataset_type='validation',
                                                      model_name=model_name)

            classification_records.append(val_classification)

            if model_name == 'Elastic Net':
                # Elastic net is the best model, therefore we will collect the feature importance across the folds
                # and report them with standard errors
                elastic_net_coefs_list.append(model.coef_.flatten())
                fold_df = pd.DataFrame({
                    "model_name": model_name,
                    "true_label": val_labels,
                    "predicted_prob": model.predict_proba(val_data)[:, 1], # positive class
                    "prediction": predictions,
                    "fold_number": fold + 1  # Use fold+1 if you want fold numbering to start at 1
                })
                elastic_net_predictions_list.append(fold_df)

    df_classifications = pd.concat(classification_records)
    df_classifications = df_classifications.sort_values(by=['model_name', 'fold'])


    df_agg_metrics = pd.DataFrame(metrics_records)
    # Reordering columns to set 'model' and 'fold' as the first columns
    df_agg_metrics = df_agg_metrics[
        ['model', 'fold'] + [col for col in df_agg_metrics.columns if col not in ['model', 'fold']]]

    # compute confidence intervals
    df_ci = {}
    metric_ci = ['sensitivity', 'specificity']
    for model_ in df_agg_metrics['model'].unique():
        df_ci[model_] = {}
        for metric in metric_ci:
            values = df_agg_metrics.loc[df_agg_metrics['model'] == model_, metric].values
            df_ci[model_][f'{metric}_ci'] = compute_confidence_interval(values)
    df_ci = pd.DataFrame.from_dict(df_ci, orient='index')
    df_ci.reset_index(inplace=True)
    df_ci.rename(columns={'index': 'model'}, inplace=True)

    # compute the average across the measures
    df_avg_metrics = df_agg_metrics.groupby(['model']).mean(numeric_only=True).reset_index()
    df_avg_metrics.drop(columns='fold',
                        inplace=True)
    df_avg_metrics = df_avg_metrics.sort_values(by='specificity', ascending=False)

    # include the confidence intervals in the measure
    df_avg_metrics = pd.merge(left=df_avg_metrics,
                              right=df_ci,
                              on='model')

    print(df_avg_metrics[['model', 'specificity_ci', 'sensitivity_ci']])

    # From the collected coefficients for the elastic net model, compute the standard erros and model coeff
    if len(elastic_net_coefs_list) > 0:
        feature_names = train_data.columns.tolist()  # Using the last fold's train_data columns
        df_elastic_net_feature_importance = collect_elastic_net_coefficients_and_std(elastic_net_coefs_list, feature_names)
    else:
        df_elastic_net_feature_importance = pd.DataFrame()

    # getting the prediction probabilities of the elasticnet model to make the net benefit curves
    df_elastic_net_predictions = pd.concat(elastic_net_predictions_list, ignore_index=True)

    return df_avg_metrics, df_classifications, df_elastic_net_feature_importance, df_elastic_net_predictions



# Fixing the issue with ElasticNetCV to properly extract feature importance and standard errors

def collect_elastic_net_coefficients_and_std(coefs_list:List[np.ndarray],
                                             feature_names):
    """
    Usage for Logistic Regression model of sklearn
    Given a list of coefficient arrays (one per fold), compute the mean and
    standard error (SE) for each feature.

    Parameters:
    - coefs_list: list of arrays, each of shape (n_features,), list of arrays, each list element are the
        coefficients obtained at each training fold
    - feature_names: list of feature names

    Returns:
    - DataFrame with columns: Feature, Mean Coefficient, Standard Error
    """
    # Stack the coefficient arrays (each row corresponds to a fold)
    coefs_array = np.vstack(coefs_list)
    mean_coefs = np.mean(coefs_array, axis=0)

    # Standard error computed as sample std dev divided by sqrt(n_folds)
    std_errors = np.std(coefs_array, axis=0, ddof=1) / np.sqrt(coefs_array.shape[0])

    # Create DataFrame
    stats_df = pd.DataFrame({
        "Feature": feature_names,
        "Mean Coefficient": mean_coefs,
        "Standard Error": std_errors
    })

    # Optionally, sort by the absolute mean coefficient value (descending)
    stats_df["Abs Mean"] = np.abs(stats_df["Mean Coefficient"])
    stats_df = stats_df.sort_values(by="Abs Mean", ascending=False).drop(columns="Abs Mean")

    return stats_df


def plot_elastic_net_model_coefficients(df_params: pd.DataFrame,
                                        title: str = None,
                                        output_path: pathlib.Path = None,
                                        ):
    """
    Generate a styled plot for the elastic net feature importance coefficients.

    Parameters:
    - df_params: DataFrame with columns 'Feature', 'Mean Coefficient', 'Standard Error'
    - output_path: Path to save the plot (optional)
    """
    import matplotlib as mpl
    # Improve overall font style
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.size'] = 12
    mpl.rcParams['axes.titlesize'] = 16
    mpl.rcParams['axes.titleweight'] = 'bold'
    mpl.rcParams['axes.labelsize'] = 14

    plt.figure(figsize=(10, 6))

    # Use a colormap to assign different colors for each feature
    cmap = plt.get_cmap("tab10")
    num_features = df_params.shape[0]
    colors = [cmap(i % 10) for i in range(num_features)]

    # Calculate x-axis limits: max value among (mean coefficient + standard error) plus margin.
    max_val = (df_params["Mean Coefficient"].abs() + df_params["Standard Error"]).max()
    plt.xlim(0, max_val * 1.1)

    # Plot horizontal bar chart with error bars
    plt.barh(df_params["Feature"],
             np.abs(df_params["Mean Coefficient"]),
             xerr=df_params["Standard Error"],
             capsize=5,
             color=colors)

    plt.xlabel("Mean Absolute Coefficient")
    plt.ylabel("Feature")
    if not title:
        title = "Elastic Net Feature Importance Across Folds"
    plt.title(title)
    plt.gca().invert_yaxis()  # Highest importance on top
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path / 'elastic_net_model_coefficients.png', dpi=300)
    plt.show()


def elastic_net_feature_importance_fixed(fitted_model, X, y):
    """
    Computes feature importance for an ElasticNet model and returns a DataFrame
    including standard errors.

    Parameters:
    - model: Trained ElasticNet model
    - X: Feature matrix (numpy array or pandas DataFrame)
    - y: Target variable (numpy array or pandas Series)

    Returns:
    - DataFrame with feature importance and standard errors
    """

    # Extract coefficients across folds
    # coefs = np.array(elastic_cv.path(X, y, l1_ratio=elastic_cv.l1_ratio_)[1])

    # Compute mean and standard deviation of coefficients across cross-validation folds
    feature_importance = np.abs(fitted_model.coef_)
    feature_names = [f"Feature {i}" for i in range(X.shape[1])]

    # mean_coefs = np.abs(coefs.mean(axis=1))
    # std_errors = coefs.std(axis=1)

    # Extract feature importance (absolute coefficients)
    feature_importance = np.abs(model.coef_)
    feature_names = [f"Feature {i}" for i in range(X.shape[1])]

    # Sort features by importance
    sorted_indices = np.argsort(feature_importance)[::-1]  # Sort descending
    sorted_importance = feature_importance[sorted_indices]
    sorted_names = [feature_names[i] for i in sorted_indices]
    sorted_std_errors = std_errors[sorted_indices]

    # Create DataFrame
    importance_df = pd.DataFrame({
        "Feature": sorted_names,
        "Importance": sorted_importance,
        "Standard Error": sorted_std_errors
    })

    # Plot feature importance with error bars
    plt.figure(figsize=(10, 5))
    plt.barh(sorted_names, sorted_importance, xerr=sorted_std_errors, color='royalblue', capsize=5)
    plt.xlabel("Absolute Coefficient Value")
    plt.ylabel("Feature")
    plt.title("Feature Importance in ElasticNet Model")
    plt.gca().invert_yaxis()  # Highest importance on top
    plt.show()

    return importance_df



models = {
    "Logistic Regression": LogisticRegression(penalty=None,
                                              solver='lbfgs',
                                              max_iter=1000,
                                              C=1),
    "Lasso": LogisticRegression(penalty='l1',
                                solver='liblinear',
                                max_iter=1000),

    "Elastic Net": LogisticRegression(penalty='elasticnet',
                                      solver='saga',
                                      l1_ratio=0.7,
                                      max_iter=1000),
    "LDA": LinearDiscriminantAnalysis(),

    "SVM": SVC(kernel="rbf",
               degree=3,
               gamma="scale",
               probability=True),

    "XGBoost": train_xgboost,

    "LogReg (ESS Only)": LogisticRegression(penalty=None,  # Model (b)
                                            solver='lbfgs',
                                            max_iter=1000,
                                            C=1),

    # we remove Age and Gender from the study, therefore we discard this model
    # "LogReg (ESS + Age + Gender)": LogisticRegression(penalty=None,  # Model (c)
    #                                                   solver='lbfgs',
    #                                                   max_iter=1000,
    #                                                   C=1),

    "Threshold on ESS": model_find_best_ess_threshold,  # Placeholder for rule-based model

    "Okun Tree": okun_decision,
}


def visualize_table(df: pd.DataFrame,
                    group_by: List[str]) -> pd.DataFrame:
    """
    Count the unique pair combinations ina dataframe within the grouped by columns
    :param df:
    :param group_by:
    :return:
    """
    df_copy = df.copy()
    print("Distribution before modification:")
    df_plot_before = df_copy.fillna('NaN')
    grouped_counts_before = df_plot_before.groupby(group_by).size().reset_index(
        name='Counts')
    print(tabulate(grouped_counts_before, headers='keys', tablefmt='grid'))
    print(f'Remaining Rows: {df_copy.shape[0]}')
    return grouped_counts_before


def load_imputed_folds(save_path: str) -> list:
    """
    Loads all saved imputed folds from disk.

    :param save_path: Directory where folds are stored.
    :return: List of dicts for each fold.
    """
    fold_files = sorted(pathlib.Path(save_path).glob("fold_*.joblib"))
    return [joblib.load(f) for f in fold_files]


def summarize_fold_consistency(imputed_folds: list,
                               feature_names: list,
                               target_name: str) -> pd.DataFrame:
    summaries = []

    for fold_idx, fold in enumerate(imputed_folds):
        for split in ['train', 'val']:
            data = fold[f'{split}_data']
            labels = fold[f'{split}_labels']

            for feature in feature_names:
                feature_vals = data[feature]
                summaries.append({
                    'fold': fold_idx,
                    'split': split,
                    'feature': feature,
                    'num_positives': labels.sum(),
                    'mean': feature_vals.mean(),
                    'std': feature_vals.std(),
                    'target_pos_rate': labels.mean()
                })

    return pd.DataFrame(summaries)