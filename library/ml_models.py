import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Optional, Dict, Tuple, List
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from scipy.optimize import minimize_scalar
from library.metrics_functions import find_best_threshold_for_predictions,compute_metrics, compute_confidence_interval
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

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


def model_find_best_ess_threshold(train_data:np.ndarray,
                                  train_labels:np.ndarray) -> float:
    """
    Finds the best threshold for classifying based on ESS, using sensitivity and specificity.
    The threshold is computed on train_data and evaluated on val_data.

    Parameters:
    - train_data: Pandas DataFrame containing the training ESS scores.
    - train_labels: True labels for training data.

    Returns:
    - best_threshold: The optimal threshold value.
    """

    def calculate_metrics(y_true, y_pred, output_dict: Optional[str] = False):
        """Calculate sensitivity, specificity, PPV, and NPV."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        if output_dict:
            return {'sensitivity': sensitivity, 'specificity': specificity, 'ppv': ppv, 'npv': npv}
        return sensitivity, specificity, ppv, npv

    # Define possible threshold values from max to min ESS
    thresholds = np.arange(train_data['ESS'].max(), train_data['ESS'].min() - 1, -1)
    thresholds = np.sort(thresholds)

    # Dictionary to store performance at each threshold
    predictions = {}

    # Iterate over all possible thresholds and compute sensitivity & specificity
    for thresh in thresholds:
        train_predicted = train_data['ESS'].apply(lambda x: 1 if x >= thresh else 0)
        sensitivity, specificity, ppv, npv = calculate_metrics(train_labels, train_predicted)
        predictions[thresh] = {'sensitivity': sensitivity, 'specificity': specificity, 'ppv': ppv, 'npv': npv}

    # Convert to DataFrame for analysis
    predictions_df = pd.DataFrame.from_dict(predictions, orient='index')

    # Find the threshold where sensitivity and specificity are closest
    predictions_df['difference'] = np.abs(predictions_df['sensitivity'] - predictions_df['specificity'])
    best_threshold = predictions_df['difference'].idxmin()
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
                     k:int=5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the stratified k-fold cross validation of each of the expected models in the dictionary of objects
    present in this library file. Compute the metrics and records which observation is a TP, TN, FN, and FP.
    :param models:
    :return:
        - Average metrics of each model across the folds with confidence intervals for specificity and sensitivity
        - classification results of each observation (TP, TN, FN, FP) with the labels and indexes from the source
    """
    data = df_model[features]
    labels = df_model[target]

    skf = StratifiedKFold(n_splits=k,
                          shuffle=True,
                          random_state=42)
    fold_indices = list(skf.split(data, labels))


    # Aggregate metrics DataFrame
    metrics_records = []
    classification_records = []  # Store all classifications
    for model_name, model in models.items():
        # model_name = "XGBoost"
        # model = models.get(model_name)
        for fold, (train_indices, val_indices) in enumerate(fold_indices):
            train_data = data.iloc[train_indices].copy()
            val_data = data.iloc[val_indices].copy()
            train_labels = labels.iloc[train_indices].values.ravel()
            val_labels = labels.iloc[val_indices].values.ravel()

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
                train_data = train_data[['ESS', 'Age', 'sex']]
                val_data = val_data[['ESS', 'Age', 'sex']]

            elif model_name == "Threshold on ESS":
                # Compute the best threshold from training data
                best_threshold = models.get("Threshold on ESS")(train_data[['ESS']], train_labels)

                # Apply threshold to validation data
                predictions = (val_data['ESS'] >= best_threshold).astype(int)

                # Compute metrics
                metrics = compute_metrics(predictions, val_labels)
                metrics.update({'fold': fold + 1, 'model': model_name, 'best_threshold': best_threshold})
                metrics_records.append(metrics)

                val_classification = classify_predictions(predictions=predictions,
                                                          labels=val_labels,
                                                          indices=val_indices,
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
                                                          indices=val_indices,
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
                                                          indices=val_indices,
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
                                                      indices=val_indices,
                                                      fold=fold + 1,
                                                      dataset_type='validation',
                                                      model_name=model_name)

            classification_records.append(val_classification)

            # EPI 219 ANALYSIS
            # Prepare input dictionary
            # models_dict = {
            #     model_name: (val_labels, y_pred_prob[:, 1]),  # pass the positive class probabilities
            # }
            # Run decision curve analysis
            # decision_curve_analysis(models_dict)
            # calibration_curve(y_true=val_labels, y_prob=y_pred_prob[:, 1])
            # CalibrationDisplay(y_true=val_labels, y_prob=y_pred_prob[:, 1])
            #
            # CalibrationDisplay.from_estimator(estimator=model, X=train_data, y=train_labels)
            # likelihood_ratios = y_pred_prob[:, 1] / y_pred_prob[:, 0]
            # brier_score = brier_score_loss(val_labels, y_pred_prob[:, 1])
            # # calculate Uncertainty
            # mean_outcome = np.mean(val_labels)
            # uncertainty = mean_outcome * (1 - mean_outcome)

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
    return df_avg_metrics, df_classifications



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

    "LogReg (ESS + Age + Gender)": LogisticRegression(penalty=None,  # Model (c)
                                                      solver='lbfgs',
                                                      max_iter=1000,
                                                      C=1),
    "Threshold on ESS": model_find_best_ess_threshold,  # Placeholder for rule-based model

    "Okun Tree": okun_decision,
}

