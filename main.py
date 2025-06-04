from config.config import config
import numpy as np
import pandas as pd
from library.ml_models import (compute_cross_validation,
                               models,
                               make_veto_dataset,
                               load_imputed_folds,
                               summarize_fold_consistency)
from library.metrics_functions import apply_veto_rule_re_classifications
from library.plot_functions import (plot_model_metrics, plot_model_metrics_specific_columns,
                                    create_venn_diagram, plot_elastic_net_model_coefficients,
                                    plot_dcurves_per_fold, ppv_curve,multi_ppv_plot_combined, plot_calibration,
                                    multi_ppv_plot, multi_calibration_plot)
from library.metrics_functions import recompute_classification, extract_metrics
from typing import Optional, Tuple, List


if __name__ == '__main__':
    # %% Read data
    # df_data = pd.read_csv(config.get('data_pre_proc_files').get('ssq_ssqdx'))
    # df_data = df_data.loc[df_data['narcolepsy'] != 'pseudo narcolepsy']
    # df_data.rename(columns={'cataplexy_clear_cut': 'NT1'}, inplace=True)

    df_data = pd.read_csv(config.get('data_pre_proc_files').get('anic_okun'))

    # df_data.rename(columns={'cataplexy_clear_cut': 'NT1'}, inplace=True)

    # %% output paths
    TEST = False
    OVERWRITE = False
    test = 'test_' if TEST else ''
    path_avg_metrics_config = config.get('results_path').get('results').joinpath(f'{test}avg_metrics_config.csv')
    path_classifications_config = config.get('results_path').get('results').joinpath(f'{test}classifications_config.csv')
    path_avg_metrics_veto = config.get('results_path').get('results').joinpath(f'{test}avg_metrics_veto.csv')
    path_avg_classifications_config_veto = config.get('results_path').get('results').joinpath(f'{test}classifications_config_veto.csv')
    path_avg_paper = config.get('results_path').get('results').joinpath(f'{test}avg_paper.csv')
    path_feature_importance = config.get('results_path').get('results').joinpath(f'{test}feature_importance.csv')
    path_model_metrics = config.get('results_path').get('results').joinpath(f'model_metrics.png')
    path_pred_prob_elastic = config.get('results_path').get('results').joinpath(f'{test}pred_prob_elasticnet.csv')
    # path_avg_paper_veto = config.get('results_path').get('results').joinpath(f'avg_paper_veto.csv')
    path_plot_best_model = config.get('results_path').get('main_model')
    # %% Select columns and drop columns with nans
    target = 'NT1 ICSD3 - TR'
    target_nt2 = target.replace('1', '2')

    categorical_var = ['sex', 'LAUGHING', 'ANGER', 'EXCITED',
                       'SURPRISED', 'HAPPY', 'EMOTIONAL', 'QUICKVERBAL', 'EMBARRAS',
                       'DISCIPLINE', 'SEX', 'DURATHLETIC', 'AFTATHLETIC', 'ELATED',
                       'STRESSED', 'STARTLED', 'TENSE', 'PLAYGAME', 'ROMANTIC',
                       'JOKING', 'MOVEDEMOT', 'KNEES', 'JAW', 'HEAD', 'HAND', 'SPEECH',
                       'DQB10602']
    continuous_var = ['Age', 'BMI', 'ESS', 'SLEEPIONSET']

    columns = list(set(categorical_var + continuous_var + [target]))
    # remove Age and Sex so the questionnaire is inclusive
    columns = [col for col in columns if not col in ['sex', 'Age']]
    df_data = df_data.loc[:, columns]
    # df_data = df_data.dropna(axis=1)
    cols_with_many_nans = df_data.columns[df_data.isna().sum() > 15]
    df_data.drop(cols_with_many_nans, axis=1, inplace=True)
    df_data.reset_index(drop=True, inplace=True)
    df_data = df_data.reindex(sorted(df_data.columns), axis=1)
    print(f'Dataset dimension: {df_data.shape}')
    # %% configuration of different models to run
    # col_ukbb = ['Age', 'BMI', 'sex', 'ESS', 'JAW', 'HEAD', 'HAND',
    #             'SPEECH', 'JOKING', 'LAUGHING', 'ANGER',
    #             'QUICKVERBAL']
    col_ukbb = ['BMI', 'ESS', 'JAW', 'HEAD', 'HAND', 'KNEES',
                'SPEECH', 'JOKING', 'LAUGHING', 'ANGER',
                'QUICKVERBAL']

    # some columns are not available because of missigness e.g., BMI
    col_ukbb_avail = [col for col in df_data.columns if col in col_ukbb]

    configurations = {
        "questionnaire": {
            'features': [col for col in df_data.columns if not col in [target, 'DQB10602']],
            'target': target,
            'dqb': False,
        },

        'questionnaire_hla': {
            'features': [col for col in df_data.columns if col != target],
            'target': target,
            'dqb': True,
        },

        'ukbb': {
            'features': col_ukbb_avail,
            'target': target,
            'dqb': False,
        },

        'ukbb_hla': {
            'features': col_ukbb_avail + ['DQB10602'],
            'target': target,
            'dqb': True,
        }

    }

    feature_set_mapper = {
        f'questionnaire': f'Full feature set (k={len(configurations.get("questionnaire").get("features"))})',
        'questionnaire_hla': f'Full Feature Set + DQB1*06:02  (k={len(configurations.get("questionnaire_hla").get("features"))})',
        'ukbb': f'Reduced Feature Set (k={len(configurations.get("ukbb").get("features"))})',
        'ukbb_hla': f'Reduced Feature Set + DQB1*06:02 (k={len(configurations.get("ukbb_hla").get("features"))})',
    }
    for key, val in feature_set_mapper.items():
        print(f'\t{key}: {val}')

    for key, dictionary in configurations.items():
        num_feature = len(dictionary.get('features'))
        print(f'{key}: {num_feature}')


    # Create a Venn diagram comparing "questionnaire" and "ukbb"
    create_venn_diagram(
        configs=configurations,
        config1="questionnaire",
        config2="ukbb",
        figure_size=(9, 8),
        set_labels=("Full Questionnaire Feature Set", "Reduces Feature Set"),
        colors=('pink', 'orange'),
        alpha=0.7,
        title="Venn Diagram of Questionnaire vs. UKBB Features"
    )

    # run the analysis if the files do not exist or it overwrite is set to True
    if not (path_avg_metrics_config.is_file() and path_classifications_config.is_file()) or OVERWRITE:
        df_feature_importance = pd.DataFrame()
        df_avg_metrics_models = {}
        k = 5
        df_classifications_configs = pd.DataFrame()
        df_metrics_configs = pd.DataFrame()
        df_elastic_pred_config = pd.DataFrame()
        for conf_name, conf_values in configurations.items():
            # conf_name = 'questionnaire_hla'
            # conf_values = configurations.get(conf_name)
            print(f'Computing configuration: {conf_name}')
            df_avg_metrics_models[conf_name] = {}
            df_model, df_dqb_veto = make_veto_dataset(df_data=df_data,
                                                      use_dqb10602=conf_values.get('dqb'))

            (df_avg_metrics,
             df_classifications,
             df_elastic_params,
             df_elastic_pred) = compute_cross_validation(models=models,
                                                     df_model=df_model,
                                                     features=conf_values.get('features'),
                                                     target=conf_values.get('target'),
                                                      k=k)

            df_classifications['hla'] = conf_values.get('dqb')
            df_classifications['config'] = conf_name
            df_classifications_configs = pd.concat([df_classifications_configs, df_classifications])

            df_avg_metrics['config'] = conf_name
            df_metrics_configs = pd.concat([df_metrics_configs, df_avg_metrics])
            # elastic net feature importance
            df_elastic_params['configuration'] = conf_name
            df_feature_importance = pd.concat([df_feature_importance, df_elastic_params])
            # elastic net predictions
            df_elastic_pred['configuration'] = conf_name
            df_elastic_pred_config = pd.concat([df_elastic_pred_config, df_elastic_pred])

        df_classifications_configs.reset_index(inplace=True, drop=True)
        df_metrics_configs.reset_index(inplace=True, drop=True)

        # organize the metrics
        model_order = df_metrics_configs.loc[df_metrics_configs['config'] == 'questionnaire', :].sort_values(by=['specificity'],
                                                                                     ascending=False).model.values
        df_metrics_configs['model'] = pd.Categorical(df_metrics_configs['model'], categories=model_order, ordered=True)
        df_metrics_configs = df_metrics_configs.sort_values(by=['config', 'specificity'], ascending=[True, False])  # ['config', 'model']
        # Save the results
        df_metrics_configs.to_csv(path_avg_metrics_config, index=False)
        # df_metrics_configs[['config', 'model', 'specificity']]
        df_classifications_configs.to_csv(path_classifications_config, index=False)
        df_feature_importance.to_csv(path_feature_importance, index=False)
        df_elastic_pred_config.to_csv(path_pred_prob_elastic, index=False)
    else:
        df_metrics_configs = pd.read_csv(path_avg_metrics_config)
        df_classifications_configs = pd.read_csv(path_classifications_config)
        df_feature_importance = pd.read_csv(path_feature_importance)
        df_elastic_pred_config = pd.read_csv(path_pred_prob_elastic)

    # sort the frames
    df_metrics_configs = df_metrics_configs.sort_values(by=['model', 'config', 'specificity'],
                                   ascending=[False, True, False]
                                   )
    # %% Apply the veto
    (df_veto_classifications,
     df_veto_avg_metrics) = apply_veto_rule_re_classifications(df_classifications=df_classifications_configs.copy(),
                                                               df_data=df_data)

    df_veto_classifications.to_csv(path_avg_classifications_config_veto, index=False)
    df_veto_avg_metrics = df_veto_avg_metrics.sort_values(by=['model', 'config', 'specificity'],
                                   ascending=[False, True, False]
                                   )
    # df_veto_avg_metrics['config'].replace(feature_set_mapper, inplace=True)
    df_veto_avg_metrics.to_csv(path_avg_metrics_veto, index=False)

    # %% Merge the avg fold metrics before and after the veto to have all in one dataset. Merge by model and config pairs
    col_veto_mapper = {col: f'{col}_veto' for col in df_veto_avg_metrics.columns if not col in ['config', 'model']}
    df_veto_avg_metrics.rename(columns=col_veto_mapper, inplace=True)

    df_avg_metrics = pd.merge(left=df_metrics_configs,
                              right=df_veto_avg_metrics,
                              on=['model', 'config'],
                              how='left')

    new_order = ['config', 'model'] + [col for col in df_avg_metrics.columns if col not in ['model', 'config']]
    df_avg_metrics = df_avg_metrics[new_order]
    df_avg_metrics = df_avg_metrics.sort_values(by=['config', 'specificity'],
                                                        ascending=[True, False])  # ['config', 'model']

    df_avg_metrics['config'].replace(feature_set_mapper, inplace=True)
    df_avg_metrics.to_csv(path_avg_paper, index=False)


    # plot_model_metrics(df_avg_metrics, palette='muted', figsize=(16, 8))

    df_avg_metrics_copy = df_avg_metrics.copy()
    df_avg_metrics_copy = df_avg_metrics_copy.rename(columns={
        'ppv_apparent': 'PPV Apparent',
        'ppv': 'PPV',
        'f1_score': 'F1 Score',
        'specificity': 'Specificity',
        'sensitivity': 'Sensitivity',
    })
    columns_to_plot = ['Specificity', 'Sensitivity', 'F1 Score', 'PPV', 'PPV Apparent'] #  'f1_score', 'npv', 'ppv']  # Example list of columns
    plot_model_metrics_specific_columns(df_avg_metrics_copy,
                       columns=columns_to_plot,
                       palette='pastel',
                       figsize=(18, 4.5),
                        output_path=path_model_metrics)



    # %%
    # Extract classification metrics

    # Apply the function to the dataset
    df_veto_classifications = recompute_classification(df_veto_classifications)

    # Initialize a list to store results
    data_list = []

    # Iterate through unique folds and models
    for model in df_classifications_configs['model_name'].unique():
        # model = 'Elastic Net'
        for fold in np.sort(df_classifications_configs['fold'].unique()):
            # fold = 1
            # Filter for current fold and model
            df_hla_true = df_classifications_configs.loc[
                (df_classifications_configs['hla'] == True) &
                (df_classifications_configs['fold'] == fold) &
                (df_classifications_configs['model_name'] == model)
                ]

            df_hla_false = df_classifications_configs.loc[
                (df_classifications_configs['hla'] == False) &
                (df_classifications_configs['fold'] == fold) &
                (df_classifications_configs['model_name'] == model)
                ]
            #
            # df_veto_rule = df_classifications_configs.loc[
            #     (df_classifications_configs['fold'] == fold) &
            #     (df_classifications_configs['model_name'] == model)
            #     ]

            # Filter now the congifucarion results for the observations the veto rule was applied
            df_hla_true_veto = df_veto_classifications.loc[
                (df_veto_classifications['hla'] == True) &
                (df_veto_classifications['fold'] == fold) &
                (df_veto_classifications['model_name'] == model) &
                (df_veto_classifications['dataset_type'] == 'validation')
                ]

            df_hla_false_veto = df_veto_classifications.loc[
                (df_veto_classifications['hla'] == False) &
                (df_veto_classifications['fold'] == fold) &
                (df_veto_classifications['model_name'] == model) &
                (df_veto_classifications['dataset_type'] == 'validation')
                ]

            # df_veto_rule_veto = df_veto_classifications.loc[
            #     (df_veto_classifications['fold'] == fold) &
            #     (df_veto_classifications['model_name'] == model) &
            #     (df_veto_classifications['dataset_type'] == 'validation')
            #     ]

            # Extract classification metrics
            # Get classification counts
            metrics_hla_true = extract_metrics(df_hla_true)
            metrics_hla_false = extract_metrics(df_hla_false)
            metrics_hla_true_veto = extract_metrics(df_hla_true_veto)
            metrics_hla_false_veto = extract_metrics(df_hla_false_veto)


            # Append to data list
            data_list.append([
                model,
                fold,

                metrics_hla_true.iloc[0]['FN'], metrics_hla_true.iloc[0]['FP'], metrics_hla_true.iloc[0]['TN'],
                metrics_hla_true.iloc[0]['TP'],

                metrics_hla_false.iloc[0]['FN'], metrics_hla_false.iloc[0]['FP'], metrics_hla_false.iloc[0]['TN'],
                metrics_hla_false.iloc[0]['TP'],

                metrics_hla_true_veto.iloc[0]['FN'], metrics_hla_true_veto.iloc[0]['FP'], metrics_hla_true_veto.iloc[0]['TN'],
                metrics_hla_true_veto.iloc[0]['TP'],

                metrics_hla_false_veto.iloc[0]['FN'], metrics_hla_false_veto.iloc[0]['FP'], metrics_hla_false_veto.iloc[0]['TN'],
                metrics_hla_false_veto.iloc[0]['TP'],
            ])

    # Create final DataFrame
    df_final = pd.DataFrame(data_list, columns=[
        'model_name',
        'Fold',

        'FN_HLA_True', 'FP_HLA_True', 'TN_HLA_True', 'TP_HLA_True',

        'FN_HLA_False', 'FP_HLA_False', 'TN_HLA_False', 'TP_HLA_False',

        'FN_HLA_True_After_Veto', 'FP_HLA_True_After_Veto', 'TN_HLA_True_After_Veto', 'TP_HLA_True_After_Veto',

        'FN_HLA_False_After_Veto', 'FP_HLA_False_After_Veto', 'TN_HLA_False_After_Veto', 'TP_HLA_False_After_Veto'
    ])
    # %%
    #       Evaluation Best Model - Elastic net
    #       Using the true, pred, pred_prob obtained for each fold and for each feature set
    #
    # %% Feature importance plot
    # Re-import necessary libraries after execution state reset

    df_feature_importance['configuration'].replace(feature_set_mapper, inplace=True)

    plot_elastic_net_model_coefficients(df_feature_importance,
                                        output_path=path_plot_best_model)


    # %% decison curve
    df_elastic_pred = pd.read_csv(path_pred_prob_elastic)
    # from dcurves import plot_graphs, my_plot_graphs
    df_elastic_pred_dcurve = df_elastic_pred.copy()

    df_elastic_pred_dcurve['configuration_formal'] = df_elastic_pred_dcurve['configuration'].map(feature_set_mapper)
    # Example usage:
    # Assume you have a DataFrame "results" with columns 'true_label' and 'pred_prob'.
    prevalence = 30 / 100000  # i.e., 0.0003
    for feature_set_config in df_elastic_pred_dcurve['configuration_formal'].unique():
        plot_dcurves_per_fold(df_results= df_elastic_pred_dcurve,
                              prevalence=prevalence,
                              configuration=feature_set_config,
                              output_path=path_plot_best_model)

    # %% Prevalance plot
    best_model = 'Elastic Net'
    # all in one figure
    # multi_ppv_plot(df_avg_metrics=df_avg_metrics,
    #                model_name=best_model,
    #                rows=2,
    #                figsize=(12,8),
    #                output_path=None)
    df_elastic_pred_config['configuration'].replace(feature_set_mapper, inplace=True)

    multi_ppv_plot_combined(df_predictions_model=df_elastic_pred_config,
                   figsize=(10,6),
                    population_prevalence = 30 / 100000,  # 0.0003
                   output_path=path_plot_best_model)



    # %% calibration plot

    # df_elastic_pred_config['configuration'].replace(feature_set_mapper, inplace=True)

    # all in one figure
    df_brier_scores = multi_calibration_plot(df_predictions=df_elastic_pred_config,
                           model_name=best_model,
                           rows=2,
                           output_path=path_plot_best_model)


    df_grouped = df_brier_scores.groupby('configuration').agg(
        brier_score=('brier_score', lambda x: f"{x.mean():.4f} ({x.std():.4f})"),
        log_loss=('log_loss', lambda x: f"{x.mean():.4f} ({x.std():.4f})"),
        auc=('auc', lambda x: f"{x.mean():.4f} ({x.std():.4f})")
    ).reset_index()

    df_grouped['configuration'].replace(feature_set_mapper, inplace=True)
    df_grouped.to_csv(path_plot_best_model.joinpath('loss_metrics_elastic_net.csv'), index=False)

    # %% Check feature distribution across the folds
    # Load folds
    imputed_folds = load_imputed_folds(save_path=r'C:\Users\giorg\OneDrive - Fundacion Raices Italo Colombianas\projects\NarcCataplexyQuestionnaire\data')

    # Extract features from one fold
    df_sample = imputed_folds[0]['train_data']
    target_name = 'target'  # Replace with your actual target column name
    # get feature names
    feature_names = [col for col in df_sample.columns if col != target_name]

    # Run summary
    df_fold_summary = summarize_fold_consistency(imputed_folds, feature_names, target_name)

    # Show first few rows
    print(df_fold_summary.head())

    # Optional: save to CSV
    df_fold_summary.to_csv('fold_consistency_summary.csv', index=False)
