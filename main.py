from config.config import config
import numpy as np
import pandas as pd
from library.ml_models import compute_cross_validation, models, make_veto_dataset
from library.metrics_functions import apply_veto_rule_re_classifications
from library.plot_functions import (plot_model_metrics, plot_model_metrics_specific_columns,
                                    create_venn_diagram, plot_elastic_net_model_coefficients, plot_dcurves_per_fold)
from library.metrics_functions import recompute_classification, extract_metrics
from typing import Optional, Tuple, List


if __name__ == '__main__':
    # %% Read data
    df_data = pd.read_csv(config.get('data_pre_proc_files').get('ssq_ssqdx'))
    df_data = df_data.loc[df_data['narcolepsy'] != 'pseudo narcolepsy']
    df_data.rename(columns={'cataplexy_clear_cut': 'NT1'}, inplace=True)

    # %% output paths
    TEST = True
    test = 'test_' if TEST else ''
    path_avg_metrics_config = config.get('results_path').get('results').joinpath(f'{test}avg_metrics_config.csv')
    path_classifications_config = config.get('results_path').get('results').joinpath(f'{test}classifications_config.csv')
    path_avg_metrics_veto = config.get('results_path').get('results').joinpath(f'{test}avg_metrics_veto.csv')
    path_avg_classifications_config_veto = config.get('results_path').get('results').joinpath(f'{test}classifications_config_veto.csv')
    path_avg_paper = config.get('results_path').get('results').joinpath(f'{test}avg_paper.csv')
    path_feature_importance = config.get('results_path').get('results').joinpath(f'{test}feature_importance.csv')
    path_pred_prob_elastic = config.get('results_path').get('results').joinpath(f'{test}pred_prob_elasticnet.csv')
    # path_avg_paper_veto = config.get('results_path').get('results').joinpath(f'avg_paper_veto.csv')

    # %% Select columns and drop columns with nans
    target = 'NT1'
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
    col_ukbb = ['BMI', 'ESS', 'JAW', 'HEAD', 'HAND',
                'SPEECH', 'JOKING', 'LAUGHING', 'ANGER', 'KNEES',
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

    # Create a Venn diagram comparing "questionnaire" and "ukbb"
    create_venn_diagram(
        configs=configurations,
        config1="questionnaire",
        config2="ukbb",
        figure_size=(6, 6),
        set_labels=("Full Questionnaire Feature Set", "Reduces Feature Set"),
        colors=('pink', 'orange'),
        alpha=0.7,
        title="Venn Diagram of Questionnaire vs. UKBB Features"
    )
    for key, dictionary in configurations.items():
        num_feature = len(dictionary.get('features'))
        print(f'{key}: {num_feature}')

    # run the analysis
    if not (path_avg_metrics_config.is_file() and path_classifications_config.is_file()):
        df_feature_importance = pd.DataFrame()
        df_avg_metrics_models = {}
        k = 5
        df_classifications_configs = pd.DataFrame()
        df_metrics_configs = pd.DataFrame()
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
        df_elastic_pred.to_csv(path_pred_prob_elastic, index=False)

    else:
        df_metrics_configs = pd.read_csv(path_avg_metrics_config)
        df_classifications_configs = pd.read_csv(path_classifications_config)
        df_feature_importance = pd.read_csv(path_feature_importance)

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

    df_avg_metrics.to_csv(path_avg_paper, index=False)


    plot_model_metrics(df_avg_metrics, palette='muted', figsize=(16, 8))



    columns_to_plot = ['f1_score', 'npv', 'sensitivity', 'specificity']  # Example list of columns
    plot_model_metrics_specific_columns(df_avg_metrics,
                       columns=columns_to_plot,
                       palette='muted',
                       figsize=(16, 4))

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
    # %% Feature importance plot
    # Re-import necessary libraries after execution state reset

    df_feature_importance['configuration'].replace({
        'questionnaire': 'Full feature set (k=27)',
        'questionnaire_hla': 'Full Feature Set + DQB1*06:02  (k=28)',
        'ukbb': 'Reduced Feature Set (k=9)',
        'ukbb_hla': 'Reduced Feature Set + DQB1*06:02 (k=10)',
    }, inplace=True)



    plot_elastic_net_model_coefficients(df_feature_importance,
                                        output_path=None,
                                        figsize=(10, 12),
                                        colormap='tab10')


    # %% decison curve
    df_elastic_pred = pd.read_csv(path_pred_prob_elastic)
    # from dcurves import plot_graphs, my_plot_graphs

    # Example usage:
    # Assume you have a DataFrame "results" with columns 'true_label' and 'pred_prob'.
    prevalence = 30 / 100000  # i.e., 0.0003
    plot_dcurves_per_fold(df_results=df_elastic_pred, prevalence=prevalence)


