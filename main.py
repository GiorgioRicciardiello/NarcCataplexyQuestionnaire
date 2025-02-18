from config.config import config
import numpy as np
import pandas as pd
from library.ml_models import compute_cross_validation, models, make_veto_dataset
from library.metrics_functions import apply_veto_rule_re_classifications
if __name__ == '__main__':
    # %% Read data
    df_data = pd.read_csv(config.get('data_pre_proc_files').get('ssq_ssqdx'))
    df_data = df_data.loc[df_data['narcolepsy'] != 'pseudo narcolepsy']
    df_data.rename(columns={'cataplexy_clear_cut': 'NT1'}, inplace=True)

    # %% output paths
    path_avg_metrics_config = config.get('results_path').get('results').joinpath(f'avg_metrics_config.csv')
    path_classifications_config = config.get('results_path').get('results').joinpath(f'classifications_config.csv')
    path_avg_metrics_veto = config.get('results_path').get('results').joinpath(f'avg_metrics_veto.csv')
    path_avg_classifications_config = config.get('results_path').get('results').joinpath(f'classifications_config_veto.csv')

    path_avg_paper = config.get('results_path').get('results').joinpath(f'avg_paper.csv')
    path_avg_paper_veto = config.get('results_path').get('results').joinpath(f'avg_paper_veto.csv')

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
    df_data = df_data.loc[:, columns]
    df_data = df_data.dropna(axis=1)
    df_data.reset_index(drop=True, inplace=True)
    df_data = df_data.reindex(sorted(df_data.columns), axis=1)
    print(f'Dataset dimension: {df_data.shape}')
    # %% configuration of different models to run
    col_ukbb = ['Age', 'BMI', 'sex', 'ESS', 'JAW', 'HEAD', 'HAND',
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
    if not (path_avg_metrics_config.is_file() and path_classifications_config.is_file()):
        df_avg_metrics_models = {}
        k = 5
        df_classifications_configs = pd.DataFrame()
        df_metrics_configs = pd.DataFrame()
        for conf_name, conf_values in configurations.items():
            print(f'Computing configuration: {conf_name}')
            df_avg_metrics_models[conf_name] = {}
            df_model, df_dqb_veto = make_veto_dataset(df_data=df_data,
                                                      use_dqb10602=conf_values.get('dqb'))

            df_avg_metrics, df_classifications = compute_cross_validation(models=models,
                                                     df_model=df_model,
                                                     features=conf_values.get('features'),
                                                     target=conf_values.get('target'),
                                                      k=k)

            df_classifications['hla'] = conf_values.get('dqb')
            df_classifications['config'] = conf_name
            df_classifications_configs = pd.concat([df_classifications_configs, df_classifications])

            df_avg_metrics['config'] = conf_name
            df_metrics_configs = pd.concat([df_metrics_configs, df_avg_metrics])

        df_classifications_configs.reset_index(inplace=True, drop=True)
        df_metrics_configs.reset_index(inplace=True, drop=True)

        # organize the metrics
        model_order = df_metrics_configs.loc[df_metrics_configs['config'] == 'questionnaire', :].sort_values(by=['specificity'],
                                                                                     ascending=False).model.values
        df_metrics_configs['model'] = pd.Categorical(df_metrics_configs['model'], categories=model_order, ordered=True)
        df_metrics_configs = df_metrics_configs.sort_values(by=['config', 'model'])
        # Save the results
        df_metrics_configs.to_csv(path_avg_metrics_config, index=False)
        df_classifications_configs.to_csv(path_classifications_config, index=False)

    else:
        df_metrics_configs = pd.read_csv(path_avg_metrics_config)
        df_classifications_configs = pd.read_csv(path_classifications_config)

    # %% Apply the veto
    (df_veto_classifications,
     df_veto_avg_metrics) = apply_veto_rule_re_classifications(df_classifications=df_classifications_configs.copy(),
                                                               df_data=df_data)

    df_veto_classifications.to_csv(path_avg_classifications_config, index=False)
    df_veto_avg_metrics.to_csv(path_avg_metrics_veto, index=False)

    # %% Organize the non-veto table for the paper
    model_order = df_metrics_configs.loc[df_metrics_configs['config'] == 'questionnaire', :].sort_values(
        by=['specificity'],
        ascending=False).model.values
    df_metrics_configs['model'] = pd.Categorical(df_metrics_configs['model'], categories=model_order, ordered=True)
    df_metrics_configs = df_metrics_configs.sort_values(by=['config', 'model'])
    # Save the results
    df_metrics_configs[['config', 'model', 'sensitivity_ci', 'specificity_ci']].to_csv(path_avg_paper, index=False)

    # %% Organize the veto table for the paper
    df_veto_res = df_veto_avg_metrics[['config', 'model', 'specificity', 'sensitivity_ci', 'specificity_ci']].copy()

    model_order = df_veto_res.loc[df_veto_res['config'] == 'questionnaire', :].sort_values(by=['specificity'],
                                                                                           ascending=False).model.values
    # Ensure 'model' column follows the predefined model_order sequence
    df_veto_res['model'] = pd.Categorical(df_veto_res['model'], categories=model_order, ordered=True)

    # Sort df_res so that within each 'config', models follow model_order
    df_veto_res = df_veto_res.sort_values(by=['config', 'model'])
    df_veto_res.to_csv(path_avg_paper_veto, index=False)

















