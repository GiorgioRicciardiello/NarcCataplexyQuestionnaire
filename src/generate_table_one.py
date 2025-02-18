"""
Use the preprocess dataset and generate the table one.
"""
from config.config import config
import numpy as np
import pandas as pd
from library.table_one import MakeTableOne

if __name__ == '__main__':
    df_data = pd.read_csv(config.get('data_pre_proc_files').get('ssq_ssqdx'))
    df_data.rename(columns={'cataplexy_clear_cut': 'NT1'}, inplace=True)

    target = 'NT1'
    categorical_var = ['sex', 'Ethnicity', 'LAUGHING', 'ANGER', 'EXCITED', 'SURPRISED', 'HAPPY', 'EMOTIONAL',
                       'QUICKVERBAL', 'EMBARRAS', 'DISCIPLINE', 'SEX', 'DURATHLETIC', 'AFTATHLETIC', 'ELATED',
                       'STRESSED', 'STARTLED', 'TENSE', 'PLAYGAME', 'ROMANTIC', 'JOKING', 'MOVEDEMOT', 'KNEES',
                       'JAW', 'HEAD', 'HAND', 'SPEECH', 'DQB10602']
    # categorical_var = list( np.sort(categorical_var))
    continuous_var = ['Age', 'BMI', 'ESS', 'DISNOCSLEEP', 'NAPS', 'SLEEPIONSET', 'ONSET']
    columns = list(set(categorical_var + continuous_var + [target]))

    #%%
    make_tab_one = MakeTableOne(df_data,
                                categorical_var=categorical_var,
                                continuous_var=continuous_var,
                                strata=target)
    df_tab_one = make_tab_one.create_table()
    df_tab_one = make_tab_one.group_variables_table(df=df_tab_one)
    df_tab_one.to_csv(config.get('results_path').get('results').joinpath('table_one.csv'), index=False,)
