"""
Use the preprocess dataset and generate the table one.
"""
from config.config import config
import numpy as np
import pandas as pd
from library.table_one import MakeTableOne
from typing import Dict, List, Optional
from scipy.stats import fisher_exact, spearmanr, mannwhitneyu, shapiro, ttest_ind
from tabulate import tabulate
def stats_test_binary_symptoms(data: pd.DataFrame,
                               columns: List[str],
                               strata_col: str = 'NT1',
                               SHOW: Optional[bool] = False
                               ) -> pd.DataFrame:
    """
    Using the binary values (with 1 as a positive response), perform Fisher's exact test
    comparing the distribution between genders.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the symptom responses and the gender column.
    columns : List[str]
        Columns to perform the statistical test.
    strata_col : str, default 'NT1'
        Column name for the gender grouping.
    SHOW : bool, default False
        print the 2x2 table for each test

    Returns
    -------
    pd.DataFrame
        DataFrame with counts, percentages, p-value and odds ratio from Fisher's exact test,
        and a column indicating the test method used.
    """
    results = []

    # Define a helper lambda to compute count and percentage for a boolean Series.
    get_counts_rates = lambda cond: (cond.sum(), cond.mean() * 100)

    if not len(data[strata_col].unique()) == 2:
        raise ValueError(f'Strata column {strata_col} has more than 2 groups')
    group0 = data[target].unique()[0]
    group1 = data[target].unique()[1]
    # Loop over each symptom.
    for col in columns:
        if col == strata_col:
            continue
        print(col)
        # Create a local copy that includes the symptom and the gender column.
        df = data[[col, strata_col]].dropna().copy()

        # Check if the symptom data are binary.
        unique_vals = set(df[col].unique())
        if unique_vals != {0, 1}:
            continue
        # For each gender group, count the number of positive responses (i.e. value == 1).
        grp0 = (df[df[strata_col] == group0][col] == 1)
        grp1 = (df[df[strata_col] == group1][col] == 1)

        group0_n, group0_rate = get_counts_rates(grp0)
        group1_n, group1_rate = get_counts_rates(grp1)
        total_n, total_rate = get_counts_rates(df[col] == 1)

        # Build the 2x2 contingency table.
        # Rows: gender groups; Columns: positive and negative responses.
        table = [
            [group0_n, df[df[strata_col] == group0].shape[0] - group0_n],
            [group1_n, df[df[strata_col] == group1].shape[0] - group1_n]
        ]
        if SHOW:
            # Define headers for the columns.
            headers = [f"{col} {group0}", f"{col} {group1}"]
            row_labels = [f"{strata_col} {group0}", f"{strata_col} {group1}"]
            # Combine row labels with table rows if desired.
            table_with_labels = [[row_labels[i]] + row for i, row in enumerate(table)]
            headers_with_labels = ["Group"] + headers
            print(tabulate(table_with_labels, headers=headers_with_labels,
                           tablefmt="grid"))

        # Perform Fisher's exact test.
        odds_ratio, p_fisher = fisher_exact(table, alternative='two-sided')
        res = {
            'Variable': col,
            # f'{strata_col} {group0} (n)': round(group0_n, 1),
            # f'{strata_col} {group0} (%)': round(group0_rate, 1),
            f'{strata_col} {group0} N, (%)': f'{round(group0_n, 1)} ({round(group0_rate, 1)})',
            # f'{strata_col} {group1} (n)': round(group1_n, 1),
            # f'{strata_col} {group1}  (%)': round(group1_rate, 1),
            f'{strata_col} {group1} N, (%)': f'{round(group1_n, 1)} ({round(group1_rate, 1)})',
            # 'Total (n)': round(total_n, 1),
            # 'Total (%)': round(total_rate, 1),
            'Total N, (%)': f'{round(total_n, 1)} ({round(total_rate, 1)})',

            'p-value': p_fisher,
            'p-value formatted': f"{p_fisher:.4f}" if p_fisher >= 0.0001 else "<0.0001",
            'Effect Size (Odds Ratio)': round(odds_ratio, 3),
            'Stat Method': "Fisher's Exact Test"
        }
        results.append(res)

    return pd.DataFrame(results)


def stats_test_continuous(data: pd.DataFrame,
                          columns: List[str],
                          strata_col: str = 'NT1',
                          SHOW: Optional[bool] = False
                          ) -> pd.DataFrame:
    """
    Perform statistical tests between continuous distributions across two groups.
    - First tests for normality using Shapiro-Wilk test.
    - If both distributions are normal, use an independent t-test.
    - Otherwise, use the Mann-Whitney U test.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the continuous symptom values and the grouping column.
    columns : List[str]
        Columns to perform the statistical test.
    strata_col : str, default 'NT1'
        Column name for the grouping.
    SHOW : bool, default False
        Print distribution summary statistics for each test.

    Returns
    -------
    pd.DataFrame
        DataFrame with mean, standard deviation, p-value, and test used.
    """
    results = []

    unique_groups = data[strata_col].dropna().unique()
    if len(unique_groups) != 2:
        raise ValueError(f'Strata column {strata_col} must have exactly 2 groups.')

    group0, group1 = unique_groups

    for col in columns:
        print(col)
        if col == strata_col:
            continue

        df = data[[col, strata_col]].dropna()
        group0_vals = df[df[strata_col] == group0][col]
        group1_vals = df[df[strata_col] == group1][col]

        if len(group0_vals) < 10 or len(group1_vals) < 10:
            continue
        # Normality test
        normal0 = shapiro(group0_vals).pvalue > 0.05
        normal1 = shapiro(group1_vals).pvalue > 0.05

        if normal0 and normal1:
            stat_test = 'Independent t-test'
            stat, p_value = ttest_ind(group0_vals, group1_vals, equal_var=False)
        else:
            stat_test = 'Mann-Whitney U test'
            stat, p_value = mannwhitneyu(group0_vals, group1_vals, alternative='two-sided')

        # Descriptive stats
        mean0, std0 = group0_vals.mean(), group0_vals.std()
        mean1, std1 = group1_vals.mean(), group1_vals.std()

        res = {
            'Variable': col,
            f'{strata_col} {group0} Mean (SD)': f'{mean0:.2f} ({std0:.2f})',
            f'{strata_col} {group1} Mean (SD)': f'{mean1:.2f} ({std1:.2f})',
            f'n {strata_col} {group0}': len(group0_vals),
            f'n {strata_col} {group1}': len(group1_vals),
            'p-value': p_value,
            'p-value formatted': f"{p_value:.4f}" if p_value >= 0.0001 else "<0.0001",
            'Stat Method': stat_test
        }

        if SHOW:
            print(f"{col} - {stat_test}: p = {p_value:.4f}")

        results.append(res)

    return pd.DataFrame(results)



if __name__ == '__main__':
    df_data = pd.read_csv(config.get('data_pre_proc_files').get('ssq_ssqdx'))
    df_data.rename(columns={'cataplexy_clear_cut': 'NT1'}, inplace=True)


    target = 'NT1'
    # Use only cases and controls
    df_data = df_data[df_data[target].isin({0, 1})]

    df_nans = df_data.isna().sum() * 100 / df_data.shape[0]
    df_nans = df_nans.reset_index(name='nan_percent')
    df_nans.sort_values(by='nan_percent', ascending=False, inplace=True)
    df_nans['nan_percent'] = df_nans['nan_percent'] .round(2)
    # define the columns dtypes
    categorical_var = ['sex',
                       'DQB10602',
                       'DURATION',
                       'HALLUC',
                       'SP',
                       'DISNOCSLEEP',
                       'Race',
                       'MEDCATA']
    # categorical_var = list( np.sort(categorical_var))
    continuous_var = ['Age',
                      'BMI',
                      'ESS',
                      'NAPS',
                      'SLEEPIONSET',
                      'ONSET',
                      'SPONSET',
                      'HHONSET',
                      'MSLT',
                      'SOREMP',
                      'SE',  # sleep latency
                      'REMLAT'
                      ]
    columns = list(set(categorical_var + continuous_var + [target]))

    #%%
    make_tab_one = MakeTableOne(df_data,
                                categorical_var=categorical_var,
                                continuous_var=continuous_var,
                                strata=target)
    df_tab_one = make_tab_one.create_table()
    df_tab_one = make_tab_one.group_variables_table(df=df_tab_one)
    df_tab_one.to_csv(config.get('results_path').get('results').joinpath('table_one_reduced.csv'), index=False,)

    for col in columns:
        print(df_data.groupby(by='NT1')[col].count())
    var = 'HHONSET'
    df_data.loc[(df_data[target] == 1) & (~df_data[var].isna()), var].shape[0]

    # %% compute statistics of significance different groups
    categorical_var = ['LAUGHING', 'ANGER', 'EXCITED',
                       'SURPRISED', 'HAPPY', 'EMOTIONAL', 'QUICKVERBAL', 'EMBARRAS',
                       'DISCIPLINE', 'SEX', 'DURATHLETIC', 'AFTATHLETIC', 'ELATED',
                       'STRESSED', 'STARTLED', 'TENSE', 'PLAYGAME', 'ROMANTIC',
                       'JOKING', 'MOVEDEMOT', 'KNEES', 'JAW', 'HEAD', 'HAND', 'SPEECH',
                       'DQB10602', 'sex', 'DISNOCSLEEP']
    categorical_var = list(np.sort(categorical_var))
    df_stats_bin = stats_test_binary_symptoms(data=df_data,
                               columns=categorical_var,
                               strata_col=target)

    df_stats_cont = stats_test_continuous(
                data=df_data,
                columns=list(continuous_var),
                strata_col='NT1',
                SHOW=True)