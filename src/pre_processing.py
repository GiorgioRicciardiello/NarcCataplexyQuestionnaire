"""
Two data sources are used the pre-processing. The sources must be merged into a single datataset.

1. Pre-process the SSQDX dataset
2. Pre-process the SSQ dataset
3. Merge the two datasets

Columns must be homogenized
"""
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy as np
import pandas as pd
from config.config import config
import ast
from config.SSI_Digitial_Questionnaire import key_mapping
import re
from typing import List, Dict, Union, Any, Tuple, Optional
from sklearn.experimental import enable_iterative_imputer  # needed to use IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

#%% Functions for the ssqdx dataset
def pre_process_ssqdx_dataset(dataset_path:pathlib.Path) -> pd.DataFrame:
    """
    Pre-process of the SSQDX dataset.
    :param dataset_path:
    :return:
    """

    def contains_a_to_f(row) -> bool:
        if 'control' in row['Dx']:
            return False
        if pd.isnull(row['Dx']):
            return False
        if 'NT1' in row['Dx']:
            return True
        if pd.Series(row['Dx']).str.contains(r'[A-F]', case=False, na=False).any():
            return True
        return False

    def set_narcolepsy(row) -> str:
        """
        Set the Narcolepsy diagnosis based on the columns DX and DQB10602.

        Diagnosis criteria:
        · 98% sure of Narcolepsy if:
            - DX (1) contains letters A, B, C, or F
            - DQB10602 = 1
        · Factual Narcolepsy if:
            - DX (1) contains letters A, B, C, or F
            - DQB10602 = 0
        · No Narcolepsy otherwise

        Parameters:
        -----------
        row : pd.Series
            A row from the DataFrame containing 'DQB10602' and 'contains_A_to_F' columns.

        Returns:
        --------
        str
            The narcolepsy diagnosis as a string.
        """
        # Check for missing values in relevant columns
        if pd.isna(row['DQB10602']) or pd.isna(row['contains_A_to_F']):
            return 'undefined'

        # Diagnosis logic based on DQB10602 and contains_A_to_F criteria
        if row['DQB10602'] == 1 and row['contains_A_to_F']:
            return 'narcolepsy'
        elif row['DQB10602'] == 0 and row['contains_A_to_F']:
            return 'factual narcolepsy'
        else:
            return 'non-narcolepsy'

    df_ssqdx = pd.read_excel(dataset_path)
    # %% Work on the dx dataset
    df_ssqdx.replace(to_replace='.', value=np.nan, inplace=True)
    df_ssqdx.rename(columns={'# NAPS': 'NAPS',
                             'DQB1*0602': 'DQB10602'}, inplace=True)
    df_ssqdx['source'] = 'dx'
    df_ssqdx['sex'] = df_ssqdx['sex'].map({'F': 0, 'M': 1})
    df_ssqdx['BMI'] = df_ssqdx['BMI'].round(1)

    df_ssqdx['Dx'] = df_ssqdx['Dx'].astype(str)
    df_ssqdx['contains_A_to_F'] = df_ssqdx.apply(contains_a_to_f, axis=1)
    df_ssqdx['narcolepsy'] = df_ssqdx.apply(set_narcolepsy, axis=1)

    # %% rename columns
    mapper_q = [
        'Q:86',
        'Q:87',
        'Q:88',
        'Q:89',
        'Q:90',
        'Q:91',
        'Q:92',
        'Q:93',
        'Q:94',
        'Q:101',
        'Q:113',
        'Q:137',
        'Q:140'
    ]
    df_ssqdx.drop(columns=mapper_q, inplace=True)
    return df_ssqdx

def pre_process_ssq_dataset(dataset_path: pathlib.Path) -> pd.DataFrame:
    """

    :param dataset_path:
    :return:
    """

    def expand_series(series: pd.Series, col_name: str) -> pd.DataFrame:
        """
        Expand the multiple response answers where a single cell in the format 111101 is parsed as a dataframe e.g.,
        ['1', '1', '1', '1', '0', '1']. For all the rows.

        Row order is preserve. A sparse dataframe is create
        :param series:
        :param col_name:
        :return:  sparse dataframe
        """
        df_exp_sparse = pd.DataFrame(series.apply(lambda x: list(x)).tolist())
        # mirror the index order
        df_exp_sparse.index = series.index
        # set the columns
        df_exp_sparse.columns = [f'{col_name}_exp_{i}' for i in range(1, df_exp_sparse.shape[1] + 1)]
        return df_exp_sparse

    def is_integer(string):
        try:
            int(string)
            return True
        except ValueError:
            return False

    def remove_non_numeric(value):
        if isinstance(value, str):
            return re.sub(r'\D', '', value)
        return value

    def string_to_numeric(value):
        if pd.isna(value):
            return value
        if isinstance(value, str):
            # Remove non-numeric characters
            numeric_string = re.sub(r'\D', '', value)
            try:
                # Convert to numeric value using literal_eval
                return ast.literal_eval(numeric_string)
            except (ValueError, SyntaxError):
                # If conversion fails, return the numeric string
                return numeric_string
        return value

    def contains_string_or_datetime(series):
        """Function to check if a column contains any string or datetime values"""
        return series.apply(lambda x: isinstance(x, (str, pd.Timestamp))).any()

    df_ssq = pd.read_excel(dataset_path)

    #  rename columns and drop unwanted
    df_ssq.drop(columns=['Name'], inplace=True)
    df_ssq.rename(columns={"Pt's Last Name": 'name_last',
                           'Full name (Last, First)': 'name',
                           "Pt's First Name": 'name_first',
                           "DOB": "date_of_birth",
                           "AGE": "age",
                           "Gender": "gender",
                           "PLACE OF BIRTH": "place_birth",
                           "ETHNIC": "ethnicity",
                           "Completed (date)": "completed",
                           "A. Clear-Cut Cataplexy": "cataplexy_clear_cut",
                           "B. Possibly": "possibility",
                           "C. Narcolepsy": "narcolepsy",
                           "D. Other Sleep Disorder": "d_other_sleep_disorder",
                           "D1. Name of Other Disorder": "d_one",
                           "D2. Name of Other Disorder": "d_two",
                           "D3. Name of Other Disorder": "d_three",
                           },
                  inplace=True)
    #  PHI formatting
    df_ssq['name'] = df_ssq['name_first'] + ' ' + df_ssq['name_last']
    df_ssq.drop(columns=['name_first', 'name_last'], inplace=True)
    df_ssq['gender'].replace({'M': 1, 'F': 0}, inplace=True)
    df_ssq['gender'] = df_ssq['gender'].astype(int)
    df_ssq['place_birth'] = df_ssq['place_birth'].str.lstrip()  # .replace(' ', '')
    df_ssq['place_birth'] = df_ssq['place_birth'].str.strip()  # .replace(' ', '')
    # %% errors
    df_ssq.replace('????', np.nan, inplace=True)
    df_ssq[97].replace({'19-30': 19.5}, inplace=True)
    df_ssq[95].replace({'18-25': 18.15}, inplace=True)
    df_ssq[46].replace({'late30s-ear40s': 35,
                        '16-26': 23}, inplace=True)
    df_ssq.loc[df_ssq['65a'] == '2,1', '65a'] = 1
    df_ssq['65a'] = df_ssq['65a'].astype(int)
    # df_ssq['83a'].astype(int)
    index_to_drop = df_ssq.loc[(df_ssq['54b'] == '0 ?') | (df_ssq['54b'] == 9)].index
    df_ssq.drop(index=index_to_drop, inplace=True)
    # there is an extra zero in:
    df_ssq.loc[956, '60b'] = '000009'  # before was 0000090'. Results extra column full of None in the next code block
    df_ssq.loc[740, '64b'] = '900000'  # before " '900000"
    df_ssq.loc[933, '72b'] = '011000'  # before '0110000'

    # sparse the dataset - Emotions and muscle weakness
    # Note: The new columns indexes make reference to the dictionary mw_experiences in the config folder
    # Select columns that match the pattern using regular expression
    pattern = r'^\d+[ab]$'
    col_emot_mw = df_ssq.filter(regex=pattern).columns
    col_emot_mw_multi_response = [col for col in col_emot_mw if
                                  df_ssq[col].apply(lambda x: isinstance(x, str)).any()]
    col_emot_mw_multi_response = [col for col in col_emot_mw_multi_response if col.endswith('b')]
    col_emot_mw_multi_response = [col for col in col_emot_mw_multi_response if
                                  ast.literal_eval(col.split('b')[0]) < 80]
    # remove the ' symbol that not all cells have
    df_ssq[col_emot_mw_multi_response] = df_ssq[col_emot_mw_multi_response].replace("'", '', regex=True)
    # all same format
    df_ssq[col_emot_mw_multi_response] = df_ssq[col_emot_mw_multi_response].astype(str)
    # expand the cells and insert the slice of the new frame
    for col in col_emot_mw_multi_response:
        df_tmp = expand_series(series=df_ssq[col], col_name=col)
        column_index = df_ssq.columns.get_loc(col)
        # squezze in the middle the new columns
        df_ssq = pd.concat([df_ssq.iloc[:, :column_index], df_tmp, df_ssq.iloc[:, column_index + 1:]], axis=1)
    # set as integer the leading columns that indicate that yes/ no response
    pattern = r'^\d+[a]$'
    col_emot_mw_yn = df_ssq.filter(regex=pattern).columns
    col_emot_mw_yn = [col for col in col_emot_mw_yn if ast.literal_eval(col.split('a')[0]) < 80]
    df_ssq[col_emot_mw_yn] = df_ssq[col_emot_mw_yn].astype(int)

    # # re-order the columns
    # columns = ['name'] + [col for col in df_ssq.columns if col != 'name']
    # df_ssq = df_ssq[columns]
    # df_ssq.reset_index(drop=True, inplace=True)

    # data type formatting
    col_unwanted = ['102a', '102b', '103a', '103b',
                    '103c', '103d', '103e', '103f',
                    '103g', '103h', '103i', '103j']
    df_ssq.drop(columns=col_unwanted, inplace=True)
    for col in df_ssq.columns[14::]:
        # print(col)
        df_ssq[col] = df_ssq[col].apply(remove_non_numeric)
    df_ssq.replace('', np.nan, inplace=True)

    for col in df_ssq.columns[14::]:
        # print(col)
        df_ssq[col] = df_ssq[col].apply(string_to_numeric)
    df_ssq = df_ssq.round(2)

    # convert as integers the columns
    # Get list of columns that do not contain any NaN values and do not have any string or datetime values
    columns_without_nan = [col for col in df_ssq.columns if
                           not df_ssq[col].isna().any() and not contains_string_or_datetime(df_ssq[col])]

    df_ssq[columns_without_nan] = df_ssq[columns_without_nan].astype(int)

    # rename all columns as strings
    pattern = r'^\d.*[a-zA-Z]$'  # string starts with a number and ends with any letter
    compiled_pattern = re.compile(pattern)
    columns = [col if isinstance(col, str) else str(col) for col in df_ssq.columns]
    df_ssq.columns = columns

    for col in df_ssq.columns:
        if is_integer(col):
            col_int = int(col)
            if col_int in key_mapping.keys():
                df_ssq.rename(columns={col: f'{col}_{key_mapping[col_int]}'}, inplace=True)
        if isinstance(col, str) and col in df_ssq.columns:
            # check if the string starts with a number and ends with any letter
            if compiled_pattern.match(col):
                num_col = col[0:2]
                ramification = col[2]
                columns_starting_with_num_col = [col for col in df_ssq.columns if str(col).startswith(num_col)]
                new_col_pattern = {col: f'{col}_{key_mapping[int(num_col)].replace("_", "-")}' for col in
                                   columns_starting_with_num_col}
                df_ssq.rename(mapper=new_col_pattern, inplace=True, axis=1)

    df_ssq['narcolepsy'] = df_ssq['narcolepsy'].astype(int)
    # Convert the 'epworth' columns to integers, ignoring NaNs
    ess_columns = df_ssq.columns.str.contains('epworth')
    df_ssq.loc[:, ess_columns] = df_ssq.loc[:, ess_columns].astype(float)
    df_ssq.loc[:, ess_columns] = df_ssq.loc[:, ess_columns].replace([9, 8], np.nan)

    for val in df_ssq.loc[:, df_ssq.columns.str.contains('epworth')].columns:
        print(f'{val}: {df_ssq[val].value_counts().to_dict()}')
        print(f'\t\t max:{max(df_ssq[val].value_counts().to_dict().keys())}')

    # ignore numbers that are used to mark missinges
    df_ssq['epworth_score'] = df_ssq.loc[:, df_ssq.columns.str.contains('epworth')].sum(skipna=True, axis=1)

    print(f'Distribution ESS Score: \n {df_ssq["epworth_score"].describe()}')

    df_ssq.columns = map(str.lower, df_ssq.columns)
    df_ssq['bmi'] = pd.to_numeric(df_ssq['bmi'], errors='coerce').round(1)
    df_ssq['age'] = pd.to_numeric(df_ssq['age'], errors='coerce').round(0)

    # drop unnamed columns
    df_ssq = df_ssq.drop(columns=df_ssq.filter(regex='^unnamed:').columns)

    col_first = [
        'ptkey',
        'gwas id',
        'name',
        'age',
        'gender',
        'date_of_birth',
        'bmi',
        'place_birth',
        'csf hcrt concentration crude (1)',
        'ethnicity',
        'completed',
        'dqb1*0602',
        'dqb1*0602 typing date',
        'cataplexy_clear_cut',
        'possibility',
        'narcolepsy',
    ]
    col_rest = [col for col in df_ssq.columns if col not in col_first]
    cols = col_first + col_rest
    assert len(cols) == df_ssq.shape[1]
    df_ssq = df_ssq[cols]
    df_ssq.reset_index(drop=True, inplace=True)

    df_ssq.rename(columns={
        'dqb1*0602': 'DQB10602',
        'dqb1*0602 typing date': 'DQB10602 date',
    }, inplace=True)

    # check the unique values per columns
    for col in df_ssq.columns:
        if df_ssq[col].nunique() > 10 or col == 'ethnicity':
            continue
        value_counts = df_ssq[col].value_counts().to_dict()
        value_counts = dict(sorted(value_counts.items()))  # Sort the dictionary by key
        print(f'{col}: \n\t{value_counts}')

    return df_ssq


# %% Functions for merging

def plot_histograms(df1, col1, df2, col2, title):
    """
    Plots histograms of two different columns from two different DataFrames side by side using Seaborn.

    Parameters:
    df1 (pd.DataFrame): The first DataFrame.
    col1 (str): The column name from the first DataFrame.
    df2 (pd.DataFrame): The second DataFrame.
    col2 (str): The column name from the second DataFrame.
    title (str): The title of the plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle(title, fontsize=16)

    # Plot histogram for the first DataFrame
    sns.histplot(df1[col1].dropna(), kde=True, ax=axes[0])
    axes[0].grid(0.7)

    # Plot histogram for the second DataFrame
    sns.histplot(df2[col2].dropna(), kde=True, ax=axes[1])
    axes[1].grid(0.7)
    plt.tight_layout()
    plt.show()


def make_new_col(row, columns_to_check: list[str]) -> int:
    return 1 if any(row[col] == 1 for col in columns_to_check) else 0

def quest76map(row):
    if pd.isna(row):
        return 0
    if 3 >= row > 0:
        return 1
    else:
        return row

def create_sleep_complaint(row) -> float:
    subset = row[['44_age_aware_sleepiness',
                  '45_sleepiness_severity_since_age',
                  '46_most_severe_sleepiness_age']]
    filtered_subset = subset[(subset >= 9) & (subset <= 99)]
    if filtered_subset.shape[0] == 1:
        return filtered_subset.iloc[0]
    elif filtered_subset.shape[0] > 1:
        return filtered_subset.mean().round(0)
    else:
        return np.nan

def set_to_zero_except_one(x):
    return 1 if x == 1 else 0

def wrangle_target_combinations(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, pd.DataFrame], Union[pd.DataFrame, Any]]:
    """
    Cleans and processes a DataFrame with narcolepsy, cataplexy, and DQB1*06:02 data to create target combinations
    and ensure distribution alignment.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing columns 'narcolepsy', 'cataplexy_clear_cut', 'DQB10602'.

    Returns:
    - tuple: Verification results, processed DataFrame.
    """

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

    def verify_dqb_distribution(df: pd.DataFrame) -> Tuple[bool, pd.DataFrame]:
        """
        Verifies if the distribution of DQB1*06:02 in narcolepsy, NT1, and non-NT1 cases aligns with expected standards:
        - 98% of NT1 cases (cataplexy_clear_cut = 1) should be DQB1*06:02 positive.
        - 23% of non-NT1 cases (cataplexy_clear_cut = 0) should be DQB1*06:02 positive and 77% negative.
        - Narcolepsy group as a whole should show DQB1*06:02 positivity aligned with NT1 and non-NT1 distributions.

        Parameters:
        - df (pd.DataFrame): DataFrame with columns 'narcolepsy', 'cataplexy_clear_cut', and 'DQB10602'.

        Returns:
        - Tuple[bool, pd.DataFrame]: A tuple containing a boolean indicating if the verification passed,
                                     and a DataFrame with the results and discrepancies.
        """
        # Results dictionary with structured keys
        results = {
            'NT1 & Narc Cases (DQB1*06:02+)': {
                'Expected_Percentage': 98,
                'Actual_Percentage': None,
                'Meets_Requirement': None
            },
            'Non-NT1 & Narc Cases (DQB1*06:02+)': {
                'Expected_Percentage': 23,
                'Actual_Percentage': None,
                'Meets_Requirement': None
            },
            'Non-NT1 & Narc Cases (DQB1*06:02-)': {
                'Expected_Percentage': 77,
                'Actual_Percentage': None,
                'Meets_Requirement': None
            },
            'Overall Narcolepsy Group (DQB1*06:02+)': {
                'Expected_Percentage': None,
                'Actual_Percentage': None,
                'Meets_Requirement': None
            }
        }

        # ---- NT1 Case Verification ----
        nt1_narc_pos_cases = df.loc[(df['cataplexy_clear_cut'] == 1) & (df['narcolepsy'] == 'narcolepsy')]
        dqb_positive_nt1_narc = nt1_narc_pos_cases['DQB10602'].value_counts(normalize=True).get(1, 0) * 100
        results['NT1 & Narc Cases (DQB1*06:02+)']['Actual_Percentage'] = dqb_positive_nt1_narc
        results['NT1 & Narc Cases (DQB1*06:02+)']['Meets_Requirement'] = dqb_positive_nt1_narc >= 98

        # ---- Non-NT1 Case Verification ----
        non_nt1_cases_narc_pos = df.loc[(df['cataplexy_clear_cut'] == 0) & (df['narcolepsy'] == 'narcolepsy')]
        dqb_positive_non_nt1_narc_pos = non_nt1_cases_narc_pos['DQB10602'].value_counts(normalize=True).get(1, 0) * 100
        dqb_negative_non_nt1_narc_pos = non_nt1_cases_narc_pos['DQB10602'].value_counts(normalize=True).get(0, 0) * 100
        results['Non-NT1 & Narc Cases (DQB1*06:02+)']['Actual_Percentage'] = dqb_positive_non_nt1_narc_pos
        results['Non-NT1 & Narc Cases (DQB1*06:02+)']['Meets_Requirement'] = abs(
            dqb_positive_non_nt1_narc_pos - 23) <= 1
        results['Non-NT1 & Narc Cases (DQB1*06:02-)']['Actual_Percentage'] = dqb_negative_non_nt1_narc_pos
        results['Non-NT1 & Narc Cases (DQB1*06:02-)']['Meets_Requirement'] = abs(
            dqb_negative_non_nt1_narc_pos - 77) <= 1

        # ---- Overall Narcolepsy Group Verification ----
        narcolepsy_cases = df.loc[(df['narcolepsy'] == 'narcolepsy') | (df['narcolepsy'] == 'factual narcolepsy')]
        dqb_positive_narcolepsy = narcolepsy_cases['DQB10602'].value_counts(normalize=True).get(1, 0) * 100
        weighted_expected = (98 * len(nt1_narc_pos_cases) + 23 * len(non_nt1_cases_narc_pos)) / max(
            len(narcolepsy_cases),
            1)
        results['Overall Narcolepsy Group (DQB1*06:02+)']['Expected_Percentage'] = weighted_expected
        results['Overall Narcolepsy Group (DQB1*06:02+)']['Actual_Percentage'] = dqb_positive_narcolepsy
        results['Overall Narcolepsy Group (DQB1*06:02+)']['Meets_Requirement'] = abs(
            dqb_positive_narcolepsy - weighted_expected) <= 1

        # Convert results to DataFrame
        results_df = pd.DataFrame.from_dict(results, orient='index').reset_index().rename(columns={'index': 'Category'})

        # Determine if all requirements are met
        verification_passed = all(results_df['Meets_Requirement'])

        # Display the results
        print("Verification of DQB1*06:02 Distribution Requirements:")
        print(tabulate(results_df, headers='keys', tablefmt='grid'))

        return verification_passed, results_df

    tabs = {}
    tab = visualize_table(df=df,
                          group_by=['source', 'narcolepsy', 'cataplexy_clear_cut', 'DQB10602'])
    tabs['tab_zero'] = tab
    # ---- Step 1: Remove Rows with All Missing Values in Key Columns ----
    # Drop rows where 'narcolepsy', 'cataplexy_clear_cut', and 'DQB10602' are all NaN.
    df = df.dropna(subset=['narcolepsy', 'cataplexy_clear_cut', 'DQB10602'], how='all')

    # ---- Step 2: Drop Rows with Undefined Narcolepsy and Missing NT1 and DQB10602 ----
    # Remove rows where 'narcolepsy' is 'undefined' and both 'cataplexy_clear_cut' and 'DQB10602' are NaN.
    nans_drop = df[(df['narcolepsy'] == 'undefined') &
                   (df['cataplexy_clear_cut'].isna()) &
                   (df['DQB10602'].isna())].index
    df = df.drop(nans_drop)
    tab_one = visualize_table(df=df, group_by=['source', 'narcolepsy', 'cataplexy_clear_cut', 'DQB10602'])
    tabs['tab_one'] = tab_one
    # ---- Step 3: Remove Rows with Defined NT1 but Missing DQB10602 ----
    # Drop rows where 'narcolepsy' is confirmed, 'cataplexy_clear_cut' is defined, but 'DQB10602' is missing.
    indexes_to_drop = df.loc[(df['narcolepsy'] == 'narcolepsy') &
                             (~df['cataplexy_clear_cut'].isna()) &
                             (df['DQB10602'].isna())].index
    df = df.drop(indexes_to_drop)
    tab_two = visualize_table(df=df, group_by=['source', 'narcolepsy', 'cataplexy_clear_cut', 'DQB10602'])
    tabs['tab_two'] = tab_two
    # ---- Step 4: Apply Control Group Rule for 23% DQB1*06:02 Positive, 77% Negative ----
    # For non-cataplexy, non-narcoleptic subjects, assign 23% as DQB1*06:02 positive, and the remaining 77% as negative.
    df_controls = df.loc[(df['narcolepsy'] == 'non-narcoleptic') &
                         (df['cataplexy_clear_cut'] == 0) &
                         (df['DQB10602'].isna()), :]
    nt1_sample_size = int(df_controls.shape[0] * 0.23)
    nt1_indices = df_controls.sample(nt1_sample_size, random_state=42).index
    df.loc[nt1_indices, 'DQB10602'] = 1
    nt1_neg_indices = list(set(df_controls.index) - set(nt1_indices))
    df.loc[nt1_neg_indices, 'DQB10602'] = 0
    tab_three = visualize_table(df, group_by=['source', 'narcolepsy', 'cataplexy_clear_cut', 'DQB10602'])
    tabs['tab_three'] = tab_three
    # ---- Step 5: Assign NT1 for Narcolepsy with DQB1*06:02 Positive ----
    # For narcolepsy patients with unknown cataplexy but DQB1*06:02 positive, assign NT1 status.
    df.loc[(df['narcolepsy'] == 'narcolepsy') &
           (df['cataplexy_clear_cut'].isna()) &
           (df['DQB10602'] == 1), 'cataplexy_clear_cut'] = 1

    # ---- Step 6: Assign Non-NT1 for Non-Narcoleptic with DQB1*06:02 Status ----
    # Set 'cataplexy_clear_cut' to 0 for non-narcoleptic subjects with either DQB1*06:02 positive or negative.
    df.loc[(df['narcolepsy'] == 'non-narcoleptic') &
           (df['cataplexy_clear_cut'].isna()) &
           (df['DQB10602'] == 0), 'cataplexy_clear_cut'] = 0
    df.loc[(df['narcolepsy'] == 'non-narcoleptic') &
           (df['cataplexy_clear_cut'].isna()) &
           (df['DQB10602'] == 1), 'cataplexy_clear_cut'] = 0
    tab_four = visualize_table(df=df, group_by=['source', 'narcolepsy', 'cataplexy_clear_cut', 'DQB10602'])
    tabs['tab_four'] = tab_four
    # ---- Step 7: Create 'Pseudo Narcolepsy' Class for Certain NT1 and Factual Narcolepsy Cases ----
    # Label certain cases as 'pseudo narcolepsy' (e.g., factual narcolepsy cases or NT1 without DQB1*06:02 positivity).
    pseudo_nt1_cases_one = df.loc[(df['narcolepsy'] == 'narcolepsy') &
                                  (df['cataplexy_clear_cut'] == 1) &
                                  (df['DQB10602'] == 0), 'cataplexy_clear_cut'].index
    pseudo_nt1_cases_two = df.loc[df['narcolepsy'] == 'factual narcolepsy', 'cataplexy_clear_cut'].index
    pseudo_nt1_cases = [*pseudo_nt1_cases_one] + [*pseudo_nt1_cases_two]
    df.loc[pseudo_nt1_cases, 'cataplexy_clear_cut'] = 2
    df.loc[pseudo_nt1_cases, 'narcolepsy'] = 'pseudo narcolepsy'
    tab_five = visualize_table(df=df, group_by=['narcolepsy', 'cataplexy_clear_cut', 'DQB10602'])
    tabs['tab_five'] = tab_five
    # ---- Step 8: Verify Distribution of DQB1*06:02 in Final Data ----
    # Check if the final distribution meets expected standards for NT1 and non-NT1 cases.
    df.loc[df['narcolepsy'] != 'pseudo narcolepsy', :]
    verification_passed, verification_results = verify_dqb_distribution(
        df.loc[df['narcolepsy'] != 'pseudo narcolepsy', :])

    return verification_results, tabs, df

def compare_imputation(df_original:pd.DataFrame,
                       df_imputed:pd.DataFrame, covariates:Dict[str, str]) -> pd.DataFrame:
    """
    Compare columns in original and imputed DataFrames, showing changes
    in value counts or descriptive statistics based on covariate type.

    Parameters:
    - df_original (pd.DataFrame): Original DataFrame with NaN values.
    - df_imputed (pd.DataFrame): Imputed DataFrame without NaN values.
    - covariates (dict): Dictionary specifying column types ('continuous' or 'ordinal').

    Returns:
    - pd.DataFrame: Comparison DataFrame with statistics before and after imputation.
    """
    comparison_data = []

    for col, col_type in covariates.items():
        if col_type == 'continuous':
            # Collect statistics for continuous variables
            before_stats = df_original[col].describe()
            after_stats = df_imputed[col].describe()
            comparison_data.append({
                'Column': col,
                'Type': 'continuous',
                'Before_Mean': before_stats['mean'],
                'After_Mean': after_stats['mean'],
                'Before_Std': before_stats['std'],
                'After_Std': after_stats['std'],
                'Before_Min': before_stats['min'],
                'After_Min': after_stats['min'],
                'Before_Max': before_stats['max'],
                'After_Max': after_stats['max']
            })

        elif col_type == 'ordinal':
            # Collect value counts for ordinal/categorical variables
            before_counts = df_original[col].value_counts(dropna=False).to_dict()
            after_counts = df_imputed[col].value_counts().to_dict()
            comparison_data.append({
                'Column': col,
                'Type': 'ordinal',
                'Before_ValueCounts': before_counts,
                'After_ValueCounts': after_counts
            })

    # Convert comparison data into a DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    return comparison_df

# %% Main
if __name__ == "__main__":
    PLOT = False
    # %% Pre-process SSQDX dataset
    df_ssqdx = pre_process_ssqdx_dataset(dataset_path=config.get('data_raw_files').get('ssqdx'))
    # %% Pre-process SSQ DATASET
    df_ssq = pre_process_ssq_dataset(dataset_path=config.get('data_raw_files').get('ssq'))
    df_ssq['narcolepsy'] = df_ssq['narcolepsy'].map({1: 'narcolepsy', 0: 'non-narcolepsy'})

    # %% Merge the two dataset
    # Mapping columns of SSQ HLA
    emotions_interest = ["LAUGHING", "QUICKVERBAL", "ANGER"]
    count_array = [str(i) for i in range(54, 74)]
    narc_columns_a = [col for narcolepsy in count_array for col in df_ssq if col.startswith(narcolepsy + 'a')]

    # POSEMOT  cataplexy triggered by positive emotions
    columns_posemot = [
        '54a_laughing-cataplexy',
        '56a_excitement-cataplexy',
        '58a_happy-memory-cataplexy',
        '60a_quick-response-cataplexy',
        '66a_elation-cataplexy',
        '63a_sexual-intercourse-cataplexy',
        '70a_exciting-game-cataplexy',
        '72a_joke-cataplexy',
    ]
    df_ssq['POSEMOT'] = df_ssq.apply(make_new_col, axis=1, args=(columns_posemot,))
    # DISNOCSLEEP  disturbed nocturnal sleep (patients complains of poor sleep at night)
    columns_dinocsleep = [
        '25_difficulty_falling_asleep_ever',
        '26_current_difficulty_falling_asleep',
    ]
    df_ssq['DISNOCSLEEP'] = df_ssq.apply(make_new_col, axis=1, args=(columns_dinocsleep,))
    # NEGEMOT  cataplexy triggered by negative emotions, anger, embarrassment , stress etc (composite of multiple
    # answers as OR)
    columns_negemot = [
        '55a_anger-cataplexy',
        '61a_embarrassment-cataplexy',
        '67a_stress-cataplexy',
        '68a_startle-cataplexy',
        '69a_tension-cataplexy'
    ]
    df_ssq['NEGEMOT'] = df_ssq.apply(make_new_col, axis=1, args=(columns_negemot,))
    # NDEMOT  CATAPLEXY triggered when remember an emotional moment (the only one I am less sure)
    columns_ndemot = [
        '55a_anger-cataplexy',
        '61a_embarrassment-cataplexy',
        '62a_disciplining-children-cataplexy',
        '67a_stress-cataplexy',
        '69a_tension-cataplexy',
        '73a_emotional-moment-cataplexy'
    ]
    df_ssq['NDEMOT'] = df_ssq.apply(make_new_col, axis=1, args=(columns_ndemot,))
    columns_movedemot = [
        '59a_emotional-memory-cataplexy',
    ]
    df_ssq['MOVEDEMOT'] = df_ssq.apply(make_new_col, axis=1, args=(columns_movedemot,))
    for col in ['POSEMOT', 'NEGEMOT', 'NDEMOT', 'DISNOCSLEEP']:
        print(f'{col}: \n{df_ssq[col].value_counts()}')

    print(df_ssq['narcolepsy'].value_counts())
    print(df_ssq['DQB10602'].value_counts())

    #  Transform column values so they match both sets
    mapper = {
        'Race': 'ethnicity',
        'sex': 'gender',
        'Age': 'age',
        'BMI': 'bmi',
        'ESS': 'epworth_score',
        'NAPS': '39_nap_frequency',
        'SLEEPIONSET': '46_age_sleep_complaint',
        # 'CATCODE': '',
        'DURATION': '84_muscle_weakness_duration',
        'FREQ': '85_muscle_weakness_frequency',
        'ONSET': '95_first_muscle_weakness_age',
        'INJURED': '101_injured_during_episode',
        # 'MEDCATA': '',
        # 'MSLT': '',
        # 'SOREMP': '',
        # 'MSLTAGE': '',
        # 'DQB10602': '',
        'DISNOCSLEEP': 'DISNOCSLEEP',
        'POSEMOT': 'POSEMOT',
        'NEGEMOT': 'NEGEMOT',
        'NDEMOT': 'NDEMOT',
        'MOVEDEMOT': 'MOVEDEMOT',
    }
    mapper_inv = {val: key for key, val in mapper.items()}
    df_ssqdx = df_ssqdx.replace(to_replace='.', value=np.nan)
    df_ssqdx['ESS'] = df_ssqdx['ESS'].astype(float)
    # clip age to the SSQHLA
    df_ssqdx['Age'] = df_ssqdx['Age'].astype(float)
    df_ssq[df_ssq['age'] < 9] = np.nan
    print(df_ssq['narcolepsy'].value_counts())

    df_ssqdx['sex'] = df_ssqdx['sex'].replace({'M': 1, 'F': 0})
    df_ssq['gender'] = df_ssq['gender'].replace({9: np.nan})

    df_ssqdx['Race'] = df_ssqdx['Race'].replace({'Caucasian ': 'Caucasian'})
    df_ssq['ethnicity'] = df_ssq['ethnicity'].replace({
        'Cauc': 'Caucasian',
        'Latino': 'Latino',
        '9': np.nan,
    })

    df_ssqdx['NAPS'].unique()
    df_ssq.loc[df_ssq['39_nap_frequency'] > 100, '39_nap_frequency'] = np.nan

    df_ssqdx['NAPS'].unique()
    df_ssq['45_sleepiness_severity_since_age'].value_counts()
    print(df_ssq['narcolepsy'].value_counts())


    # age sleep complaints columns are not perfect, feature engineer by combining and estimating
    df_ssqdx['SLEEPIONSET'] = df_ssqdx['SLEEPIONSET'].replace({'33': 33})
    print(df_ssq['narcolepsy'].value_counts())

    df_ssq['46_age_sleep_complaint'] = df_ssq.apply(create_sleep_complaint, axis=1)
    print(df_ssq['narcolepsy'].value_counts())

    if PLOT:
        sns.histplot(df_ssq['46_age_sleep_complaint'].dropna(), kde=True)
        plt.grid(alpha=0.7)
        plt.tight_layout()
        plt.show()
        plot_histograms(df1=df_ssqdx,
                        df2=df_ssq,
                        col1=f'SLEEPIONSET',
                        col2=f'46_age_sleep_complaint',
                        title=f'SSQHLA SLEEPIONSET {df_ssqdx.shape[0]} \nVs \nSSQ 46_age_sleep_complaint {df_ssq.shape[0]}')

    # Duration is ordinal -> 5-30sec, 13sec-2min, 2-10min, 10min++ -> 0,1,2,3
    df_ssqdx['DURATION'].unique()
    df_ssq['84_muscle_weakness_duration'] = df_ssq['84_muscle_weakness_duration'].map({0: 0, 1: 1, 2: 2, 3: 3})
    if PLOT:
        sns.histplot(df_ssq['84_muscle_weakness_duration'].dropna(), kde=True)
        plt.grid(alpha=0.7)
        plt.tight_layout()
        plt.show()

    # frequency, in the ssq is okay is they are not much since the majority are controls
    # df_ssqdx['FREQ'].unique()
    df_ssq['85_muscle_weakness_frequency'] = df_ssq['85_muscle_weakness_frequency'].map({0: 0, 1: 1, 2: 2, 3: 3})
    if PLOT:
        sns.histplot(df_ssq['85_muscle_weakness_frequency'].dropna(), kde=True)
        plt.grid(alpha=0.7)
        plt.tight_layout()
        plt.show()

    df_ssqdx['ONSET'].unique()
    df_ssq['95_first_muscle_weakness_age'].unique()
    df_ssq.loc[df_ssq['95_first_muscle_weakness_age'] > 100, '95_first_muscle_weakness_age'] = np.nan
    if PLOT:
        sns.histplot(df_ssq['95_first_muscle_weakness_age'].dropna(), kde=True)
        plt.grid(alpha=0.7)
        plt.tight_layout()
        plt.show()

    # df_ssqdx['INJURED'].unique()
    df_ssq['101_injured_during_episode'].unique()
    df_ssq.loc[df_ssq['101_injured_during_episode'] > 2, '101_injured_during_episode'] = np.nan
    if PLOT:
        sns.histplot(df_ssq['101_injured_during_episode'].dropna(), kde=True)
        plt.grid(alpha=0.7)
        plt.tight_layout()
        plt.show()

    if PLOT:
        # comapre the columns side-by-side in a single dataframe and making histogram plots
        df_comparison_dist = pd.DataFrame()
        for col_ssqhla, col_ssq in mapper.items():
            desc_ssqhla = df_ssqdx[col_ssqhla].describe()
            desc_ssq = df_ssq[col_ssq].describe()
            combined_desc = pd.concat([desc_ssqhla, desc_ssq], axis=1)
            combined_desc.columns = ['ssqhla', 'ssq']
            combined_desc['columns'] = f'{col_ssqhla}, {col_ssq}'
            df_comparison_dist = pd.concat([df_comparison_dist, combined_desc])

        for col_ssqhla, col_ssq in mapper.items():
            plot_histograms(df1=df_ssqdx,
                            df2=df_ssq,
                            col1=f'{col_ssqhla}',
                            col2=f'{col_ssq}',
                            title=f'SSQHLA {col_ssqhla} {df_ssqdx.loc[~df_ssqdx[col_ssqhla].isna(), col_ssqhla].shape[0]} '
                                  f'\nVs \nSSQ {col_ssq} {df_ssq.loc[~df_ssq[col_ssq].isna(), col_ssq].shape[0]}')
    # slice the frame with the columns of interest (intersection between SSQ and SSQHLA)
    df_ssq_to_merge = df_ssq[[*mapper.values()]].copy()
    # include the multiple target columns
    df_ssq_to_merge['DQB10602'] = df_ssq['DQB10602']
    df_ssq_to_merge['narcolepsy'] = df_ssq['narcolepsy']
    # df_ssq_to_merge['target'] = df_ssq['target']
    df_ssq_to_merge['cataplexy_clear_cut'] = df_ssq['cataplexy_clear_cut']

    print(df_ssq_to_merge['DQB10602'].value_counts())
    print(df_ssq_to_merge['narcolepsy'].value_counts())
    # print(df_ssq_to_merge['target'].value_counts())
    print(df_ssq_to_merge['cataplexy_clear_cut'].value_counts())
    # %% Emotions and muscle weakness equivalence (Mapping between the two datasets)
    emotions_ssq_ssqhla = {
        '56a_excitement-cataplexy': 'EXCITED',
        '73a_emotional-moment-cataplexy': 'MOVEDEMOT',
        '62a_disciplining-children-cataplexy': 'DISCIPLINE',
        '65a_post-athletic-activities-cataplexy': 'AFTATHLETIC',
        '72a_joke-cataplexy': 'JOKING',
        '61a_embarrassment-cataplexy': 'EMBARRAS',
        '55a_anger-cataplexy': 'ANGER',
        '63a_sexual-intercourse-cataplexy': 'SEX',
        '70a_exciting-game-cataplexy': 'PLAYGAME',
        '54a_laughing-cataplexy': 'LAUGHING',
        '68a_startle-cataplexy': 'STARTLED',
        '66a_elation-cataplexy': 'ELATED',
        '59a_emotional-memory-cataplexy': 'EMOTIONAL',
        '60a_quick-response-cataplexy': 'QUICKVERBAL',
        '64a_athletic-activities-cataplexy': 'DURATHLETIC',
        '71a_romantic-moment-cataplexy': 'ROMANTIC',
        '67a_stress-cataplexy': 'STRESSED',
        '69a_tension-cataplexy': 'TENSE',
        '58a_happy-memory-cataplexy': 'HAPPY',
        '57a_surprise-cataplexy': 'SURPRISED'
    }
    # df_emotions = pd.DataFrame.from_dict(emotions_ssq_ssqhla, orient='index', columns=['Emotion'])
    # df_emotions = df_emotions.reset_index().rename(columns={'index': 'Event'})

    for emotion_ssq_, emotions_ssq_ssqhla_ in emotions_ssq_ssqhla.items():
        df_ssq_to_merge[emotions_ssq_ssqhla_] = df_ssq[emotion_ssq_]

    df_ssq_to_merge['JAW'] = df_ssq[f'{77}_{key_mapping.get(77)}']
    df_ssq_to_merge['KNEES'] = df_ssq[f'{76}_{key_mapping.get(76)}'].apply(quest76map)
    df_ssq_to_merge['HEAD'] = df_ssq[f'{78}_{key_mapping.get(78)}']
    df_ssq_to_merge['HAND'] = df_ssq[f'{79}_{key_mapping.get(79)}']


    # falling experience with experience 6
    col_exp_falling = [col for col in df_ssq.columns if '_exp_6' in col]
    df_ssq_to_merge['FALL'] = df_ssq[col_exp_falling].any(axis=1).astype(int)

    # rename the columns
    df_ssq_to_merge.rename(mapper_inv, inplace=True, axis=1)
    df_ssq_to_merge['DQB10602'].value_counts()
    # %% include the missing columns (not measure from the questionnaire)
    df_ssqdx.shape
    df_ssq_to_merge.shape
    for col in df_ssqdx.columns:
        if not col in df_ssq_to_merge.columns:
            # print(col)
            df_ssq_to_merge[col] = np.nan
    # %% complete the merge
    df_ssqdx['source'] = 'ssqdx'
    df_ssq_to_merge['source'] = 'ssq'
    df_data = pd.concat([df_ssqdx, df_ssq_to_merge], axis=0)
    df_data.reset_index(inplace=True, drop=True)
    # df_data.rename(columns={'DQB10602': 'target'}, inplace=True)
    if PLOT:
        palette = sns.color_palette("Set2",
                                    n_colors=len(df_data['source'].unique()))
        # Create a 2x2 grid, but adjust the layout so the third plot spans both columns in the second row
        fig, ax = plt.subplots(nrows=2,
                               ncols=2,
                               figsize=(10, 8),
                               sharey=False,
                               gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [1, 1]})
        # Top-left: DQB10602
        sns.countplot(data=df_data, x='DQB10602', hue='source', palette=palette, ax=ax[0, 0])
        ax[0, 0].set_title(f'DQB10602 In Complete Dataset\n{df_data.DQB10602.value_counts().to_dict()}')
        ax[0, 0].grid(alpha=.7)
        # Top-right: Narcolepsy
        sns.countplot(data=df_data, x='cataplexy_clear_cut', hue='source', palette=palette, ax=ax[0, 1])

        ax[0, 1].set_title(f'Merged DQB and Narcolepsy In Complete Dataset\n'
                            f'{df_data["cataplexy_clear_cut"].value_counts().to_dict()}')
        ax[0, 1].grid(alpha=.7)
        # Span the third plot across both columns in the second row
        ax_bottom = fig.add_subplot(2, 1, 2)
        sns.countplot(data=df_data, x='narcolepsy', hue='source', palette=palette, ax=ax_bottom)
        ax_bottom.set_title(f'Narcolepsy In Complete Dataset\n{df_data.narcolepsy.value_counts().to_dict()}')
        ax_bottom.grid(alpha=.7)
        # Remove the unused axes
        ax[1, 0].remove()  # Remove the empty subplot
        ax[1, 1].remove()  # Remove the empty subplot
        plt.tight_layout()
        plt.show()


    # %% Check data types on emotions and muscle weaknesses
    # keep them as binary responses yes/no
    emotions_ssqhla = {
        "LAUGHING", "EXCITED", "HAPPY", "QUICKVERBAL", "SEX", "ELATED", "PLAYGAME",
        "JOKING", "NEGEMOT", "ANGER", "EMBARRAS", "DISCIPLINE", "STRESSED", "TENSE",
        "NDEMOT", "SURPRISED", "EMOTIONAL", "DURATHLETIC", "AFTATHLETIC", "STARTLED",
        "ROMANTIC", "MOVEDEMOT"
    }
    # muscle weakness
    mw_ssqhla = {'KNEES', 'JAW', 'HEAD', 'HAND', 'SPEECH'}

    for col in emotions_ssqhla:
        # print(f'\n{col}: {df_data[col].value_counts()}')
        df_data[col] = df_data[col].apply(set_to_zero_except_one)

    for col in mw_ssqhla:
        # print(f'\n{col}: {df_data[col].value_counts()}')
        df_data[col] = df_data[col].apply(set_to_zero_except_one)

    # %% organize columns
    cols_head = [
        "Race",
        "Ethnicity",
        "sex",
        "Age",
        "BMI",
    ]
    cols_tail = [
        "CATCODE",
        "MEDCATA",
        "MSLT",
        "SOREMP",
        "MSLTAGE",
        "SE",
        "REMLAT",
        "RDI",
        "PLMIND",
        "Dx",
        "contains_A_to_F",
        "narcolepsy",
        "cataplexy_clear_cut",
        "DQB10602",
        'source'
    ]
    col_middle = [col for col in df_data.columns if not (col in cols_head or col in cols_tail)]
    columns = cols_head + col_middle + cols_tail
    df_data = df_data[columns]


    # %% Filter and Clean the diagnosis
    df_data['narcolepsy'] = df_data['narcolepsy'].replace({'non-narcolepsy': 'non-narcoleptic'})
    # harmonize the narcolepsy column
    # assert the unique values
    assert df_data['narcolepsy'].value_counts().shape[0] == 4
    assert df_data['DQB10602'].value_counts().shape[0] == 2
    assert df_data['cataplexy_clear_cut'].value_counts().shape[0] == 2

    verification_results,tabs, df_data = wrangle_target_combinations(df=df_data)

    # %% Imputation Age, Sex and ESS
    categorical_var = ['sex',
                       'Ethnicity',
                       'LAUGHING', 'ANGER', 'EXCITED','SURPRISED', 'HAPPY',
                       'EMOTIONAL', 'QUICKVERBAL', 'EMBARRAS', 'DISCIPLINE',
                       'SEX', 'DURATHLETIC', 'AFTATHLETIC', 'ELATED', 'STRESSED',
                       'STARTLED', 'TENSE', 'PLAYGAME', 'ROMANTIC', 'JOKING', 'MOVEDEMOT',
                       'KNEES', 'JAW', 'HEAD', 'HAND', 'SPEECH',
                       # 'DURATION', 'CATSEVER',
                       # 'ONSET', 'HALLUC', 'HHONSET', 'HHSEVERITY', 'SP', 'SPSEVER', 'SPONSET',
                       # 'SPMEDS', 'FREQ', 'INJURED', 'POSEMOT', 'NEGEMOT', 'NDEMOT', 'CATCODE',
                       # 'MEDCATA', 'MSLT', 'SOREMP', 'MSLTAGE', 'SE', 'REMLAT', 'RDI', 'PLMIND'
                       # 'cataplexy_sampled', 'cataplexy'
                       ]

    continuous_var = ['Age', 'BMI', 'ESS', 'DISNOCSLEEP', 'NAPS', 'SLEEPIONSET', 'ONSET']
    columns = list(set(categorical_var + continuous_var))

    df_nan_count = df_data.isna().sum(0)
    df_impute = df_data.drop(columns=df_nan_count[df_nan_count > 15].index)

    covariates_list = continuous_var + categorical_var
    covariates_list = [cov for cov in covariates_list if cov in df_impute.columns]
    # Create the covariates dictionary with appropriate types
    covariates = {cov: 'continuous' for cov in continuous_var if cov in covariates_list}
    covariates.update({cov: 'ordinal' for cov in categorical_var if cov in covariates_list})

    # Identify columns that will be imputed
    imputed_columns = df_impute[covariates_list].isnull().any()
    imputed_columns_list = imputed_columns[imputed_columns].index.tolist()

    imputer = IterativeImputer(
        estimator=BayesianRidge(),  # Model for the imputation process
        max_iter=100,  # Maximum number of imputation rounds
        tol=1e-3,  # Convergence tolerance
        n_nearest_features=2,  # Use 2 nearest features to estimate missing values
        initial_strategy="mean",  # Initial strategy to start filling missing values
        imputation_order="ascending"  # Order of imputation (ascending by missing values)
    )

    # Fit the imputer on the data and transform it
    imputed_data = imputer.fit_transform(df_impute[covariates_list])
    df_imputed = pd.DataFrame(imputed_data, columns=covariates_list, index=df_data.index)
    # Round the ordinal columns
    for col, col_type in covariates.items():
        if col_type == 'ordinal':
            df_imputed[col] = np.round(df_imputed[col]).astype(int)
    # compare
    imputed_columns = {key: covariates.get(key) for key in imputed_columns_list}
    df_comparison_results = compare_imputation(df_data, df_imputed, imputed_columns)

    # include the remaining columns
    col_missing = [col for col in df_data.columns if not col in df_imputed.columns]
    df_imputed[col_missing] = df_data[col_missing]

    # one observation that narcolepsy is nan
    df_imputed = df_imputed.loc[~df_imputed['narcolepsy'].isna(), :]
    # Reorder df_imputed to have the same column order as df_data
    df_imputed = df_imputed[df_data.columns]
    assert df_imputed['cataplexy_clear_cut'].isna().sum() == 0
    assert df_imputed['narcolepsy'].isna().sum() == 0
    assert df_imputed['DQB10602'].isna().sum() == 0

    #%% save dataset
    df_imputed.to_csv(config.get('data_pre_proc_files').get('ssq_ssqdx'), index=False)







