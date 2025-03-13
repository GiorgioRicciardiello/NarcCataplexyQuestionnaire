import pandas as pd
from typing import  Optional, Tuple, List
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import numpy as np
import pathlib
import matplotlib as mpl
import textwrap
from library.my_dcurves import my_plot_graphs


def create_venn_diagram(
    configs,
    config1,
    config2,
    figure_size=(6, 6),
    set_labels=None,
    colors=('pink', 'orange'),
    alpha=0.7,
    title="Venn Diagram"
):
    """
    Creates and displays a Venn diagram comparing the feature sets
    from two configurations in a given dictionary.

    :param configs: dict
        Dictionary of configurations, each containing a 'features' list.
    :param config1: str
        Key in the configs dict for the first configuration.
    :param config2: str
        Key in the configs dict for the second configuration.
    :param figure_size: tuple
        Width and height of the figure (in inches).
    :param set_labels: tuple or None
        Labels for the two sets. If None, uses (config1, config2).
    :param colors: tuple
        Colors for the two circles (e.g., ('pink', 'orange')).
    :param alpha: float
        Transparency for the circles (0.0 = fully transparent, 1.0 = solid).
    :param title: str
        Title of the plot.
    """

    # 1. Extract the feature sets
    features1 = set(configs[config1]['features'])
    features2 = set(configs[config2]['features'])

    # 2. Compute intersections and differences
    only_config1 = features1.difference(features2)
    only_config2 = features2.difference(features1)
    common_features = features1.intersection(features2)

    # 3. Create the figure and axes
    fig, ax = plt.subplots(figsize=figure_size)

    # 4. Create the Venn diagram
    #    If set_labels is None, we default to config1 and config2
    if set_labels is None:
        set_labels = (config1, config2)
    v = venn2(
        [features1, features2],
        set_labels=set_labels,
        set_colors=colors,
        alpha=alpha,
        subset_label_formatter=lambda x: ''  # remove default counts
    )

    # 5. Increase the font size for the set labels
    for text in v.set_labels:
        if text:
            text.set_fontsize(14)

    # 6. Increase the font size for the subset labels (the numeric counts)
    #    Since we removed them, this is optional
    for text in v.subset_labels:
        if text:
            text.set_fontsize(12)

    # 7. Manually place the feature names in each region

    # Region unique to config1: label '10'
    label_10 = v.get_label_by_id('10')
    if label_10:
        pos_10 = label_10.get_position()
        ax.text(
            pos_10[0], pos_10[1],
            "\n".join(sorted(only_config1)),
            fontsize=10, ha='center', va='center', color='black'
        )

    # Region unique to config2: label '01'
    label_01 = v.get_label_by_id('01')
    if label_01:
        pos_01 = label_01.get_position()
        ax.text(
            pos_01[0], pos_01[1],
            "\n".join(sorted(only_config2)),
            fontsize=10, ha='center', va='center', color='black'
        )

    # Common region: label '11'
    label_11 = v.get_label_by_id('11')
    if label_11:
        pos_11 = label_11.get_position()
        ax.text(
            pos_11[0], pos_11[1],
            "\n".join(sorted(common_features)),
            fontsize=10, ha='center', va='center', color='black'
        )

    # 8. Set plot title and layout
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.show()



def plot_model_metrics(df: pd.DataFrame,
                       palette: Optional[str] = 'muted',
                       figsize: Optional[Tuple] = (16, 8)):
    """
    Plot F1 score and NPV for each model and configuration in two vertically-stacked subplots.

    The figure is wider but shorter in height. A shared legend (based on 'model') is placed
    between the two plots. The plots use seaborn's style and theme for a publication-quality appearance.

    Parameters:
        df (pd.DataFrame): DataFrame with at least the following columns: 'model', 'config', 'F1', 'npv'.
    """
    # Sort data by 'model' and 'config' for consistent ordering.
    df = df.sort_values(by=['model', 'config'])

    # Set seaborn theme for a clean look.
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    # Create a figure with two rows and one column.
    # A wide but low figure: adjust the figsize as needed (width, height).
    fig, axes = plt.subplots(nrows=2,
                             ncols=1,
                             figsize=figsize,
                             )
    fig.subplots_adjust(hspace=2.0,
                        # wspace=2
                        )
    # --- First Plot: F1 Score ---
    ax1 = axes[0]
    sns.barplot(
        data=df,
        x='config',
        y='f1_score',
        hue='model',
        ax=ax1,
        # ci='sd',  # Optionally, show standard deviation as error bars.
        palette=palette
    )
    # ax1.set_title("F1 Score Comparison")
    # ax1.set_xlabel("Configuration")
    ax1.set_ylabel("F1 Score")
    # Remove the legend from this axis.
    ax1.get_legend().remove()

    # Optionally annotate each bar with its F1 value
    for container in ax1.containers:
        ax1.bar_label(container,
                      fmt='%.2f',
                      padding=3,
                      fontweight='bold',
                      fontsize=8
                      )

    # --- Second Plot: NPV ---
    ax2 = axes[1]
    sns.barplot(
        data=df,
        x='config',
        y='npv',
        hue='model',
        ax=ax2,
        # ci='sd',
        palette=palette
    )
    # ax2.set_title("NPV Comparison")
    ax2.set_xlabel("Configuration")
    ax2.set_ylabel("NPV")
    # Remove the legend from this axis.
    ax2.get_legend().remove()

    # --- Create a Single Shared Legend ---
    # Obtain handles and labels from one of the plots (both are using the same hue).
    handles, labels = ax1.get_legend_handles_labels()

    # Place the legend in the figure. Here, we use bbox_to_anchor to position it between the subplots.
    # You can adjust (0.5, 0.5) if you want a different position.
    fig.legend(handles, labels,
               loc='upper center',
               bbox_to_anchor=(0.5, 0.5),
               ncol=len(labels),
               frameon=False,
               title=None)  # Explicitly set title to None to remove it
    # Optionally annotate each bar with its npv value
    for container in axes[1].containers:
        axes[1].bar_label(container,
                          fmt='%.2f',
                          padding=3,
                          fontweight='bold',
                          fontsize=8
                          )

    # Adjust layout to make room for the legend.
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()



def plot_model_metrics_specific_columns(df: pd.DataFrame,
                                        columns: List[str],
                                        palette: Optional[str] = 'muted',
                                        figsize: Optional[Tuple] = (16, 8)):
    """
    Plot specified metrics for each model and configuration in vertically-stacked subplots.

    One subplot is created per column specified in the 'columns' parameter. A single shared legend
    (based on 'model') is placed above the first subplot.

    Parameters:
        df (pd.DataFrame): DataFrame with at least 'model', 'config', and the columns specified in 'columns'.
        columns (List[str]): List of column names to plot (e.g., ['f1_score', 'npv', 'accuracy']).
        palette (str, optional): Seaborn color palette name.
        figsize (Tuple, optional): Figure size (width, height).
    """
    # Sort data by 'model' and 'config' for consistent ordering
    df = df.sort_values(by=['model', 'config'])

    # Set seaborn theme for a clean look
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    # Create figure with dynamic number of rows based on columns length
    n_rows = len(columns)
    fig, axes = plt.subplots(nrows=n_rows,
                             ncols=1,
                             figsize=(figsize[0], figsize[1] * n_rows / 2))  # Adjust height based on rows
    fig.subplots_adjust(hspace=2.0)  # Consistent vertical spacing

    # Ensure axes is iterable even for a single subplot
    if n_rows == 1:
        axes = [axes]

    # Plot each metric in its own subplot
    for i, col in enumerate(columns):
        ax = axes[i]
        sns.barplot(
            data=df,
            x='config',
            y=col,
            hue='model',
            ax=ax,
            palette=palette
        )
        # Set labels
        ax.set_ylabel(col.upper())
        if i == n_rows - 1:  # Only set xlabel for the bottom plot
            ax.set_xlabel("Configuration")
        else:
            ax.set_xlabel("")

        # Remove individual legends
        ax.get_legend().remove()

        # Annotate bars
        for container in ax.containers:
            ax.bar_label(container,
                         fmt='%.2f',
                         padding=3,
                         fontweight='bold',
                         fontsize=8)

    # Create a single shared legend at the top of the first figure
    handles, labels = axes[0].get_legend_handles_labels()  # Get legend info from first plot
    fig.legend(handles, labels,
               loc='upper center',  # Position at the top center
               bbox_to_anchor=(0.5, 1.0),  # Anchor to top of figure (adjusted from 0.5)
               ncol=len(labels) / 2,  # Horizontal layout
               frameon=False,  # No frame around legend
               title=None)  # No legend title

    # Adjust layout to accommodate legend above the plot
    plt.tight_layout(rect=[0, 0, 1, 0.9])  # Leave space at the top for legend
    plt.show()



def plot_elastic_net_model_coefficients(df_params: pd.DataFrame,
                                        output_path: pathlib.Path = None):
    """
    Generate a styled plot for the elastic net feature importance coefficients.

    Parameters:
    - df_params: DataFrame with columns 'Feature', 'Mean Coefficient', 'Standard Error', 'configuration'
    - output_path: Path to save the plot (optional)
    """

    # Improve overall font style
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.size'] = 12
    mpl.rcParams['axes.titlesize'] = 16
    mpl.rcParams['axes.titleweight'] = 'bold'
    mpl.rcParams['axes.labelsize'] = 14

    # Get unique configurations and compute a global maximum for consistent x-axis limits.
    configurations = df_params["configuration"].unique()
    n_configs = len(configurations)
    global_max = (df_params["Mean Coefficient"].abs() + df_params["Standard Error"]).max()

    # mark if the param is negative
    df_params['is_negative'] = df_params['Mean Coefficient'] < 0

    # Identify all unique features and assign each one a color from a colormap.
    unique_features = df_params["Feature"].unique()
    cmap = plt.get_cmap("tab20c")
    feature_to_color = {}
    for i, feat in enumerate(unique_features):
        feature_to_color[feat] = cmap(i % 10)  # cycle through 10 distinct colors

    # Create subplots: one for each configuration.
    fig, axes = plt.subplots(
        1,
        n_configs,
        figsize=(4 * n_configs, 8),
        sharey=True
    )

    # If there's only one configuration, make axes iterable.
    if n_configs == 1:
        axes = [axes]

    # Plot each configuration in a separate subplot.
    for ax, config in zip(axes, configurations):
        df_plot = df_params.loc[df_params["configuration"] == config, :]

        # Assign color to each row based on the feature.
        # colors = [feature_to_color[f] for f in df_plot["Feature"]]

        # Assign color: black if is_negative is True, otherwise use feature_to_color mapping.
        colors = ["black" if is_neg else feature_to_color[feat]
                  for feat, is_neg in zip(df_plot["Feature"], df_plot["is_negative"])]


        # Set consistent x-axis limit across all subplots.
        ax.set_xlim(0, global_max * 1.1)

        # Plot the horizontal bar chart with error bars.
        bars = ax.barh(
            df_plot["Feature"],
            np.abs(df_plot["Mean Coefficient"]),
            xerr=df_plot["Standard Error"],
            capsize=5,
            color=colors
        )

        ax.invert_yaxis()  # Highest importance on top
        ax.grid(True, linestyle="--", alpha=1)

        # Wrap long configuration strings onto multiple lines.
        wrapped_config = textwrap.fill(config, width=20)
        ax.set_title(wrapped_config)

        # Annotate a star inside each bar for features that are negative in this configuration.
        # We use the bar container to determine each bar's position.
        offset = global_max * 0.02  # small offset relative to global_max
        for bar, (_, row) in zip(bars, df_plot.iterrows()):
            if row["is_negative"]:
                # The x position is near the right end of the bar (using data coordinates).
                bar_width = bar.get_width()
                x = bar_width + offset
                # y coordinate is the center of the bar.
                y = bar.get_y() + bar.get_height() / 2
                ax.text(x, y, '*', color='red', va='center', ha='center', fontsize=20)

    # Set one common x-label and y-label for all subplots.
    fig.supxlabel("Mean Absolute Coefficient", fontsize=12)
    fig.supylabel("Feature", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # Optionally save the figure
    if output_path:
        plt.savefig(output_path / 'elastic_net_model_coefficients.png', dpi=300)
    plt.show()


# %% Dcurves functions
def prepare_net_benefit_df(
        results: pd.DataFrame,
        prevalence: float,
        thresholds: np.ndarray = None,
        model_name: str = "Elastic Net"
) -> pd.DataFrame:
    """
    Prepare a DataFrame of net benefit values for an Elastic Net model, the "all", and "none" strategies.

    Parameters
    ----------
    results : pd.DataFrame
        DataFrame containing at least the columns 'true_label' and 'predicted_prob'.
    prevalence : float
        Prevalence of the event (e.g., 30 per 10000 would be 0.003).
    thresholds : np.ndarray, optional
        Array of threshold probabilities. If None, defaults to np.linspace(0.01, 0.99, 99).
    model_name : str, optional
        Name of the model to use in the output DataFrame (default is "Elastic Net").

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns: "threshold", "model", "net_benefit" that can be passed to plot_graphs().

    Notes
    -----
    For the "all" strategy:
      net_benefit = prevalence - (1 - prevalence) * (threshold / (1 - threshold))

    For the "none" strategy:
      net_benefit = 0 for all thresholds.

    For the Elastic Net model, net benefit is computed as:
      net_benefit = TP/N - FP/N * (threshold / (1 - threshold))
    where TP and FP are the counts of true positives and false positives at the threshold.
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    # Make a copy of the input DataFrame to avoid modifying the original
    data = results.copy()
    N = len(data)

    net_benefit_data = []

    # Compute net benefit for the Elastic Net model at each threshold
    for thr in thresholds:
        # Create binary predictions using the threshold
        data['pred_thr'] = (data['predicted_prob'] >= thr).astype(int)
        TP = ((data['true_label'] == 1) & (data['pred_thr'] == 1)).sum()
        FP = ((data['true_label'] == 0) & (data['pred_thr'] == 1)).sum()

        net_benefit = TP / N - FP / N * (thr / (1 - thr))

        net_benefit_data.append({
            'threshold': thr,
            'model': model_name,
            'net_benefit': net_benefit
        })

    # Compute net benefit for the "all" strategy
    for thr in thresholds:
        net_benefit_all = prevalence - (1 - prevalence) * (thr / (1 - thr))
        net_benefit_data.append({
            'threshold': thr,
            'model': 'all',
            'net_benefit': net_benefit_all
        })

    # Compute net benefit for the "none" strategy (always 0)
    for thr in thresholds:
        net_benefit_data.append({
            'threshold': thr,
            'model': 'none',
            'net_benefit': 0
        })

    plot_df = pd.DataFrame(net_benefit_data)
    return plot_df


def plot_dcurves_per_fold(df_results: pd.DataFrame, prevalence: float):
    """
    Plots decision curves for each validation fold, displaying net benefit curves along with
    the sample sizes (total, cases, and controls) in each fold.

    This function dynamically determines a grid layout for subplots based on the number of
    unique folds in the input DataFrame. For each fold, it:
      - Filters the results for the current fold.
      - Computes the number of total samples, cases (true_label==1), and controls (true_label==0).
      - Prepares the plotting data using the provided prevalence and a helper function
        `prepare_net_benefit_df`.
      - Generates a net benefit plot on the designated subplot axis using `my_plot_graphs`.
      - Sets the subplot title to include the fold number and sample counts.
    Any extra axes in the subplot grid (if the grid is larger than the number of folds) are hidden.

    Parameters
    ----------
    df_results : pd.DataFrame
        DataFrame containing the decision curve results, including at least the columns
        'fold_number' and 'true_label'. It should also have any columns required by
        `prepare_net_benefit_df` for preparing the net benefit data.
    prevalence : float
        The prevalence value used in the calculation of net benefit (e.g., 0.003).

    Returns
    -------
    None
        The function displays the generated figure with subplots, but does not return any value.
    """
    unique_folds = np.sort(df_results['fold_number'].unique())
    num_folds = len(unique_folds)

    # --- Determine subplot grid dynamically ---
    # If you want to manually specify rows and columns, set nrows or ncols to an integer.
    # Otherwise, leave them as None to have the grid computed automatically.
    nrows = None  # e.g., set nrows = 2 if you want two rows
    ncols = None  # e.g., set ncols = 3 if you want three columns

    if nrows is None and ncols is None:
        nrows = int(np.floor(np.sqrt(num_folds)))
        ncols = int(np.ceil(num_folds / nrows))
    elif nrows is None:
        nrows = int(np.ceil(num_folds / ncols))
    elif ncols is None:
        ncols = int(np.ceil(num_folds / nrows))

    # Create the figure and axes grid
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))

    # Flatten the axes array for easier iteration (if there's more than one subplot)
    if num_folds > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    # --- Plot for each fold ---
    for ax, fold in zip(axes, unique_folds):
        # Filter data for the current fold
        fold_data = df_results.loc[df_results['fold_number'] == fold]

        # Count the number of samples, cases, and controls
        num_total = len(fold_data)
        # Assumes that a column "true_label" exists and that 1 indicates a case.
        cases = fold_data[fold_data["true_label"] == 1].shape[0]
        controls = fold_data[fold_data["true_label"] == 0].shape[0]

        # Prepare the DataFrame to be plotted
        df_plot_curve = prepare_net_benefit_df(
            results=fold_data,
            prevalence=prevalence
        )

        # Generate the plot on the provided axis
        my_plot_graphs(
            plot_df=df_plot_curve,
            graph_type="net_benefit",
            y_limits=(-0.05, 0.5),
            ax=ax
        )

        # Set the subplot title with fold, total samples, cases, and controls
        ax.set_title(f"Fold {fold}\nTotal: {num_total}, Cases: {cases}, Controls: {controls}")

    # If there are extra axes (when the grid is larger than the number of folds), hide them.
    for extra_ax in axes[len(unique_folds):]:
        extra_ax.axis('off')

    plt.tight_layout()
    plt.show()



