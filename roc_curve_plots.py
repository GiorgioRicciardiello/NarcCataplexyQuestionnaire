from lifelines.fitters.npmle import npmle

from config.config import config
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from pathlib import Path
from typing import Optional, Tuple
import math
from tabulate import tabulate
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import math
from sklearn.metrics import roc_curve, auc
from pathlib import Path
import seaborn as sns
from typing import Optional, Tuple
from matplotlib.lines import Line2D
from sklearn.metrics import confusion_matrix, roc_curve, auc
from scipy.stats import norm
from typing import Dict, List


def compute_best_thresholds(
        df: pd.DataFrame,
        dataset_type: str = 'full',
        target_fpr: float = 0.01,  # the lower the higher the specifcity
        prevalence: float = 30 / 100000
) -> pd.DataFrame:
    """
    Given a “long” DataFrame with columns:
        - 'config'         (feature‐set name)
        - 'model_name'     (string)
        - 'true_label'     (0/1)
        - 'predicted_prob' (float [0, 1])
        - 'dataset_type'   (e.g., 'full', 'validation', etc.)

    Returns a DataFrame with one row per (config, model_name) computed only on
    rows where dataset_type == 'full'. The returned columns are:
        - 'config'
        - 'model_name'
        - 'auc'                  (area under the ROC curve)

        # Youden’s J optimal threshold
        - 'best_threshold_j'
        - 'sensitivity_j'
        - 'specificity_j'
        - 'ppv_j'                (apparent PPV at best_threshold_j)
        - 'ppvprev_j'            (prevalence‐adjusted PPV at best_threshold_j)

        # FPR‐constrained threshold (FPR ≤ target_fpr)
        - 'best_threshold_fpr'
        - 'sensitivity_fpr'
        - 'specificity_fpr'
        - 'ppv_fpr'              (apparent PPV at best_threshold_fpr)
        - 'ppvprev_fpr'          (prevalence‐adjusted PPV at best_threshold_fpr)

    Any group with all‐NaN predicted_prob (on the 'full' subset) will be skipped.

    After computing, the returned DataFrame is sorted within each 'config' by
    'specificity_j' in descending order.

    Args:
        df (pd.DataFrame): Must contain the required columns.
        target_fpr (float): Maximum allowed false positive rate (e.g. 0.01 for 1% FPR)
                            when selecting threshold by specificity priority.
        prevalence (float): Prevalence to use for prevalence‐adjusted PPV calculation.

    Returns:
        pd.DataFrame: One row per (config, model_name) with all the columns listed above,
                      sorted within each config by descending specificity_j.
    """
    records: List[Dict] = []

    # 1) Filter to the “full” dataset only
    df_full = df[df['dataset_type'] == dataset_type]

    # 2) Group by (config, model_name)
    grouped = df_full.groupby(['config', 'model_name'])
    for (config, model_name), subset in grouped:
        y_true = subset['true_label'].to_numpy()
        y_prob = subset['predicted_prob'].to_numpy()

        # Skip if all probabilities are NaN
        if pd.isna(y_prob).all():
            continue

        # 3) Compute ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        # --- Youden’s J criterion ---
        j_stats = tpr - fpr
        best_idx_j = j_stats.argmax()
        best_thresh_j = thresholds[best_idx_j]
        sens_j = tpr[best_idx_j]
        spec_j = 1.0 - fpr[best_idx_j]

        # Confusion counts at best_thresh_j
        y_pred_j = (y_prob >= best_thresh_j).astype(int)
        tn_j, fp_j, fn_j, tp_j = confusion_matrix(y_true, y_pred_j).ravel()
        ppv_j = tp_j / (tp_j + fp_j) if (tp_j + fp_j) > 0 else 0.0
        # Prevalence‐adjusted PPV at best_thresh_j
        denom_j = (sens_j * prevalence) + ((1 - spec_j) * (1 - prevalence))
        ppvprev_j = (sens_j * prevalence) / denom_j if denom_j > 0 else 0.0

        # --- Constrain FPR to ≤ target_fpr, maximize sensitivity ---
        valid_idx = np.where(fpr <= target_fpr)[0]
        if valid_idx.size > 0:
            best_idx_fpr = valid_idx[np.argmax(tpr[valid_idx])]
        else:
            # If no threshold meets fpr ≤ target_fpr, choose threshold whose FPR is closest
            best_idx_fpr = np.argmin(np.abs(fpr - target_fpr))

        best_thresh_fpr = thresholds[best_idx_fpr]
        sens_fpr = tpr[best_idx_fpr]
        spec_fpr = 1.0 - fpr[best_idx_fpr]

        # Confusion counts at best_thresh_fpr
        y_pred_fpr = (y_prob >= best_thresh_fpr).astype(int)
        tn_fpr, fp_fpr, fn_fpr, tp_fpr = confusion_matrix(y_true, y_pred_fpr).ravel()
        ppv_fpr = tp_fpr / (tp_fpr + fp_fpr) if (tp_fpr + fp_fpr) > 0 else 0.0
        # Prevalence‐adjusted PPV at best_thresh_fpr
        denom_fpr = (sens_fpr * prevalence) + ((1 - spec_fpr) * (1 - prevalence))
        ppvprev_fpr = (sens_fpr * prevalence) / denom_fpr if denom_fpr > 0 else 0.0

        # 4) Append results for this (config, model_name)
        records.append({
            'config': config,
            'model_name': model_name,
            'auc': float(roc_auc),

            'best_threshold_fpr': float(best_thresh_fpr),  # chosen under a false‐positive‐rate constraint
            'sensitivity_fpr': float(sens_fpr),  # chosen under a false‐positive‐rate constraint
            'specificity_fpr': float(spec_fpr),  # chosen under a false‐positive‐rate constraint
            'ppv_fpr': float(ppv_fpr),  # chosen under a false‐positive‐rate constraint
            'ppvprev_fpr': float(ppvprev_fpr),  # chosen under a false‐positive‐rate constraint

            'best_threshold_j': float(best_thresh_j),  # maximizing Youden’s J
            'sensitivity_j': float(sens_j),  # maximizing Youden’s J
            'specificity_j': float(spec_j),  # maximizing Youden’s J
            'ppv_j': float(ppv_j),  # maximizing Youden’s J
            'ppvprev_j': float(ppvprev_j),  # maximizing Youden’s J

        })

    # 5) Build DataFrame and sort
    df_best = pd.DataFrame.from_records(records)

    # Sort within each config by specificity_j (descending)
    df_best = df_best.sort_values(
        by=['config', 'specificity_j'],
        ascending=[True, False]
    ).reset_index(drop=True)

    return df_best


def plot_model_logits_distribution_by_config(
    df: pd.DataFrame,
    model_name: str,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 12),
    bins: int = 30
) -> None:
    """
    For a given model_name, create a grid of subplots—one cell per feature set (config).
    Each cell contains two small subplots side-by-side:
      (a) A scatter of predicted probabilities (logits) for the 'full' dataset,
          colored by true_label (0=Control, 1=NT1), with a horizontal line at the best threshold.
      (b) A histogram showing the distribution of predicted probabilities for Control vs. NT1,
          with a vertical line at the same best threshold.

    The grid dimensions are chosen to be as square as possible based on the number of configs.

    Expects df to have columns:
      - 'config'         (feature‐set name)
      - 'model_name'     (string)
      - 'dataset_type'   (string; e.g. 'full', 'validation')
      - 'true_label'     (0 or 1)
      - 'predicted_prob' (float between 0 and 1)

    Args:
        df (pd.DataFrame):
            A “long” DataFrame where each row is one sample’s prediction for a given model/config.
            Required columns: ['config', 'model_name', 'dataset_type', 'true_label', 'predicted_prob'].

        model_name (str):
            The name of the model (e.g., 'Elastic Net') for which to plot all configs.

        output_path (Path | None):
            If provided, save the combined figure to this path (PNG). Otherwise, display it.

        figsize (tuple):
            Overall figure size in inches, e.g. (12, 12). Height will scale by the number of rows.

        bins (int):
            Number of bins for each histogram subplot.
    """
    # Filter to 'full' dataset and specified model_name
    df_model = df[
        (df['dataset_type'] == 'full') &
        (df['model_name'] == model_name)
    ].copy()

    configs = df_model['config'].unique()
    if len(configs) == 0:
        raise ValueError(f"No data for model '{model_name}' on dataset_type='full'.")

    n_configs = len(configs)
    # Compute grid size: ncols ~ sqrt(n_configs), nrows appropriately
    ncols = math.ceil(math.sqrt(n_configs))
    nrows = math.ceil(n_configs / ncols)

    fig = plt.figure(figsize=figsize)
    outer_grid = GridSpec(nrows,
                          ncols,
                          figure=fig,
                          # wspace=0.1,
                          # hspace=0.1
                          )

    for idx, config in enumerate(configs):
        # Determine row/col in the outer grid
        row = idx // ncols
        col = idx % ncols
        # Create a 1x2 sub‐grid within this outer cell
        inner_grid = GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer_grid[row, col], wspace=0.3, hspace=0.0
        )

        # Axes for scatter (left) and histogram (right)
        ax_scatter = fig.add_subplot(inner_grid[0, 0])
        ax_hist    = fig.add_subplot(inner_grid[0, 1])

        # Subset to this config & model on the full dataset
        df_fc = df_model[df_model['config'] == config]
        if df_fc.empty:
            ax_scatter.axis('off')
            ax_hist.axis('off')
            continue

        y_true = df_fc['true_label'].to_numpy()
        y_prob = df_fc['predicted_prob'].to_numpy()

        # Compute ROC + best threshold
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        j_stats = tpr - fpr
        best_idx = j_stats.argmax()
        best_thresh = thresholds[best_idx]

        # --- Scatter subplot ---
        sorted_idx = np.argsort(y_prob)
        y_prob_sorted = y_prob[sorted_idx]
        y_true_sorted = y_true[sorted_idx]
        colors = np.where(y_true_sorted == 1, 'tab:orange', 'tab:blue')

        ax_scatter.scatter(
            np.arange(len(y_prob_sorted)),
            y_prob_sorted,
            c=colors,
            edgecolors='k',
            alpha=0.7
        )
        ax_scatter.axhline(
            best_thresh,
            color='red',
            linestyle='--',
            linewidth=1.5
        )
        ax_scatter.set_ylim([-0.02, 1.02])
        ax_scatter.set_xticks([])
        ax_scatter.set_ylabel("Pred. Prob.", fontsize=9)
        ax_scatter.set_title(f"{config}\n(AUC={roc_auc:.3f})", fontsize=10)

        # --- Histogram subplot ---
        prob_control = y_prob[y_true == 0]
        prob_nt1     = y_prob[y_true == 1]

        ax_hist.hist(
            prob_control,
            bins=bins,
            density=True,
            alpha=0.5,
            color='tab:blue',
            label='Control'
        )
        ax_hist.hist(
            prob_nt1,
            bins=bins,
            density=True,
            alpha=0.5,
            color='tab:orange',
            label='NT1'
        )
        ax_hist.axvline(
            best_thresh,
            color='red',
            linestyle='--',
            linewidth=1.5
        )
        ax_hist.set_xlim([-0.02, 1.02])
        ax_hist.set_xlabel("Pred. Prob.", fontsize=9)
        ax_hist.set_ylabel("Density", fontsize=9)
        ax_hist.legend(fontsize=8)

    plt.suptitle(f"{model_name} — Full Dataset Logits & Distributions by Feature Set", fontsize=14, y=1.02)
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    plt.show()


if __name__ == '__main__':
    TEST = True
    # %% input paths 
    test_flag = 'test_' if TEST else ''
    base_path = config.get('results_path').get('results')
    # pred and prob of full dataset concat with k-folds
    path_models_pred_prob            = base_path.joinpath(f'{test_flag}pred_prob_all_models.csv')
    path_output = base_path.joinpath(f'poster_ppt')
    path_output.mkdir(parents=True, exist_ok=True)
    # %%
    df_all_model_probs = pd.read_csv(path_models_pred_prob)


    # model_names = df_all_model_probs['model_name'].unique()
    # dataset_type = 'full'
    # mask = (
    #         (df_all_model_probs['model_name'] == model_names[0])
    #         &
    #         (df_all_model_probs['dataset_type'] == dataset_type)
    # )
    # filtered = df_all_model_probs[mask]

    # %% Ful dataset best thresholds

    def plot_roc_curves(
            df: pd.DataFrame,
            zoom: Optional[Tuple[float, float]] = (1.02, -0.02),
            output_path: Optional[Path] = None,
            dataset_type: str = 'full',
            target_fpr: float = 0.01,
            figsize: Tuple[int, int] = (10, 8),
            fontsize_legend:int=11,
            fontsize_lbl:int=16,
    ) -> None:
        """
        Plot ROC curves for each unique 'model_name' in `df` for each feature set ('config'),
        circle the point corresponding to the threshold that maximizes Youden’s J (tpr - fpr),
        and mark with an 'X' the point corresponding to the threshold that enforces FPR ≤ target_fpr.
        Include both threshold values in the legend.

        Expects `df` to have columns:
          - 'config'         (feature‐set name)
          - 'model_name'     (string)
          - 'true_label'     (0/1)
          - 'predicted_prob' (float in [0, 1])
          - 'dataset_type'   (e.g., 'full', 'validation', etc.)

        Args:
            df (pd.DataFrame):
                A “long” DataFrame where each row corresponds to one sample's
                prediction for a particular model/config. Required columns:
                ['config', 'model_name', 'dataset_type', 'true_label', 'predicted_prob'].

            zoom (tuple):
                (x_max, y_min) to restrict ROC axes. E.g. zoom=(0.1, 0.9) sets
                xlim=[0, 0.1], ylim=[0.9, 1.02].

            output_path (Path | None):
                If given, save the figure as PNG to this path; otherwise, just show it.

            dataset_type (str):
                Which subset of df to plot (usually 'full').

            target_fpr (float):
                Maximum allowed FPR (e.g., 0.01 for 1%). Marks with 'X' the point where
                FPR ≤ target_fpr and TPR is maximized.

            figsize (tuple):
                Figure size in inches, e.g. (10, 8).
        """
        configs = df['config'].unique()
        n = len(configs)
        if n == 0:
            return

        # Determine grid dimensions: ncols = ceil(sqrt(n)), nrows = ceil(n / ncols)
        ncols = math.ceil(math.sqrt(n))
        nrows = math.ceil(n / ncols)

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=figsize,
            squeeze=False
        )

        for idx, feature_set in enumerate(configs):
            row = idx // ncols
            col = idx % ncols
            ax = axes[row][col]

            # Subset to this feature_set and dataset_type
            subset_fs = df[(df['config'] == feature_set) & (df['dataset_type'] == dataset_type)]
            if subset_fs.empty:
                ax.axis('off')
                continue


            for model in subset_fs['model_name'].unique():
                subset = subset_fs[subset_fs['model_name'] == model]
                y_true = subset['true_label'].to_numpy()
                y_prob = subset['predicted_prob'].to_numpy()
                N = len(y_true)
                n_cases = int(y_true.sum())
                n_controls = N - n_cases

                if np.isnan(y_prob).all():
                    continue

                # Compute ROC + AUC
                fpr, tpr, thresholds = roc_curve(y_true, y_prob)
                roc_auc = auc(fpr, tpr)

                # Youden’s J = tpr – fpr → best index
                j_stats = tpr - fpr
                best_idx_j = j_stats.argmax()
                best_fpr_j = fpr[best_idx_j]
                best_tpr_j = tpr[best_idx_j]
                best_thresh_j = thresholds[best_idx_j]

                # FPR‐constrained threshold: max sensitivity with fpr ≤ target_fpr
                valid_idx = np.where(fpr <= target_fpr)[0]
                if valid_idx.size > 0:
                    best_idx_fpr = valid_idx[np.argmax(tpr[valid_idx])]
                else:
                    best_idx_fpr = np.argmin(np.abs(fpr - target_fpr))
                best_fpr_fpr = fpr[best_idx_fpr]
                best_tpr_fpr = tpr[best_idx_fpr]
                best_thresh_fpr = thresholds[best_idx_fpr]

                # Plot ROC line
                line, = ax.plot(
                    fpr, tpr,
                    lw=1.4,
                    label=(
                        f"{model} (AUC={roc_auc:.3f}, "
                        f"Jthr (o)={best_thresh_j:.3f}, "
                        f"FPRthr (*)={best_thresh_fpr:.3f})"
                    ),
                    alpha=0.7,
                )
                color = line.get_color()

                # Circle Youden‐J optimum
                ax.scatter(
                    [best_fpr_j],
                    [best_tpr_j],
                    s=80,
                    facecolors='none',
                    edgecolors=color,
                    linewidths=1.3
                )
                # Mark FPR‐constrained point with 'X'
                ax.scatter(
                    [best_fpr_fpr],
                    [best_tpr_fpr],
                    marker='*',
                    c=color,
                    s=80,
                    linewidths=1.3
                )

            # Diagonal random‐guess line
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=1)

            # Axis limits & labels
            ax.set_xlim([0, zoom[0]])
            ax.set_ylim([zoom[1], 1.02])
            ax.set_xlabel("False Positive Rate", fontsize=fontsize_lbl)
            ax.set_ylabel("True Positive Rate", fontsize=fontsize_lbl)
            ax.set_title(
                f"{feature_set}\nN={N}; NT1={n_cases}; Controls={n_controls}",
                fontsize=12
            )
            ax.legend(loc="lower right", fontsize=fontsize_legend)
            ax.grid(alpha=0.6)

        # Hide any unused subplots
        total_plots = nrows * ncols
        for empty_idx in range(n, total_plots):
            r = empty_idx // ncols
            c = empty_idx % ncols
            axes[r][c].axis('off')

        plt.tight_layout()

        if output_path is not None:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')

        plt.show()

    plot_roc_curves(df_all_model_probs,
                    output_path=path_output.joinpath(f'roc_curve.png'),
                    target_fpr=0.04,
                    zoom=(0.06, 0.9),
                    figsize=(16, 12),
                    fontsize_legend=11,
                    fontsize_lbl=16)

    # %% Visualize the logits
    plot_model_logits_distribution_by_config(
        df=df_all_model_probs,
        model_name='Elastic Net',
        output_path=None,
        figsize=(18, 10),
        bins=50
    )
    # %% =====================================================================
    # compute the best treshold
    # =======================================================================
    best_thresholds_max_spec = compute_best_thresholds(df=df_all_model_probs,
                                                       target_fpr=0.004,  # fewer false positives (higher specificity).
                                                       prevalence= 30 / 100000,
                                                       dataset_type='full')

    # print(tabulate(best_thresholds_max_spec, headers=best_thresholds_max_spec.columns))

    print(
        tabulate(
            best_thresholds_max_spec,
            headers=best_thresholds_max_spec.columns.tolist(),
            tablefmt="psql",
            showindex=False
        )
    )


    # %% Cross validations
    def plot_roc_curves_cv(
            df: pd.DataFrame,
            output_dir: Optional[Path] = None,
            zoom: Tuple[float, float] = (1.02, -0.02),
            figsize: Tuple[int, int] = (10, 8)
    ) -> None:
        """
        For each cross-validation fold in `df`, create one figure containing a grid of subplots—
        one subplot per feature set (df['config']). Within each subplot, plot ROC curves for all
        models in that feature set/fold (dataset_type='validation'), circle the “best” threshold
        (max Youden’s J), and include that threshold in the legend.

        Expects `df` with columns:
          - 'config'         (feature-set name, e.g. "questionnaire", "ukbb", etc.)
          - 'model_name'     (string)
          - 'true_label'     (0/1)
          - 'predicted_prob' (float in [0, 1])
          - 'dataset_type'   (e.g. 'validation' or 'full')
          - 'fold'           (int)

        Args:
            df (pd.DataFrame):
                “Long” DataFrame where each row is one sample’s prediction for a given
                (config, model, fold). Required columns:
                    ['config', 'model_name', 'true_label', 'predicted_prob', 'dataset_type', 'fold'].

            output_dir (Path | None):
                If provided, save each fold’s figure as a PNG into this directory (named
                "roc_fold_{fold}.png"). If None, just show the figures interactively.

            zoom (tuple):
                (x_max, y_min) to restrict axis ranges. E.g. zoom=(0.1, 0.9) → xlim=[0,0.1], ylim=[0.9,1.02].

            figsize (tuple):
                Base figure size (width, height). The function will multiply height
                by the number of grid rows to keep subplots readable.
        """
        # Identify unique validation folds
        folds = sorted(df.loc[df['dataset_type'] == 'validation', 'fold_number'].dropna().unique())
        if not folds:
            return

        # Identify all feature sets
        configs = df['config'].unique()

        for fold in folds:
            # Subset to this fold and validation set
            df_fold = df[(df['dataset_type'] == 'validation') & (df['fold_number'] == fold)]
            if df_fold.empty:
                continue

            n = len(configs)
            ncols = math.ceil(math.sqrt(n))
            nrows = math.ceil(n / ncols)

            # Scale figure height by nrows
            width, height = figsize
            fig, axes = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=(width, height * nrows),
                squeeze=False
            )

            for idx, feature_set in enumerate(configs):
                row = idx // ncols
                col = idx % ncols
                ax = axes[row][col]

                subset_fs = df_fold[df_fold['config'] == feature_set]
                if subset_fs.empty:
                    # If no data for this feature_set in this fold, hide axis
                    ax.axis('off')
                    continue

                for model in subset_fs['model_name'].unique():
                    sub = subset_fs[subset_fs['model_name'] == model]
                    y_true = sub['true_label'].to_numpy()
                    y_prob = sub['predicted_prob'].to_numpy()

                    if np.isnan(y_prob).all():
                        continue

                    # Compute ROC + AUC
                    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
                    roc_auc = auc(fpr, tpr)

                    # Youden’s J = tpr – fpr → find best index
                    j_stats = tpr - fpr
                    best_idx = j_stats.argmax()
                    best_fpr = fpr[best_idx]
                    best_tpr = tpr[best_idx]
                    best_thresh = thresholds[best_idx]

                    # Interpolate for a smoother line
                    fpr_smooth = np.linspace(0, 1, 200)
                    tpr_smooth = np.interp(fpr_smooth, fpr, tpr)

                    # Plot the smoothed ROC line
                    line, = ax.plot(
                        fpr_smooth, tpr_smooth,
                        lw=1.4,
                        label=f"{model} (AUC={roc_auc:.3f}, thr={best_thresh:.3f})"
                    )
                    color = line.get_color()

                    # Circle best-threshold point (using original fpr/tpr)
                    ax.scatter(
                        [best_fpr],
                        [best_tpr],
                        s=80,
                        facecolors='none',
                        edgecolors=color,
                        linewidths=1.3
                    )

                # Plot random‐guess diagonal
                ax.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=1)

                # Axis limits & labels
                ax.set_xlim([0, zoom[0]])
                ax.set_ylim([zoom[1], 1.02])
                ax.set_xlabel("False Positive Rate", fontsize=10)
                ax.set_ylabel("True Positive Rate", fontsize=10)
                ax.set_title(f"Fold {fold}\n{feature_set}'", fontsize=12)
                ax.legend(loc="lower right", fontsize=8)
                ax.grid(alpha=0.6)

            # Hide any empty subplots
            total_plots = nrows * ncols
            for empty_idx in range(len(configs), total_plots):
                r = empty_idx // ncols
                c = empty_idx % ncols
                axes[r][c].axis('off')

            plt.tight_layout()

            if output_dir is not None:
                output_dir.mkdir(parents=True, exist_ok=True)
                save_path = output_dir / f"roc_fold_{fold}.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')

            plt.show()


    plot_roc_curves_cv(df_all_model_probs,
                    output_dir=None,
                    zoom=(0.1, 0.9),
                    figsize=(12, 6))



    # %% =====================================================================
    # compute final metrics with best tresholds computes from the full dataset
    # =======================================================================
    def _compute_metrics(y_pred: np.ndarray,
                         y_true: np.ndarray,
                         prevalence: float = None) -> Dict[str, float]:
        """
        Compute classification metrics (no CIs) given binary predictions and true labels.
        """
        if prevalence is None:
            prevalence = 30 / 100000

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        # Prevalence-adjusted PPV
        denominator = (sensitivity * prevalence) + ((1 - specificity) * (1 - prevalence))
        ppv = (sensitivity * prevalence) / denominator if denominator > 0 else 0

        return {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'accuracy': accuracy,
            'ppv_apparent': precision,  # “apparent” PPV = precision on fold
            'f1_score': f1_score,
            'npv': npv,
            'fpr': fpr,
            'fnr': fnr,
            'ppv': ppv  # prevalence-adjusted
        }


    def _compute_confidence_interval(values: List[float]) -> str:
        """
        Given a list of metric values (one per fold), compute and return a 95% CI string.
        Format: "mean\n(lower, upper)" with four decimal places.
        """
        arr = np.array(values)
        mean_val = arr.mean()
        std_err = arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0
        margin = norm.ppf(0.975) * std_err
        lower = max(0.0, mean_val - margin)
        upper = min(1.0, mean_val + margin)
        return f"{mean_val:.4f}\n({lower:.4f}, {upper:.4f})"


    def compute_metrics_across_folds_with_best_thresholds(
            df_probs: pd.DataFrame,
            best_thresholds: pd.DataFrame,
            best_threshold_col:str='best_threshold_fpr',
    ) -> pd.DataFrame:
        """
        Re-compute metrics (with 95% CIs) across validation folds for each (config, model_name),
        using the best_threshold found on the 'full' dataset. Returns a DataFrame with one row
        per (config, model_name) and columns:
            - config
            - model_name
            - sensitivity_ci
            - specificity_ci
            - accuracy_ci
            - ppv_apparent_ci
            - f1_score_ci
            - npv_ci
            - fpr_ci
            - fnr_ci
            - ppv_ci

        Args:
            df_probs: DataFrame with columns [
                'config', 'model_name', 'dataset_type', 'fold',
                'true_label', 'predicted_prob', ...
            ] including cross-validation folds.

            best_thresholds: DataFrame from compute_best_thresholds(), with columns [
                'config', 'model_name', 'best_threshold', 'auc'
            ] computed on dataset_type == 'full'.

        Returns:
            pd.DataFrame with metrics+CI per (config, model_name).
        """
        records: List[Dict] = []

        # Filter to validation rows only
        df_val = df_probs[df_probs['dataset_type'] == 'validation'].copy()

        # Merge best_threshold onto df_val so each row has its config/model's threshold
        df_merged = pd.merge(
            df_val,
            best_thresholds[['config', 'model_name', best_threshold_col]],
            on=['config', 'model_name'],
            how='left'
        )

        # Drop any rows where best_threshold is missing (e.g. no full-data threshold)
        df_merged = df_merged.dropna(subset=[best_threshold_col])

        # Now group by config & model_name & fold to compute metrics per fold
        grouped_folds = df_merged.groupby(['config', 'model_name', 'fold_number'])
        # We will collect metrics per-fold in a nested dict: metrics_per_group[(config,model)][metric] = [list of values]
        metrics_per_group: Dict[Tuple[str, str], Dict[str, List[float]]] = {}

        for (config, model_name, fold), sub in grouped_folds:
            y_true = sub['true_label'].to_numpy()
            y_prob = sub['predicted_prob'].to_numpy()
            threshold = sub[best_threshold_col].iloc[0]  # same for all rows in this group

            # Convert probabilities to binary predictions using threshold
            y_pred = (y_prob >= threshold).astype(int)

            # Compute metrics on this fold
            m = _compute_metrics(y_pred, y_true)

            key = (config, model_name)
            if key not in metrics_per_group:
                # initialize lists for each metric
                metrics_per_group[key] = {
                    'sensitivity': [],
                    'specificity': [],
                    'accuracy': [],
                    'ppv_apparent': [],
                    'f1_score': [],
                    'npv': [],
                    'fpr': [],
                    'fnr': [],
                    'ppv': []
                }

            # append this fold's metrics
            for metric_name, metric_val in m.items():
                metrics_per_group[key][metric_name].append(metric_val)

        # Now, for each (config, model_name), compute mean+CI for each metric
        for (config, model_name), metric_lists in metrics_per_group.items():
            record: Dict[str, str] = {
                'config': config,
                'model_name': model_name
            }
            # Compute CI for each metric
            for metric_name, values in metric_lists.items():
                ci_str = _compute_confidence_interval(values)
                record[f'{metric_name}_ci'] = ci_str

            records.append(record)

        return pd.DataFrame.from_records(records)


    df_metrics_with_ci = compute_metrics_across_folds_with_best_thresholds(
        df_probs=df_all_model_probs,
        best_thresholds=best_thresholds_max_spec
    )

    print(
        tabulate(
            df_metrics_with_ci,
            headers=df_metrics_with_ci.columns.tolist(),
            tablefmt="psql",
            showindex=False
        )
    )

    # bar plot
    def _parse_ci_string(ci_str: str) -> Tuple[float, float, float]:
        """
        Given a CI string of the form "mean\n(lower, upper)", parse and return
        (mean, lower, upper) as floats.
        """
        # Split on newline
        mean_part, range_part = ci_str.strip().split('\n')
        mean_val = float(mean_part)
        # Remove parentheses and split on comma
        range_part = range_part.strip().strip('()')
        lower_str, upper_str = [s.strip() for s in range_part.split(',')]
        lower_val = float(lower_str)
        upper_val = float(upper_str)
        return mean_val, lower_val, upper_val



    def plot_model_metrics_with_ci(
            df: "pd.DataFrame",
            metrics: List[str],
            palette: Optional[str] = 'muted',
            figsize: Optional[Tuple[int, int]] = (16, 8),
            output_path: Optional[str] = None
    ) -> None:
        """
        Plot specified metrics (with 95% CIs) for each model and configuration.
        Bars represent mean*100, error bars represent CI range*100.
        All annotations in a given config sit at the same height (group_max + 0.5),
        while the y-axis bottom is set via np.min(mean_min).

        df must contain:
          - 'config'       (feature-set name)
          - 'model_name'   (model name)
          - for each metric in `metrics`, a "<metric>_ci" column formatted as "mean\n(lower, upper)"

        Args:
          df (pd.DataFrame)
          metrics (List[str]): e.g. ['sensitivity', 'specificity', 'ppv', 'f1_score']
          palette (str):       seaborn palette name
          figsize (tuple):     (width, height) inches per subplot
          output_path (str|None): if provided, save to this path; otherwise show()
        """

        # 1) Check required columns
        if 'model_name' not in df.columns or 'config' not in df.columns:
            raise ValueError("DataFrame must contain 'model_name' and 'config' columns.")

        # 2) Sort for consistent ordering
        df = df.sort_values(by=['model_name', 'config']).reset_index(drop=True)

        # 3) Extract unique configs & models
        configs = sorted(df['config'].unique())
        models = sorted(df['model_name'].unique())

        # 4) Parse CI columns into a nested dict:
        #    data_dict[metric][(config, model)] = (mean*100, lower*100, upper*100)
        data_dict: Dict[str, Dict[Tuple[str, str], Tuple[float, float, float]]] = {}
        for metric in metrics:
            ci_col = f"{metric}_ci"
            if ci_col not in df.columns:
                raise ValueError(f"Column '{ci_col}' not found in DataFrame.")
            data_dict[metric] = {}
            for _, row in df.iterrows():
                cfg = row['config']
                mdl = row['model_name']
                ci_str = row[ci_col]
                if isinstance(ci_str, str) and ci_str.strip():
                    mean_val, low_val, high_val = _parse_ci_string(ci_str)
                    data_dict[metric][(cfg, mdl)] = (
                        mean_val * 100.0,
                        low_val * 100.0,
                        high_val * 100.0
                    )
                else:
                    data_dict[metric][(cfg, mdl)] = (0.0, 0.0, 0.0)

        # 5) Collect all lower/upper bounds to set y-axis bottom (if needed elsewhere)
        all_lowers, all_uppers = [], []
        for metric in metrics:
            for (cfg, mdl), (m_val, l_val, u_val) in data_dict[metric].items():
                all_lowers.append(l_val)
                all_uppers.append(u_val)
        global_min_lower = min(all_lowers) if all_lowers else 0.0
        # (We will compute mean_min per subplot instead for setting ylim.)

        # 6) Prepare figure with one subplot per metric
        n_metrics = len(metrics)
        fig, axes = plt.subplots(
            nrows=n_metrics,
            ncols=1,
            figsize=(figsize[0], figsize[1] * n_metrics),
            squeeze=False
        )
        sns.set_theme(style="whitegrid", context="talk", palette=palette)

        # 7) Assign colors
        palette_colors = sns.color_palette(palette, n_colors=len(models))
        model_to_color = {models[i]: palette_colors[i] for i in range(len(models))}

        # 8) Loop over each metric
        for i, metric in enumerate(metrics):
            ax = axes[i][0]
            bar_width = 0.8 / len(models)

            # Build mean_min list so we can set the bottom of y-axis later
            mean_min = []

            # Precompute “group_max” for each config so that all annotations align
            group_max: Dict[str, float] = {}
            group_min: Dict[str, float] = {}
            for cfg in configs:
                max_for_cfg = max(
                    data_dict[metric].get((cfg, mdl), (0.0, 0.0, 0.0))[0]
                    for mdl in models
                )
                min_for_cfg = min(
                    data_dict[metric].get((cfg, mdl), (0.0, 0.0, 0.0))[0]
                    for mdl in models
                )
                group_max[cfg] = max_for_cfg
                group_min[cfg] = min_for_cfg - 0.5

            # X‐axis positions
            x_indices = np.arange(len(configs))

            # Now draw each model’s bars and label them at group_max
            for j, model in enumerate(models):
                means, err_l, err_u = [], [], []
                for cfg in configs:
                    m_val, l_val, u_val = data_dict[metric].get((cfg, model), (0.0, 0.0, 0.0))
                    means.append(m_val)
                    err_l.append(m_val - l_val)
                    err_u.append(u_val - m_val)

                # Keep track of the minimum mean across all bars (for y-axis bottom)
                mean_min.append(np.min(means))

                # Compute bar positions
                offsets = (j - (len(models) - 1) / 2) * bar_width
                bar_positions = x_indices + offsets

                # Draw bars
                ax.bar(
                    bar_positions,
                    means,
                    width=bar_width,
                    color=model_to_color[model],
                    edgecolor='black',
                    label=model
                )
                # Draw error bars
                ax.errorbar(
                    bar_positions,
                    means,
                    yerr=np.vstack([err_l, err_u]),
                    fmt='none',
                    ecolor='black',
                    capsize=5,
                    linewidth=1.2
                )

                # Annotate each bar at the group_max height (so all labels in this config align)
                for idx_cfg, xpos in enumerate(bar_positions):
                    cfg_name = configs[idx_cfg]
                    y_annot = group_max[cfg_name] + 0.5
                    ax.text(
                        xpos,
                        y_annot,
                        f"{means[idx_cfg]:.3f}",
                        ha='center',
                        va='bottom',
                        fontsize=12,
                        fontweight='bold',
                        color='black',
                        bbox={
                            'facecolor': 'white',
                            'edgecolor': 'none',
                            'pad': 2.6
                        }
                    )

            # 9) Now set y-axis limits exactly as requested
            ax.set_xticks(x_indices)
            ax.set_xticklabels(configs, rotation=0, ha='center', fontsize=20)
            ax.set_ylabel(f"{metric.replace('_', ' ').upper()} (%)", fontsize=16)
            # Bottom of y-axis = min(mean_min); Top = 100.5
            ax.set_ylim(np.min(mean_min)-4, 100)
            # ax.set_ylim(np.min(group_min[cfg_name]), 100)
            ax.set_title(f"{metric.replace('_', ' ').upper()}", fontsize=24)
            ax.grid(axis='y', alpha=0.4)

        # 10) Shared legend at top
        handles = [Line2D([0], [0], color=model_to_color[m], lw=4) for m in models]
        fig.legend(
            handles,
            models,
            loc='upper center',
            bbox_to_anchor=(0.5, 1.0),
            ncol=len(models),
            frameon=False,
            fontsize=20
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.show()


    df_metrics_with_ci_plot = df_metrics_with_ci.copy()
    df_metrics_with_ci_plot['config'] = df_metrics_with_ci_plot['config'].str.replace('+', '+\n')
    df_metrics_with_ci_plot = df_metrics_with_ci_plot.loc[df_metrics_with_ci_plot['model_name'] != 'LogReg (ESS Only)', :]
    plot_model_metrics_with_ci(
        df=df_metrics_with_ci_plot,
        metrics=['sensitivity', 'specificity', 'ppv', 'f1_score'],
        palette='pastel',
        figsize=(24, 5),
        output_path=path_output.joinpath(f'cv_bar_plots.png'), # "model_metrics_with_ci.png"
    )
