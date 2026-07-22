'''
NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECT WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any
from pathlib import Path


class Report:
    """
    Encapsulates the results of a detection limits analysis and provides
    methods to save the results in a structured format.
    """

    def __init__(self):
        # Structure: { "MetricName": { "AccuracyMetric": DataFrame } }
        self.mappings: Dict[str, Dict[str, pd.DataFrame]] = {}
        self.summary_stats: Dict[str, Any] = {}
        self.human_snr_threshold: float = 5.0
        self.log_scale: bool = True

    def set_plot_options(self, human_snr_threshold: float = 5.0, log_scale: bool = True):
        """Store plotting/analysis options used when saving report artifacts."""
        self.human_snr_threshold = float(human_snr_threshold)
        self.log_scale = bool(log_scale)

    def add_mapping(self, metric_name: str, accuracy_metric: str, df: pd.DataFrame):
        """Add a mapping between an SNR metric and an AI accuracy metric."""
        if metric_name not in self.mappings:
            self.mappings[metric_name] = {}
        self.mappings[metric_name][accuracy_metric] = df

    def save(
        self,
        output_path: str,
        save_plots: bool = True,
        save_csvs: bool = False,
        legacy_layout: bool = False,
    ):
        """
        Saves the report results to the specified output path.

        Args:
            output_path (str): Path to the directory where the report will be saved.
            save_plots (bool): Whether to save plots.
            save_csvs (bool): Whether to save mapping CSVs. Disabled by default
                to keep looped/library use efficient and in-memory.
            legacy_layout (bool): Save files in the historic flat layout
                expected by older workflows.
        """
        out_dir = Path(output_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save individual CSV mappings.
        if save_csvs:
            csv_dir = out_dir if legacy_layout else (out_dir / "mappings")
            if not legacy_layout:
                csv_dir.mkdir(exist_ok=True)
            for metric_name, acc_map in self.mappings.items():
                for acc_metric, df in acc_map.items():
                    filename = f"{metric_name}_vs_{acc_metric}.csv"
                    df.to_csv(csv_dir / filename, index=False)

        # Save individual PNG mappings.
        if save_plots:
            self._save_individual_figures(out_dir)

        # Save a compact summary figure for quick overview.
        if save_plots and not legacy_layout:
            self._generate_summary_figure(out_dir / "summary_report.png")

        self._compute_summary_stats()

        print(f"INFO: Report saved successfully to {out_dir}")

    def _generate_summary_figure(self, figure_path: Path):
        """Generates a single combined figure containing all mappings."""
        if not self.mappings:
            return

        metrics = list(self.mappings.keys())
        acc_metrics = list(next(iter(self.mappings.values())).keys())

        num_metrics = len(metrics)
        num_acc = len(acc_metrics)

        fig, axes = plt.subplots(num_metrics, num_acc,
                                 figsize=(4 * num_acc, 4 * num_metrics),
                                 squeeze=False)

        for i, metric_name in enumerate(metrics):
            for j, acc_metric in enumerate(acc_metrics):
                ax = axes[i, j]
                df = self.mappings[metric_name].get(acc_metric)

                if df is not None:
                    # Assuming the first column is the accuracy metric and second is the SNR metric
                    x = df.iloc[:, 0]
                    y = df.iloc[:, 1]
                    y_to_plot = self._transform_y_values(y)
                    ax.set_xlabel(acc_metric)
                    ax.plot(x, y_to_plot, 'o-', markersize=4, label='Data')
                    ax.axhline(
                        self._transform_scalar(self.human_snr_threshold),
                        color='red',
                        linestyle='--',
                        linewidth=1,
                        label=f"Human SNR threshold={self.human_snr_threshold:g}",
                    )
                    if self.log_scale:
                        ax.set_ylabel(f"ln({metric_name})")
                        ax.set_title(f"ln({metric_name}) vs {acc_metric}")
                    else:
                        ax.set_ylabel(metric_name)
                        ax.set_title(f"{metric_name} vs {acc_metric}")
                    ax.grid(True, linestyle='--', alpha=0.7)
                    ax.legend(fontsize=8)
                else:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center')

        plt.tight_layout()
        plt.savefig(figure_path)
        plt.close()

    def _save_individual_figures(self, output_dir: Path):
        """Save one figure per metric/accuracy pair, including std error bars."""
        for metric_name, acc_map in self.mappings.items():
            for acc_metric, df in acc_map.items():
                plt.figure()
                x = df.iloc[:, 0]
                y = df.iloc[:, 1]
                yerr = df.iloc[:, 2] if df.shape[1] > 2 else None
                y_to_plot = self._transform_y_values(y)

                if yerr is not None and self.log_scale:
                    safe_y = np.where(y.to_numpy() > 0, y.to_numpy(), 1e-10)
                    yerr = np.asarray(yerr)
                    yerr = np.where(yerr >= safe_y, safe_y * 0.95, yerr)
                    yerr_to_plot = np.log(safe_y + yerr) - np.log(safe_y)
                else:
                    yerr_to_plot = yerr

                if yerr is not None:
                    plt.errorbar(
                        x,
                        y_to_plot,
                        yerr=yerr_to_plot,
                        fmt='o-',
                        capsize=5,
                        ecolor='black',
                        markerfacecolor='blue',
                        markersize=4,
                        label='Data with std dev',
                    )
                else:
                    plt.plot(x, y_to_plot, 'o-', markersize=4)

                plt.axhline(
                    self._transform_scalar(self.human_snr_threshold),
                    color='red',
                    linestyle='--',
                    linewidth=1,
                    label=f'Human SNR threshold={self.human_snr_threshold:g}',
                )

                plt.xlabel(acc_metric)
                if self.log_scale:
                    plt.ylabel(f"ln({metric_name})")
                    plt.title(f"ln({metric_name})=f(AI model {acc_metric})")
                else:
                    plt.ylabel(metric_name)
                    plt.title(f"{metric_name}=f(AI model {acc_metric})")
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend()
                plt.tight_layout()

                figure_filename = f"{metric_name}_vs_{acc_metric}.png"
                plt.savefig(output_dir / figure_filename)
                plt.close()

    def _transform_y_values(self, values):
        """Apply configured y-axis transform for SNR values."""
        if not self.log_scale:
            return values
        safe_values = np.where(np.asarray(values) > 0, np.asarray(values), 1e-10)
        return np.log(safe_values)

    def _transform_scalar(self, value: float) -> float:
        if not self.log_scale:
            return value
        return float(np.log(value if value > 0 else 1e-10))

    def _compute_summary_stats(self):
        """Compute threshold-oriented summary metrics for each mapping."""
        summary = {}
        for metric_name, acc_map in self.mappings.items():
            summary[metric_name] = {}
            for acc_metric, df in acc_map.items():
                x = df.iloc[:, 0]
                y = df.iloc[:, 1]
                idx = (y - self.human_snr_threshold).abs().idxmin()
                summary[metric_name][acc_metric] = {
                    "human_snr_threshold": self.human_snr_threshold,
                    "ai_value_at_threshold": float(x.loc[idx]),
                    "snr_nearest_threshold": float(y.loc[idx]),
                }

        self.summary_stats = summary
