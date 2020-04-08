import datetime
import inspect
import numpy as np
from pyseir.parameters.parameter_ensemble_generator import ParameterEnsembleGenerator
from pyseir.reports.pdf_report import PDFReport
from pyseir import load_data
import matplotlib.pyplot as plt
from pyseir.reports.names import compartment_to_name_map


class CountyReport:
    """
    Generate a county level report showing detailed data on different scenarios.

    Parameters
    ----------
    fips: str
        County fips code.
    model_ensemble: list(SEIRModel)
        Models from the ensemble.
    county_outputs: dict
        Dictionary of structure (in yaml syntax)
            suppression_policy__0.5:
                t_list: array of timestamps
                HICU: For compartment S we have a bunch of properties
                    S_ci50: the median posterior model.  Other condidence intervals are available.
                    ...
                    peak_value_ciXX:
                    peak_time_ciXX:
                    ...
                    # For HICU, HVent, HGen, capacities are relevant, so we also calculate surge windows.
                    capacity: estimated capacity for this compartment.
                    surge_start: array surge window starts for each model in the ensemble
                    surge_end:
    filename: str
        Where to save the report.
    summary: dict
        Summary attrs to print on the first page of the report.
    xlim: tuple
        Limits in days since simulation start to plot.
    """

    def __init__(self, fips, model_ensemble, county_outputs, filename, summary, xlim=(0, 360)):
        self.fips = fips
        self.county_outputs = county_outputs
        self.model_ensemble = model_ensemble
        self.county_metadata = load_data.load_county_metadata_by_fips(fips)
        self.summary = summary
        _county_case_data = load_data.load_county_case_data()
        self.county_case_data = _county_case_data[_county_case_data['fips'] == fips]
        self.report = PDFReport(filename=filename)
        self.xlim = xlim

    def generate_and_save(self):
        """
        Name says it all.
        """
        self.write_header_pages()
        self.plot_seir_distributions()
        self.report.close()

    def write_header_pages(self):
        """
        Generate header pages with the county summary and PDF summaries.
        """
        self.report.write_text_page(self.summary,
                                    title=f'PySEIR COVID19 Estimates\n{self.county_metadata["county"]} County, {self.county_metadata["state"]}',
                                    page_heading=f'Generated {self.summary["date_generated"]}', body_fontsize=6,
                                    title_fontsize=12)

        self.report.write_text_page(inspect.getsource(ParameterEnsembleGenerator.sample_seir_parameters), title='PySEIR Model Ensemble Parameters')

    def plot_seir_distributions(self, xlim=(0, 360)):
        """
        Generate plots for each suppression policy containing distributions and
        peak information for each model compartment.

        Parameters
        ----------
        xlim: tuple
            Limits in days since simulation start to plot.
        """
        # Add a sample model from the ensemble.
        fig = self.model_ensemble[0].plot_results(xlim=xlim)
        fig.suptitle(f'PySEIR COVID19 Estimates: {self.county_metadata["county"]} County, {self.county_metadata["state"]}. 'f'SAMPLE OF MODEL ENSEMBLE', fontsize=16)
        self.report.add_figure(fig)

        suppression_policies = [key for key in self.county_outputs.keys() if key.startswith('suppression_policy')]

        for suppression_policy in suppression_policies:
            output = self.county_outputs[suppression_policy]
            compartments = list(output.keys())
            compartments.remove('t_list')

            # -----------------------------------
            # Plot each compartment distribution
            # -----------------------------------
            fig = plt.figure(figsize=(20, 24))
            fig.suptitle(f'PySEIR COVID19 Estimates: {self.county_metadata["county"]} County, {self.county_metadata["state"]}. '
                         f'\nSupression Policy={suppression_policy} (1=No Suppression)' , fontsize=16)
            for i_plot, compartment in enumerate(compartments):
                plt.subplot(5, 5, i_plot + 1)
                plt.plot(output['t_list'], output[compartment]['ci_50'], color='steelblue',
                         linewidth=3, label=compartment_to_name_map[compartment])
                plt.fill_between(output['t_list'], output[compartment]['ci_32'], output[compartment]['ci_68'], alpha=.3, color='steelblue')
                plt.fill_between(output['t_list'], output[compartment]['ci_5'], output[compartment]['ci_95'], alpha=.3, color='steelblue')
                plt.yscale('log')
                plt.ylim(1e0)
                plt.xlim(0, 360)
                plt.grid(True, which='both', alpha=0.3)
                plt.xlabel('Days Since Case 0')

                # Circular import :(
                from pyseir.ensembles.ensemble_runner import compartment_to_capacity_attr_map
                if compartment in compartment_to_capacity_attr_map:
                    percentiles = np.percentile([getattr(m, compartment_to_capacity_attr_map[compartment]) for m in self.model_ensemble], (5, 32, 50, 68, 95))
                    plt.hlines(percentiles[2], *plt.xlim(), label='ICU Capacity', color='darkseagreen')
                    plt.hlines([percentiles[0], percentiles[4]], *plt.xlim(), color='darkseagreen', linestyles='-.', alpha=.4)
                    plt.hlines([percentiles[1], percentiles[3]], *plt.xlim(), color='darkseagreen', linestyles='--', alpha=.2)


                # Plot data
                if compartment in ['D', 'total_deaths'] and len(self.county_case_data) > 0:
                    plt.errorbar((self.county_case_data.date - self.summary['t0']).dt.days,
                                  self.county_case_data.deaths, yerr=np.sqrt(self.county_case_data.deaths),
                             linestyle='-', label='Deaths Observed', marker='o', markersize=4)
                if compartment in ['I'] and len(self.county_case_data) > 0:
                    plt.errorbar((self.county_case_data.date - self.summary['t0']).dt.days,
                             self.county_case_data.cases, yerr=np.sqrt(self.county_case_data.cases  ), linestyle='-',
                             label='Cases Observed', marker='o', markersize=4, color='firebrick')

                plt.legend()
                self._plot_dates(log=False)

            # -----------------------------
            # Plot peak Timing
            # -----------------------------
            color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color'] + list('bgrcmyk')
            marker_cycle = ['o', 's', '+', 'd', 'o'] * 4

            plt.subplot(5, 5, len(compartments) + 1)
            for i, compartment in enumerate(['E', 'A', 'I', 'HGen', 'HICU', 'HVent', 'general_admissions_per_day',
                                             'icu_admissions_per_day', 'direct_deaths_per_day', 'total_deaths_per_day']):
                median = output[compartment]['peak_time_ci50']
                ci5, ci95 = output[compartment]['peak_time_ci5'], output[compartment]['peak_time_ci95']
                ci32, ci68 = output[compartment]['peak_time_ci32'], output[compartment]['peak_time_ci68']
                plt.scatter(median, i, label=compartment_to_name_map[compartment], c=color_cycle[i], marker=marker_cycle[i])
                plt.fill_betweenx([i-.3, i+.3], [ci32, ci32], [ci68, ci68], alpha=.3, color=color_cycle[i],)
                plt.fill_betweenx([i-.1, i+.1], [ci5, ci5], [ci95, ci95], alpha=.3, color=color_cycle[i])
            self._plot_dates(log=False)
            plt.legend(loc=(1.05, 0.0))
            plt.grid(True, which='both', alpha=0.3)
            plt.xlabel('Peak Time After $t_0(C=5)$ [Days]')
            plt.yticks([])

            # -----------------------------
            # Plot peak capacity
            # -----------------------------
            plt.subplot(5, 5, len(compartments) + 3)
            for i, compartment in enumerate(['E', 'A', 'I', 'R', 'D', 'total_deaths',
                                             'direct_deaths_per_day', 'total_deaths_per_day', 'HGen', 'HICU', 'HVent',
                                             'HGen_cumulative', 'HICU_cumulative', 'HVent_cumulative',
                                             'general_admissions_per_day', 'icu_admissions_per_day']):
                median = output[compartment]['peak_value_ci50']
                ci5, ci95 = output[compartment]['peak_value_ci5'], output[compartment]['peak_value_ci95']
                ci32, ci68 = output[compartment]['peak_value_ci32'], output[compartment]['peak_value_ci68']
                plt.scatter(median, i, label=compartment_to_name_map[compartment], c=color_cycle[i], marker=marker_cycle[i])
                plt.fill_betweenx([i-.3, i+.3], [ci32, ci32], [ci68, ci68], alpha=.3, color=color_cycle[i])
                plt.fill_betweenx([i-.1, i+.1], [ci5, ci5], [ci95, ci95], alpha=.3, color=color_cycle[i])
                plt.xscale('log')

            plt.vlines(self.county_metadata['total_population'], *plt.ylim(), label='Entire Population', alpha=0.5, color='g')
            plt.vlines(self.county_metadata['total_population'] * 0.65, *plt.ylim(), label='Approx. Herd Immunity',
                       alpha=0.5, color='purple', linestyles='--', linewidths=2)
            plt.legend(loc=(1, -0.1))
            plt.grid(True, which='both', alpha=0.3)
            plt.xlabel('Value at Peak')
            plt.yticks([])

            self.report.add_figure(fig)

    def _plot_dates(self, log=True):
        """
        Helper function to add date plots.

        Parameters
        ----------
        log: bool
            If True, shift y-positioning of labels based on a log scale.
        """
        low_limit = plt.ylim()[0]
        if log:
            upp_limit = 1 * np.log(plt.ylim()[1])
        else:
            upp_limit = 1 * plt.ylim()[1]

        for month in range(4, 11):
            dt = datetime.datetime(day=1, month=month, year=2020)
            offset = (dt - self.summary['t0']).days
            plt.vlines(offset, low_limit, upp_limit, color='firebrick', alpha=.4, linestyles=':')
            plt.text(offset, low_limit*1.3, dt.strftime('%B'), rotation=90, color='firebrick', alpha=0.6)
