import matplotlib.pyplot as plt
from scipy.optimize import minimize, dual_annealing, basinhopping, differential_evolution


class PolicyOptimizer:
    """
    Fit a suppression policy to minimize total death count.

    Note that it is apparent that many suppression polices (especially piecewise
    are either not convex and hence converge with significant differences run to
    run. Ideally policies are based more on parametric forms.

    So far, global optimizers do not seem to have done very well...

    Note also that full suppression is usually preferable *IF* the time period
    of the simulation is too short since the next peak will be pushed out
    several years.  In the case of COVID, this is not realistic since we have
    pandemic conditions already.

    Parameters
    ----------
    seir_model_class: class
        The simulation class to use in the loss function.
    seir_model_args: dict
        The model arguments passed to init the model class on each iteration.
    parametric_policy: callable
        parametric_policy(x0, t_list) should return a callable(t) that produces
        the suppression level at time t.
    parametric_policy_kwargs: dict
        Additional kwargs passed to parametric policy.
    x0: array-like
        Initial guess at optimal policy.
    optimization_bounds: list(list)
        Bounds are usually provided sine suppression levels cannot realistically
        vary outside [0, 3]. This is a list of lists e.g. ((0, 3), (0, 3)). on
        parameters.  A given suppression policy may implement these internally.
    """

    def __init__(self,
                 seir_model_class,
                 seir_model_args,
                 parametric_policy,
                 x0,
                 parametric_policy_kwargs=None,
                 optimization_bounds=None):

        self.seir_model_class = seir_model_class
        self.seir_model_args = seir_model_args
        self.parametric_policy = parametric_policy
        self.parametric_policy_kwargs = parametric_policy_kwargs

        self.x0 = x0
        self.optimization_bounds = optimization_bounds

        self.fit_results = dict(
            total_deaths=[],
            D=[],
            deaths_from_ventilator_limits=[],
            deaths_from_icu_bed_limits=[],
            deaths_from_hospital_bed_limits=[]
        )
        self.minimization_results = None
        self.best_model = None

    def _loss_function(self, x):
        """
        Parameters
        ----------
        x: array-like
            Parameters passed to the suppression policy.

        Returns
        -------
        loss: float
            Loss to minimize.
        """
        model = self.seir_model_class(
            **self.seir_model_args,
            suppression_policy=self.parametric_policy(
                x, t_list=self.seir_model_args['t_list'], **self.parametric_policy_kwargs)
        )

        model.run()

        # Store array of run results
        for key in ('total_deaths', 'D', 'deaths_from_hospital_bed_limits',
                    'deaths_from_icu_bed_limits', 'deaths_from_ventilator_limits'):
            self.fit_results[key].append(model.results[key][-1])

        # This may get memory hungry so leaving out for now...
        # self.fit_results['models'].append(model)

        # We can also add a small Gaussian Prior Toward No Distancing Policy
        # (i.e. suppression_level=1) to stabilize the Fit
        # This prior could be refined to be an alternative outcome such as economic incentives
        loss = self.fit_results['total_deaths'][-1]
                #+ 10 * self.fit_results['total_deaths'][-1] * np.average((x - 1) ** 2)
        return loss

    def run(self, minimize_kwargs=dict(tol=0.01, method=None)):
        """
        Minimize the death rate and select the best performing model.

        Parameters
        ----------
        tol: tolerance to pass to the optimizer.

        Returns
        -------
        minimization_results: dict
            Results dict from scipy.optimize.minimize.
        """
        self.minimization_results = minimize(
            self._loss_function,
            x0=self.x0,
            bounds=self.optimization_bounds,
            **minimize_kwargs)

        self.best_model = self.seir_model_class(
            **self.seir_model_args,
            suppression_policy=self.parametric_policy(
                self.minimization_results['x'],
                t_list=self.seir_model_args['t_list'],
                **self.parametric_policy_kwargs)
        )
        self.best_model.run()

        return self.minimization_results

    def plot_optimal_model(self, **kwargs):
        """
        Plots the optimial model results and policy.

        Parameters
        ----------
        **kwargs: passed to plot_results.
        """
        self.best_model.plot_results(**kwargs)

    def plot_loss(self, y_scale='linear'):
        """
        Plot the loss function by

        Parameters
        ----------
        y_scale: str
            Passed to matplotlib plt.yscale
        """
        plt.figure(figsize=(8, 8))

        evals = range(len(self.fit_results['total_deaths']))
        plt.plot(evals, self.fit_results['total_deaths'], label='Total Deaths', linestyle='')
        plt.plot(evals, self.fit_results['D'], label='Direct COVID Deaths')
        plt.plot(evals, self.fit_results['deaths_from_hospital_bed_limits'], label='Deaths from General Bed Limits')
        plt.plot(evals, self.fit_results['deaths_from_icu_bed_limits'], label='Deaths from ICU Bed Limits')
        plt.plot(evals, self.fit_results['deaths_from_ventilator_limits'], label='Deaths from Ventilator Bed Limits')

        plt.legend()
        plt.grid(which='both')
        plt.yscale(y_scale)
        plt.xlabel('Optimization Iteration', fontsize=14)
        plt.ylabel('Deaths', fontsize=14)
