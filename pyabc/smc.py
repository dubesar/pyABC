"""
Parallel Approximate Bayesian Computation - Sequential Monte Carlo
==================================================================

The ABCSMC class is the most central class of the pyABC package.
Most of the other classes serve to configure it. (I.e. the other classes
implement a Strategy pattern.)
"""

import datetime
import logging
from typing import List, Callable, TypeVar
import numpy as np
import scipy as sp
import pandas as pd
import copy
from typing import Union

from .distance import Distance, PNormDistance, to_distance
from .epsilon import Epsilon, MedianEpsilon
from .model import Model
from .population import Particle
from .transition import Transition, MultivariateNormalTransition
from .random_variables import RV, ModelPerturbationKernel, Distribution
from .storage import History
from .populationstrategy import PopulationStrategy
from .pyabc_rand_choice import fast_random_choice
from .model import SimpleModel
from .populationstrategy import ConstantPopulationSize
from .platform_factory import DefaultSampler
from .acceptor import Acceptor, UniformAcceptor, SimpleFunctionAcceptor
from .sampler import Sampler


logger = logging.getLogger("ABC")

model_output = TypeVar("model_output")


def identity(x):
    return x


class ABCSMC:
    """
    Approximate Bayesian Computation - Sequential Monte Carlo (ABCSMC).

    This is an implementation of an ABCSMC algorithm similar to
    [#tonistumpf]_.


    Parameters
    ----------

    models: list of models, single model, list of functions, or single function
       * If models is a function, then the function should have a single
         parameter, which is of dictionary type, and should return a single
         dictionary, which contains the simulated data.
       * If models is a list of functions, then the first point applies to
         each function.
       * Models can also be a list of Model instances or a single
         Model instance.

       This model's output is passed to the summary statistics calculation.
       Per default, the model is assumed to already return the calculated
       summary statistics. Accordingly, the default summary_statistics
       function is just the identity. Note that the sampling and evaluation of
       particles happens in the model's methods, so overriding these offers a
       great deal of flexibility, in particular the freedom to use or ignore
       the distance_function, summary_statistics, and eps parameters here.

    parameter_priors: List[Distribution]
        A list of prior distributions for the models' parameters.
        Each list entry is the prior distribution for the corresponding model.

    distance_function: Distance, optional
        Measures the distance of the tentatively sampled particle to the
        measured data.

    population_size: int, PopulationStrategy, optional
        Specify the size of the population.
        If ``population_specification`` is an ``int``, then the size is
        constant. Adaptive population sizes are also possible by passing a
        :class:`pyabc.populationstrategy.PopulationStrategy` object.
        The default is 100 particles per population.

    summary_statistics: Callable[[model_output], dict]
        A function which takes the raw model output as returned by
        any ot the models and calculates the corresponding summary
        statistics. Note that the default value is just the identity
        function. I.e. the model is assumed to already calculate
        the summary statistics. However, in a model selection setting
        it can make sense to have the model produce some kind or raw output
        and then take the same summary statistics function for all the models.

    model_prior: RV, optional
        A random variable giving the prior weights of the model classes.
        The default is a uniform prior over the model classes,
        ``RV("randint", 0, len(models))``.

    model_perturbation_kernel: ModelPerturbationKernel
        Kernel which governs with which probability to switch from one
        model to another model for a given sample while generating proposals
        for the subsequent population from the current population.

    transitions: List[Transition], Transition, optional
        A list of :class:`pyabc.transition.Transition` objects
        or a single :class:`pyabc.transition.Transition` in case
        of a single model. Defaults to multivariate normal transitions for
        every model.

    eps: Epsilon, optional
        Accepts any :class:`pyabc.epsilon.Epsilon` subclass.
        The default is the :class:`pyabc.epsilon.MedianEpsilon` which adapts
        automatically. The object passed here determines how the acceptance
        threshold scheduling is performed.

    sampler: Sampler, optional
        In some cases, a mapper implementation will require initialization
        to run properly, e.g. database connection, grid setup, etc..
        The sampler is an object that encapsulates this information.
        The default sampler :class:`pyabc.sampler.MulticoreEvalParallelSampler`
        will parallelize across the cores of a single
        machine only.

    acceptor: Acceptor, optional
        Takes a distance function, summary statistics and an epsilon threshold
        to decide about acceptance of a particle. Argument accepts any subclass
        of :class:`pyabc.acceptor.Acceptor`, or a function convertible to an
        acceptor. Defaults to a :class:`pyabc.acceptor.UniformAcceptor`.


    Attributes
    ----------

    max_number_particles_for_distance_update: int
        Defaults to 1000. Set this to the maximum number of particles that an
        adaptive distance measure (e.g. AdaptivePNormDistance) uses to update
        itself each iteration.

    stop_if_only_single_model_alive: bool
        Defaults to False. Set this to true if you want to stop ABCSMC
        automatically as soon as only a single model has survived.


    .. [#tonistumpf] Toni, Tina, and Michael P. H. Stumpf.
                  “Simulation-Based Model Selection for Dynamical
                  Systems in Systems and Population Biology”.
                  Bioinformatics 26, no. 1, 104–10, 2010.
                  doi:10.1093/bioinformatics/btp619.
    """

    def __init__(self,
                 models: Union[List[Model], Model],
                 parameter_priors: Union[List[Distribution],
                                         Distribution, Callable],
                 distance_function: Union[Distance, Callable] = None,
                 population_size: Union[PopulationStrategy, int] = 100,
                 summary_statistics: Callable[[model_output], dict] = identity,
                 model_prior: RV = None,
                 model_perturbation_kernel: ModelPerturbationKernel = None,
                 transitions: List[Transition] = None,
                 eps: Epsilon = None,
                 sampler: Sampler = None,
                 acceptor: Acceptor = None):

        if not isinstance(models, list):
            models = [models]
        models = list(map(SimpleModel.assert_model, models))
        self.models = models

        if not isinstance(parameter_priors, list):
            parameter_priors = [parameter_priors]
        self.parameter_priors = parameter_priors

        # sanity checks
        if len(self.models) != len(self.parameter_priors):
            raise AssertionError(
                "Number models and number parameter priors have to agree.")

        if distance_function is None:
            distance_function = PNormDistance()
        self.distance_function = to_distance(distance_function)

        self.summary_statistics = summary_statistics

        if model_prior is None:
            model_prior = RV("randint", 0, len(self.models))
        self.model_prior = model_prior

        if model_perturbation_kernel is None:
            model_perturbation_kernel = ModelPerturbationKernel(
                len(self.models), probability_to_stay=.7)
        self.model_perturbation_kernel = model_perturbation_kernel

        if transitions is None:
            transitions = [MultivariateNormalTransition()
                           for _ in self.models]
        if not isinstance(transitions, list):
            transitions = [transitions]
        self.transitions = transitions  # type: List[Transition]

        if eps is None:
            eps = MedianEpsilon(median_multiplier=1)
        self.eps = eps

        if isinstance(population_size, int):
            population_size = ConstantPopulationSize(
                population_size)
        self.population_strategy = population_size

        if sampler is None:
            sampler = DefaultSampler()
        self.sampler = sampler

        if acceptor is None:
            acceptor = UniformAcceptor()
        self.acceptor = SimpleFunctionAcceptor.assert_acceptor(acceptor)

        self.stop_if_only_single_model_alive = False
        self.max_number_particles_for_distance_update = 1000
        self.x_0 = None
        self.history = None
        self._initial_population = None
        self.minimum_epsilon = None
        self.max_nr_populations = None
        self.min_acceptance_rate = None

    def __getstate__(self):
        state_red_dict = self.__dict__.copy()
        del state_red_dict['sampler']
        return state_red_dict

    def new(self, db: str,
            observed_sum_stat: dict = None,
            *,
            gt_model: int = None,
            gt_par: dict = None,
            meta_info=None) -> int:
        """
        Make a new ABCSMC run.

        Parameters
        ----------

        db: str
            Has to be a valid SQLAlchemy database identifier.
            This indicates the database to be used (and created if necessary
            and possible) for the ABC-SMC run.

            To use an in-memory database pass "sqlite://".
            Note that in-memory databases are only available on the master
            mode. If workers are started on different nodes they won't be
            able to access the database. This should not be a problem
            in most scenarios. The in-memory option is mainly useful for
            benchmarking (and maybe) for testing.

        observed_sum_stat : dict, optional
            This is the really important parameter here. It is of the
            form ``{'statistic_1': val_1, 'statistic_2': val_2, ... }``.

            The dictionary provided here represents the measured data.
            Particle during ABCSMC sampling are compared against the
            summary statistics provided here.

            The summary statistics' values can be integers, floats,
            strings and everything which is a numpy array or can be
            converted to one (e.g. lists).
            In addition, pandas.DataFrames can also be used as summary
            statistics.
            **Note that storage of pandas DataFrames in pyABC's database
            is still considered experimental.**

            This parameter is optional, as the distance function might
            implement comparison to the observed data on its own.
            Not giving this parameter is equivalent to passing an empty
            dictionary ``{}``.

        gt_model: int, optional
            This is only meta data stored to the database, but not actually
            used for the ABCSMC algorithm. If you want to predict your ABCSMC
            procedure against synthetic samples, you can use
            this parameter to indicate the ground truth model number.
            This helps with further analysis. If you use actually measured data
            (and don't know the ground truth) you don't have to set this.

        gt_par: dict, optional
            Similar to ``ground_truth_model``, this is only for recording
            purposes in the database, but not used in the ABCSMC algorithm.
            This stores the parameters of the ground truth model
            if it was synthetically obtained.
            Don't give this parameter if you don't know the ground truth.

        meta_info: dict, optional
            Can contain an arbitrary number of keys, only for recording
            purposes. Store arbitrary
            meta information in this dictionary. Can be used for really
            anything.
            This dictionary is stored in the database.

        Returns
        -------

        run_id: int
            The history.id, which is the id under which the generated ABCSMC
            run entry in the database can be identified.
        """
        # record observed summary statistics
        if observed_sum_stat is None:
            observed_sum_stat = {}
        self.x_0 = observed_sum_stat

        # initialize history object
        self.history = History(db)

        if gt_par is None:
            gt_par = {}

        # save configuration data to database
        model_names = [model.name for model in self.models]
        self.history.store_initial_data(gt_model,
                                        meta_info,
                                        observed_sum_stat,
                                        gt_par,
                                        model_names,
                                        self.distance_function.to_json(),
                                        self.eps.to_json(),
                                        self.population_strategy.to_json())

        # return id generated in store_initial_data
        return self.history.id

    def load(self, db: str,
             abc_id: int = 1,
             observed_sum_stat: dict = None) -> int:
        """
        Load an ABC-SMC run for continuation.

        Parameters
        ----------

        db: str
            A SQLAlchemy database identifier pointing to the database from
            which to continue a run.

        abc_id: int, optional
            The id of the ABC-SMC run in the database which is to be continued.
            The default is 1. If more than one ABC-SMC run is stored, use
            the ``abc_id`` parameter to indicate which one to continue.

        observed_sum_stat: dict, optional
            The observed summary statistics. This field should be used only if
            the summary statistics cannot be reproduced exactly from the
            database (in particular when they are no numpy or pandas objects,
            e.g. when they were generated in R). If None, then the summary
            statistics are read from the history.
        """
        self.history = History(db)
        self.history.id = abc_id

        # extract observed sum stats from input or history
        if observed_sum_stat is None:
            observed_sum_stat = self.history.observed_sum_stat()
        self.x_0 = observed_sum_stat

        # just return the id again
        return self.history.id

    def _initialize_dist_eps_acc(self, t: int):
        """
        Called once at the start of run(). This function either, if available,
        takes the last population from the history, or generates a
        sample population from the prior. Then, it calls the initialize()
        functions of the distance, epsilon, and acceptor.

        Note that a calibration sample is only taken if required by any of
        the tools.

        Parameters
        ----------

        t: int
            Time point for which to initialize (i.e. the time point at which
            to do the first population). Usually 0 or history.max_t + 1.
        """
        def get_initial_sum_stats():
            population = self._get_initial_population(t)
            # only the accepted sum stats are available initially
            sum_stats = population.get_accepted_sum_stats()
            return sum_stats

        def get_initial_weighted_distances():
            population = self._get_initial_population(t)

            def distance_to_ground_truth(x, par):
                return self.distance_function(x, self.x_0, t, par)

            population.update_distances(distance_to_ground_truth)
            weighted_distances = population.get_weighted_distances()
            return weighted_distances

        # initialize dist, eps, acc (order important)
        self.distance_function.initialize(
            t, get_initial_sum_stats, self.x_0)
        self.eps.initialize(
            t, get_initial_weighted_distances)
        self.acceptor.initialize(
            t, get_initial_weighted_distances, self.max_nr_populations,
            self.distance_function, self.x_0)

    def _get_initial_population(self, t: int) -> (List[float], List[dict]):
        """
        Get initial samples, either from the last population stored in history,
        or via sampling sum stats from the prior. This can be used to calibrate
        the distance function or the epsilon.

        The history must have been initialized already. This function fills the
        private property _initial_population.

        .. warning::
            The sample is cached. Thus, the function can be called repeatedly
            without further computational overhead.
        """
        if self._initial_population is None:
            if self.history.n_populations > 0:
                # extract latest population from database
                population = self.history.get_population()
            else:
                # sample
                population = self._sample_from_prior(t)
                # update number of samples in calibration
                self.history.update_nr_samples(
                    History.PRE_TIME, self.sampler.nr_evaluations_)
            self._initial_population = population

        return self._initial_population

    def _create_simulate_from_prior_function(self, t: int):
        """
        Similar to _create_simulate_function, apart here we sample from the
        prior and accept all.
        """
        model_prior = self.model_prior
        parameter_priors = self.parameter_priors
        models = self.models
        summary_statistics = self.summary_statistics

        # simulation function, simplifying some parts compared to later

        def simulate_one():
            # sample model
            m = int(model_prior.rvs())
            # sample parameter
            theta = parameter_priors[m].rvs()
            # simulate summary statistics
            model_result = models[m].summary_statistics(
                t, theta, summary_statistics)
            # sampled from prior, so all have uniform weight
            weight = 1.0
            # remember sum stat as accepted
            accepted_sum_stats = [model_result.sum_stats]
            # distance will be computed after initialization of the
            # distance function
            accepted_distances = [np.inf]
            # all are happy and accepted
            accepted = True

            return Particle(
                m=m,
                parameter=theta,
                weight=weight,
                accepted_sum_stats=accepted_sum_stats,
                accepted_distances=accepted_distances,
                rejected_sum_stats=[],
                rejected_distances=[],
                accepted=accepted)

        return simulate_one

    def _sample_from_prior(self, t: int) -> List[dict]:
        """
        Only sample from prior and return results without changing
        the history of the distance function or the epsilon.
        """
        # create simulate function
        simulate_one = self._create_simulate_from_prior_function(t)

        logger.info(f"Calibration sample before t={t}.")

        # call sampler
        sample = self.sampler.sample_until_n_accepted(
            self.population_strategy.nr_particles, simulate_one,
            all_accepted=True)

        # extract accepted population
        population = sample.get_accepted_population()

        return population

    def _create_simulate_function(self, t: int):
        """
        Create a simulation function which performs the sampling of parameters,
        simulation of data and acceptance checking, and which is then passed
        to the sampler.

        Parameters
        ----------
        t: int
            Time index

        Returns
        -------
        simulate_one: callable
            Function that samples parameters, simulates data, and checks
            acceptance.

        .. note::
            For some of the samplers, the sampling function needs to be
            serialized in order to be transported to where the sampling
            happens. Therefore, the returned function should be light, and
            in particular not contain references to the ABCSMC class.
        """
        # cache model_probabilities to not query the database so often
        model_probabilities = self.history.get_model_probabilities(
            self.history.max_t)

        m = sp.array(model_probabilities.index)
        p = sp.array(model_probabilities.p)

        model_prior = self.model_prior
        parameter_priors = self.parameter_priors
        model_perturbation_kernel = self.model_perturbation_kernel
        transitions = self.transitions
        nr_samples_per_parameter = \
            self.population_strategy.nr_samples_per_parameter
        models = self.models
        summary_statistics = self.summary_statistics
        distance_function = self.distance_function
        eps = self.eps
        acceptor = self.acceptor
        x_0 = self.x_0

        # simulation function
        def simulate_one():
            parameter = ABCSMC._generate_valid_proposal(
                t, m, p,
                model_prior,
                parameter_priors,
                model_perturbation_kernel,
                transitions)
            particle = ABCSMC._evaluate_proposal(
                *parameter,
                t,
                model_probabilities,
                nr_samples_per_parameter,
                models,
                summary_statistics,
                distance_function,
                eps,
                acceptor,
                x_0,
                model_prior,
                parameter_priors,
                model_perturbation_kernel,
                transitions)
            return particle

        return simulate_one

    @staticmethod
    def _generate_valid_proposal(
            t, m, p,
            model_prior,
            parameter_priors,
            model_perturbation_kernel,
            transitions):
        """
        Sample a parameter for a model.

        Parameters
        ----------
        t: Population number
        m: Indices of alive models
        p: Probabilities of alive models

        Returns
        -------

        Model, parameter.

        """
        # first generation
        if t == 0:  # sample from prior
            m_ss = int(model_prior.rvs())
            theta_ss = parameter_priors[m_ss].rvs()
            return m_ss, theta_ss

        # later generation
        while True:  # find m_s and theta_ss, valid according to prior
            if len(m) > 1:
                index = fast_random_choice(p)
                m_s = m[index]
                m_ss = model_perturbation_kernel.rvs(m_s)
                # theta_s is None if the population m_ss has died out.
                # This can happen since the model_perturbation
                # _kernel can return  a model nr which has died out.
                if m_ss not in m:
                    continue
            else:
                m_ss = m[0]
            theta_ss = transitions[m_ss].rvs()

            if (model_prior.pmf(m_ss)
                    * parameter_priors[m_ss].pdf(theta_ss) > 0):
                return m_ss, theta_ss

    @staticmethod
    def _evaluate_proposal(
            m_ss, theta_ss,
            t,
            model_probabilities,
            nr_samples_per_parameter,
            models,
            summary_statistics,
            distance_function,
            eps,
            acceptor,
            x_0,
            model_prior,
            parameter_priors,
            model_perturbation_kernel,
            transitions) -> Particle:
        """
        Corresponds to Sampler.simulate_one. Data for the given parameters
        theta_ss are simulated, summary statistics computed and evaluated.

        This is where the actual model evaluation happens.
        """

        # from here, theta_ss is valid according to the prior

        accepted_sum_stats = []
        accepted_distances = []
        rejected_sum_stats = []
        rejected_distances = []
        accepted_weights = []

        for _ in range(nr_samples_per_parameter):
            model_result = models[m_ss].accept(
                t,
                theta_ss,
                summary_statistics,
                distance_function,
                eps,
                acceptor,
                x_0)
            if model_result.accepted:
                accepted_sum_stats.append(model_result.sum_stats)
                accepted_distances.append(model_result.distance)
                accepted_weights.append(model_result.weight)
            else:
                rejected_sum_stats.append(model_result.sum_stats)
                rejected_distances.append(model_result.distance)

        accepted = len(accepted_sum_stats) > 0

        if accepted:
            weight = ABCSMC._calc_proposal_weight(
                accepted_distances, m_ss, theta_ss,
                accepted_weights, t,
                model_probabilities,
                model_prior,
                parameter_priors,
                nr_samples_per_parameter,
                model_perturbation_kernel,
                transitions)
        else:
            weight = 0

        return Particle(
            m=m_ss,
            parameter=theta_ss,
            weight=weight,
            accepted_sum_stats=accepted_sum_stats,
            accepted_distances=accepted_distances,
            rejected_sum_stats=rejected_sum_stats,
            rejected_distances=rejected_distances,
            accepted=accepted)

    @staticmethod
    def _calc_proposal_weight(
            distance_list,
            m_ss,
            theta_ss,
            acceptance_weights,
            t,
            model_probabilities,
            model_prior,
            parameter_priors,
            nr_samples_per_parameter,
            model_perturbation_kernel,
            transitions):
        """
        Calculate the weight for the generated parameter.
        """
        if t == 0:
            weight = (len(distance_list)
                      / nr_samples_per_parameter)
        else:
            model_factor = sum(
                row.p * model_perturbation_kernel.pmf(m_ss, m)
                for m, row in model_probabilities.iterrows())
            particle_factor = transitions[m_ss].pdf(
                pd.Series(dict(theta_ss)))
            normalization = model_factor * particle_factor
            if normalization == 0:
                print('normalization is zero!')
            # reflects stochasticity of the model
            fraction_accepted_runs_for_single_parameter = (
                len(distance_list)
                / nr_samples_per_parameter)
            weight = (model_prior.pmf(m_ss)
                      * parameter_priors[m_ss].pdf(theta_ss)
                      * fraction_accepted_runs_for_single_parameter
                      / normalization)

            # account for acceptance weights
            # TODO This is only valid for single samples (see #54)
            weight *= np.prod(acceptance_weights)

        return weight

    def run(self,
            minimum_epsilon: float = 0.,
            max_nr_populations: int = np.inf,
            min_acceptance_rate: float = 0.,
            **kwargs) -> History:
        """
        Run the ABCSMC model selection until either of the stopping
        criteria is met.

        Parameters
        ----------

        minimum_epsilon: float, optional (default = 0.0)
            Stop if epsilon is smaller than minimum epsilon specified here.

        max_nr_populations: int, optional (default = np.inf)
            The maximum number of populations. Stop if this number is reached.

        min_acceptance_rate: float, optional (default = 0.0)
            Minimal allowed acceptance rate. Sampling stops if a population
            has a lower rate.


        Population after population is sampled and particles which are close
        enough to the observed data are accepted and added to the next
        population.
        If an adaptive Epsilon is specified (this is the default), then
        the acceptance threshold decreases from population to population
        automatically in a data dependent way.

        Sampling of further populations is stopped, when either of the three
        stopping criteria is met:

            * the maximum number of populations ``max_nr_populations``
              is reached,
            * the acceptance threshold for the last sampled population was
              smaller than ``minimum_epsilon``,
            * or the acceptance rate dropped below ``acceptance_rate``.

        The value of ``minimum_epsilon`` determines the quality of the ABCSMC
        approximation. The smaller the better. But sampling time also increases
        with decreasing ``minimum_epsilon``.

        This method can be called repeatedly to sample further populations
        after sampling was stopped once.
        """
        # argument handling
        if len(kwargs) > 1:
            raise TypeError("Keyword arguments are not allowed.")

        self.minimum_epsilon = minimum_epsilon
        self.max_nr_populations = max_nr_populations
        self.min_acceptance_rate = min_acceptance_rate

        # sample from prior to calibrate distance, epsilon, and acceptor
        self._initialize_dist_eps_acc(self.history.max_t + 1)

        t0 = self.history.max_t + 1
        self.history.start_time = datetime.datetime.now()
        # not saved as attribute b/c Mapper of type
        # "ipython_cluster" is not pickleable

        # configure sampler by whoever wants to
        self.distance_function.configure_sampler(self.sampler)

        # run loop over time points
        t_max = t0 + max_nr_populations
        for t in range(t0, t_max):

            # get epsilon for generation t
            current_eps = self.eps(t)
            logger.info(f"t: {t}, eps: {current_eps}.")

            # do some adaptations
            self._fit_transitions(t)
            self._adapt_population_size(t)

            # create simulate function
            simulate_one = self._create_simulate_function(t)

            logger.debug(f"Now submitting population {t}.")

            # perform the sampling
            sample = self.sampler.sample_until_n_accepted(
                self.population_strategy.nr_particles, simulate_one)

            # retrieve accepted population
            population = sample.get_accepted_population()
            logger.debug(f"Population {t} done.")

            # save to database before making any changes to the population
            nr_evaluations = self.sampler.nr_evaluations_
            model_names = [model.name for model in self.models]
            self.history.append_population(
                t, current_eps, population, nr_evaluations,
                model_names)
            logger.debug(
                f"Total samples up to t = {t}: "
                f"{self.history.total_nr_simulations}.")

            # prepare next iteration

            # acceptance rate
            pop_size = len(population.get_list())
            acceptance_rate = pop_size / nr_evaluations
            logger.info(f"Acceptance rate: {pop_size} / {nr_evaluations} = "
                        f"{acceptance_rate:.4e}.")

            # update distance function
            partial_sum_stats = sample.first_n_sum_stats(
                self.max_number_particles_for_distance_update)
            df_updated = self.distance_function.update(
                t + 1, partial_sum_stats)

            # compute distances with the new distance measure
            if df_updated:
                def distance_to_ground_truth(x, par):
                    return self.distance_function(x, self.x_0, t + 1, par)

                population.update_distances(distance_to_ground_truth)

            # update epsilon
            self.eps.update(t + 1, population.get_weighted_distances())

            # update acceptor
            self.acceptor.update(
                t + 1, population.get_weighted_distances(),
                self.distance_function, acceptance_rate)

            # check early termination conditions
            if (current_eps <= minimum_epsilon
                    or (self.stop_if_only_single_model_alive
                        and self.history.nr_of_models_alive() <= 1)
                    or acceptance_rate < min_acceptance_rate):
                break

        # end of run loop

        # close session and store end time
        self.history.done()

        # return used history object
        return self.history

    def _adapt_population_size(self, t):
        """
        Adapt population size based on the employed population strategy.

        Parameters
        ----------

        t: int
            Time for which to adapt the population size.
        """
        if t == 0:  # we need a particle population to do the fitting
            return

        w = self.history.get_model_probabilities(
            self.history.max_t)["p"].values

        # make a copy in case the population strategy messes with
        # the transitions
        # WARNING: the deepcopy also copies the random states of scipy.stats
        # distributions
        copied_transitions = copy.deepcopy(self.transitions)
        self.population_strategy.adapt_population_size(copied_transitions, w)

    def _fit_transitions(self, t):
        """
        Fit the density estimator.

        Parameters
        ----------
        t: int
            Time for which to update the kernel density estimator.
        """

        if t == 0:  # we need a particle population to do the fitting
            return

        for m in self.history.alive_models(t - 1):
            particles, w = self.history.get_distribution(m, t - 1)
            self.transitions[m].fit(particles, w)
