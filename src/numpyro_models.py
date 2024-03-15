import itertools
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import arviz as az
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from jax import random
from numpyro.handlers import reparam
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.infer.reparam import LocScaleReparam
from pandas.core.api import DataFrame as DataFrame
from scipy.stats import beta


class AbstractNumpyroModel(ABC):
    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def model(self):
        pass

    @abstractmethod
    def generate_reparam_config(self):
        pass

    @abstractmethod
    def generate_arviz_data(self):
        pass


class BaseNumpyroModel(AbstractNumpyroModel):
    """
    Provides some hopefully generic numpyro model methods.
    """

    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        rng_key=None,
        intensity_variable_name: str = "contrast",
        experiment_conditions: Optional[List[str]] = [],
        group_variables: Optional[List[str]] = [],
        use_reparam: bool = True,
    ) -> None:
        self.intensity_variable_name = intensity_variable_name
        self.experiment_conditions = experiment_conditions
        self.group_variables = group_variables
        if data is not None:
            self.data = data
            (
                self.data,
                self.grouping_ids,
            ) = PsychometricFunctionWrapper.create_grouping_ids(
                data=self.data,
                experiment_conditions=self.experiment_conditions,
                group_variables=self.group_variables,
            )
        else:
            self.data = None
            self.grouping_ids = None
        if rng_key is None:
            self.rng_key = random.PRNGKey(0)
        else:
            self.rng_key = rng_key

        if use_reparam:
            self.model = reparam(self.model, config=self.generate_reparam_config())
        else:
            self.model = self.model

        self.arviz_data = None

    def sample(
        self,
        num_samples: int = 1000,
        num_warmup: int = 1000,
        num_chains: int = 4,
        model_kwargs: Dict = {},
        nuts_kwargs: Dict = {},
        create_arviz_data: bool = True,
        arviz_kwargs: Dict = {},
    ):
        nuts_kernel = NUTS(self.model, **nuts_kwargs)
        self.mcmc = MCMC(
            nuts_kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
        )

        # split RNG: https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#jax-prng
        self.rng_key, rng_key_ = random.split(self.rng_key)
        self.mcmc.run(rng_key_, data=self.data, **model_kwargs)
        posterior_samples = self.mcmc.get_samples()
        self.posterior_samples = posterior_samples
        if create_arviz_data:
            self.arviz_data = self.generate_arviz_data(**arviz_kwargs)

    def predict(
        self,
        data: pd.DataFrame = None,
        prior: bool = False,
        n_prior_samples=50,
        sample_obs: bool = True,
        model_kwargs: dict = {},
    ):
        if data is None:
            data = self.data
        if prior:
            posterior_samples = None
        else:
            posterior_samples = self.posterior_samples

        predictive = Predictive(
            self.model,
            num_samples=n_prior_samples,  # ignored if posterior_samples is not None
            posterior_samples=posterior_samples,
        )

        self.rng_key, rng_key_ = random.split(self.rng_key)
        samples = predictive(rng_key_, data=data, sample_obs=sample_obs, **model_kwargs)
        return samples

    def generate_arviz_data(
        self,
        data: pd.DataFrame = None,
        n_prior_samples: int = 1000,
        sample_obs: bool = True,
        model_kwargs: dict = {},
        from_numpyro_kwargs: dict = {},
    ):
        """
        Create data usable by arviz

        Eventually might be good to unify this with the predict method
        so that we use only arviz InferenceData objects for everything.

        kwargs are passed to arviz.from_numpyro.
        """
        if data is None:
            data = self.data

        posterior_predictive = self.predict(
            data=data, prior=False, sample_obs=sample_obs, model_kwargs=model_kwargs
        )

        prior = self.predict(
            data=data,
            prior=True,
            n_prior_samples=n_prior_samples,
            sample_obs=sample_obs,
            model_kwargs=model_kwargs,
        )

        arviz_data = az.from_numpyro(
            self.mcmc,
            prior=prior,
            posterior_predictive=posterior_predictive,
            **from_numpyro_kwargs,
        )
        return arviz_data

    def generate_reparam_config(self) -> dict:
        return {}

    @staticmethod
    def create_grouping_ids(
        data: pd.DataFrame,
        experiment_conditions: List[str] = [],
        group_variables: List[str] = [],
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Create a dictionary of grouping ids

        So that it's easier to tell which condition corresponds to
        which parameter index in the posterior samples.

        """

        grouping_ids = {}

        def cats_to_dict(x):
            return dict(enumerate(x.cat.categories))

        if experiment_conditions:
            # is not empty
            for cond in experiment_conditions:
                data[cond] = data[cond].astype("category")
                data[cond + "_id"] = data[cond].cat.codes  # assign integer id
                grouping_ids[cond] = cats_to_dict(data[cond])

        if group_variables:
            for group in group_variables:
                data[group] = data[group].astype("category")
                data[group + "_id"] = data[group].cat.codes
                grouping_ids[group] = cats_to_dict(data[group])

        grouping_ids = grouping_ids
        return data, grouping_ids


class PsychometricFunctionWrapper(BaseNumpyroModel):
    """
    A light wrapper class for organising models

    Psychometric function parameterisation from Schütt et al. (2016)
    (see Table A.1), except we use binomial instead of beta-binomial.

    The idea is that this class helps us organise model objects
    together in a shared namespace without abstracting too much
    away from the basic numpyro API.

    The intention is for the user to define the actual model later.

    This can be done using an inherited class.

    I would like to also do it by passing a function to
    the model argument.

    Example of the latter:
        def my_model(data = None, **kwargs):
            blah blah model

        m1 = PsychometricFunctionWrapper(model=my_model, data=my_data)
        m1.sample()

    However this use isn't quite working yet. Lose passing of
    self.data and end up with NoneType errors.

    """

    @staticmethod
    def weibull_width_constant(min: float = 0.05, max: float = 0.95):
        return jnp.log(-jnp.log(min)) - jnp.log(-jnp.log(max))

    @staticmethod
    def weibull_fn(stimulus_intensity, m, w, min: float = 0.05, max: float = 0.95):
        c = PsychometricFunctionWrapper.weibull_width_constant(min=min, max=max)
        return 1.0 - jnp.exp(
            jnp.log(0.5) * jnp.exp(c * (jnp.log(stimulus_intensity) - m) / w)
        )

    @staticmethod
    def squished_weibull(
        stimulus_intensity,
        m,
        w,
        gamma,
        lam,
        log_w: bool = False,
        fn_positive: bool = True,
    ):
        if log_w:
            w = jnp.exp(w)
        if not fn_positive:
            w = -w

        S = PsychometricFunctionWrapper.weibull_fn(stimulus_intensity, m, w)
        return gamma + (1 - gamma - lam) * S


class SinglePsychometricFunction(PsychometricFunctionWrapper):
    """
    A single condition, single participant model
    """

    def model(
        self,
        data: pd.DataFrame = None,
        function_positive: bool = True,
        sample_obs: bool = True,
    ):
        if data is None:
            data = self.data

        stimulus_intensity = data[self.intensity_variable_name].values

        m = numpyro.sample("m", dist.Normal(1.0, 2.0))
        log_w = numpyro.sample("log_w", dist.Normal(1.0, 1.0))
        gamma = 0.5
        # cant use a truncated Beta because truncated distribution
        # must have real number support.
        lam = numpyro.sample(
            "lam",
            dist.TruncatedDistribution(
                base_dist=dist.Normal(0.03, 0.03), low=0.0, high=0.15
            ),
        )
        psi = numpyro.deterministic(
            "psi",
            self.squished_weibull(
                stimulus_intensity=stimulus_intensity,
                m=m,
                w=log_w,
                gamma=gamma,
                lam=lam,
                log_w=True,
                fn_positive=function_positive,
            ),
        )
        if sample_obs:
            n_trials = data["n_trials"].values
            n_successes = data["n_successes"].values
            numpyro.sample("obs", dist.Binomial(n_trials, psi), obs=n_successes)


class ParticipantsAllConditions(PsychometricFunctionWrapper):
    """
    Multiple participants acting in all conditions

    Threshold and width both fitted to each condition in a hierarchical model.
    The effect of different images is not considered here.

    The interaction designation specifies that we have separate thresholds for
    each cell of the experimental design.

    """

    def model(
        self,
        data: pd.DataFrame = None,
        function_positive: bool = True,
        sample_obs: bool = True,
        include_participant_re: bool = True,
    ):
        if data is None:
            data = self.data

        stimulus_intensity = data[self.intensity_variable_name].values
        image_condition_id = data["image_condition_id"].values
        grain_id = data["grain_id"].values
        participant_id = data["participant_id"].values
        n_image_conditions = len(data["image_condition_id"].unique())
        n_grains = len(data["grain_id"].unique())
        n_participants = len(data["participant_id"].unique())

        # threshold parameters
        # basic structure per cell is cond + subj_delta
        m_mu = numpyro.sample("m_mu", dist.Normal(1.0, 1.0))
        m_sigma = numpyro.sample("m_sigma", dist.LogNormal(-1.0, 1.0))

        ## cond
        with numpyro.plate("n_image_conditions", n_image_conditions, dim=-2):
            with numpyro.plate("n_grain", n_grains, dim=-1):
                m_cond = numpyro.sample(
                    "m_cond",
                    dist.Normal(m_mu, m_sigma),
                )

        ## participant offset
        m_delta_sd_participant = numpyro.sample(
            "m_delta_sd_participant", dist.LogNormal(-1.0, 1.0)
        )
        with numpyro.plate("n_participants", n_participants, dim=-3):
            with numpyro.plate("n_image_conditions", n_image_conditions, dim=-2):
                with numpyro.plate("n_grain", n_grains, dim=-1):
                    m_delta_participant = numpyro.sample(
                        "m_delta_participant", dist.Normal(0.0, m_delta_sd_participant)
                    )

        # width parameters
        log_w_mu = numpyro.sample("log_w_mu", dist.Normal(1.0, 1.0))
        log_w_sigma = numpyro.sample("log_w_sigma", dist.LogNormal(-1.0, 1.0))

        ## cond
        with numpyro.plate("n_image_conditions", n_image_conditions, dim=-2):
            with numpyro.plate("n_grain", n_grains, dim=-1):
                log_w_cond = numpyro.sample(
                    "log_w_cond",
                    dist.Normal(log_w_mu, log_w_sigma),
                )

        ## participant offset
        log_w_delta_sd_participant = numpyro.sample(
            "log_w_delta_sd_participant", dist.LogNormal(-1.0, 1.0)
        )
        with numpyro.plate("n_participants", n_participants, dim=-3):
            with numpyro.plate("n_image_conditions", n_image_conditions, dim=-2):
                with numpyro.plate("n_grain", n_grains, dim=-1):
                    log_w_delta_participant = numpyro.sample(
                        "log_w_delta_participant",
                        dist.Normal(0.0, log_w_delta_sd_participant),
                    )

        # lapse rate parameter
        lam_mu = numpyro.sample("lam_mu", dist.Beta(1.0, 10.0))
        lam_sigma = numpyro.sample("lam_sigma", dist.LogNormal(-1.0, 1.0))

        with numpyro.plate("n_participants", n_participants):
            lam_participant = numpyro.sample(
                "lam_participant",
                dist.TruncatedDistribution(
                    base_dist=dist.Normal(lam_mu, lam_sigma), low=0.0, high=0.2
                ),
            )

        ## lower bound fixed by design
        gamma = 0.5

        # mean-centering deltas for each experimental condition
        for i in range(n_image_conditions):
            for j in range(n_grains):
                m_delta_participant = m_delta_participant.at[:, i, j].set(
                    m_delta_participant[:, i, j]
                    - jnp.mean(m_delta_participant[:, i, j])
                )
                log_w_delta_participant = log_w_delta_participant.at[:, i, j].set(
                    log_w_delta_participant[:, i, j]
                    - jnp.mean(log_w_delta_participant[:, i, j])
                )

        if include_participant_re:
            participant_delta_factor = 1.0
        else:
            participant_delta_factor = 0.0

        m_final = m_cond[image_condition_id, grain_id] + (
            participant_delta_factor
            * m_delta_participant[participant_id, image_condition_id, grain_id]
        )
        log_w_final = log_w_cond[image_condition_id, grain_id] + (
            participant_delta_factor
            * log_w_delta_participant[participant_id, image_condition_id, grain_id]
        )

        psi = numpyro.deterministic(
            "psi",
            self.squished_weibull(
                stimulus_intensity=stimulus_intensity,
                m=m_final,
                w=log_w_final,
                gamma=gamma,
                lam=lam_participant[participant_id],
                log_w=True,
                fn_positive=function_positive,
            ),
        )
        if sample_obs:
            n_trials = data["n_trials"].values
            n_successes = data["n_successes"].values
            numpyro.sample("obs", dist.Binomial(n_trials, psi), obs=n_successes)

    def generate_reparam_config(self) -> dict:
        reparam_config = {
            "m_cond": LocScaleReparam(0),
            "m_delta_participant": LocScaleReparam(0),
            "log_w_cond": LocScaleReparam(0),
            "log_w_delta_participant": LocScaleReparam(0),
        }
        return reparam_config

    def generate_arviz_data(
        self,
        data: pd.DataFrame = None,
        n_prior_samples: int = 500,
        model_kwargs: dict = {},
    ):
        coords = {
            "participant": list(self.grouping_ids["participant"].values()),
            "image_condition": list(self.grouping_ids["image_condition"].values()),
            "grain": list(self.grouping_ids["grain"].values()),
        }
        dims = {
            "m_cond": ["image_condition", "grain"],
            "m_delta_participant": ["participant", "image_condition", "grain"],
            "log_w_cond": ["image_condition", "grain"],
            "log_w_delta_participant": ["participant", "image_condition", "grain"],
            "lam_participant": ["participant"],
        }
        return super().generate_arviz_data(
            data=data,
            n_prior_samples=n_prior_samples,
            model_kwargs=model_kwargs,
            from_numpyro_kwargs={"coords": coords, "dims": dims},
        )


class ParticipantsAllConditionsAndImages(PsychometricFunctionWrapper):
    """
    Multiple participants acting in all conditions

    Threshold and width both fitted to each condition in a hierarchical model.
    Images add an additional offset to both threshold and width.

    """

    def model(
        self,
        data: pd.DataFrame = None,
        function_positive: bool = True,
        sample_obs: bool = True,
        include_participant_re: bool = True,
        include_image_re: bool = True,
    ):
        if data is None:
            data = self.data

        stimulus_intensity = data[self.intensity_variable_name].values
        image_condition_id = data["image_condition_id"].values
        grain_id = data["grain_id"].values
        participant_id = data["participant_id"].values
        image_name_id = data["image_name_id"].values

        n_image_conditions = len(data["image_condition_id"].unique())
        n_grains = len(data["grain_id"].unique())
        n_participants = len(data["participant_id"].unique())
        n_images = len(data["image_name_id"].unique())

        # threshold parameters
        # basic structure per cell is cond + subj_delta + image delta
        m_mu = numpyro.sample("m_mu", dist.Normal(1.0, 1.0))
        m_sigma = numpyro.sample("m_sigma", dist.LogNormal(-1.0, 1.0))

        ## cond
        with numpyro.plate("n_image_conditions", n_image_conditions, dim=-2):
            with numpyro.plate("n_grain", n_grains, dim=-1):
                m_cond = numpyro.sample(
                    "m_cond",
                    dist.Normal(m_mu, m_sigma),
                )

        ## participant offset
        m_delta_sd_participant = numpyro.sample(
            "m_delta_sd_participant", dist.LogNormal(-1.0, 1.0)
        )
        with numpyro.plate("n_participants", n_participants, dim=-3):
            with numpyro.plate("n_image_conditions", n_image_conditions, dim=-2):
                with numpyro.plate("n_grain", n_grains, dim=-1):
                    m_delta_participant = numpyro.sample(
                        "m_delta_participant", dist.Normal(0.0, m_delta_sd_participant)
                    )

        ## image offset
        m_delta_sd_image = numpyro.sample("m_delta_sd_image", dist.LogNormal(-1.0, 1.0))
        with numpyro.plate("n_images", n_images):
            # note how unlike m_delta_participant, doesn't depend on condition.
            m_delta_image = numpyro.sample(
                "m_delta_image", dist.Normal(0.0, m_delta_sd_image)
            )

        # width parameters
        log_w_mu = numpyro.sample("log_w_mu", dist.Normal(1.0, 1.0))
        log_w_sigma = numpyro.sample("log_w_sigma", dist.LogNormal(-1.0, 1.0))

        ## cond
        with numpyro.plate("n_image_conditions", n_image_conditions, dim=-2):
            with numpyro.plate("n_grain", n_grains, dim=-1):
                log_w_cond = numpyro.sample(
                    "log_w_cond",
                    dist.Normal(log_w_mu, log_w_sigma),
                )

        ## participant offset
        log_w_delta_sd_participant = numpyro.sample(
            "log_w_delta_sd_participant", dist.LogNormal(-1.0, 1.0)
        )
        with numpyro.plate("n_participants", n_participants, dim=-3):
            with numpyro.plate("n_image_conditions", n_image_conditions, dim=-2):
                with numpyro.plate("n_grain", n_grains, dim=-1):
                    log_w_delta_participant = numpyro.sample(
                        "log_w_delta_participant",
                        dist.Normal(0.0, log_w_delta_sd_participant),
                    )

        ## image offset
        log_w_delta_sd_image = numpyro.sample(
            "log_w_delta_sd_image", dist.LogNormal(-1.0, 1.0)
        )
        with numpyro.plate("n_images", n_images):
            # note how unlike m_delta_participant, doesn't depend on condition.
            log_w_delta_image = numpyro.sample(
                "log_w_delta_image", dist.Normal(0.0, log_w_delta_sd_image)
            )

        # lapse rate parameter
        lam_mu = numpyro.sample("lam_mu", dist.Beta(1.0, 10.0))
        lam_sigma = numpyro.sample("lam_sigma", dist.LogNormal(-1.0, 1.0))

        with numpyro.plate("n_participants", n_participants):
            lam_participant = numpyro.sample(
                "lam_participant",
                dist.TruncatedDistribution(
                    base_dist=dist.Normal(lam_mu, lam_sigma), low=0.0, high=0.2
                ),
            )

        ## lower bound fixed by design
        gamma = 0.5

        # mean-centering deltas for each experimental condition
        for i in range(n_image_conditions):
            for j in range(n_grains):
                m_delta_participant = m_delta_participant.at[:, i, j].set(
                    m_delta_participant[:, i, j]
                    - jnp.mean(m_delta_participant[:, i, j])
                )
                log_w_delta_participant = log_w_delta_participant.at[:, i, j].set(
                    log_w_delta_participant[:, i, j]
                    - jnp.mean(log_w_delta_participant[:, i, j])
                )

        m_delta_image = m_delta_image - jnp.mean(
            m_delta_image
        )  # don't need the "at" syntax here because we're not indexing
        log_w_delta_image = log_w_delta_image - jnp.mean(
            log_w_delta_image
        )  # don't need the "at" syntax here because we're not indexing

        if include_participant_re:
            participant_delta_factor = 1.0
        else:
            participant_delta_factor = 0.0

        if include_image_re:
            image_delta_factor = 1.0
        else:
            image_delta_factor = 0.0

        m_final = (
            m_cond[image_condition_id, grain_id]
            + (
                participant_delta_factor
                * m_delta_participant[participant_id, image_condition_id, grain_id]
            )
            + (image_delta_factor * m_delta_image[image_name_id])
        )
        log_w_final = (
            log_w_cond[image_condition_id, grain_id]
            + (
                participant_delta_factor
                * log_w_delta_participant[participant_id, image_condition_id, grain_id]
            )
            + (image_delta_factor * log_w_delta_image[image_name_id])
        )

        psi = numpyro.deterministic(
            "psi",
            self.squished_weibull(
                stimulus_intensity=stimulus_intensity,
                m=m_final,
                w=log_w_final,
                gamma=gamma,
                lam=lam_participant[participant_id],
                log_w=True,
                fn_positive=function_positive,
            ),
        )
        if sample_obs:
            n_trials = data["n_trials"].values
            n_successes = data["n_successes"].values
            numpyro.sample("obs", dist.Binomial(n_trials, psi), obs=n_successes)

    def generate_reparam_config(self) -> dict:
        reparam_config = {
            "m_cond": LocScaleReparam(0),
            "m_delta_participant": LocScaleReparam(0),
            "m_delta_image": LocScaleReparam(0),
            "log_w_cond": LocScaleReparam(0),
            "log_w_delta_participant": LocScaleReparam(0),
            "log_w_delta_image": LocScaleReparam(0),
        }
        return reparam_config

    def generate_arviz_data(
        self,
        data: pd.DataFrame = None,
        n_prior_samples: int = 500,
        model_kwargs: dict = {},
    ):
        coords = {
            "participant": list(self.grouping_ids["participant"].values()),
            "image_condition": list(self.grouping_ids["image_condition"].values()),
            "grain": list(self.grouping_ids["grain"].values()),
            "image_name": list(self.grouping_ids["image_name"].values()),
        }
        dims = {
            "m_cond": ["image_condition", "grain"],
            "m_delta_participant": ["participant", "image_condition", "grain"],
            "m_delta_image": ["image_name"],
            "log_w_cond": ["image_condition", "grain"],
            "log_w_delta_participant": ["participant", "image_condition", "grain"],
            "log_w_delta_image": ["image_name"],
            "lam_participant": ["participant"],
        }
        return super().generate_arviz_data(
            data=data,
            n_prior_samples=n_prior_samples,
            model_kwargs=model_kwargs,
            from_numpyro_kwargs={"coords": coords, "dims": dims},
        )


# class ParticipantsAndImagesInteraction(PsychometricFunctionWrapper):
#     """
#     Multiple participants acting in all conditions

#     Threshold and width both fitted to each condition in a hierarchical model.
#     We also fit a separate threshold for each image.

#     The interaction designation specifies that we have separate thresholds for
#     each cell of the experimental design.

#     We needed to change the structure a bit from above.

#     """

#     def model(
#         self,
#         data: pd.DataFrame = None,
#         function_positive: bool = True,
#         sample_obs: bool = True,
#         include_participant_re: bool = True,
#         include_image_re: bool = True,
#     ):
#         if data is None:
#             data = self.data

#         stimulus_intensity = data[self.intensity_variable_name].values
#         congruent_id = data["congruent_id"].values
#         before_after_id = data["before_after_id"].values
#         image_id = data["template_id"].values
#         participant_id = data["participant_ID_id"].values
#         n_image_conditions = len(data["congruent_id"].unique())
#         n_grain = len(data["before_after_id"].unique())
#         n_images = len(data["template_id"].unique())
#         n_participants = len(data["participant_ID_id"].unique())

#         # threshold parameters
#         # basic structure per cell is cond + subj_delta + image_delta
#         m_mu = numpyro.sample("m_mu", dist.Normal(-2.0, 2.0))
#         m_sigma = numpyro.sample("m_sigma", dist.LogNormal(-1.0, 1.0))

#         ## cond
#         with numpyro.plate("n_image_conditions", n_image_conditions, dim=-2):
#             with numpyro.plate("n_grain", n_grain, dim=-1):
#                 m_cond = numpyro.sample(
#                     "m_cond",
#                     dist.Normal(m_mu, m_sigma),
#                 )

#         ## participant offset
#         m_delta_sd_participant = numpyro.sample(
#             "m_delta_sd_participant", dist.LogNormal(-1.0, 1.0)
#         )
#         with numpyro.plate("n_participants", n_participants, dim=-3):
#             with numpyro.plate("n_image_conditions", n_image_conditions, dim=-2):
#                 with numpyro.plate("n_grain", n_grain, dim=-1):
#                     m_delta_participant = numpyro.sample(
#                         "m_delta_participant", dist.Normal(0.0, m_delta_sd_participant)
#                     )

#         ## image offset
#         m_delta_sd_image = numpyro.sample("m_delta_sd_image", dist.LogNormal(-1.0, 1.0))
#         with numpyro.plate("n_images", n_images):
#             # note how unlike m_delta_participant, doesn't depend on condition.
#             m_delta_image = numpyro.sample(
#                 "m_delta_image", dist.Normal(0.0, m_delta_sd_image)
#             )

#         # width parameters
#         log_w_mu = numpyro.sample("log_w_mu", dist.Normal(0.0, 1.0))
#         log_w_sigma = numpyro.sample("log_w_sigma", dist.LogNormal(-1.0, 1.0))

#         ## cond
#         with numpyro.plate("n_image_conditions", n_image_conditions, dim=-2):
#             with numpyro.plate("n_grain", n_grain, dim=-1):
#                 log_w_cond = numpyro.sample(
#                     "log_w_cond",
#                     dist.Normal(log_w_mu, log_w_sigma),
#                 )

#         ## participant offset
#         log_w_delta_sd_participant = numpyro.sample(
#             "log_w_delta_sd_participant", dist.LogNormal(-1.0, 1.0)
#         )
#         with numpyro.plate("n_participants", n_participants, dim=-3):
#             with numpyro.plate("n_image_conditions", n_image_conditions, dim=-2):
#                 with numpyro.plate("n_grain", n_grain, dim=-1):
#                     log_w_delta_participant = numpyro.sample(
#                         "log_w_delta_participant",
#                         dist.Normal(0.0, log_w_delta_sd_participant),
#                     )

#         # lapse rate parameter
#         lam_mu = numpyro.sample("lam_mu", dist.Beta(1.0, 10.0))
#         lam_sigma = numpyro.sample("lam_sigma", dist.LogNormal(-1.0, 1.0))

#         with numpyro.plate("n_participants", n_participants):
#             lam_participant = numpyro.sample(
#                 "lam_participant",
#                 dist.TruncatedDistribution(
#                     base_dist=dist.Normal(lam_mu, lam_sigma), low=0.0, high=0.2
#                 ),
#             )

#         ## lower bound fixed by design
#         gamma = 0.5

#         # mean-centering deltas for each experimental condition
#         m_delta_image = m_delta_image - jnp.mean(
#             m_delta_image
#         )  # don't need the "at" syntax here because we're not indexing
#         for i in range(n_image_conditions):
#             for j in range(n_grain):
#                 m_delta_participant = m_delta_participant.at[:, i, j].set(
#                     m_delta_participant[:, i, j]
#                     - jnp.mean(m_delta_participant[:, i, j])
#                 )
#                 log_w_delta_participant = log_w_delta_participant.at[:, i, j].set(
#                     log_w_delta_participant[:, i, j]
#                     - jnp.mean(log_w_delta_participant[:, i, j])
#                 )

#         if include_participant_re:
#             participant_delta_factor = 1.0
#         else:
#             participant_delta_factor = 0.0

#         if include_image_re:
#             image_delta_factor = 1.0
#         else:
#             image_delta_factor = 0.0

#         m_final = (
#             m_cond[congruent_id, before_after_id]
#             + (
#                 participant_delta_factor
#                 * m_delta_participant[participant_id, congruent_id, before_after_id]
#             )
#             + (image_delta_factor * m_delta_image[image_id])
#         )
#         log_w_final = log_w_cond[congruent_id, before_after_id] + (
#             participant_delta_factor
#             * log_w_delta_participant[participant_id, congruent_id, before_after_id]
#         )

#         psi = numpyro.deterministic(
#             "psi",
#             self.squished_weibull(
#                 stimulus_intensity=stimulus_intensity,
#                 m=m_final,
#                 w=log_w_final,
#                 gamma=gamma,
#                 lam=lam_participant[participant_id],
#                 log_w=True,
#                 fn_positive=function_positive,
#             ),
#         )
#         if sample_obs:
#             n_trials = data["n_trials"].values
#             n_successes = data["n_successes"].values
#             numpyro.sample("obs", dist.Binomial(n_trials, psi), obs=n_successes)

#     def generate_reparam_config(self) -> dict:
#         reparam_config = {
#             "m_cond": LocScaleReparam(0),
#             "m_delta_participant": LocScaleReparam(0),
#             "m_delta_image": LocScaleReparam(0),
#             "log_w_cond": LocScaleReparam(0),
#             "log_w_delta_participant": LocScaleReparam(0),
#         }
#         return reparam_config

#     def generate_arviz_data(
#         self,
#         data: pd.DataFrame = None,
#         n_prior_samples: int = 500,
#         model_kwargs: dict = {},
#     ):
#         coords = {
#             "participant": list(self.grouping_ids["participant_ID"].values()),
#             "template": list(self.grouping_ids["template"].values()),
#             "congruent": list(self.grouping_ids["congruent"].values()),
#             "before_after": list(self.grouping_ids["before_after"].values()),
#         }
#         dims = {
#             "m_cond": ["congruent", "before_after"],
#             "m_delta_participant": ["participant", "congruent", "before_after"],
#             "m_delta_image": ["template"],
#             "log_w_cond": ["congruent", "before_after"],
#             "log_w_delta_participant": ["participant", "congruent", "before_after"],
#             "lam_participant": ["participant"],
#         }
#         return super().generate_arviz_data(
#             data=data,
#             n_prior_samples=n_prior_samples,
#             model_kwargs=model_kwargs,
#             from_numpyro_kwargs={"coords": coords, "dims": dims},
#         )


def expand_grid(data_dict):
    rows = itertools.product(*data_dict.values())
    return pd.DataFrame.from_records(rows, columns=data_dict.keys())


def group_bernoulli_trials_into_binomial(
    df: pd.DataFrame,
    stimulus_intensity: str,
    experiment_conditions: Optional[List[str]] = [],
    group_variables: Optional[List[str]] = [],
    bernoulli_outcome_column: str = "hit",
    use_rule_of_succession: bool = True,
) -> pd.DataFrame:
    """
    Group bernoulli trials into binomial trials.

    This function takes a dataframe containing one bernoulli trial
    per row, and groups it into a dataframe containing one binomial
    description (n_successes, n_trials) for each combination of
    grouping varibles.

    The method will also return 90% beta distribution confidence intervals,
    optionally using a rule-of-succession correction (add one success and one failure
    to every cell). The returned upper and lower limits for plotting
    use the maximum and minimum of the proportion correct and the beta distribution
    (so that the ci is always bounded 0 or 1, but we get some sense of uncertainty
    on the proportion correct).

    Args:
        df (pd.DataFrame): Dataframe containing bernoulli trials.
        stimulus_intensity: the variable names of the stimulus intensity values.
        experiment_conditions (List[str], optional): A list of variable names of experiment condition(s).
        group_variables (List[str], optional): A list of variable names of group-level values (e.g. participant, image).
        bernoulli_outcome_column (str, optional): The name of the column used for denoting success/failure.
                                                Defaults to "hit".
        use_rule_of_succession (bool, optional): If true, add one success and one failure to allow Beta distribution to be defined.
                                                Defaults to True.

    Returns:
        pd.DataFrame: Dataframe with one row per combination of grouping variables.
    """

    if use_rule_of_succession:
        rule_of_succession = 1
    else:
        rule_of_succession = 0

    group_list = [stimulus_intensity] + [*experiment_conditions] + [*group_variables]

    grouped_df = (
        df.groupby(group_list)
        .agg(
            n_successes=(bernoulli_outcome_column, "sum"),
            n_trials=(bernoulli_outcome_column, len),
            proportion_correct=(bernoulli_outcome_column, "mean"),
        )
        .assign(
            n_failures=lambda x: x["n_trials"] - x["n_successes"],
            # beta confidence intervals
            beta_lower=lambda x: beta.ppf(
                0.05,
                x["n_successes"] + rule_of_succession,
                x["n_failures"] + rule_of_succession,
            ),
            beta_mid=lambda x: beta.ppf(
                0.5,
                x["n_successes"] + rule_of_succession,
                x["n_failures"] + rule_of_succession,
            ),
            beta_upper=lambda x: beta.ppf(
                0.95,
                x["n_successes"] + rule_of_succession,
                x["n_failures"] + rule_of_succession,
            ),
            plot_lo=lambda x: np.min(
                [x["proportion_correct"], x["beta_lower"]], axis=0
            ),
            plot_hi=lambda x: np.max(
                [x["proportion_correct"], x["beta_upper"]], axis=0
            ),
        )
        .reset_index()
    )

    for v in [*experiment_conditions, *group_variables]:
        grouped_df[v] = grouped_df[v].astype("category")

    return grouped_df


def plot_psychometric_function(
    x: jnp.ndarray, df, post_pred_mean, post_pred_hpdi, ax=None, prior_samples_psi=None
):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 4))

    ax.hlines(0.5, x.min(), x.max(), "k", "--", alpha=0.5)

    if prior_samples_psi is not None:
        # user provided prior samples; plot.
        ax.plot(x, prior_samples_psi.T, alpha=0.05, ls="-", color="g")

    ax.fill_between(
        x, post_pred_hpdi[0, :], post_pred_hpdi[1, :], alpha=0.3, interpolate=True
    )
    ax.plot(x, post_pred_mean, label="posterior mean")
    ax.errorbar(
        df["reach"],
        df["proportion_correct"],
        yerr=[
            df["proportion_correct"] - df["plot_lo"],
            df["plot_hi"] - df["proportion_correct"],
        ],
        fmt="none",
        alpha=0.3,
    )
    ax.plot(
        "reach",
        "proportion_correct",
        data=df,
        marker="o",
        linestyle="",
        markeredgecolor="k",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Reach")
    ax.set_ylabel("Proportion correct")
    return ax
