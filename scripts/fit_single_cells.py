"""
A script to fit each cell of the example 
dataset with a psychometric function.
"""

# %%

import lzma
import pickle
from itertools import product

import jax.numpy as jnp
import numpyro
import pandas as pd
from jax import random
from numpyro.diagnostics import hpdi

from src.helpers import get_top_directory, preprocess_data
from src.numpyro_models import (
    SinglePsychometricFunction,
    group_bernoulli_trials_into_binomial,
)

numpyro.set_platform("cpu")
numpyro.set_host_device_count(4)


rng_key = random.PRNGKey(42)
top_dir = get_top_directory()
df = preprocess_data()

# %%

# create a binomial grouped data structure:
grouped_df = group_bernoulli_trials_into_binomial(
    df,
    stimulus_intensity="reach",
    experiment_conditions=["grain", "image_condition"],
    group_variables=["participant"],
    bernoulli_outcome_column="correct",
    use_rule_of_succession=True,
)
grouped_df = grouped_df.dropna()

# subset for testing:
# grouped_df = grouped_df.loc[
#     (grouped_df.participant == "participant_1")
#     & ((grouped_df.grain == 5.0) | (grouped_df.grain == 1.0))
#     & (grouped_df.image_condition == "scenes")
# ]
grouped_df


# %%

model_dict = {}
pred_df = pd.DataFrame()

for participant, grain, image_condition in product(
    grouped_df.participant.unique(),
    grouped_df.grain.unique(),
    grouped_df.image_condition.unique(),
):
    print(f"Fitting {participant}, {grain}, {image_condition}...\n")
    subset_df = grouped_df.loc[
        (grouped_df.participant == participant)
        & (grouped_df.grain == grain)
        & (grouped_df.image_condition == image_condition)
    ]
    if subset_df.shape[0] == 0:
        print(f"No data for {participant}, {grain}, {image_condition}")
    else:
        m1 = SinglePsychometricFunction(
            data=subset_df, rng_key=rng_key, intensity_variable_name="reach"
        )
        m1.sample(num_warmup=2000, num_samples=1000, num_chains=4)

        # generate samples for plotting:
        x = jnp.logspace(jnp.log10(0.3), jnp.log10(16), 51)
        new_df = pd.DataFrame(
            {
                "reach": x,
                "participant": participant,
                "grain": grain,
                "image_condition": image_condition,
            }
        )
        prior_samples = m1.predict(data=new_df, prior=True, sample_obs=False)
        posterior_samples = m1.predict(data=new_df, prior=False, sample_obs=False)
        post_pred_mean = posterior_samples["psi"].mean(axis=0)
        post_pred_hpdi = hpdi(posterior_samples["psi"], 0.9)
        new_df["post_mean"] = post_pred_mean
        new_df["post_lower"] = post_pred_hpdi[0]
        new_df["post_upper"] = post_pred_hpdi[1]
        pred_df = pd.concat([pred_df, new_df])

        model_dict[(participant, grain, image_condition)] = {
            "model": m1,
            "prior_samples": prior_samples,
            "posterior_samples": posterior_samples,
        }


# %%
# Save output in compressed file:
output_file = top_dir / "results" / "single_psychometric_fits.xz"
print(f"Saving output to {output_file}")
with lzma.open(output_file, "wb") as f:
    pickle.dump((model_dict, pred_df), f)

print("Done!")
# can be re-loaded with:
# with lzma.open(output_file, "r") as f:
#     tmp = f.read()
#     model_dict, pred_df = pickle.loads(tmp)
