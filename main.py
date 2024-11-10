import numpy as np
import cmdstanpy


def simulate_regions(n_timesteps: int, n_regions: int) -> np.ndarray:
    rng = np.random.RandomState(0)
    time = np.arange(n_timesteps)
    rfx_int = rng.normal(loc=0, scale=0.2, size=n_regions)
    rfx_slope = rng.normal(loc=0, scale=0.2, size=n_regions)

    expectation = np.exp(
        (0.5 + rfx_slope) * np.sin(2 * np.pi * time.reshape(-1, 1) / 7) + 2.5 + rfx_int
    )

    return expectation


def create_counterfactuals(expectation: np.ndarray, lift=1.05) -> np.ndarray:
    n_timesteps, n_regions = expectation.shape
    rng = np.random.RandomState(0)

    noise = rng.multivariate_normal(
        mean=np.zeros(n_regions), cov=np.eye(n_regions), size=(2, n_timesteps)
    )

    Y0 = expectation + noise[0]
    Y1 = lift * expectation + noise[1]

    return Y0, Y1


def create_geo_data(Y0: np.ndarray, Y1: np.ndarray, treat_time: int = 30, n_treated=5):
    n_timesteps, n_regions = Y0.shape

    treated_times = np.arange(n_timesteps) >= treat_time
    treated_regions = np.arange(n_regions) < n_treated

    yc0 = Y0[np.ix_(~treated_times, ~treated_regions)]
    yc1 = Y0[np.ix_(treated_times, ~treated_regions)]
    yt0 = Y0[np.ix_(~treated_times, treated_regions)]
    yt1 = Y1[np.ix_(treated_times, treated_regions)]
    return (yc0, yc1, yt0, yt1)


if __name__ == "__main__":
    expectation = simulate_regions(n_timesteps=7 * 8, n_regions=50)

    Y0, Y1 = create_counterfactuals(expectation)

    yc0, yc1, yt0, yt1 = create_geo_data(Y0, Y1)

    stan_data = dict(
        n_timesteps_pre=len(yc0),
        n_timesteps_post=len(yc1),
        n_control_regions=yc1.shape[1],
        n_treatment_regions=yt0.shape[1],
        y_control_pre=yc0,
        y_control_post=yc1,
        y_treatment_pre=yt0,
        y_treatment_post=yt1,
        sample=0,
    )

    model = cmdstanpy.CmdStanModel(stan_file="geo_model.stan")
    fit = model.sample(data=stan_data)

    print(fit.stan_variables().keys())
