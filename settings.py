import numpy as np


class Settings:
    popularity = None

    loss_alpha: float = None
    loss_beta: float = None
    loss_scale: float = None
    loss_percentile: float = None

    metrics_alpha: float = None
    metrics_beta: float = None
    metrics_gamma: float = None
    metrics_scale: float = None
    metrics_percentile: float = None
    max_y_aux_popularity: float = None


def set_parameters(
        popularity,

        loss_alpha,
        loss_beta,
        loss_scale,
        loss_percentile,

        metrics_alpha,
        metrics_beta,
        metrics_gamma,
        metrics_scale,
        metrics_percentile
):
    Settings.popularity = popularity

    Settings.loss_alpha = loss_alpha
    Settings.loss_beta = loss_beta
    Settings.loss_scale = loss_scale
    Settings.loss_percentile = loss_percentile

    Settings.metrics_alpha = metrics_alpha
    Settings.metrics_beta = metrics_beta
    Settings.metrics_gamma = metrics_gamma
    Settings.metrics_scale = metrics_scale
    Settings.metrics_percentile = metrics_percentile

    domain = np.linspace(0, 1, 1000)
    codomain = [y_aux_popularity(x) for x in domain]
    Settings.max_y_aux_popularity = max(codomain)


def y_aux_popularity(x):
    f = 1 / (Settings.metrics_beta * np.sqrt(2 * np.pi))
    y = np.tanh(Settings.metrics_alpha * x) + \
        Settings.metrics_scale * f * np.exp(-1 / (2 * (Settings.metrics_beta ** 2)) * (x - Settings.metrics_percentile) ** 2)
    return y


def y_popularity(x):
    return y_aux_popularity(x) / Settings.max_y_aux_popularity
