import numpy as np
from matplotlib import pyplot as plt, cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

from Conferences.SIGIR.CMN_our_interface.Pinterest.PinterestICCVReader import PinterestICCVReader


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

    new_loss = False


def get_popularity(matrix):
    return matrix.A.sum(axis=0)


def get_percentile(popularity_array, k):
    # print('Luciano > popularity_array:', popularity_array)
    sorted_popularity_array = np.sort(popularity_array)
    index = int(round(popularity_array.shape[0] * k / 100))
    # print('Luciano > len:', popularity_array.shape[0])
    # print('Luciano > index:', index)
    percentile = sorted_popularity_array[index]
    # print('Luciano > percentile:', percentile)
    return percentile


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
        metrics_percentile,
        new_loss
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

    Settings.new_loss = new_loss

    domain = np.linspace(0, 1, 1000)
    codomain = [y_aux_popularity(x) for x in domain]
    Settings.max_y_aux_popularity = max(codomain)

    print('SETTINGS -------------------------------')
    print('Settings.loss_alpha:', Settings.loss_alpha)
    print('Settings.loss_beta:', Settings.loss_beta)
    print('Settings.loss_scale:', Settings.loss_scale)
    print('Settings.loss_percentile:', Settings.loss_percentile)

    print('Settings.metrics_alpha:', Settings.metrics_alpha)
    print('Settings.metrics_beta:', Settings.metrics_beta)
    print('Settings.metrics_gamma:', Settings.metrics_gamma)
    print('Settings.metrics_scale:', Settings.metrics_scale)
    print('Settings.metrics_percentile:', Settings.metrics_percentile)

    print('Settings.new_loss:', Settings.new_loss)


def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    return y


def y_aux_popularity(x):
    f = 1 / (Settings.metrics_beta * np.sqrt(2 * np.pi))
    y = np.tanh(Settings.metrics_alpha * x) + \
        Settings.metrics_scale * f * np.exp(-1 / (2 * (Settings.metrics_beta ** 2)) * (x - Settings.metrics_percentile) ** 2)
    return y


def y_popularity(x):
    y = y_aux_popularity(x) / Settings.max_y_aux_popularity
    return y


def y_position(x, cutoff):
    y = sigmoid(-x * Settings.metrics_gamma / cutoff) + 0.5
    return y


def y_custom(popularity, position, cutoff):
    y = y_popularity(popularity) * y_position(position, cutoff)
    return y


if __name__ == "__main__":
    print("Testing settings")

    dataset = PinterestICCVReader()

    URM_train = dataset.URM_train.copy()

    popularity = get_popularity(URM_train)

    min_value = np.min(popularity)
    max_value = np.max(popularity)
    gap = max_value - min_value

    popularity = (popularity - min_value) / gap

    set_parameters(
        popularity=popularity,
        loss_alpha=200,
        loss_beta=0.02,
        loss_scale=1,
        loss_percentile=get_percentile(popularity, 45),

        metrics_alpha=100,
        metrics_beta=0.03,
        metrics_gamma=5,
        metrics_scale=1 / 15,
        metrics_percentile=0.45,
        new_loss=False
    )

    print('percentile 45:', get_percentile(popularity, 45))
    print('percentile 99:', get_percentile(popularity, 99))
    print('percentile 1:', get_percentile(popularity, 1))
    print('max pop:', np.max(popularity))
    print('min pop:', np.min(popularity))
    print('n. 99:', np.sum(popularity <= 0.4116222760290557))
    print('n. 1:', np.sum(popularity > 0.4116222760290557))
    print('len:', len(popularity))
    print('check:', np.sum(popularity <= 0.4116222760290557) + np.sum(popularity > 0.4116222760290557))

    cutoff = 5
    points = 1000

    x_1 = np.linspace(0, 1, points)
    x_2 = np.linspace(0, cutoff, points)

    y_1 = y_popularity(x_1)
    y_2 = y_position(x_2, cutoff)

    plt.figure()
    plt.plot(x_1, y_1)
    plt.plot(x_2, y_2)
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    plt.show()

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make data.
    x_1, x_2 = np.meshgrid(x_1, x_2)
    z = y_custom(x_1, x_2, cutoff)
    # print(z)
    print('max_value:', np.max(z))

    # Plot the surface.
    surf = ax.plot_surface(x_1, x_2, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(0, 1)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=2)

    plt.show()
