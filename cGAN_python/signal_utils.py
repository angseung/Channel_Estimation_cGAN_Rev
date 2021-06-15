import numpy as np
from scipy.signal import convolve

def rayleigh_channel(channel_shape = None, distance_rate = None, sig_power = None, profile = None):
    if (type(channel_shape) is int):
        channel_shape = [channel_shape, 1]

    elif (type(channel_shape) is not list):
        raise TypeError("'channel_shape' should be int or list type value!")

    (multipath, c) = channel_shape

    if (c != 1):
        row = c

    else:
        row = 1

    if (profile is None):
        ch_profile = np.exp(-1 * np.array(range(1, multipath + 1)) / 5)

    else:
        ch_profile = profile
        (_, multipath) = profile.shape

    ch_profile = np.ones((row, 1)) * ch_profile / np.sum(ch_profile, axis = 0)

    ch_coef = (np.random.randn(row, multipath) + 1j * np.random.randn(row, multipath)) * np.sqrt(0.5)
    ch_coef = np.multiply(ch_coef, np.sqrt(ch_profile))

    if (sig_power is not None):
        exp_beta = 3
        sigma = 10
        shadowing = np.random.normal(loc = 0, scale = 1, size = (1, 1)) * sigma

        path_loss = -10 * exp_beta * np.log10(distance_rate)
        loss = 10 ** ((path_loss + shadowing.item()) / 10)

        ch_coef = ch_coef * np.sqrt(loss) * np.srqt(sig_power)

        rx_power = loss * sig_power

    else:
        rx_power = 1

    return (ch_coef, rx_power)


def rayleigh_channel_v2(channel_shape = None, distance_rate = None, sig_power = None,
                        mode = None, profile = None):
    ## Initialize params using args
    if (profile is None):
        profile = 0

    if (mode is None):
        mode = 1

    if (sig_power is None):
        sig_power = 1

    if (distance_rate is None):
        distance_rate = 1

    if (type(channel_shape) == int):
        channel_shape = [channel_shape, 1]

    elif (type(channel_shape) != list):
        raise TypeError("'channel_shape' should be int or list type value!")

    ## Check multipath shape
    (multipath, c) = channel_shape

    if (c != 1):
        row = c

    else:
        row = 1

    ## Set profile value
    if (profile == 0):
        ## Use defualt channel profile
        ch_profile = np.exp(-1 * np.array(range(1, multipath + 1)) / 5)

    else:
        ## Use provided profile arg
        ch_profile = profile
        (_, multipath) = profile.shape

    ## Normalize channel profile
    ch_profile = np.ones((row, 1)) * ch_profile / np.sum(ch_profile, axis = 0)

    ## Multifly profile with random matrix
    rnd = (np.random.randn(row, multipath) + 1j * np.random.randn(row, multipath)) * np.sqrt(0.5)

    ## Apply Racian model
    if (mode == 2):
        # LOS power rate
        K = np.zeros((row, multipath))
        K[:, 0] = 10 ** (15 / 10)

        # Apply power rate
        rnd = np.sqrt(K / (K + 1)) * np.sqrt(1 / (K + 1))

    ch_coef = rnd * np.sqrt(ch_profile)

    ## Apply distance attenuation
    exp_beta = 3 # Default 3~4, ideal 2
    sigma = 0 # Default 5~14, LTE 10
    shadowing = np.random.normal(loc = 0, scale = 1, size = (1, 1)) * sigma

    path_loss = -10 * exp_beta * np.log10(distance_rate)
    loss = 10 ** ((path_loss + shadowing.item()) / 10)

    ch_coef = ch_coef * np.sqrt(loss) * np.sqrt(sig_power)

    rx_power = loss * sig_power

    return (ch_coef, rx_power)


def awgn_noise(hx = [], snr = 1):
    (p, q) = hx.shape
    noise_power = 10 ** (-snr / 10)
    n = (np.random.randn(p, q) + 1j * np.random.randn(p, q)) * np.sqrt(noise_power / 2)
    y = hx + n

    return (y, noise_power)


def base_mod(data = [], mod_scheme = 2):
    if (mod_scheme == 1):
        mod_data = 2 * data - 1

    elif (mod_scheme == 2):
        (mp_row, mp_col) = data.shape

        if (np.mod(mp_col, 2) != 0):
            raise ValueError

        odd_data = data[:, 0 : mp_col : 2]
        odd_data = 1j * (2 * odd_data - 1)

        even_data = data[:, 1 : mp_col + 1 : 2]
        even_data = 2 * even_data - 1

        mod_data = 0.7071 * (odd_data + even_data)

    elif (mod_scheme == 4):
        (mp_row, mp_col) = data.shape

        while (np.mod(mp_col, 4) != 0):
            raise ValueError

        first_data = data[:, 0 : mp_col - 2 : 4]
        first = 4 * first_data - 2

        second_data = data[:, 1 : mp_col - 1 : 4]
        second = 2 * (second_data != first_data) - 1

        third_data = data[:, 2 : mp_col : 4]
        third = 1j * (4 * third_data - 2)

        fourth_data = data[:, 3 : mp_col + 1 : 4]
        fourth = 1j * (2 * (fourth_data != third_data) - 1)

        mod_data = (first + second + third + fourth) * 0.3162

    return mod_data


def base_demod(mod_data = [], mod_scheme = 2):
    if (type(mod_data) == list):
        mod_data = np.array(mod_data)

    RE = np.real(mod_data)
    IM = np.imag(mod_data)

    if (mod_scheme == 1):
        demod_data = 1 * (mod_data > 0)

    elif (mod_scheme == 2):
        (mp_row, mp_col) = mod_data.shape

        odd_data = 1 * (IM > 0)
        even_data = 1 * (RE > 0)

        temp = np.vstack((odd_data, even_data))

        demod_data = temp.reshape((mp_row, 2 * mp_col), order = 'F')

    elif (mod_scheme is 4):
        (mp_row, mp_col) = mod_data.shape

        first = 1 * (RE > 0)
        second = 1 * (np.abs(RE) < 0.6325)
        third = 1 * (IM > 0)
        fourth = 1 * (np.abs(IM) < 0.6325)

        temp = np.vstack((first, second, third, fourth))

        demod_data = temp.reshape((mp_row, 4 * mp_col), order = 'F')

    return demod_data


def conv(x = [], h = []):
    try:
        if ((x.ndim != 1) or (h.ndim != 1)):
            x = x.flatten()
            h = h.flatten()
    except:
        raise TypeError("Input x and h MUST BE numpy ndarray!!")

    conv_data = convolve(x, h)

    return conv_data