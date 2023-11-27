import math
import random as rand


def triangle_function(x: int, c: float = 1.0, steps: int = 4):
    """ creates output in a triangle pattern """
    t = [i/steps for i in range(steps+1)] + \
        [(steps-(i+1))/steps for i in range(steps-1)]
    return c*t[x % len(t)]


def generate_single_longterm_data(
    series_length: int,
    amount_steps: int = None,
    noise: bool = True,
    horizon: int = 20
):
    if amount_steps is None:
        amount_steps = rand.randint(5, 15)

    pure_training_data = [
        triangle_function(x, steps=amount_steps)
        for x in range(series_length)
    ]
    if noise:
        new_data_series = [
            x + rand.gauss(0, 0.02)
            for x in pure_training_data
        ]
    else:
        new_data_series = pure_training_data
    new_label_series = [
        triangle_function(x, steps=amount_steps)
        for x in range(series_length, series_length+horizon)
    ]
    return new_data_series, new_label_series


def generate_longterm_data(
    amount: int,
    variable_steps: bool = False,
    noise: bool = False,
    horizon: int = 20
):
    """ generates a data set of series with variable length between 5 and 50 """
    min_len = 12
    max_len = 100
    amount_steps = 40

    data = []
    labels = []
    for _ in range(amount):
        amount_steps = rand.randint(7, 50)
        series_length = rand.randint(min_len, max_len)
        new_data_series, new_label_series = generate_single_longterm_data(
            series_length,
            amount_steps=amount_steps if not variable_steps else None,
            noise=noise,
            horizon=horizon
        )
        data.append(new_data_series)
        labels.append(new_label_series)
    return (data, labels)
