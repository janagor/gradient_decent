from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from autograd import grad
from typing import Callable, Tuple, List
from cec2017.functions import f1, f2, f3


def booth(x):
    return np.square(x[0] + 2 * x[1] - 7) + np.square(2 * x[0] + x[1] - 5)


@dataclass
class Point:
    point: annotations.Union(list, np.ndarray)


@dataclass
class Params:
    x_start: Point
    learning_rate: float
    max_iter: int = 1000
    error: float = 0.01
    bounds: annotations.Tuple[float, float] = (float("-inf"), float("inf"))


@dataclass
class Output:
    x_min: np.array
    y_min: np.array
    x_vals: np.ndarray
    f_vals: np.ndarray
    iter: int


def steepest_descent(f: Callable, params: Params) -> Output:
    x = np.array(params.x_start)
    beta = params.learning_rate
    error = params.error
    max_iter = params.max_iter
    bounds = params.bounds

    X = np.array([x])
    Y = np.array([f(x)])

    grad_fct = grad(f)
    gradinet = grad_fct(x)
    change = beta * gradinet
    x = x - change
    i = 0

    while (
        np.linalg.norm(f(x) - f(X[-1])) > error
        and np.linalg.norm(gradinet) >= error
        and i < max_iter
    ):
        i += 1
        X = np.append(X, [x], axis=0)
        Y = np.append(Y, [f(x)], axis=0)
        gradinet = grad_fct(x)
        change = beta * gradinet
        x = x - change
        print(f"{i}: {f(x)}: {x}")
        for j, val in list(enumerate(x)):
            if val < bounds[0]:
                x[j] = bounds[0]
            elif val > bounds[1]:
                x[j] = bounds[1]
    output = Output(X[-1], f(X[-1]), X, Y, i)
    return output


def function_plotter(
    f: Callable,
    max_x: int,
    plot_step: float,
    x: np.ndarray,
    arrows_num: int = 5,
    dimentionality: int = 2,
    axis_indexes: Tuple[int] = (0, 1),
    file: str = None,
) -> None:

    x_arr = np.arange(-max_x, max_x, plot_step)
    y_arr = np.arange(-max_x, max_x, plot_step)
    X, Y = np.meshgrid(x_arr, y_arr)
    Z = np.empty(X.shape)
    temp = np.zeros(dimentionality)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            temp[axis_indexes[0]] = X[i, j]
            temp[axis_indexes[1]] = Y[i, j]
            Z[i, j] = f(temp)

    plt.contour(X, Y, Z, 20)
    step_diff = int(len(x) / arrows_num)
    cur_step = 0
    for i in range(arrows_num - 1):
        start_x = x[cur_step][0]
        start_y = x[cur_step][1]
        dx = x[cur_step + step_diff][0] - x[cur_step][0]
        dy = x[cur_step + step_diff][1] - x[cur_step][1]
        plt.arrow(
            start_x, start_y,
            dx, dy,
            head_width=1, head_length=2,
            fc="k", ec="k"
        )
        cur_step += step_diff
    plt.title("gradient decent steps")
    plt.xlabel(f"x[{axis_indexes[0]}]")
    plt.ylabel(f"x[{axis_indexes[1]}]")
    if file is not None:
        plt.savefig(f'{file}')
    plt.show()


plot_type = ["linear", "logarithmic"]


def result_value_plotter(
    f: Callable,
    X: List[np.ndarray],
    betas: List[float],
    file: str = None,
    plt_type: str = "linear",
):
    if plt_type not in plot_type:
        raise ValueError
    i = 0
    for x, beta in zip(X, betas):
        i += 1
        step_num = len(x)
        time = np.arange(step_num)
        vals = list(map(lambda xx: f(xx), x))
        if plt_type != plot_type[1]:
            plt.plot(time, np.array(vals), '-', label=f'{beta}')
        else:
            plt.plot(time, np.log(np.array(vals)), '-', label=f'{beta}')
    title1 = "Value of function in every step of\n gradient decent."
    plt.title(title1 + f" {plt_type.capitalize()}")
    plt.ylabel("value")
    plt.xlabel("step")
    plt.legend()
    if file is not None:
        plt.savefig(f'{file}')
    plt.show()


def test_booth():
    UPPER_BOUND = 100
    DIMENSIONALITY = 2
    x = np.random.uniform(-UPPER_BOUND, UPPER_BOUND, size=DIMENSIONALITY)
    betas = np.arange(1e-1, 1.25e-1, 0.05e-1)
    values = []
    for beta in betas:
        params = Params(x, beta, 60, 0.00000001, (-100, 100))
        X = steepest_descent(booth, params)
        values.append(X.x_vals)
    result_value_plotter(booth, values, betas, "booth_vals.png")

    params = Params(x, 1.1e-1, 1000, 0.001, (-100, 100))
    X = steepest_descent(booth, params)
    function_plotter(
        booth, 100, 1, X.x_vals, int(len(X.x_vals) / 2),
        DIMENSIONALITY, (0, 1), "booth.png"
    )


def test_CEC_f1():
    UPPER_BOUND = 100
    DIMENSIONALITY = 10
    x = np.random.uniform(-UPPER_BOUND, UPPER_BOUND, size=DIMENSIONALITY)
    values = []
    betas = np.arange(1e-11, 1.5e-9, 2e-10)
    betas = np.arange(1e-10, 1.5e-8, 2e-9)
    for beta in betas:
        params = Params(x, beta, 300, 0.000001, (-100, 100))
        X = steepest_descent(f1, params)
        values.append(X.x_vals)
    result_value_plotter(f1, values, betas, "f1_vals_log.png", "logarithmic")
    result_value_plotter(f1, values, betas, "f1_vals_lin.png")

    plot_dimentions = [(1, 8), (0, 7), (4, 5)]

    params = Params(x, 1e-8, 2000, 1e-9, (-100, 100))
    X = steepest_descent(f1, params)
    for num, dimentions in list(enumerate(plot_dimentions)):
        xtrans = np.transpose(X.x_vals)
        first = xtrans[dimentions[0]]
        second = xtrans[dimentions[1]]
        res = np.array([first, second])
        b = np.transpose(res)

        function_plotter(
            f1, 100, 1, b, len(first),
            DIMENSIONALITY, (dimentions[0], dimentions[1]), f"f1_3_{num}.png"
        )


def test_CEC_f2():
    UPPER_BOUND = 100
    DIMENSIONALITY = 10
    x = np.random.uniform(-UPPER_BOUND, UPPER_BOUND, size=DIMENSIONALITY)
    values = []
    betas = np.arange(1e-18, 1e-16, 2e-17)

    for beta in betas:
        params = Params(x, beta, 800, 0.000001, (-100, 100))
        X = steepest_descent(f2, params)
        values.append(X.x_vals)
    result_value_plotter(f2, values, betas, "f2_vals_log.png", "logarithmic")
    result_value_plotter(f2, values, betas, "f2_vals_lin.png")

    plot_dimentions = [(1, 8), (0, 7), (4, 5)]
    params = Params(x, 1.1e-17, 20000, 0.00001, (-100, 100))
    X = steepest_descent(f2, params)
    for num, dimentions in list(enumerate(plot_dimentions)):
        xtrans = np.transpose(X.x_vals)
        first = xtrans[dimentions[0]]
        second = xtrans[dimentions[1]]
        res = np.array([first, second])
        b = np.transpose(res)

        function_plotter(
            f2, 100, 1, b, len(first),
            DIMENSIONALITY, (dimentions[0], dimentions[1]), f"f2_2_{num}.png"
        )


def test_CEC_f3():
    UPPER_BOUND = 100
    DIMENSIONALITY = 10
    x = np.random.uniform(-UPPER_BOUND, UPPER_BOUND, size=DIMENSIONALITY)
    values = []
    betas = np.arange(1e-10, 1e-8, 2e-9)

    for beta in betas:
        params = Params(x, beta, 800, 0.000001, (-100, 100))
        X = steepest_descent(f3, params)
        values.append(X.x_vals)
    result_value_plotter(f3, values, betas, "f3_vals_log.png", "logarithmic")
    result_value_plotter(f3, values, betas, "f3_vals_lin.png")

    plot_dimentions = [(1, 8), (0, 7), (4, 5)]
    params = Params(x, 1e-9, 1000, 0.00001, (-100, 100))
    X = steepest_descent(f3, params)
    for num, dimentions in list(enumerate(plot_dimentions)):
        xtrans = np.transpose(X.x_vals)
        first = xtrans[dimentions[0]]
        second = xtrans[dimentions[1]]
        res = np.array([first, second])
        b = np.transpose(res)

        function_plotter(
            f3, 100, 1, b, len(first),
            DIMENSIONALITY, (dimentions[0], dimentions[1]), f"f3_3_{num}.png"
        )
    print(x)


def test_all():
    test_booth()
    test_CEC_f1()
    test_CEC_f2()
    test_CEC_f3()


if __name__ == "__main__":
    test_all()
