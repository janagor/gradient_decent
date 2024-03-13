# Jan Górski 324960
from __future__ import annotations
from typing import Tuple, List, Any
import numpy as np
import time
from dataclasses import dataclass
from matplotlib import pyplot as plt


@dataclass
class StatisticResults:
    average: float
    std_dev: float
    min: int
    max: int


@dataclass
class KnapsackParameters:
    obj_num: int
    mass_range: Tuple[int]
    val_range: Tuple[int]


@dataclass
class SolutionEvaluation:
    res_all: int
    time_all: float
    res_pm_ratio: int
    time_pm_ration: float


@dataclass
class ComparisonSolversData:
    optimal_sols_ratio: float
    max_abs_error: float
    max_rel_error: float
    av_abs_error: float
    av_rel_error: float
    std_dev_abs_error: float
    std_dev_rel_error: float


@dataclass
class TimeEvaluation:
    max: float
    min: float
    av: float
    std_dev: float


def get_comp_sols_data(
    optimal_sols_ratio: float,
    iter_num: int,
    abs_errors: np.ndarray[float],
    rel_errors: np.ndarray[float]
):
    return ComparisonSolversData(
        float(optimal_sols_ratio) / float(iter_num),
        np.max(abs_errors),
        np.max(rel_errors),
        np.average(abs_errors),
        np.average(rel_errors),
        np.std(abs_errors),
        np.std(rel_errors),
    )


def get_time_ev_data(times):
    return TimeEvaluation(
        np.max(times),
        np.min(times),
        np.average(times),
        np.std(times)
    )


def knap_gen(
    obj_num: int, mass_range: Tuple[int, int], val_range: Tuple[int, int]
) -> Tuple[np.ndarray[int], float, np.ndarray[int]]:

    rng = np.random.default_rng()
    m = rng.integers(mass_range[0], mass_range[1], obj_num)
    M = np.sum(m) / 2
    p = rng.integers(val_range[0], val_range[1], obj_num)
    return (m, M, p)


def pow_set(s: List[Any]) -> List[Any]:
    ps = []
    size = len(s)
    for i in range(1 << size):
        subset = []
        for j in range(size):
            if 1 << j & i:
                subset.append(s[j])
        ps.append(subset)
    return ps


def knap_solver_all(masses: np.ndarray[int], vals: np.ndarray[int], max_accept) -> int:

    knaps = zip(masses, vals)
    pos_sols = pow_set(list(knaps))
    pos_sols.remove([])
    maximum = 0
    for sol in pos_sols:
        sol_val = 0
        sol_mass = 0
        for m, p in sol:
            sol_val += p
            sol_mass += m
        if sol_val > maximum and sol_mass <= max_accept:
            maximum = sol_val
    return maximum


def knap_solver_pm_ratio(
    masses: np.ndarray[int], vals: np.ndarray[int], max_accept
) -> int:

    ratios = list(np.array(vals) / np.array(masses))
    knaps = list(zip(masses, vals, ratios))
    knaps = sorted(knaps, key=lambda knaps: knaps[2])
    knaps.reverse()
    in_val = 0
    in_mass = 0
    sol = []
    for mass, val, _ in knaps:
        if in_mass + mass <= max_accept:
            sol.append([mass, val])
            in_val += val
            in_mass += mass
        if in_mass == max_accept:
            break
    return in_val


def evaluate_time_and_solution(
    obj_num,
    mass_range,
    val_range
) -> SolutionEvaluation:

    m, M, p = knap_gen(obj_num, mass_range, val_range)
    start_all = time.process_time()
    res_all = knap_solver_all(m, p, M)
    end_all = time.process_time()
    total_all = end_all - start_all
    time_all = total_all

    start_pm_ratio = time.process_time()
    res_pm_ratio = knap_solver_pm_ratio(m, p, M)
    end_pm_ratio = time.process_time()
    time_pm_ratio = end_pm_ratio - start_pm_ratio

    return SolutionEvaluation(
        res_all, time_all, res_pm_ratio, time_pm_ratio
    )


def compare_knap_solvs(
    iter_num: int, obj_num: int,
    mass_range: Tuple[int, int], val_range: Tuple[int, int]
) -> Tuple[ComparisonSolversData, TimeEvaluation, TimeEvaluation]:
    optimal_sols_ratio = 0
    abs_errors = np.zeros(iter_num)
    rel_errors = np.zeros(iter_num)

    times_all = np.zeros(iter_num)
    times_pm_ratio = np.zeros(iter_num)
    sols_all = np.zeros(iter_num)
    sols_pm_ratio = np.zeros(iter_num)

    for i in range(iter_num):
        evaluation = evaluate_time_and_solution(obj_num, mass_range, val_range)
        sols_all[i] = evaluation.res_all
        times_all[i] = evaluation.time_all
        sols_pm_ratio[i] = evaluation.res_pm_ratio
        times_pm_ratio[i] = evaluation.time_pm_ration

        if sols_all[i] == sols_pm_ratio[i]:
            optimal_sols_ratio += 1

    abs_errors = sols_all - sols_pm_ratio
    rel_errors = abs_errors / sols_all
    comp_sols_data = get_comp_sols_data(
        optimal_sols_ratio, iter_num, abs_errors, rel_errors
    )

    return (
        comp_sols_data,
        get_time_ev_data(times_all),
        get_time_ev_data(times_pm_ratio)
    )


def compare_knap_solvs_different_sizes(
    iter_num: int,
    obj_num_range: Tuple[int, int],
    mass_range: Tuple[int, int],
    val_range: Tuple[int, int],
) -> Tuple[np.ndarray[int], float, np.ndarray[int]]:
    comparison_results = []
    for obj_num in range(obj_num_range[0], obj_num_range[1] + 1):
        comparison_results.append(compare_knap_solvs(
            iter_num,
            obj_num,
            mass_range,
            val_range)
        )
    return comparison_results


plot_type = ["linear", "logarithmic"]


def result_plotter(
    obj_num_range: Tuple[int, int],
    results: List[Tuple[
        ComparisonSolversData, TimeEvaluation, TimeEvaluation
    ]],
    file: str = None,
    plt_type: str = "linear",
):
    if plt_type not in plot_type:
        raise ValueError
    times_all = list(map(
        lambda time_eval: time_eval[1].av, results
    ))
    times_pm_ratio = list(map(
        lambda time_eval: time_eval[2].av, results
    ))
    elems_num = np.arange(obj_num_range[0], obj_num_range[1] + 1)
    if plt_type != plot_type[1]:
        plt.plot(elems_num, times_all, "o", label="all")
        plt.plot(elems_num, times_pm_ratio, "o", label="pm_ratio")
    else:
        plt.plot(elems_num, np.log(times_all), "o", label="all")
        plt.plot(elems_num, np.log(times_pm_ratio), "o", label="pm_ratio")
    title1 = "Computing time to elements number\n"
    title2 = f"ratio in rapsack problem. {plt_type.capitalize()}."

    plt.title(title1 + title2)
    plt.xlabel("Number of elements")
    plt.ylabel("Computing time [s]")
    plt.legend()
    if file is not None:
        plt.savefig(f'{file}')
    plt.show()


if __name__ == "__main__":
    MASSES_RANGE = (900, 1000)
    VALUES_RANGE = (900, 1000)
    ELEMENTS_RANGE = (1, 12)
    ITER_NUM = 100
    results = compare_knap_solvs_different_sizes(
        ITER_NUM, ELEMENTS_RANGE, MASSES_RANGE, VALUES_RANGE
    )
    result_plotter(ELEMENTS_RANGE, results, "test_log.png", "logarithmic")
    result_plotter(ELEMENTS_RANGE, results, "test_lin.png")
