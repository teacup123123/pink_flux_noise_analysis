import dataclasses
import itertools

import numpy as np
import pylab as pl
import pickle as pk

from standalone import *
import scipy.io as sio

A_Phi0 = 1.2e-6


@dataclasses.dataclass
class fq_hyperbolic_transition:
    Delta: float
    Ip: float

    def qubit_freq_GHz(self, offset_Phi0):
        return np.sqrt(self.Delta ** 2 + (2 * self.Ip * Phi0 * offset_Phi0 / h / 1e9) ** 2)


def analytic_Gamma2R(fq: fq_hyperbolic_transition, offset_from_opti_Phi0, A_Phi0, ln_factor=14):
    return (2 * fq.Ip / hbar) ** 2 * \
           (Phi0 * offset_from_opti_Phi0) / (2e9 * pi * fq.Delta) \
           * (A_Phi0 * Phi0) * np.sqrt(ln_factor)


def analytic_Gamma2E(fq: fq_hyperbolic_transition, offset_from_opti_Phi0, A_Phi0):
    return (2 * fq.Ip / hbar) ** 2 * \
           (Phi0 * offset_from_opti_Phi0) / (2e9 * pi * fq.Delta) \
           * (A_Phi0 * Phi0) * np.sqrt(np.log(2))


def save():
    tir = 1_300_000_000  # ns
    tend = 4 * tir
    tuv = 650  # ns
    ln_factor = np.log(tir / tuv)
    print(f'ln_factor={ln_factor}')

    PLOT = False
    # PLOT = True

    ng1of = NoiseGen1OverF(t_step_ns=tuv, n_fft=tend // tuv)

    t2rs = []
    t2es = []
    formula2Rs = []
    formula2Es = []
    # x_Phi0s = np.linspace(100e-6, 1e-6, 12)
    x_Phi0 = 0.0

    fq = fq_hyperbolic_transition(5.0, 300e-9)

    def logscale(a, b, n):
        return np.logspace(np.log10(a), np.log10(b), n)

    deltas = logscale(8.0, 2.5, 5)
    ips = logscale(170e-9, 400e-9, 5)
    A_Phi0s = logscale(0.7e-6, 2.5e-6, 5)
    dia = list(itertools.product(deltas, ips, A_Phi0s))

    for d, i, a in (dia):
        flux_trajectory_Phi0, *_ = ng1of.generate(A2=a ** 2, t_cut_off_ns=tir)
        fq.Delta = d
        fq.Ip = i
        print(f'================{d, i, a}================')
        formula2R = analytic_Gamma2R(fq, x_Phi0, A_Phi0, ln_factor=ln_factor)
        formula2E = analytic_Gamma2E(fq, x_Phi0, A_Phi0)
        (t2r, t2e) = qb_plot_t2s_at(fq,
                                    t_step_ns=tuv,
                                    t_observation_ns=tuv * 1000,
                                    t_cut_off_ns=tir,
                                    t_total_ns=tend,

                                    generated=flux_trajectory_Phi0,

                                    A_amplitude_Phi0=A_Phi0, offset_Phi0=x_Phi0,

                                    laziness=16 * 4,

                                    plot=PLOT
                                    )
        (t2r, t2e) = map(float, (t2r, t2e))

        print(f'formula2R = {formula2R / 1e9} GHz = {1e9 / formula2R} ns')
        print(f'      t2r = {1 / t2r} GHz = {t2r} ns')
        print(f'formula2E = {formula2E / 1e9} GHz = {1e9 / formula2E} ns')
        print(f'      t2e = {1 / t2e} GHz = {t2e} ns')
        t2rs.append(t2r)
        t2es.append(t2e)
        formula2Rs.append(formula2R)
        formula2Es.append(formula2E)
        if PLOT: pl.show()
    data = {
        'formula2Rs': formula2Rs,
        'formula2Es': formula2Es,
        't2rs': t2rs,
        't2es': t2es,
        'dia': dia,
    }
    with open('data125.pickle', 'wb') as f:
        pk.dump(data, f)


if __name__ == '__main__':
    with open('data125.pickle', 'rb') as f:
        data = pk.load(f)
        dia = data['dia']
        t2rs = data['t2rs']
        t2es = data['t2es']
        dia, t2rs, t2es = map(np.array, (dia, t2rs, t2es))
        gamma2rs = 1e9 / t2rs
        gamma2es = 1e9 / t2es
        d, i, a = dia.T
        conste = gamma2es / ((i * a * Phi0 / h) ** 2 / (d * 1e9))
        constr = gamma2rs / ((i * a * Phi0 / h) ** 2 / (d * 1e9))
        print()
