import dataclasses

import numpy as np
import pylab as pl

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


if __name__ == '__main__':
    tir = 1_000_000_000 // 5  # ns
    tend = 5 * tir
    tuv = 1000 // 5  # ns
    ln_factor = np.log(tir / tuv)
    print(f'ln_factor={ln_factor}')

    PLOT = False
    # PLOT = True

    ng1of = NoiseGen1OverF(t_step_ns=tuv, n_fft=tend // tuv)
    flux_trajectory_Phi0, *_ = ng1of.generate(A2=A_Phi0 ** 2, t_cut_off_ns=tir)

    t2rs = []
    t2es = []
    formula2Rs = []
    formula2Es = []
    # x_Phi0s = np.linspace(100e-6, 1e-6, 12)
    x_Phi0s = np.logspace(np.log10(100e-6), np.log10(0.1e-6), 25)
    fq = fq_hyperbolic_transition(5.0, 300e-9)

    sio.savemat('noise.mat', {
        'flux_trajectory_Phi0': flux_trajectory_Phi0,
        'tend': tend,
        'tir': tir,
        'tuv': tuv,
        'fq_Delta': fq.Delta,
        'fq_freq_t': fq.qubit_freq_GHz(flux_trajectory_Phi0),
        'fq_Ip': fq.Ip,
    })

    for x_Phi0 in x_Phi0s:
        print(f'================{x_Phi0}================')
        formula2R = analytic_Gamma2R(fq, x_Phi0, A_Phi0, ln_factor=ln_factor)
        formula2E = analytic_Gamma2E(fq, x_Phi0, A_Phi0)
        (t2r, t2e) = qb_plot_t2s_at(fq,
                                    t_step_ns=tuv,
                                    t_observation_ns=tuv * 1000,
                                    t_cut_off_ns=tir,
                                    t_total_ns=tend,

                                    generated=flux_trajectory_Phi0,

                                    A_amplitude_Phi0=A_Phi0, offset_Phi0=x_Phi0,
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
    sio.savemat('for_matlab.mat', {
        'formula2Rs': formula2Rs,
        'formula2Es': formula2Es,
        't2rs': t2rs,
        't2es': t2es,
        'x_Phi0s': x_Phi0s,
        'A_Phi0': A_Phi0,
    })
