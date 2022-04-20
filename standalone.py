from typing import Protocol, TypeVar
import dataclasses
import time

import matplotlib.pyplot as pl
import numpy as np
import pylab as pl
import scipy.optimize as opt
import tqdm

Phi0 = 2.0678e-15
''' in units of Weber (Wb) = T/m^2'''
hbar = 1.0545718e-34
h = 6.62607004e-34
pi = np.pi
phi0 = Phi0 / 2 / pi
u0 = 4e-7 * pi


# noinspection DuplicatedCode,PyPep8Naming
@dataclasses.dataclass
class NoiseGen1OverF:
    """Noise generator of pink noise, """
    t_step_ns: float
    n_fft: int = 16_000

    @property
    def t_total_ns(self):
        return self.n_fft * self.t_step_ns

    @t_total_ns.setter
    def t_total_ns(self, value):
        """beware, this set is only approximate, coz of ceil function"""
        self.n_fft = int(np.ceil(value / self.t_step_ns))

    @property
    def fft_freqs_GHz(self):
        return np.fft.fftfreq(self.n_fft, d=self.t_step_ns)

    @property
    def fft_ts_ns(self):
        return np.arange(self.n_fft) * self.t_step_ns

    def power_spectrum_from_fft(self, fft):
        """
        numerical observation:
        fft = sum( X(t_i) phase(t_i) ,i) = n_fft * <X(t_i)phase(t_i)>

        The following two expressions are the same:
        |n_fft * <X(t_i)phase(t_i)>|^2 * dt / (2 pi n_fft) ~ n_fft dt X^2 / (2pi)
        1/(2pi) * integral[<X(t_i)X(0)phase(t_i)>dt] ~  n_fft dt X^2 / (2pi)
        """
        return np.abs(fft) ** 2 * self.t_step_ns / (2 * np.pi * self.n_fft)

    def power_spectrum(self, temperal_all):
        """
        brute-force calculation of power_spectrum, using the expression
        1/(2pi) * integral[<X(t_i)X(0)phase(t_i)>dt] ~ n_fft dt X^2 / (2pi)
        """
        auto_correlation = np.zeros(self.n_fft)
        for i in tqdm.tqdm(range(self.n_fft), total=self.n_fft):
            move = i
            auto_correlation += temperal_all[move] * np.roll(temperal_all, -move) / self.n_fft
        power_spectral = np.real(np.fft.fft(auto_correlation) * self.t_step_ns / 2 / np.pi)
        return power_spectral

    def generate(self, A2, t_cut_off_ns=None,
                 fixed_radial_cutoff=False,  # should the Fourrier component amplitudes be A2/f? or just the envelope?
                 return_fft_coef=True,
                 return_power_spectrum=True
                 ):
        print('generating noise')
        frequencies_GHz = self.fft_freqs_GHz
        df = (frequencies_GHz[1] - frequencies_GHz[0])
        fcutoff_GHz = 1. / t_cut_off_ns if t_cut_off_ns is not None else df
        filter = np.sqrt(A2 / np.maximum(np.abs(frequencies_GHz), fcutoff_GHz) * self.n_fft / self.t_step_ns)
        fft_coeffs = (np.random.randn(self.n_fft) + 1j * np.random.randn(self.n_fft))
        if fixed_radial_cutoff:
            fft_coeffs *= np.sqrt(2) / np.abs(fft_coeffs)  # now NORMALIZED in amplitude with random phase
        fft_coeffs = fft_coeffs * filter
        fft_coeffs[0] = 0.  # it is not our job to place an offset

        temperal_samples_all = np.fft.ifft(fft_coeffs)
        temperal_samples_all = temperal_samples_all.real
        fft_coeffs = np.fft.fft(temperal_samples_all) if return_fft_coef else None
        power_spectrum = self.power_spectrum_from_fft(fft_coeffs) if return_power_spectrum else None
        print('generated noise')
        return (temperal_samples_all,) + (fft_coeffs,) * return_fft_coef + (power_spectrum,) * return_power_spectrum


T = TypeVar('T')


class PhiDependentTransition(Protocol):
    def qubit_freq_GHz(self, frustration_Phi0: T) -> T:
        ...


@dataclasses.dataclass
class powerlaw:
    """represents an transition, such that the k-th derivative is stored under ck"""
    c0_GHz: float = 0.
    c1_GHz_o_Phi0: float = 0.
    c2_GHz_o_Phi02: float = 1.

    def qubit_freq_GHz(self, frustration_Phi0):
        return self.c0_GHz + \
               self.c1_GHz_o_Phi0 * frustration_Phi0 + \
               0.5 * self.c2_GHz_o_Phi02 * frustration_Phi0 ** 2


def qb_plot_t2s_at(transition: PhiDependentTransition,
                   t_step_ns=200,  # smallest unit of time in ns, corresponds to high frequency cutoff
                   t_observation_ns=200_000,  # interval on which Gamma2(R or E) will be observed
                   t_cut_off_ns=200_000 * 256,  # manual low-f cutoff of the generation
                   t_total_ns=200_000 * 2048,
                   # duration of noise generated, will correspond to fft frequency resolution
                   A_amplitude_Phi0=1.0e-06,  # noise amplitude, in units of Phi0
                   generated=None,  # to bypass the noise-generation with existing noise array

                   offset_Phi0=0.,  # offsets the noise array before calculating transition(t)

                   plot=False,
                   laziness=16,  # laziness of the sweeping-window ensemble averaging
                   return_curves=False,
                   ):
    # see how all parameters are in temperal domain, for easy intuitive comparisons,
    # plz check a posteriori that all scales are well chosen

    ng1of = NoiseGen1OverF(t_step_ns=t_step_ns, n_fft=t_total_ns // t_step_ns)

    # choose one out of the presets
    if generated is None:
        flux_trajectory_Phi0, *_ = ng1of.generate(A2=A_amplitude_Phi0 ** 2, t_cut_off_ns=t_cut_off_ns)
    else:
        flux_trajectory_Phi0 = generated

    flux_trajectory_Phi0: np.ndarray  # temporal sampled returned as np array
    # choose one out of the presets

    print('calculating qubit freq(t)')
    dfs_GHz = transition.qubit_freq_GHz(flux_trajectory_Phi0 + offset_Phi0)
    print('calculated qubit freq(t)')
    dfs_GHz -= np.mean(dfs_GHz)  # so we get less shaky phase in the end

    tobs_tstep = int(t_observation_ns // t_step_ns)  # how many steps in the observation period

    print('integrating trajectory')
    dfs_GHz = np.concatenate([[0], dfs_GHz, dfs_GHz])  # notice how we repeat the signals two times
    # this wrapping solves

    cumsum_radian: np.ndarray = np.cumsum(dfs_GHz) * (t_step_ns * 2 * np.pi)

    print(f'{cumsum_radian.size} items in cumsum, end = {cumsum_radian[-1]:.3e} (should be 0. coz of df-=mean(df))')
    print('integrated trajectory')

    print('sweeping ensemble-averaging window')
    time.sleep(0.3)

    def sweep():
        # these two will become ndarrays buffers for ensemble averaging
        ramseyAccu, echoAccu_interpulse = 0., 0.

        ttot_tstep = ng1of.n_fft
        tot_sz = 0
        for i in tqdm.tqdm(range(0, tobs_tstep, laziness), total=len(range(0, tobs_tstep, laziness))):
            _sz = (((ttot_tstep - i) // tobs_tstep) + ((ttot_tstep - i) % tobs_tstep > 0)) * tobs_tstep
            tot_sz += _sz
            _cumsum_radian = cumsum_radian[i:i + _sz]
            _cumsum_radian.shape = (_sz // tobs_tstep, tobs_tstep)
            phaseR = _cumsum_radian - _cumsum_radian[:, 0].reshape((_sz // tobs_tstep, 1))
            ramseyAccu = ramseyAccu + np.exp(1j * phaseR).mean(axis=0)

            if _cumsum_radian.shape[0] % 2 == 1:  # echo requires
                _cumsum_radian = np.delete(_cumsum_radian, -1, axis=0)

            #
            double = _cumsum_radian.reshape((_sz // tobs_tstep // 2, tobs_tstep, 2))
            single = double.reshape((_sz // tobs_tstep // 2, 2, tobs_tstep))
            single = single[:, 0, :].squeeze()
            phaseE = single * 2 - (double[:, :, 0] + single[:, 0].reshape((single.shape[0], 1)))
            echoAccu_interpulse = echoAccu_interpulse + np.exp(1j * phaseE).mean(axis=0)
        ramseyAccu /= ramseyAccu[0]
        echoAccu_interpulse /= echoAccu_interpulse[0]
        return ramseyAccu, echoAccu_interpulse

    ramseyAccu, echoAccu_interpulse = sweep()
    times = np.arange(tobs_tstep) * t_step_ns

    found_t2s = False
    try:
        idx_t2e = np.argmin(np.abs(np.abs(echoAccu_interpulse) - 1. / np.e))
        idx_t2r = np.argmin(np.abs(np.abs(ramseyAccu) - 1. / np.e))
        t2e = opt.fsolve(lambda x: np.interp(x, times, np.abs(echoAccu_interpulse) - 1. / np.e), times[idx_t2e]) * 2
        t2r = opt.fsolve(lambda x: np.interp(x, times, np.abs(ramseyAccu) - 1. / np.e), times[idx_t2r])
        found_t2s = True
    except:
        t2e = t2r = np.nan  # not available

    if plot:
        pl.figure(figsize=(12, 7))
        pl.plot(times, np.angle(ramseyAccu) / 2 / np.pi, 'r--')
        pl.plot(times, np.abs(ramseyAccu), '-|r')
        pl.plot(times * 2, np.angle(echoAccu_interpulse), 'b--')
        pl.plot(times * 2, np.abs(echoAccu_interpulse), '-|b')
        pl.legend(['complexe phase of c(t) Ramsey', 'amplitude of c(t) Ramsey',
                   'complexe phase of c(t) Echo', 'amplitude of c(t) Echo'], loc='lower left')
        if found_t2s:
            pl.plot([t2e, t2e], [-1, 1], '--b')
            pl.plot([t2r, t2r], [-1, 1], '--r')
        pl.plot([times.min(), times.max()], [1 / np.e] * 2, '--k')
        pl.xlim(0, t_observation_ns), pl.ylim(-1., 1.)
        pl.title(
            f'{offset_Phi0}' +
            (f'E{t2e}_ns = {1e6 / (t2e)}kHz; R{t2r}_ns = {1e6 / t2r}kHz' if found_t2s else '')
        )

    return (t2r, t2e) + (times, np.abs(ramseyAccu), np.abs(echoAccu_interpulse)) * return_curves


def main():
    def qb_spectrum(Delta, Ip, offset=0.):
        def fun(x):
            return np.sqrt(Delta ** 2 + (2 * Ip * Phi0 * (x + offset) / h / 1e9) ** 2)

        return fun

    Delta = 6.
    Ip = 300e-9

    # checking that we have not made mistakes over the derivative calculations
    offset = 30e-6 * 0
    sp = qb_spectrum(Delta, Ip, offset)
    c1_sp = (sp(1e-8) - sp(-1e-8)) / (2e-8)
    c2_sp = (sp(1e-8) + sp(-1e-8) - 2 * sp(0)) / (1e-8) ** 2
    c1 = (2 * Ip * Phi0 / (h * Delta * 1e9)) ** 2 * Delta * offset
    c2 = (2 * Ip * Phi0 / (h * Delta * 1e9)) ** 2 * Delta

    # printing some of the settings and analytical predictions of gamma1 and gamma2
    A_Phi0 = 2e-6
    print(f'c2={c2}')
    print(f'c1={c1}')
    print(f'A_Phi0={A_Phi0}')
    print(f'gamma2[kHz]={11.5959 * (A_Phi0 * Phi0) ** 2 * (c2 / Phi0 ** 2) * 1e6}')
    print(f'gamma1[kHz]='
          f'{(2 * Ip / hbar) ** 2 * (offset * Phi0) / (2 * pi * Delta * 1e12) * (A_Phi0 * Phi0) * np.sqrt(np.log(2))}')

    # checking that a power law approximates the full spectrum well locally
    qb = powerlaw(c0_GHz=sp(0), c1_GHz_o_Phi0=c1 * 1, c2_GHz_o_Phi02=c2 * 1)
    print(qb.qubit_freq_GHz(-40e-6), sp(-40e-6))
    print(qb.qubit_freq_GHz(-20e-6), sp(-20e-6))
    print(qb.qubit_freq_GHz(0.), sp(0.))
    print(qb.qubit_freq_GHz(20e-6), sp(20e-6))
    print(qb.qubit_freq_GHz(40e-6), sp(40e-6))

    qb_plot_t2s_at(qb, A_amplitude_Phi0=A_Phi0, plot=True)
    pl.show()


if __name__ == '__main__':
    main()
