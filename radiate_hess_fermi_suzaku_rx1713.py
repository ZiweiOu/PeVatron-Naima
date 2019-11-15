import numpy as np
import astropy.units as u
from astropy.io import ascii
import naima
from naima.models import InverseCompton, Synchrotron, ExponentialCutoffPowerLaw,PionDecay,Bremsstrahlung
import sys
import matplotlib.pyplot as plt

# Read fake data

# fit Fermi and HESS

datax = ascii.read('Suzaku_XIS_data_short.dat')
datal = ascii.read('spectrum_RX1713_Fermi.dat')
datah = ascii.read('spectrum_RX1713_HESS.dat')

# Model definition
def radiation(pars, data):
    # the distance of source in unit of kpc
    distance    = 1 * u.kpc
    # Match parameters to ECPL properties, and give them the appropriate units
    amplitude   = 10**(47.+pars[0]) / u.TeV # !!!note the paramter range
    e_alpha     = pars[1]
    p_alpha     = pars[2]
    e_cutoff    = (10**pars[3]) * u.TeV
    p_cutoff    = (10**pars[4]) * u.TeV
    B           = pars[5] * u.uG
    Kep         = pars[6]
    N_H         = pars[7]*u.cm**-3


 # Initialize instances of the particle distribution and radiative models
	 # Kep, the electron/proton ratio at 1 TeV
    ECPLE = ExponentialCutoffPowerLaw(Kep*amplitude, 1 * u.TeV, e_alpha, e_cutoff)
    ECPLP = ExponentialCutoffPowerLaw(amplitude, 1 * u.TeV, p_alpha, p_cutoff)

    SYN = Synchrotron(ECPLE, B)

    # construct model
    IC  = InverseCompton(ECPLE,seed_photon_fields=['CMB', ['FIR', 26.5 * u.K, 0.415 * u.eV / u.cm**3]])
		# IC  = InverseCompton(ECPLE, seed_photon_fields=['CMB', 'FIR', 'NIR', ['SSC', Esy, phn_sy]])
    PI  = PionDecay(ECPLP, N_H)

    # compute flux at the energies given in data['energy']
    model = ( IC.sed(data, distance=distance) +
                    + SYN.sed(data, distance=distance)
                    + PI.sed(data,distance=distance)).to(data['flux'].unit)


		# Prepare an energy array for saving the particle distribution
    proton_energy = np.logspace(9, 15, 100) * u.eV
    proton_dist = PI.particle_distribution(proton_energy) # returning the particle energy density in units of number of protons per unit energy

    # Compute and save total energy in protons above 1 TeV
    Wp = PI.compute_Wp(Epmin=1 * u.GeV) # total energy in electrons between energies Eemin and Eemax

    return model, (proton_energy, proton_dist), Wp


# Prior definition
def lnprior(pars):

    # Limit parameters range
    #note that, the start point must satisfy these limit !!!
    logprob = naima.uniform_prior(pars[0], -2, 4)\
                + naima.uniform_prior(pars[1], 1, 4)\
                + naima.uniform_prior(pars[2], 1, 4)\
                + naima.uniform_prior(pars[3], -2, 3)\
                + naima.uniform_prior(pars[4], -1, 1)\
                + naima.uniform_prior(pars[5], 0, 100)\
                + naima.uniform_prior(pars[6], 0.0001, 1)\
                + naima.uniform_prior(pars[7], 0.001, 0.1)

    return logprob


if __name__ == '__main__':

    # set start point

    # Fermi and hess initial parameters
    p0 = np.array(( 0.19, 2.41, 1.8, 1.64, 0.37, 14.7, 0.22, 0.023 ))


    # set parameter name
    labels = ['log10(norm)', 'e_alpha','p_alpha', 'e_cutoff','p_cutoff', 'B', 'Kep', 'N_H']

    # Run sampler to find best fit values
    sampler, pos = naima.run_sampler(data_table=[datax, datal, datah],
                                     p0=p0,
                                     labels=labels,
                                     model=radiation,
                                     prior=lnprior,
                                     nwalkers=100,
                                     nburn=50,
                                     nrun=50,
                                     threads=16,
                                     guess=False,
                                     prefit=True,
                                     interactive=False)

    # Save run results to HDF5 file (can be read later with naima.read_run)
    naima.save_run('model_Suzaku_Fermi_HESSS.hdf5', sampler)

    ## Diagnostic plots with labels for the metadata blobs
    
    naima.save_diagnostic_plots(
        'RX1713_HESS_Fermi_Suzaku',
        sampler,
        sed=True,
        last_step=False,
        blob_labels=['RX J1713.7-3946 Photon spectrum of Suzaku, Fermi and HESS',
                     'RX J1713.7-3946 Proton energy distribution of Suzaku, Fermi and HESS',
                     'RX J1713.7-3946 $W_p (E_p>1\, \mathrm{GeV})$ of Suzaku, Fermi and HESS'])

    naima.save_results_table('RX1713_HESS_Fermi_Suzaku', sampler) # format='ascii.ecsv'
