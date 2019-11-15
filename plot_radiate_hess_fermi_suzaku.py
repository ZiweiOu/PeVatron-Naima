import naima
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from naima.models import InverseCompton, Synchrotron, ExponentialCutoffPowerLaw,PionDecay,Bremsstrahlung

# best fit parameters

# Fermi and DC-1

p = [47+0.19, 2.41, 1.8, 1.64, 0.37, 14.7, 0.22, 0.023 ]


distance    = 1             *u.kpc
ampitude    = 10**p[0]      /u.TeV
e_alpha     = p[1]
p_alpha     = p[2]
e_cutoff    = 10**p[3]      *u.TeV
p_cutoff    = 10**p[4]      *u.TeV
B           = p[5]          *u.uG
Kep         = p[6]
N_H         = p[7]          *u.cm**-3


ECPLE = naima.models.ExponentialCutoffPowerLaw(Kep*ampitude,1*u.TeV, e_alpha , e_cutoff)
ECPLP = naima.models.ExponentialCutoffPowerLaw(ampitude, 1*u.TeV, p_alpha , p_cutoff)


IC  = naima.models.InverseCompton(ECPLE,seed_photon_fields = ['CMB',['FIR',26.5*u.K,0.415*u.eV*u.cm**-3]],Eemin=1*u.MeV)
SYN = naima.models.Synchrotron(ECPLE, B)
BRE = Bremsstrahlung(ECPLE,N_H,Eemin=1000*u.MeV)
PI  = naima.models.PionDecay(ECPLP,nh=N_H)


energy  = np.logspace(-8,14,100)*u.eV
energy_bre  = np.logspace(4,14,1000)*u.eV

sed_IC  = IC.sed(energy, distance=distance)
sed_SYN = SYN.sed(energy, distance=distance)
sed_PI  = PI.sed(energy, distance=distance)

sed_sum = sed_IC + sed_SYN + sed_PI

We = IC.compute_We(Eemin=1 * u.TeV)
Wp = PI.compute_Wp(Epmin=1 * u.TeV)
print( ' We	: ', We)
print( ' Wp	: ', Wp)
print( ' norm	: ', ampitude)
print( ' e_alpha: ', e_alpha)
print( ' p_alpha: ', p_alpha)
print( ' e_cut	: ', e_cutoff)
print( ' p_cut	: ', p_cutoff)
print( ' B	: ', B)
print( ' Kep    : ', Kep)
print( ' Nh     : ', N_H)

# Plot figure
#plt.figure(figsize=(8,5))
#plt.rc('font', family='sans')
#plt.rc('mathtext', fontset='custom')

# load data from file
''' that the skiprows set to 11 is just suitable for this data file '''


x1,y1,yerr1 =np.loadtxt('spectrum_RX1713_Fermi.dat' ,skiprows=11,unpack=True)
x2,y2,yerr2 =np.loadtxt('spectrum_RX1713_HESS.dat' ,skiprows=11,unpack=True)
x3,y3,yerr3 =np.loadtxt('Suzaku_XIS_data_short.dat' ,skiprows=11,unpack=True)

plt.errorbar(x1,y1,yerr1,fmt='o',color='blue')
plt.errorbar(x2,y2,yerr2,fmt='o',color='red')
plt.errorbar(x3,y3,yerr3,fmt='o',color='orange')

plt.loglog(energy,sed_PI,lw=2,label='PI',c=naima.plot.color_cycle[0])
plt.loglog(energy,sed_IC,lw=2,label='IC ',c=naima.plot.color_cycle[1])
plt.loglog(energy,sed_SYN,lw=2,label='Sync',c=naima.plot.color_cycle[2])


plt.loglog(energy,sed_sum,lw=2,label='sum',c=naima.plot.color_cycle[0])

plt.xlabel('Photon energy [{0}]'.format(energy.unit.to_string('latex_inline')))
plt.ylabel('$E^2dN/dE$ [{0}]'.format(sed_SYN.unit.to_string('latex_inline')))
plt.ylim(1e-15, 1e-6)
plt.yscale('log')
plt.xscale('log')
plt.title("RX J1731.7-3946 SED of Suzaku, Fermi and HESS")
plt.tight_layout()
plt.legend(loc='best')
plt.savefig('RX1731_Suzaku_Fermi_HESS.eps',format = 'eps',dpi=100, bbox_inches='tight')
plt.show()

