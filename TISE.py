import numpy as np
from scipy.fftpack import ifft
from scipy.fftpack import fft


class Wavefunction(object):
    
    def __init__(self, x, psiX0, Vx,
                 k0 = None, hbar=1, m=1, t0=0.0):
     
       
        self.x, psiX0, self.Vx = map(np.asarray, (x, psiX0, Vx))
        N = self.x.size
        assert self.x.shape == (N,)
        assert psiX0.shape == (N,)
        assert self.Vx.shape == (N,)

        # Attributes
        self.hbar = hbar
        self.m = m
        self.t = t0
        self.dt_ = None
        self.N = len(x)
        self.dx = self.x[1] - self.x[0]
        self.dk = 2 * np.pi / (self.N * self.dx)

        
        if k0 == None:
            self.k0 = -0.5 * self.N * self.dk
        else:
            self.k0 = k0
        self.k = self.k0 + self.dk * np.arange(self.N)

        self.psi_x = psiX0
        self.kFromX()

        self.xEvolveHalf = None
        self.xEvolve = None
        self.kEvolve = None
        self.psi_x_line = None
        self.psi_k_line = None
        self.Vx_line = None
        
    def getDt(self):
        return self.dt

    def setDt(self, dt):
        if dt != self.dt_:
            self.dt_ = dt
            self.xEvolveHalf = np.exp(-0.5 * 1j * self.Vx/ self.hbar * dt )
            self.xEvolve = self.xEvolveHalf * self.xEvolveHalf
            self.kEvolve = np.exp(-0.5 * 1j * self.hbar /self.m * (self.k * self.k) * dt)

    def setPsiX(self, psi_x):
        self.psi_x_mod = (psi_x * np.exp(-1j * self.k[0] * self.x)
                          * self.dx / np.sqrt(2 * np.pi))

    def getPsiX(self):
        return (self.psi_x_mod* np.exp(1j * self.k[0] * self.x)
                * np.sqrt(2 * np.pi) / self.dx)

    def setPsiK(self, psi_k):
        self.psi_k_mod = psi_k * np.exp(1j * self.x[0]
                                        * self.dk * np.arange(self.N))

    def getPsiK(self):
        return self.psi_k_mod * np.exp(-1j * self.x[0] * 
                                        self.dk * np.arange(self.N))
    
    
    psi_x = property(getPsiX, setPsiX)
    psi_k = property(getPsiK, setPsiK)
    dt = property(getDt, setDt)

    def kFromX(self):
        self.psi_k_mod = fft(self.psi_x_mod)

    def xFromK(self):
        self.psi_x_mod = ifft(self.psi_k_mod)

    def timeForward(self, dt, steps = 1):
        self.dt = dt

        if steps > 0:
            self.psi_x_mod *= self.xEvolveHalf

        for i in range(steps-1):
            self.kFromX()
            self.psi_k_mod *= self.kEvolve
            self.xFromK()
            self.psi_x_mod *= self.xEvolve

        self.xFromK()
        self.psi_k_mod *= self.kEvolve

        self.xFromK()
        self.psi_x_mod *= self.xEvolveHalf

        self.kFromX

        self.t += dt * steps