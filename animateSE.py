import numpy as np
import TISE
import Gauss
import matplotlib.pyplot as plt
from matplotlib import animation as anim


def theta(x):
    x = np.asarray(x)
    y = np.zeros(x.shape)
    y[x > 0] = 1.0
    return y

def boundaries(x, width, height):
    return height * (theta(x) - theta(x - width))

# Time
dt = 0.01
steps = 50
t_max = 120
edges = int(t_max / float(steps * dt))
hbar = 1.0   
m = 1.9      
# In x
N = 2 ** 11
dx = 0.1
x = dx * (np.arange(N) - 0.5 * N)
# Potential
V0 = 1.5
L = hbar / np.sqrt(2 * m * V0)
a = 3 * L
x0 = -60* L
Vx = boundaries(x, a, V0)
Vx[x < -98] = 1e6
Vx[x > 98] = 1e6

# Initial Conditions
p0 = np.sqrt(2 * m * 0.2 * V0)
dp2 = p0 * p0 * 1./80
d = hbar / np.sqrt(2 * dp2)

k0 = p0 / hbar
v0 = p0 / m
psiX0 = Gauss.xGauss(x, d, x0, k0)

# Schrodinger equation
SE = TISE.Wavefunction(x=x,
                psiX0=psiX0,
                Vx=Vx,
                hbar=hbar,
                m=m,
                k0=-28)

# Plot
fig = plt.figure()
xlim = (-100, 100)
klim = (-5, 5)
ymin = 0
ymax = V0
ax1 = fig.add_subplot(111, xlim=xlim,ylim=(ymin - 0.15 * (ymax - ymin),ymax + 0.2 * (ymax - ymin)))
psi_x_line, = ax1.plot([], [], c='g', label=r'$|\psi(x)|$')
Vx_line, = ax1.plot([], [], c='k', label=r'$V(x)$')
center_line = ax1.axvline(0, c='k', ls=':',label = r"$x_0 + v_0t$")
title = ax1.set_title("")
ax1.legend(prop=dict(size=12))
ax1.set_xlabel('$x$')
ax1.set_ylabel(r'$|\psi(x)|$')
Vx_line.set_data(SE.x, SE.Vx)

# Animation
def init():
    psi_x_line.set_data([], [])
    Vx_line.set_data([], [])
    center_line.set_data([], [])

    title.set_text("")
    return (psi_x_line, Vx_line, center_line, title)

def animate(i):
    SE.timeForward(dt, steps)
    psi_x_line.set_data(SE.x, 4 * abs(SE.psi_x))
    Vx_line.set_data(SE.x, SE.Vx)
    center_line.set_data(2 * [x0 + SE.t * p0 / m], [0, 1])

   
    title.set_text("t = %.2f" % SE.t)
    return (psi_x_line, Vx_line, center_line,  title)

anim = anim.FuncAnimation(fig, animate, init_func=init,
                               frames=edges, interval=100, blit=True)


plt.show()