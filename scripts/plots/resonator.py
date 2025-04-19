# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "matplotlib",
#     "pmlib @ file:///${PROJECT_ROOT}/"
# ]
# ///
"""Linear stiff string plot."""

from math import log, pi, sqrt

import matplotlib.animation as ani
import matplotlib.pyplot as plt
import numpy as np

from pmlib.resonators import BoundaryCondition, Endpoint, LinearResonator1D


# Global parameters
SR = 44100  # sample rate (Hz)
B = 0.0001  # inharmonicity parameter (>0)
F0 = 100  # fundamental (Hz)
CTR = 0.1  # center location
WID = 0.05  # width of excitation
U0 = 1  # maximum initial displacement
V0 = 0  # velocity
loss = (100, 10), (1000, 8)  # loss (frequency (Hz), T60 (s))

# Derived parameters
GAMMA = 2 * F0
kappa = sqrt(B) * (GAMMA / pi)

# Scheme loss parameters
zeta = (
    (-(GAMMA**2) + sqrt((GAMMA**4) + 4 * (kappa**2) * ((2 * pi * loss[0][0]) ** 2)))
    / (2 * (kappa**2)),
    (-(GAMMA**2) + sqrt((GAMMA**4) + 4 * (kappa**2) * ((2 * pi * loss[1][0]) ** 2)))
    / (2 * (kappa**2)),
)
sig = (
    6 * log(10) * (-zeta[1] / loss[0][1] + zeta[0] / loss[1][1]) / (zeta[0] - zeta[1]),
    6 * log(10) * (1 / loss[0][1] - 1 / loss[1][1]) / (zeta[0] - zeta[1]),
)

LinearResonator1D.sample_rate(SR)

# Create and build our resonator model
resonator = LinearResonator1D(
    GAMMA,
    kappa,
    sig,
    {Endpoint.LEFT: BoundaryCondition.FREE, Endpoint.RIGHT: BoundaryCondition.CLAMPED},
)

resonator.build()

# Create raised cosine
xax = np.arange(0, resonator.n + 1).reshape(resonator.n + 1, 1) / resonator.n
ind = np.sign(
    np.maximum(np.multiply(-(xax - (CTR + WID / 2)), xax - (CTR - WID / 2)), 0)
)
rc = np.multiply(0.5 * ind, 1 + np.cos(2 * pi * (xax - CTR) / WID))

# Initialise our state vectors
u2 = U0 * rc
u1 = (U0 + V0 / SR) * rc
u = np.zeros((resonator.n + 1, 1))

fig, ax = plt.subplots()
string = ax.plot(xax.flatten(), u.flatten())[0]
ax.set_ylim([-0.5, 0.5])
ax.set_xlabel("location divided by length")
ax.set_ylabel("displacement")
ax.set_title("Linear Stiff String Simulation")


def update(_: int):
    """Main loop."""
    u[0 : resonator.n + 1] = resonator.b * u1 + resonator.c * u2

    string.set_ydata(u.flatten())

    u2[0 : resonator.n + 1] = u1[0 : resonator.n + 1]
    u1[0 : resonator.n + 1] = u[0 : resonator.n + 1]

    return string


animation = ani.FuncAnimation(fig=fig, func=update, frames=40, interval=30)

plt.show()
