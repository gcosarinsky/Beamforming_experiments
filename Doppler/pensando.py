import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

fs = 100
f0 = 1.3
df = 0.00
prf = 0.07  # 5 khz
n_prf_cycles = 10

t_prf_cycle = np.arange(0, 1/prf, 1/fs)
ascan = np.sin(2*np.pi*(f0 + df)*t_prf_cycle)
q = np.tile(ascan, n_prf_cycles)
t_total = np.arange(q.size) / fs
q0 = np.sin(2*np.pi*(f0 + df)*t_total)
ref_signal = np.sin(2*np.pi*f0*t_total)

fig, ax = plt.subplots(2, 1)
ax[0].plot(q0*ref_signal)
ax[1].plot(q*ref_signal)