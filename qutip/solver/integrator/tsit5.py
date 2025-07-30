"""
Provide a cython implimentation verner 'most-efficient'
order 7 runge-Kutta method.
See https://www.sfu.ca/~jverner/
"""
# Verner 7 Efficient
# https://www.sfu.ca/~jverner/RKV76.IIa.Efficient.00001675585.081206.CoeffsOnlyFLOAT
__all__ = ["tsit5_coeff"]
import numpy as np
order = 5
rk_step = 7
rk_extra_step = 7
denseout_order = 0

bh = np.zeros(rk_step, dtype=np.float64)
a = np.zeros((rk_extra_step, rk_extra_step), dtype=np.float64)
b = np.zeros(rk_step, dtype=np.float64)
c = np.zeros(rk_extra_step, dtype=np.float64)
e = np.zeros(rk_step, dtype=np.float64)
bi = None

c[0] = 0.
c[1] = .161
c[2] = .327
c[3] = .9
c[4] = .9800255409045097
c[5] = 1.
c[6] = 1.

b[0] =  .09646076681806523
b[1] =  .01
b[2] =  .4798896504144996
b[3] =  1.379008574103742
b[4] = -3.290069515436081
b[5] =  2.324710524099774
b[6] = 0.

bh[0] =  .001780011052226
bh[1] =  .000816434459657
bh[2] = -.007880878010262
bh[3] =  .144711007173263
bh[4] = -.582357165452555
bh[5] =  .458082105929187
bh[6] = 1/66

a[1, 0] = c[1]

a[2, 1] = 0.3354806554923570
a[2, 0] = c[2] - a[2, 1]

a[3, 1] = -6.359448489975075
a[3, 2] = 4.362295432869581
a[3, 0] = c[3] - a[3, 2] - a[3, 1]

a[4, 1] = -11.74888356406283
a[4, 2] = 7.495539342889836
a[4, 3] = -0.09249506636175525
a[4, 0] = c[4] - a[4, 3] - a[4, 2] - a[4, 1]

a[5, 1] = -12.92096931784711
a[5, 2] = 8.159367898576159
a[5, 3] = -0.071584973281401
a[5, 4] = -0.02826905039406838
a[5, 0] = c[5] - a[5, 4] - a[5, 3] - a[5, 2] - a[5, 1]

a[6, :6] = b[:6]

for i in range(rk_step):
    e[i] = b[i] - bh[i]

tsit5_coeff = {'order': order, 'a': a, 'b': b, 'c': c, 'e': e, 'bi': bi}
