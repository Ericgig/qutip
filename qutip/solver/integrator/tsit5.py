"""
Tsitouras's 5/4 order Runge-Kutta method

Runge–Kutta pairs of order 5(4) satisfying only the first column
simplifying assumption,
Ch. Tsitouras,
Computers & Mathematics with Applications, Vol 62, Issue 2, 770-775
Jan 2011
"""

__all__ = ["tsit5_coeff"]
import numpy as np
order = 5
rk_step = 7

a = np.zeros((rk_step, rk_step), dtype=np.float64)
b = np.zeros(rk_step, dtype=np.float64)
c = np.zeros(rk_step, dtype=np.float64)
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

e[0] =  .17800110522257773e-2
e[1] =  .8164344596567463e-3
e[2] = -.7880878010261994e-2
e[3] =  .1447110071732629
e[4] = -.5823571654525552
e[5] =  .45808210592918686
e[6] = -1/66

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

tsit5_coeff = {'order': order, 'a': a, 'b': b, 'c': c, 'e': e, 'bi': bi}
