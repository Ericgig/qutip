
import numpy as np
from scipy.fftpack import fft
import scipy.optimize as opt
from scipy import interpolate
from scipy.special import erf

class filter:
    def __init__(self):
        self.num_x = 0
        self.t_step = 0
        self.num_ctrl = 0
        self.x_max = None
        self.x_min = None

    def __call__(self, x):
        return x

    def reverse(self, gradient):
        return gradient

    def init_timeslots(self, times, tau, T, t_step, num_x, num_ctrl):
        """
        Generate the timeslot duration array 'tau' based on the evo_time
        and num_tslots attributes, unless the tau attribute is already set
        in which case this step in ignored
        Generate the cumulative time array 'time' based on the tau values
        """
        if times is not None:
            t_step = len(times)-1
            T = times[-1]
            time = times
        elif tau is not None:
            t_step = len(tau)
            T = np.sum(tau)
            time = np.cumsum(np.insert(tau,0,0))
        else:
            time = np.linspace(0,T,t_step+1)
        self.num_x = t_step
        self.t_step = t_step
        self.num_ctrl = num_ctrl
        return (t_step, num_ctrl), time

    def set_amp_bound(self, amp_lbound=None, amp_ubound=None):
        self.x_min = amp_lbound
        self.x_max = amp_ubound

    def _compute_xlim(self):
        if self.x_max is None and self.x_min is None:
            return

        if self.x_max is not None:
            if isinstance(self.x_max, list):
                self.x_max = np.array(self.x_max)
            if isinstance(self.x_max, (int, float)):
                self.x_max = self.x_max*np.ones((self.num_x,self.num_ctrl))
            elif isinstance(self.x_max, np.ndarray):
                if self.x_max.shape != (self.num_x,self.num_ctrl):
                    raise Exception("shape of the amb_bound not right")

        else:
            self.x_max = np.array([[None]*self.num_ctrl]*self.num_x)

        if self.x_min is not None:
            if isinstance(self.x_min, list):
                self.x_min = np.array(self.x_min)
            if isinstance(self.x_min, (int, float)):
                self.x_min = self.x_min*np.ones((self.num_x,self.num_ctrl))
            elif isinstance(self.x_min, np.ndarray):
                if self.x_min.shape != (self.num_x,self.num_ctrl):
                    raise Exception("shape of the amb_bound not right")
        else:
            self.x_min = np.array([[None]*self.num_ctrl]*self.num_x)

    def get_xlimit(self):
        self._compute_xlim()
        if self.x_max is None and self.x_min is None:
            return None  # No constrain
        xmax = self.x_max.astype(object)
        xmax[np.isinf(xmax)] = None
        xmax = list(self.x_max.flatten())
        xmin = self.x_min.astype(object)
        xmin[np.isinf(xmin)] = None
        xmin = list(self.x_min.flatten())
        return zip(xmin,xmax)

    def reverse_state(self, target):
        x_shape = (self.num_x, self.num_ctrl)
        xx = np.zeros(x_shape)

        def diff(y):
            yy = self(y.reshape(x_shape))
            return np.sum((yy-target)**2)

        def gradiant(y):
            yy = self(y.reshape(x_shape))
            grad = self.reverse((yy-target)*2)
            return grad.reshape(np.prod(x_shape))

        rez = opt.minimize(fun=diff, jac=gradiant, x0=xx)
        return rez.x.reshape(x_shape)


class pass_througth(filter):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        return x

    def reverse(self, gradient):
        return gradient

    def init_timeslots(self, times=None, tau=None, T=1, t_step=10,
                             num_x=1, num_ctrl=1):
        """
        Generate the timeslot duration array 'tau' based on the evo_time
        and num_tslots attributes, unless the tau attribute is already set
        in which case this step in ignored
        Generate the cumulative time array 'time' based on the tau values
        """
        if times is not None:
            t_step = len(times)-1
            T = times[-1]
            time = times
        elif tau is not None:
            t_step = len(tau)
            T = np.sum(tau)
            time = np.cumsum(np.insert(tau,0,0))
        else:
            time = np.linspace(0,T,t_step+1)

        self.num_x = t_step
        self.t_step = t_step
        self.num_ctrl = num_ctrl
        self.times = time
        return (t_step, num_ctrl), time


class fourrier(filter):
    """
        Pulse described as fourrier modes
        u(t) = a(j)*sin(pi*j*t/T)
        Takes a linearly spaced time and num_x <= len(times)
        When num_x < t_step, only the num_x lower mode are kept.
        (low-pass filter)
    """
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        u = np.zeros((self.t_step, self.num_ctrl))
        s = np.zeros((self.t_step*2-2, self.num_ctrl))
        s[:self.num_x,:] = x
        for j in range(self.num_ctrl):
            u[:,j] = fft(s[:,j]).imag[:self.t_step]
        return u

    def reverse(self, gradient):
        x = np.zeros((self.num_x, self.num_ctrl))
        s = np.zeros((self.t_step*2-2, self.num_ctrl))
        s[:self.t_step,:] = gradient
        for j in range(self.num_ctrl):
            x[:,j] = fft(s[:,j]).imag[:self.num_x]
        return x/self.t_step

    def init_timeslots(self, times=None, tau=None, T=1, t_step=None,
                             num_x=None, num_ctrl=1):
        self.num_ctrl = num_ctrl

        if times is not None:
            if not np.allclose(np.diff(times), times[1]-times[0]):
                raise Exception("Times must be equaly distributed")
            else:
                self.t_step = len(times)-1
                T = times[-1]
        elif tau is not None:
            if not np.allclose(np.diff(tau), tau[0]):
                raise Exception("tau must be all equal")
            else:
                self.t_step = len(tau)
                T = np.sum(tau)
        elif t_step is not None:
            self.t_step = t_step
        elif num_x is not None:
            self.t_step = num_x
        else:
            self.t_step = 10

        if num_x is None:
            self.num_x = self.t_step
        else:
            self.num_x = num_x

        return (self.num_x, self.num_ctrl), \
            np.linspace(0, T, self.t_step+1)


class spline(filter):
    """
    Should this be a special case of convolution?
    Zero padding first and last?
    """
    def __init__(self, overSampleRate=5):
        super().__init__()
        self.N = overSampleRate
        self.dt = 1/overSampleRate

    def __call__(self, x):
        u = np.zeros((self.t_step, self.num_ctrl))

        for j in range(self.num_ctrl):
            for i in range(0,self.N//2):
                tt = (i+0.5)*self.dt+0.5

                u[i,j] += x[0,j]*tt*(0.5+tt*(2-tt*1.5))
                u[i,j] += x[1,j]*tt*tt*0.5*(tt-1)

                u[i+self.N,j] += x[0,j]*(1+tt*tt*(tt*1.5-2.5))
                u[i+self.N,j] += x[1,j]*tt*(0.5+tt*(2-tt*1.5))
                u[i+self.N,j] += x[2,j]*tt*tt*0.5*(tt-1)

                for t in range(2,self.num_x-1):
                    tout = t*self.N+i
                    u[tout,j] += x[t-2,j]*tt*(-0.5+tt*(1-tt*0.5))
                    u[tout,j] += x[t-1,j]*(1+tt*tt*(tt*1.5-2.5))
                    u[tout,j] += x[t  ,j]*tt*(0.5+tt*(2-tt*1.5))
                    u[tout,j] += x[t+1,j]*tt*tt*0.5*(tt-1)

                tout = self.num_x*self.N-1*self.N+i
                u[tout,j] += x[self.num_x-3,j]*tt*(-0.5+tt*(1-tt*0.5))
                u[tout,j] += x[self.num_x-2,j]*(1+tt*tt*(tt*1.5-2.5))
                u[tout,j] += x[self.num_x-1,j]*tt*(0.5+tt*(2-tt*1.5))


            for i in range(self.N//2,self.N):
                tt = (i+0.5)*self.dt-0.5

                u[i,j] += x[0,j]*(1+tt*tt*(tt*1.5-2.5))
                u[i,j] += x[1,j]*tt*(0.5+tt*(2-tt*1.5))
                u[i,j] += x[2,j]*tt*tt*0.5*(tt-1)

                for t in range(1,self.num_x-2):
                    tout = t*self.N+i
                    u[tout,j] += x[t-1,j]*tt*(-0.5+tt*(1-tt*0.5))
                    u[tout,j] += x[t  ,j]*(1+tt*tt*(tt*1.5-2.5))
                    u[tout,j] += x[t+1,j]*tt*(0.5+tt*(2-tt*1.5))
                    u[tout,j] += x[t+2,j]*tt*tt*0.5*(tt-1)

                tout = self.num_x*self.N-2*self.N+i
                u[tout,j] += x[self.num_x-3,j]*tt*(-0.5+tt*(1-tt*0.5))
                u[tout,j] += x[self.num_x-2,j]*(1+tt*tt*(tt*1.5-2.5))
                u[tout,j] += x[self.num_x-1,j]*tt*(0.5+tt*(2-tt*1.5))

                tout = self.num_x*self.N-1*self.N+i
                u[tout,j] += x[self.num_x-2,j]*tt*(-0.5+tt*(1-tt*0.5))
                u[tout,j] += x[self.num_x-1,j]*(1+tt*tt*(tt*1.5-2.5))

        return u

    def reverse(self, gradient):
        gx = np.zeros((self.num_x, self.num_ctrl))
        u = gradient

        for j in range(self.num_ctrl):
            for i in range(0,self.N//2):
                tt = (i+0.5)*self.dt+0.5

                gx[0,j] += u[i,j]*tt*(0.5+tt*(2-tt*1.5))
                gx[1,j] += u[i,j]*tt*tt*0.5*(tt-1)

                gx[0,j] += u[i+self.N,j]*(1+tt*tt*(tt*1.5-2.5))
                gx[1,j] += u[i+self.N,j]*tt*(0.5+tt*(2-tt*1.5))
                gx[2,j] += u[i+self.N,j]*tt*tt*0.5*(tt-1)

                for t in range(2,self.num_x-1):
                    tout = t*self.N+i
                    gx[t-2,j] += u[tout,j]*tt*(-0.5+tt*(1-tt*0.5))
                    gx[t-1,j] += u[tout,j]*(1+tt*tt*(tt*1.5-2.5))
                    gx[t  ,j] += u[tout,j]*tt*(0.5+tt*(2-tt*1.5))
                    gx[t+1,j] += u[tout,j]*tt*tt*0.5*(tt-1)

                tout = self.num_x*self.N-1*self.N+i
                gx[self.num_x-3,j] += u[tout,j]*tt*(-0.5+tt*(1-tt*0.5))
                gx[self.num_x-2,j] += u[tout,j]*(1+tt*tt*(tt*1.5-2.5))
                gx[self.num_x-1,j] += u[tout,j]*tt*(0.5+tt*(2-tt*1.5))

        for j in range(self.num_ctrl):
            for i in range(self.N):
                tt = (i+0.5)*self.dt-0.5

                gx[0,j] += u[i,j]*(1+tt*tt*(tt*1.5-2.5))
                gx[1,j] += u[i,j]*tt*(0.5+tt*(2-tt*1.5))
                gx[2,j] += u[i,j]*tt*tt*0.5*(tt-1)

                for t in range(1,self.num_x-2):
                    tout = t*self.N+i
                    gx[t-1,j]+= u[tout,j]*tt*(-0.5+tt*(1-tt*0.5))
                    gx[t  ,j]+= u[tout,j]*(1+tt*tt*(tt*1.5-2.5))
                    gx[t+1,j]+= u[tout,j]*tt*(0.5+tt*(2-tt*1.5))
                    gx[t+2,j]+= u[tout,j]*tt*tt*0.5*(tt-1)

                tout = self.num_x*self.N-2*self.N+i
                gx[self.num_x-3,j]+= u[tout,j]*tt*(-0.5+tt*(1-tt*0.5))
                gx[self.num_x-2,j]+= u[tout,j]*(1+tt*tt*(tt*1.5-2.5))
                gx[self.num_x-1,j]+= u[tout,j]*tt*(0.5+tt*(2-tt*1.5))

                tout = self.num_x*self.N-1*self.N+i
                gx[self.num_x-2,j]+= u[tout,j]*tt*(-0.5+tt*(1-tt*0.5))
                gx[self.num_x-1,j]+= u[tout,j]*(1+tt*tt*(tt*1.5-2.5))
        return gx/self.N/2

    def init_timeslots(self, times=None, tau=None, T=1, t_step=None,
                       num_x=None, num_ctrl=1):
        """
        Times/tau correspond to the timeslot before the interpollation.
        """

        self.num_ctrl = num_ctrl

        if times is not None:
            if not np.allclose(np.diff(times), times[1]-times[0]):
                raise Exception("Times must be equaly distributed")
            else:
                self.num_x = len(times)-1
                self.t_step = self.num_x * self.N
                T = times[-1]
        elif tau is not None:
            if not np.allclose(np.diff(tau), tau[0]):
                raise Exception("tau must be all equal")
            else:
                self.num_x = len(tau)
                self.t_step = self.num_x * self.N
                T = np.sum(tau)
        elif t_step is not None:
            self.num_x = t_step
            self.t_step = self.num_x * self.N
        else:
            self.num_x = 10
            self.t_step = self.num_x * self.N
        self.times = np.linspace(0, T, self.t_step+1)
        return (self.num_x, self.num_ctrl), self.times

    def _compute_xlim(self):
        if self.x_max is None and self.x_min is None:
            return

        if self.x_max is not None:
            if isinstance(self.x_max, list):
                self.x_max = np.array(self.x_max)
            if isinstance(self.x_max, (int, float)):
                self.x_max = self.x_max*np.ones((self.num_x,self.num_ctrl))
            elif isinstance(self.x_max, np.ndarray):
                if self.x_max.shape != (self.num_x,self.num_ctrl):
                    x_max = np.zeros((self.num_x,self.num_ctrl))
                    xx = np.linspace(0, self.times[-1], self.num_x+1)
                    xnew = (xx[1:] + xx[:-1]) * 0.5
                    t = (self.times[1:] + self.times[:-1]) * 0.5
                    for i in range(self.num_ctrl):
                        intf = interpolate.splrep(self.x_max[:,i], t, s=0)
                        x_max[:,i] = interpolate.splev(xnew, intf, der=0)
                    self.x_max = x_max
        else:
            self.x_max = np.array([[None]*self.num_ctrl]*self.num_x)

        if self.x_min is not None:
            if isinstance(self.x_min, list):
                self.x_min = np.array(self.x_min)
            if isinstance(self.x_min, (int, float)):
                self.x_min = self.x_min*np.ones((self.num_x,self.num_ctrl))
            elif isinstance(self.x_min, np.ndarray):
                if self.x_min.shape != (self.num_x,self.num_ctrl):
                    x_min = np.zeros((self.num_x,self.num_ctrl))
                    xx = np.linspace(0, self.times[-1], self.num_x+1)
                    xnew = (xx[1:] + xx[:-1]) * 0.5
                    t = (self.times[1:] + self.times[:-1]) * 0.5
                    for i in range(self.num_ctrl):
                        intf = interpolate.splrep(self.x_min[:,i], t, s=0)
                        x_min[:,i] = interpolate.splev(xnew, intf, der=0)
                    self.x_min = x_min
        else:
            self.x_min = np.array([[None]*self.num_ctrl]*self.num_x)


class gaussian(filter):
    def __init__(self, overSampleRate):
        super().__init__()
        self.N = overSampleRate
        self.dt = 1/overSampleRate

    def set_bandwith(self, omega):
        if isinstance(omega, (int, float)):
            omega = [omega] * self.num_ctrl
        dt = self.times[1]
        Dt = dt * self.N
        DDt = Dt*0.5
        self.T = np.zeros((len(self.times)-1, self.num_x, self.num_ctrl))
        time = (self.times[:-1] + self.times[1:]) * 0.5
        for i in range(self.num_ctrl):
            for j,t in enumerate(time):
                for tt in range(self.num_x):
                    T = (t-tt*Dt)*0.5
                    self.T[j,tt,i] = (erf(omega[i]*T)-erf(omega[i]*(T-DDt)))*0.5

    def __call__(self, x):
        return np.einsum('ijk,jk->ik', self.T, x)

    def reverse(self, gradient):
        return np.einsum('ijk,ik->jk', self.T, gradient)

    def init_timeslots(self, times=None, tau=None, T=1, t_step=None,
                       num_x=None, num_ctrl=1):
        """
        Times/tau correspond to the timeslot before the interpollation.
        """

        self.num_ctrl = num_ctrl

        if times is not None:
            if not np.allclose(np.diff(times), times[1]-times[0]):
                raise Exception("Times must be equaly distributed")
            else:
                self.num_x = len(times)-1
                self.t_step = self.num_x * self.N
                T = times[-1]
        elif tau is not None:
            if not np.allclose(np.diff(tau), tau[0]):
                raise Exception("tau must be all equal")
            else:
                self.num_x = len(tau)
                self.t_step = self.num_x * self.N
                T = np.sum(tau)
        elif t_step is not None:
            self.num_x = t_step
            self.t_step = self.num_x * self.N
        else:
            self.num_x = 10
            self.t_step = self.num_x * self.N
        self.times = np.linspace(0, T, self.t_step+1)
        return (self.num_x, self.num_ctrl), self.times




def get_filter():
    pass
