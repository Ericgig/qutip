
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy import interpolate
from scipy.fftpack import fft
from scipy.special import erf

"""
    Contain a transfer_functions class that allows the optimization variables
    of the grape algorithm to not be directly the amplitudes of the control
    fields.

    Optimal control methods for rapidly time-varying Hamiltonians, 2011
    Motzoi, F. and Gambetta, J. M. and Merkel, S. T. and Wilhelm, F. K.
    PhysRevA.84.022307, https://link.aps.org/doi/10.1103/PhysRevA.84.022307

    The available transfer_functions are:
        pass_througth:
            optimization variables are the amplitudes of the control fields

        fourrier:
            The amplitudes of the control fields is obtained by the fourrier
            series of the optimization variables.
            u[t] = x[i] * sin(t*(i+1)*pi/times[-1])
            The number of frequency used is set during initiation.

        spline:
            The amplitudes of the control fields is the spline interpolation
            of the optimization variables.
            The number of sampling per timeslices and the start and end
            value are set at initiation.

        gaussian:
            Represent a Gaussian filtering in the frequency domain.
            At initiation, the reference frequency, sampling rate, start
            and end values can be set.
"""

class transfer_functions:
    """A class for representing transfer functions, between optimization
    variables of the grape algorithm and the amplitudes of the control fields

    Parameters
    ----------
    num_x : array_like
        Data for vector/matrix representation of the quantum object.
    num_ctrls : list
        Dimensions of object used for tensor products.
    times : list
        Shape of underlying data structure (matrix shape).
    xtimes
    x_max
    x_min

    Attributes
    ----------
    ...

    Methods
    -------
    __call__(x):
        Return the amplitudes (u) from the optimisation variables (x).

    reverse_state(target):
        Return the x which best match the provided amplitudes.

    gradient_u2x(self, gradient):
        return the gradient of the x from a gradient in the u basis.

    set_times(times, num_ctrls=1):
        Set the times for the control amplitudes and return
        (shape of the optimisation variables), times of the control amplitudes
        * Depending on the transfer functions, the times of the control
          amplitudes and optimisation variables can differ.

    set_amp_bound(amp_lbound=None, amp_ubound=None):
        Input the amplitude bounds: (int, float) or list

    plotPulse(x):
        For the optimisation variables (x), plot the resulting pulse.

    get_xlimit():
        Return the bound for the optimisation variables in the format fitting
        scipy optimize.

    reverse_state(amplitudes):
        Obtain the best fitting optimisation variables to match the given
        amplitudes.
    """

    def __init__(self):
        self.num_x = 0
        self.num_ctrls = 0
        self.times = []
        self.xtimes = []
        self.x_max = None
        self.x_min = None

    def __call__(self, x):
        """
        return the amplitudes of the control fields
        from the optimization variables.
        """
        return x

    def gradient_u2x(self, gradient):
        """
        Obtain the grandient of the optimization variables from the
        gradient of the amplitude of the control fields
        """
        return gradient

    def init_timeslots_old(self, times=None, tau=None, T=1,
                       t_step=10, num_x=None, num_ctrls=1):
         if times is not None:
             t_step = len(times)-1
             T = times[-1]
             time = times
             self.times = times
         elif tau is not None:
             t_step = len(tau)
             T = np.sum(tau)
             self.times = np.cumsum(np.insert(tau,0,0))
         else:
             t_step = 10
             self.times = np.linspace(0,T,t_step+1)
         self.num_x = t_step
         self.t_step = t_step
         self.num_ctrls = num_ctrls
         return (t_step, num_ctrls), self.times

    def set_times(self, times, num_ctrls=1):
        """
        Generate the timeslot duration array 'tau' based on the evo_time
        and num_tslots attributes, unless the tau attribute is already set
        in which case this step in ignored
        Generate the cumulative time array 'time' based on the tau values
        """
        if isinstance(times, list):
            times = np.array(times)
        if not isinstance(times, np.ndarray):
            raise Exception("times must be a list or np.array")
        if not np.all(np.diff(times)>=0):
            raise Exception("times must be sorted")
        self.num_x = len(times)-1
        self.num_ctrls = num_ctrls
        self.times = times
        self.xtimes = times
        return (self.num_x, num_ctrls), times

    def set_amp_bound(self, amp_lbound=None, amp_ubound=None):
        """
        Input the amplitude bounds
        * For some transfer functions (fourrier)
          take the bound on the optimisation variables.
        """
        if amp_lbound is not None and not \
                isinstance(amp_lbound, (int, float, list, np.ndarray)):
            raise Exception("bounds must be a real or list of real")
        if amp_ubound is not None and not \
                isinstance(amp_ubound, (int, float, list, np.ndarray)):
            raise Exception("bounds must be a real or list of real")
        self.x_min = amp_lbound
        self.x_max = amp_ubound

    def _compute_xlim(self):
        if self.x_max is None and self.x_min is None:
            return

        if self.x_max is not None:
            if isinstance(self.x_max, list):
                self.x_max = np.array(self.x_max)
            if isinstance(self.x_max, (int, float)):
                self.x_max = self.x_max*np.ones((self.num_x,self.num_ctrls))
            elif isinstance(self.x_max, np.ndarray):
                if self.x_max.shape == (self.num_x,self.num_ctrls):
                    pass
                elif self.x_max.shape == (self.num_x) or \
                        self.x_max.shape == (self.num_x, 1):
                    self.x_max = np.einsum("i,j->ij",
                                                 np.ones(self.num_x),
                                                 self.x_max)
                else:
                    raise Exception("shape of the amb_ubound not right")
        else:
            self.x_max = np.array([[None]*self.num_ctrls]*self.num_x)

        if self.x_min is not None:
            if isinstance(self.x_min, list):
                self.x_min = np.array(self.x_min)
            if isinstance(self.x_min, (int, float)):
                self.x_min = self.x_min*np.ones((self.num_x,self.num_ctrls))
            elif isinstance(self.x_min, np.ndarray):
                if self.x_min.shape == (self.num_x,self.num_ctrls):
                    pass
                elif self.x_min.shape == (self.num_x) or \
                        self.x_min.shape == (self.num_x, 1):
                    self.x_min = np.einsum("i,j->ij",
                                                 np.ones(self.num_x),
                                                 self.x_min)
                else:
                    raise Exception("shape of the amb_lbound not right")
        else:
            self.x_min = np.array([[None]*self.num_ctrls]*self.num_x)

    def plotPulse(self, x):
        """
        Plot the control amplitudes corresponding
        to the given optimisation variables.
        """
        u = self(x)
        t, dt = (self.times[:-1]+self.times[1:])*0.5, np.diff(self.times)
        xt, dxt = (self.xtimes[:-1]+self.xtimes[1:])*0.5, np.diff(self.xtimes)
        for i in range(self.num_ctrls):
            plt.bar(t, u[:,i], dt)
            plt.bar(xt, x[:,i], dxt, fill=False)
            plt.show()

    def originalTimesAmps(self, x):
        """
        Amps as a function to the given times.
        """
        return x, self.xtimes

    def interpolatedAmpsAndTimes(self, x):
        """
        Amps as a function to the given times.
        """
        return self(x), self.times

    def get_xlimit(self):
        """
        Return the bound for the optimisation variables in the format fitting
        scipy optimize.
        """
        self._compute_xlim()
        if self.x_max is None and self.x_min is None:
            return None  # No constrain
        xmax = self.x_max.astype(object)
        xmax[np.isinf(self.x_max)] = None
        xmax = list(self.x_max.flatten())
        xmin = self.x_min.astype(object)
        xmin[np.isinf(self.x_min)] = None
        xmin = list(self.x_min.flatten())
        return zip(xmin,xmax)

    def reverse_state(self, amplitudes=None, times=None, targetfunc=None):
        """
        Return the best fitting optimisation variables from amplitudes.
        2 calling method
            amplitudes: np.array(Nt, num_ctrls)
            times: np.array(Nt)
        or
            targetfunc: callable
                targetfunc(times, num_ctrls):
                    return amplitudes(len(times), num_ctrls)
        """
        x_shape = (self.num_x, self.num_ctrls)
        xx = np.zeros(x_shape)
        target_time = (self.times[1:]+self.times[:-1])*0.5

        if amplitudes is not None:
            amplitudes = np.array(amplitudes)
            if times is None:
                times = target_time
                if len(times) != amplitudes.shape[0]:
                    raise Exception("Please add times")
            else:
                times = np.array(times)
                if amplitudes.shape[0]+1 == len(times):
                    times = (times[1:]+times[:-1])*0.5
                elif amplitudes.shape[0] == len(times):
                    raise Exception("Length of times do not fit the "
                                    "shape of amplitudes")

            if len(amplitudes.shape) == 1 or amplitudes.shape[1] == 1:
                amplitudes = amplitudes.reshape((amplitudes.shape[0], 1))
                amplitudes = np.einsum("i,j->ij", amplitudes,
                                       np,ones(self.num_ctrls))

            if amplitudes.shape[1] != self.num_ctrls:
                raise Exception("amplitudes' shape must be (Nt, num_ctrls)")

            if np.all(times == target_time):
                target = amplitudes
            else:
                target = np.zeros((len(target_time),self.num_ctrls))
                for i in range(self.num_ctrls):
                    tck = interpolate.splrep(times, amplitudes[:,i], s=0)
                    target[:,i] = interpolate.splev(target_time, tck, der=0)

        elif targetfunc is not None:
            try:
                target = targetfunc(target_time, self.num_ctrls)
                if target.shape != (len(target_time), self.num_ctrls):
                    raise Exception()
            except e:
                raise Exception(targetfunc.__name__ +" call failed:\n"
                                "expected signature:\n"
                                "targetfunc(times, num_ctrls) -> amplitudes "
                                "shape = (Nt, num_ctrls)")
        else:
            # no target, set to zeros
            target = np.zeros((len(target_time), self.num_ctrls))

        def diff(y):
            yy = self(y.reshape(x_shape))
            return np.sum((yy-target)**2)

        def gradient(y):
            yy = self(y.reshape(x_shape))
            grad = self.gradient_u2x((yy-target)*2)
            return grad.reshape(np.prod(x_shape))

        rez = opt.minimize(fun=diff, jac=gradient, x0=xx)
        return rez.x.reshape(x_shape)

class pass_througth(transfer_functions):
    """
    The optimisation variables are the amplitudes.
    """
    def __init__(self):
        super().__init__()
        self.name = "Pass_througth"

class fourrier(transfer_functions):
    """
        Pulse described as fourrier modes
        u(t) = sum_j x(j)*sin(pi*j*t/T); 1 <= j<= num_x
        The number of frequency is set at initialisation.
    """
    def __init__(self, num_x=None):
        super().__init__()
        self.name = "Fourrier"
        self.num_x = num_x

    def __call__(self, x):
        u = np.zeros((self.t_step, self.num_ctrls))
        s = np.zeros((self.t_step*2-2, self.num_ctrls))
        s[1:self.num_x+1,:] = x
        for j in range(self.num_ctrls):
            u[:,j] = -fft(s[:,j]).imag[:self.t_step]
        return u

    def gradient_u2x(self, gradient):
        x = np.zeros((self.num_x, self.num_ctrls))
        s = np.zeros((self.t_step*2-2, self.num_ctrls))
        s[:self.t_step,:] = gradient
        for j in range(self.num_ctrls):
            x[:,j] = -fft(s[:,j]).imag[1:self.num_x+1]
        return x

    def init_timeslots_old(self, times=None, tau=None, T=1, t_step=None,
                             num_x=None, num_ctrls=1):
        self.num_ctrls = num_ctrls
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
        self.times = np.linspace(0, T, self.t_step+1)
        return (self.num_x, self.num_ctrls), self.times

    def set_times(self, times, num_ctrls=1):
        self.num_ctrls = num_ctrls
        if not np.allclose(np.diff(times), times[1]-times[0]):
            raise Exception("Times must be equaly distributed")
        self.num_ctrls = num_ctrls
        self.times = times
        self.xtimes = times
        self.t_step = len(times)-1
        if self.num_x is None:
            self.num_x = min([10,self.t_step//10])
        return (self.num_x, self.num_ctrls), self.times

    def _compute_xlim(self):
        if self.x_max is None and self.x_min is None:
            return
        if self.x_max is not None:
            if isinstance(self.x_max, list):
                self.x_max = np.array(self.x_max)
            if isinstance(self.x_max, (int, float)):
                self.x_max = self.x_max*np.ones((self.num_x,self.num_ctrls))
            elif isinstance(self.x_max, np.ndarray):
                if self.x_max.shape != (self.num_x,self.num_ctrls):
                    raise Exception("fourrier: wrong bounds shape")
        else:
            self.x_max = np.array([[None]*self.num_ctrls]*self.num_x)

        if self.x_min is not None:
            if isinstance(self.x_min, list):
                self.x_min = np.array(self.x_min)
            if isinstance(self.x_min, (int, float)):
                self.x_min = self.x_min*np.ones((self.num_x,self.num_ctrls))
            elif isinstance(self.x_min, np.ndarray):
                if self.x_min.shape != (self.num_x,self.num_ctrls):
                    raise Exception("fourrier: wrong bounds shape")
        else:
            self.x_min = np.array([[None]*self.num_ctrls]*self.num_x)

    def plotPulse(self, x):
        u = self(x)
        t, dt = (self.times[:-1]+self.times[1:])*0.5, np.diff(self.times)
        for i in range(self.num_ctrls):
            plt.bar(np.arange(self.num_x), x[:,i], 0.7)
            plt.title("Fourier series")
            plt.show()
            plt.bar(t, u[:,i], dt)
            plt.title("Amplidutes")
            plt.show()

    def originalTimesAmps(self, x):
        """
        Amps as a function to the given times.
        """
        return self(x), self.xtimes

class spline(transfer_functions):
    """
    spline interpolation to smoote the pulse.
    options set a initialisation:
        overSampleRate: number of timeslice of the amplitudes for each x block.
        start, end: amplitudes at the boundaries of the time range.

    The amplitudes time range is wider than the given times.
    """
    def __init__(self, overSampleRate=5, start=0., end=0.):
        super().__init__()
        self.N = overSampleRate
        self.dt = 1/overSampleRate
        self.boundary = [start, end]
        self.name = "Spline"

    def make_T(self):
        self.T = np.zeros((len(self.times)-1, self.num_x))
        boundLenght = 2*self.N - self.N%2
        self.front = np.zeros(2*self.N)
        self.back = np.zeros(boundLenght)
        dtt = 0.5 + 0.5 *(self.N%2)
        for i in range(0,self.N):
            tt = (i+dtt)*self.dt
            self.front[i] = tt*(-0.5+tt*(1-tt*0.5)) * self.boundary[0]
            self.front[i] += (1+tt*tt*(tt*1.5-2.5)) * self.boundary[0]
            self.T[0*self.N+i, 0] = tt*(0.5+tt*(2-tt*1.5))
            self.T[0*self.N+i, 1] = tt*tt*0.5*(tt-1)

            self.front[self.N+i] = tt*(-0.5+tt*(1-tt*0.5)) * self.boundary[0]
            self.T[1*self.N+i, 0] = (1+tt*tt*(tt*1.5-2.5))
            self.T[1*self.N+i, 1] = tt*(0.5+tt*(2-tt*1.5))
            self.T[1*self.N+i, 2] = tt*tt*0.5*(tt-1)

            for t in range(2,self.num_x-1):
                tout = t*self.N+i
                self.T[tout, t-2] = tt*(-0.5+tt*(1-tt*0.5))
                self.T[tout, t-1] = (1+tt*tt*(tt*1.5-2.5))
                self.T[tout, t  ] = tt*(0.5+tt*(2-tt*1.5))
                self.T[tout, t+1] = tt*tt*0.5*(tt-1)

            tout = (self.num_x-1)*self.N+i
            self.T[tout, self.num_x-3] = tt*(-0.5+tt*(1-tt*0.5))
            self.T[tout, self.num_x-2] = (1+tt*tt*(tt*1.5-2.5))
            self.T[tout, self.num_x-1] = tt*(0.5+tt*(2-tt*1.5))
            self.back[i] = tt*tt*0.5*(tt-1) * self.boundary[1]

            tout = (self.num_x-0)*self.N+i
            if tout != len(self.times)-1:
                self.T[tout, self.num_x-2] = tt*(-0.5+tt*(1-tt*0.5))
                self.T[tout, self.num_x-1] = (1+tt*tt*(tt*1.5-2.5))
                self.back[self.N+i] = tt*(0.5+tt*(2-tt*1.5)) * self.boundary[1]
                self.back[self.N+i] += tt*tt*0.5*(tt-1) * self.boundary[1]
        self.front = self.front.reshape((self.N*2,1))
        self.back = self.back.reshape((boundLenght,1))

    def __call__(self, x):
        u = np.einsum('ij,jk->ik', self.T, x)
        boundLenght = 2*self.N - self.N%2
        u[:2*self.N,:] += self.front
        u[-boundLenght:,:] += self.back
        return u

    def gradient_u2x(self, gradient):
        return np.einsum('ij,ik->jk', self.T, gradient)

    def init_timeslots_old(self, times=None, tau=None, T=1, t_step=None,
                       num_x=None, num_ctrls=1):
        """
        Times/tau correspond to the timeslot before the interpolation.
        """

        self.num_ctrls = num_ctrls

        if times is not None:
            if not np.allclose(np.diff(times), times[1]-times[0]):
                raise Exception("Times must be equaly distributed")
            elif times[0] != 0:
                raise Exception("Times must start at 0")
            else:
                self.num_x = len(times)-1
                #self.t_step = (self.num_x+1) * self.N
                T = times[-1]
        elif tau is not None:
            if not np.allclose(np.diff(tau), tau[0]):
                raise Exception("tau must be all equal")
            else:
                self.num_x = len(tau)
                #self.t_step = (self.num_x+1) * self.N
                T = np.sum(tau)
        elif t_step is not None:
            self.num_x = t_step
            #self.t_step = (self.num_x+1) * self.N
        else:
            self.num_x = 10
            #self.t_step = (self.num_x+1) * self.N
        dt = T/self.num_x/self.N
        extra_t = (self.N//2)
        dtt = 0#0.5 - (self.N - (self.N//2)*2)*0.5
        self.times = np.linspace(-dt*(extra_t+dtt), T + dt*(extra_t+dtt),
                                 self.num_x*self.N + 2*extra_t + 1)
        self.xtimes = np.linspace(0, T, self.num_x+1)
        self.make_T()
        return (self.num_x, self.num_ctrls), self.times

    def set_times(self, times, num_ctrls=1):
        """
        Times/tau correspond to the timeslot before the interpolation.
        """
        self.num_ctrls = num_ctrls
        if not np.allclose(np.diff(times), times[1]-times[0]):
            raise Exception("Times must be equaly distributed")
        elif times[0] != 0:
            raise Exception("Times must start at 0")

        self.num_x = len(times)-1
        T = times[-1]
        dt = T/self.num_x/self.N
        extra_t = (self.N//2)
        self.xtimes = times
        self.times = np.linspace(-dt*(extra_t), T + dt*(extra_t),
                                 self.num_x*self.N + 2*extra_t + 1)
        self.make_T()
        return (self.num_x, self.num_ctrls), self.times

    def _compute_xlim(self):
        if self.x_max is None and self.x_min is None:
            return

        if self.x_max is not None:
            if isinstance(self.x_max, list):
                self.x_max = np.array(self.x_max)
            if isinstance(self.x_max, (int, float)):
                self.x_max = self.x_max*np.ones((self.num_x,self.num_ctrls))
            elif isinstance(self.x_max, np.ndarray):
                if self.x_max.shape != (self.num_x,self.num_ctrls):
                    x_max = np.zeros((self.num_x,self.num_ctrls))
                    xx = np.linspace(0, self.times[-1], self.num_x+1)
                    xnew = (xx[1:] + xx[:-1]) * 0.5
                    t = (self.times[1:] + self.times[:-1]) * 0.5
                    for i in range(self.num_ctrls):
                        intf = interpolate.splrep(self.x_max[:,i], t, s=0)
                        x_max[:,i] = interpolate.splev(xnew, intf, der=0)
                    self.x_max = x_max
        else:
            self.x_max = np.array([[None]*self.num_ctrls]*self.num_x)

        if self.x_min is not None:
            if isinstance(self.x_min, list):
                self.x_min = np.array(self.x_min)
            if isinstance(self.x_min, (int, float)):
                self.x_min = self.x_min*np.ones((self.num_x,self.num_ctrls))
            elif isinstance(self.x_min, np.ndarray):
                if self.x_min.shape != (self.num_x,self.num_ctrls):
                    x_min = np.zeros((self.num_x,self.num_ctrls))
                    xx = np.linspace(0, self.times[-1], self.num_x+1)
                    xnew = (xx[1:] + xx[:-1]) * 0.5
                    t = (self.times[1:] + self.times[:-1]) * 0.5
                    for i in range(self.num_ctrls):
                        intf = interpolate.splrep(self.x_min[:,i], t, s=0)
                        x_min[:,i] = interpolate.splev(xnew, intf, der=0)
                    self.x_min = x_min
        else:
            self.x_min = np.array([[None]*self.num_ctrls]*self.num_x)

class gaussian(transfer_functions):
    """
    Represent square function filtered through a gaussian filter.
    Options set a initialisation:
        omega: (float, list) bandwitdh of the
        overSampleRate: number of timeslice of the amplitudes for each x block.
        start, end: amplitudes at the boundaries of the time range.
        bound_type = (code, number): control the number of time slice of padding
                                    before and after the original time range.
            code:
                "n": n extra slice of dt/overSampleRate
                "x": n extra slice of dt
                "w": go until a dampening of erf(n) (default, n=2)

    The amplitudes time range is wider than the given times.
    """

    def __init__(self, omega=1, overSampleRate=5, start=0., end=0.,
                 bound_type=("w",2)):
        super().__init__()
        self.N = overSampleRate
        self.dt = 1/overSampleRate
        self.boundary = [start, end]
        self.omega = omega
        self.bound_type = bound_type
        self.name = "Gaussian"

    def make_T(self):
        if isinstance(self.omega, (int, float)):
            omega = [self.omega] * self.num_ctrls
        else:
            omega = self.omega
        if isinstance(self.boundary[0], (int, float)):
            start = [self.boundary[0]] * self.num_ctrls
        else:
            start = self.boundary[0]
        if isinstance(self.boundary[1], (int, float)):
            end = [self.boundary[1]] * self.num_ctrls
        else:
            end = self.boundary[1]

        Dxt = (self.xtimes[1]-self.xtimes[0])*0.25
        self.T = np.zeros((len(self.times)-1, self.num_x, self.num_ctrls))
        self.cte = np.zeros((len(self.times)-1, self.num_ctrls))
        time = (self.times[:-1] + self.times[1:]) * 0.5
        xtime = (self.xtimes[:-1] + self.xtimes[1:]) * 0.5
        for i in range(self.num_ctrls):
            for j,t in enumerate(time):
                self.cte[j,i] =(0.5-0.5*erf(omega[i]*0.5*t))*start[i]
                self.cte[j,i] +=(0.5+0.5*erf(omega[i]*0.5*(t-self.xtimes[-1])))*end[i]
                for k, xt in enumerate(xtime):
                    T = (t-xt)*0.5
                    self.T[j,k,i] = (erf(omega[i]*(T+Dxt))-erf(omega[i]*(T-Dxt)))*0.5

    def __call__(self, x):
        return np.einsum('ijk,jk->ik', self.T, x)+self.cte

    def gradient_u2x(self, gradient):
        return np.einsum('ijk,ik->jk', self.T, gradient)

    def set_times(self, times, num_ctrls=1):
        """
        Times/tau correspond to the timeslot before the interpolation.
        """
        self.num_ctrls = num_ctrls
        if not np.allclose(np.diff(times), times[1]-times[0]):
            raise Exception("Times must be equaly distributed")
        elif times[0] != 0:
            raise Exception("Times must start at 0")
        self.num_x = len(times)-1
        T = times[-1]
        dt = T/self.num_x/self.N
        #self.x_times = np.linspace(0, T, self.num_x+1)
        if self.bound_type[0] == "x":
            extra_t = self.bound_type[1] * self.N
        elif self.bound_type[0] == "n":
            extra_t = self.bound_type[1]
        else:
            if isinstance(self.omega, (int, float)):
                omega = [self.omega] * self.num_ctrls
            else:
                omega = self.omega
            extra_t = []
            for w in omega:
                extra_t += [2*self.bound_type[1]/w/dt]
            extra_t = int(max(extra_t))

        self.xtimes = times
        self.times = np.linspace(-dt*extra_t, T + dt*extra_t,
                                 self.num_x * self.N + 2*extra_t + 1)
        self.make_T()
        return (self.num_x, self.num_ctrls), self.times

    def init_timeslots_old(self, times=None, tau=None, T=1, t_step=None,
                       num_x=None, num_ctrls=1):
        """
        Times/tau correspond to the timeslot before the interpolation.
        """

        self.num_ctrls = num_ctrls

        if times is not None:
            if not np.allclose(np.diff(times), times[1]-times[0]):
                raise Exception("Times must be equaly distributed")
            elif times[0] != 0:
                raise Exception("Times must start at 0")
            else:
                self.num_x = len(times)-1
                T = times[-1]
        elif tau is not None:
            if not np.allclose(np.diff(tau), tau[0]):
                raise Exception("tau must be all equal")
            else:
                self.num_x = len(tau)
                T = np.sum(tau)
        elif t_step is not None:
            self.num_x = t_step
        else:
            self.num_x = 10

        dt = T/self.num_x/self.N
        #self.x_times = np.linspace(0, T, self.num_x+1)
        if self.bound_type[0] == "x":
            extra_t = self.bound_type[1] * self.N
        elif self.bound_type[0] == "n":
            extra_t = self.bound_type[1]
        else:
            if isinstance(self.omega, (int, float)):
                omega = [self.omega] * self.num_ctrls
            else:
                omega = self.omega
            extra_t = []
            for w in omega:
                extra_t += [2*self.bound_type[1]/w/dt]
            extra_t = max(extra_t)
        self.times = np.linspace(-dt*extra_t, T + dt*extra_t,
                                 self.num_x * self.N + 2*extra_t + 1)
        self.xtimes = np.linspace(0, T, self.num_x+1)
        self.make_T()
        return (self.num_x, self.num_ctrls), self.times

    def _compute_xlim(self):
        if self.x_max is None and self.x_min is None:
            return

        if self.x_max is not None:
            if isinstance(self.x_max, list):
                self.x_max = np.array(self.x_max)
            if isinstance(self.x_max, (int, float)):
                self.x_max = self.x_max*np.ones((self.num_x,self.num_ctrls))
            elif isinstance(self.x_max, np.ndarray):
                if self.x_max.shape != (self.num_x,self.num_ctrls):
                    raise Exception("shape of the amb_bound not right")
        else:
            self.x_max = np.array([[None]*self.num_ctrls]*self.num_x)

        if self.x_min is not None:
            if isinstance(self.x_min, list):
                self.x_min = np.array(self.x_min)
            if isinstance(self.x_min, (int, float)):
                self.x_min = self.x_min*np.ones((self.num_x,self.num_ctrls))
            elif isinstance(self.x_min, np.ndarray):
                if self.x_min.shape != (self.num_x,self.num_ctrls):
                    raise Exception("shape of the amb_bound not right")
        else:
            self.x_min = np.array([[None]*self.num_ctrls]*self.num_x)




class PulseGenCrab(transfer_functions):
    """

    """

    def __init__(self, guess_pulse=None, ramping_pulse=None,
                 guess_pulse_action='MODULATE'):
        self.num_x = 0
        self.num_ctrls = 0
        self.times = []
        self.xtimes = []

        self.apply_bound = False
        self.x_max = None
        self.x_min = None
        self._bound_scale_cond = None
        self._bound_mean = []
        self._bound_scale = []
        self.scaling = 1.0

        self.guess_pulse = guess_pulse
        self.ramping_pulse = ramping_pulse
        self.guess_pulse_func = None
        self.guess_pulse_action = guess_pulse_action
        self.name = "PulseGenCrab"
        self.init_guess_pulse()

    def set_times(self, times, num_ctrls=1):
        self.num_ctrls = num_ctrls
        self.times = (times[:-1]+times[1:])*0.5
        self.xtimes = times
        nt = len(self.times)
        if self.guess_pulse is None:
            self.guess_pulse = np.ones((nt, self.num_ctrls))
        if self.ramping_pulse is None:
            self.ramping_pulse = np.ones(nt)
        if self.guess_pulse.shape != (nt, self.num_ctrls):
            self.guess_pulse = np.einsum("i,j->ij", self.guess_pulse,
                                                    np.ones(self.num_ctrls))
        self._init_bounds()
        return (self.num_x, self.num_ctrls), self.times

    def _apply_ramping_pulse(self, pulse):
        for i in range(self.num_ctrls):
            pulse[:,i] *= self.ramping_pulse
        return pulse

    def init_guess_pulse(self):
        if self.guess_pulse_action is None:
            self.guess_pulse_action = 'MODULATE'
        if self.guess_pulse_action.upper() == 'MODULATE':
            self.guess_pulse_func = self.guess_pulse_modulate
        else:
            self.guess_pulse_func = self.guess_pulse_add

    def guess_pulse_add(self, pulse):
        pulse = pulse + self.guess_pulse
        return pulse

    def guess_pulse_modulate(self, pulse):
        pulse = (1.0 + pulse)*self.guess_pulse
        return pulse

    def set_amp_bound(self, amp_lbound=None, amp_ubound=None):
        """
        Input the amplitude bounds
        * For some transfer functions (fourrier)
          take the bound on the optimisation variables.
        """
        if amp_ubound is not None and not \
                isinstance(amp_ubound, (int, float, list, np.ndarray)):
            raise Exception("bounds must be a real or a list of real numbers")
        if amp_ubound is not None and not \
                isinstance(amp_ubound, (int, float, list, np.ndarray)):
            raise Exception("bounds must be a real or a list of real numbers")
        self.x_min = amp_lbound
        self.x_max = amp_ubound

    def _init_bounds(self):
        if self.x_min is None and self.x_max is None:
            # no bounds to apply
            self._bound_scale_cond = None

        elif self.x_min is None:
            if isinstance(self.x_max, (int, float)):
                self.x_max = [self.x_max]*self.num_ctrls
            if len(self.x_max) != self.num_ctrls:
                raise Exception("Bound for CRAB, "
                                "must be one or a list of num_ctrls values")
            # only upper bound
            self.apply_bound = True
            self._bound_scale_cond = "_BSC_GT_MEAN"
            for xmax in self.x_max:
                if xmax > 0:
                    self._bound_mean += [0.0]
                    self._bound_scale += [xmax]
                else:
                    self._bound_scale += [self.scaling*self.num_coeffs + \
                                self.get_guess_pulse_scale()]
                    self._bound_mean += [-abs(self._bound_scale) + xmax]

        elif self.x_max is None:
            if isinstance(self.x_min, (int, float)):
                self.x_min = [self.x_min]*self.num_ctrls
            if len(self.x_min) != self.num_ctrls:
                raise Exception("Bound for CRAB, "
                                "must be one or a list of num_ctrls values")
            # only lower bound
            self.apply_bound = True
            self._bound_scale_cond = "_BSC_LT_MEAN"
            for xmin in self.x_min:
                if xmin < 0:
                    self._bound_mean += [0.0]
                    self._bound_scale += [abs(xmin)]
                else:
                    self._bound_scale += [self.scaling*self.num_coeffs + \
                                self.get_guess_pulse_scale()]
                    self._bound_mean += [abs(self._bound_scale) + xmin]

        else:
            if isinstance(self.x_min, (int, float)):
                self.x_min = [self.x_min]*self.num_ctrls
            if len(self.x_min) != self.num_ctrls:
                raise Exception("Bound for CRAB, "
                                "must be one or a list of num_ctrls values")
            if isinstance(self.x_max, (int, float)):
                self.x_max = [self.x_max]*self.num_ctrls
            if len(self.x_max) != self.num_ctrls:
                raise Exception("Bound for CRAB, "
                                "must be one or a list of num_ctrls values")
            # lower and upper bounds
            self.apply_bound = True
            self._bound_scale_cond = "_BSC_ALL_LT0"
            for xmin, xmax in zip(self.x_min, self.x_max):
                if xmin == 0:
                    # can touch the lower bound
                    self._bound_mean += [0.0]
                    self._bound_scale += [xmax]
                else:
                    self._bound_mean += [0.5*(xmax + xmin)]
                    self._bound_scale += [0.5*(xmax - xmin)]
                    self._bound_scale_cond = "_BSC_ALL"

        self._bound_mean = np.array(self._bound_mean)
        self._bound_mean = np.einsum("i,j->ij",
                                     np.ones(len(self.times)),
                                     self._bound_mean)
        self._bound_scale = np.array(self._bound_scale)
        self._bound_scale = np.einsum("i,j->ij",
                                      np.ones(len(self.times)),
                                      self._bound_scale)


    def get_guess_pulse_scale(self):
        scale = 0.0
        if self.guess_pulse is not None:
            scale = max(np.amax(self.guess_pulse) - np.amin(self.guess_pulse),
                        np.amax(self.guess_pulse))
        return scale

    def _apply_bounds(self, pulse):
        """
        Scaling the amplitudes using the tanh function if there are bounds
        """
        if self._bound_scale_cond == "_BSC_ALL":
            pulse = (pulse-self._bound_mean)/self._bound_scale
            pulse = np.tanh(pulse)*self._bound_scale + self._bound_mean

        elif self._bound_scale_cond == "_BSC_ALL_LT0":
            pulse[pulse<0.0] = 0.
            pulse = ( np.tanh(pulse/self._bound_scale)
                                  *self._bound_scale)

        elif self._bound_scale_cond == "_BSC_GT_MEAN":
            scale_where = pulse > self._bound_mean
            pulse[scale_where] = (pulse[scale_where]-self._bound_mean)/self._bound_scale
            pulse[scale_where] = (np.tanh(pulse[scale_where])*self._bound_scale
                                        + self._bound_mean)

        elif self._bound_scale_cond == "_BSC_LT_MEAN":
            scale_where = pulse < self._bound_mean
            pulse[scale_where] = (pulse[scale_where]-self._bound_mean)/self._bound_scale
            pulse[scale_where] = (np.tanh(pulse[scale_where])*self._bound_scale
                                        + self._bound_mean)

        return pulse

class PulseGenCrabFourier(PulseGenCrab):
    """
    Generates a pulse using the Fourier basis functions, i.e. sin and cos

    Attributes
    ----------
    freqs : float array[num_coeffs]
        Frequencies for the basis functions
    randomize_freqs : bool
        If True (default) the some random offset is applied to the frequencies
    """
    def __init__(self, num_x=10, opt_freq=False, randomize_freqs=False,
                 guess_pulse=None, ramping_pulse=None, guess_pulse_action='MODULATE'):
        super().__init__(guess_pulse=guess_pulse, ramping_pulse=ramping_pulse,
                         guess_pulse_action=guess_pulse_action)
        self.num_tslots = 0
        self.num_x = num_x
        self.ff = 2*np.pi
        self.opt_freq = opt_freq
        self.randomize_freqs = randomize_freqs
        self.name = "PulseGenCrabFourier"

    def set_times(self, times, num_ctrls=1):
        super().set_times(times, num_ctrls)
        self.num_tslots = len(self.times)
        self.init_freqs()
        if self.opt_freq:
            return (self.num_x*3, self.num_ctrls), self.times
        else:
            return (self.num_x*2, self.num_ctrls), self.times

    def init_freqs(self):
        """
        Generate the frequencies
        These are the Fourier harmonics with a uniformly distributed
        random offset
        """
        self.freqs = np.empty(self.num_x)
        self.ff = 2*np.pi / self.xtimes[-1]
        for i in range(self.num_x):
            self.freqs[i] = self.ff*(i + 1)

        if self.randomize_freqs:
            self.freqs += (np.random.random(self.num_x) - 0.5)*self.ff

    def _apply_freq_bounds(self, freqs):
        #freqs = (freqs-self.freqs)/self.ff
        freqs = np.tanh(freqs)*self.ff + self.freqs
        return freqs

    def __call__(self, x):
        """
        Generate a pulse using the Fourier basis with the freqs and
        coeffs attributes.

        Parameters
        ----------
        coeffs : float array[num_coeffs, num_basis_funcs]
            The basis coefficient values
            If given this overides the default and sets the attribute
            of the same name.
        """

        """if not self._pulse_initialised:
            self.init_pulse()"""

        pulse = np.zeros((self.num_tslots, self.num_ctrls))
        for i in range(self.num_ctrls):
            if self.opt_freq:
                coeffs = x[:,i].reshape((self.num_x, 3))
                freqs = coeffs[:,2]
                freqs = self._apply_freq_bounds(freqs)
            else:
                coeffs = x[:,i].reshape((self.num_x, 2))
                freqs = self.freqs
            for j in range(self.num_x):
                phase = freqs[j]*self.times
                pulse[:,i] += coeffs[j, 0]*np.sin(phase) +\
                              coeffs[j, 1]*np.cos(phase)

        if self.guess_pulse_func:
            pulse = self.guess_pulse_func(pulse)
        if self.ramping_pulse is not None:
            pulse = self._apply_ramping_pulse(pulse)
        if self.apply_bound:
            pulse = self._apply_bounds(pulse)
        return pulse

    def plotPulse(self, x):
        u = self(x)
        t, dt = (self.times, np.diff(self.xtimes))
        for i in range(self.num_ctrls):
            #plt.bar(np.arange(self.num_x), x[:,i], 0.7)
            #plt.title("Fourier series")
            #plt.show()
            plt.bar(t, u[:,i], dt)
            plt.title("Amplidutes")
            plt.show()
