
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
    num_ctrl : list
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

    set_times(times, num_ctrl=1):
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
        self.num_ctrl = 0
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
                       t_step=10, num_x=None, num_ctrl=1):
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
         self.num_ctrl = num_ctrl
         return (t_step, num_ctrl), self.times

    def set_times(self, times, num_ctrl=1):
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
        self.num_ctrl = num_ctrl
        self.times = times
        self.xtimes = times
        return (self.num_x, num_ctrl), times

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
                self.x_max = self.x_max*np.ones((self.num_x,self.num_ctrl))
            elif isinstance(self.x_max, np.ndarray):
                if self.x_max.shape != (self.num_x,self.num_ctrl):
                    raise Exception("shape of the amb_ubound not right")
        else:
            self.x_max = np.array([[None]*self.num_ctrl]*self.num_x)

        if self.x_min is not None:
            if isinstance(self.x_min, list):
                self.x_min = np.array(self.x_min)
            if isinstance(self.x_min, (int, float)):
                self.x_min = self.x_min*np.ones((self.num_x,self.num_ctrl))
            elif isinstance(self.x_min, np.ndarray):
                if self.x_min.shape != (self.num_x,self.num_ctrl):
                    raise Exception("shape of the amb_lbound not right")
        else:
            self.x_min = np.array([[None]*self.num_ctrl]*self.num_x)

    def plotPulse(self, x):
        """
        Plot the control amplitudes corresponding
        to the given optimisation variables.
        """
        u = self(x)
        t, dt = (self.times[:-1]+self.times[1:])*0.5, np.diff(self.times)
        xt, dxt = (self.xtimes[:-1]+self.xtimes[1:])*0.5, np.diff(self.xtimes)
        for i in range(self.num_ctrl):
            plt.bar(t, u[:,i], dt)
            plt.bar(xt, x[:,i], dxt, fill=False)
            plt.show()

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
            amplitudes: np.array(Nt, num_ctrl)
            times: np.array(Nt)
        or
            targetfunc: callable
                targetfunc(times, num_ctrl):
                    return amplitudes(len(times), num_ctrl)
        """
        x_shape = (self.num_x, self.num_ctrl)
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
                                       np,ones(self.num_ctrl))

            if amplitudes.shape[1] != self.num_ctrl:
                raise Exception("amplitudes' shape must be (Nt, num_ctrl)")

            if times == target_time:
                target = amplitudes
            else:
                target = np.zeros((len(target_time),self.num_ctrl))
                for i in range(self.num_ctrl):
                    tck = interpolate.splrep(times, amplitudes[:,i], s=0)
                    target[:,i] = interpolate.splev(target_time, tck, der=0)

        elif targetfunc is not None:
            try:
                target = targetfunc(target_time, self.num_ctrl)
                if target.shape != (len(target_time), self.num_ctrl):
                    raise Exception()
            except e:
                raise Exception(targetfunc.__name__ +" call failed:\n"
                                "expected signature:\n"
                                "targetfunc(times, num_ctrl) -> amplitudes "
                                "shape = (Nt, num_ctrl)")
        else:
            # no target, set to zeros
            target = np.zeros((len(target_time), self.num_ctrl))

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
    def __init__(self, num_x):
        super().__init__()
        self.name = "Fourrier"
        self.num_x = num_x

    def __call__(self, x):
        u = np.zeros((self.t_step, self.num_ctrl))
        s = np.zeros((self.t_step*2-2, self.num_ctrl))
        s[1:self.num_x+1,:] = x
        for j in range(self.num_ctrl):
            u[:,j] = -fft(s[:,j]).imag[:self.t_step]
        return u

    def gradient_u2x(self, gradient):
        x = np.zeros((self.num_x, self.num_ctrl))
        s = np.zeros((self.t_step*2-2, self.num_ctrl))
        s[:self.t_step,:] = gradient
        for j in range(self.num_ctrl):
            x[:,j] = -fft(s[:,j]).imag[1:self.num_x+1]
        return x

    def init_timeslots_old(self, times=None, tau=None, T=1, t_step=None,
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
        self.times = np.linspace(0, T, self.t_step+1)
        return (self.num_x, self.num_ctrl), self.times

    def set_times(self, times, num_ctrl=1):
        self.num_ctrl = num_ctrl
        if not np.allclose(np.diff(times), times[1]-times[0]):
            raise Exception("Times must be equaly distributed")
        self.num_ctrl = num_ctrl
        self.times = times
        self.xtimes = times
        self.t_step = len(times)-1
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
                    raise Exception("fourrier: wrong bounds shape")
        else:
            self.x_max = np.array([[None]*self.num_ctrl]*self.num_x)

        if self.x_min is not None:
            if isinstance(self.x_min, list):
                self.x_min = np.array(self.x_min)
            if isinstance(self.x_min, (int, float)):
                self.x_min = self.x_min*np.ones((self.num_x,self.num_ctrl))
            elif isinstance(self.x_min, np.ndarray):
                if self.x_min.shape != (self.num_x,self.num_ctrl):
                    raise Exception("fourrier: wrong bounds shape")
        else:
            self.x_min = np.array([[None]*self.num_ctrl]*self.num_x)

    def plotPulse(self, x):
        u = self(x)
        t, dt = (self.times[:-1]+self.times[1:])*0.5, np.diff(self.times)
        for i in range(self.num_ctrl):
            plt.bar(np.arange(self.num_x), x[:,i], 0.7)
            plt.title("Fourier series")
            plt.show()
            plt.bar(t, u[:,i], dt)
            plt.title("Amplidutes")
            plt.show()

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
                       num_x=None, num_ctrl=1):
        """
        Times/tau correspond to the timeslot before the interpolation.
        """

        self.num_ctrl = num_ctrl

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
        return (self.num_x, self.num_ctrl), self.times

    def set_times(self, times, num_ctrl=1):
        """
        Times/tau correspond to the timeslot before the interpolation.
        """
        self.num_ctrl = num_ctrl
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
            omega = [self.omega] * self.num_ctrl
        else:
            omega = self.omega
        if isinstance(self.boundary[0], (int, float)):
            start = [self.boundary[0]] * self.num_ctrl
        else:
            start = self.boundary[0]
        if isinstance(self.boundary[1], (int, float)):
            end = [self.boundary[1]] * self.num_ctrl
        else:
            end = self.boundary[1]

        Dxt = (self.xtimes[1]-self.xtimes[0])*0.25
        self.T = np.zeros((len(self.times)-1, self.num_x, self.num_ctrl))
        self.cte = np.zeros((len(self.times)-1, self.num_ctrl))
        time = (self.times[:-1] + self.times[1:]) * 0.5
        xtime = (self.xtimes[:-1] + self.xtimes[1:]) * 0.5
        for i in range(self.num_ctrl):
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

    def set_times(self, times, num_ctrl=1):
        """
        Times/tau correspond to the timeslot before the interpolation.
        """
        self.num_ctrl = num_ctrl
        if not np.allclose(np.diff(times), times[1]-times[0]):
            raise Exception("Times must be equaly distributed")
        elif times[0] != 0:
            raise Exception("Times must start at 0")
        self.num_x = len(times)-1
        T = times[-1]
        dt = T/self.num_x/self.N
        self.x_times = np.linspace(0, T, self.num_x+1)
        if self.bound_type[0] == "x":
            extra_t = self.bound_type[1] * self.N
        elif self.bound_type[0] == "n":
            extra_t = self.bound_type[1]
        else:
            if isinstance(self.omega, (int, float)):
                omega = [self.omega] * self.num_ctrl
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
        return (self.num_x, self.num_ctrl), self.times

    def init_timeslots_old(self, times=None, tau=None, T=1, t_step=None,
                       num_x=None, num_ctrl=1):
        """
        Times/tau correspond to the timeslot before the interpolation.
        """

        self.num_ctrl = num_ctrl

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
        self.x_times = np.linspace(0, T, self.num_x+1)
        if self.bound_type[0] == "x":
            extra_t = self.bound_type[1] * self.N
        elif self.bound_type[0] == "n":
            extra_t = self.bound_type[1]
        else:
            if isinstance(self.omega, (int, float)):
                omega = [self.omega] * self.num_ctrl
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
