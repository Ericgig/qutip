
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
        #self.x_times = np.linspace(0, T, self.num_x+1)
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




class PulseGenCrab(transfer_functions):
    """

    """

    def __init__(self, guess_pulse=None, ramping_pulse=None):
        super().__init__()
        self.guess_pulse = None
        self.ramping_pulse = None
        self.guess_pulse_func = None
        self.apply_bound = False
        self.guess_pulse_action = 'MODULATE'
        self.name = "PulseGenCrab"

    def __call__(self, x):
        return None

    def gradient_u2x(self, gradient):
        return None

    def set_times(self, times, num_ctrl=1):
        self.num_ctrl = num_ctrl
        self.num_x = len(times)-1
        T = times[-1]
        self.times = times
        return (self.num_x, self.num_ctrl), self.times

    def _apply_ramping_pulse(self, pulse):
        return pulse*self.ramping_pulse

    def init_guess_pulse(self):
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

    def _init_bounds(self):
        add_guess_pulse_scale = False
        if self.x_min is None and self.x_max is None:
            # no bounds to apply
            self._bound_scale_cond = None
        elif self.x_min is None:
            # only upper bound
            if self.x_max > 0:
                self._bound_mean = 0.0
                self._bound_scale = self.x_max
            else:
                add_guess_pulse_scale = True
                self._bound_scale = self.scaling*self.num_coeffs + \
                            self.get_guess_pulse_scale()
                self._bound_mean = -abs(self._bound_scale) + self.x_max
            self._bound_scale_cond = "_BSC_GT_MEAN"

        elif self.x_max is None:
            # only lower bound
            if self.x_min < 0:
                self._bound_mean = 0.0
                self._bound_scale = abs(self.x_min)
            else:
                self._bound_scale = self.scaling*self.num_coeffs + \
                            self.get_guess_pulse_scale()
                self._bound_mean = abs(self._bound_scale) + self.x_min
            self._bound_scale_cond = "_BSC_LT_MEAN"

        else:
            # lower and upper bounds
            self._bound_mean = 0.5*(self.x_max + self.x_min)
            self._bound_scale = 0.5*(self.x_max - self.x_min)
            self._bound_scale_cond = "_BSC_ALL"

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
            pulse = np.tanh(pulse)*self._bound_scale + self._bound_mean
            return pulse
        elif self._bound_scale_cond == "_BSC_GT_MEAN":
            scale_where = pulse > self._bound_mean
            pulse[scale_where] = (np.tanh(pulse[scale_where])*self._bound_scale
                                        + self._bound_mean)
            return pulse
        elif self._bound_scale_cond == "_BSC_LT_MEAN":
            scale_where = pulse < self._bound_mean
            pulse[scale_where] = (np.tanh(pulse[scale_where])*self._bound_scale
                                        + self._bound_mean)
            return pulse
        else:
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
    def __init__(self, guess_pulse=None, ramping_pulse=None):
        super().__init__()
        self.guess_pulse = None
        self.ramping_pulse = None
        self.guess_pulse_func = None
        self.apply_bound = False
        self.guess_pulse_action = 'MODULATE'
        self.opt_freq = False
        self.name = "PulseGenCrabFourier"

    def init_freqs(self):
        """
        Generate the frequencies
        These are the Fourier harmonics with a uniformly distributed
        random offset
        """
        self.freqs = np.empty(self.num_coeffs)
        ff = 2*np.pi / self.pulse_time
        for i in range(self.num_coeffs):
            self.freqs[i] = ff*(i + 1)

        if self.randomize_freqs:
            self.freqs += np.random.random(self.num_coeffs) - 0.5

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

        for i in range(self.num_ctrls):
            coeffs = x[:,i].reshape(self.coeff_shape)
            if self.opt_freq:
                freqs = coeffs[:,2]
            else:
                freqs = self.freqs

        pulse = np.zeros(self.num_tslots)
        for i in range(self.num_coeffs):
            phase = freqs[i]*self.times
            pulse += coeffs[i, 0]*np.sin(phase) + coeffs[i, 1]*np.cos(phase)

        if self.guess_pulse_func:
            pulse = self.guess_pulse_func(pulse)
        if self.ramping_pulse is not None:
            pulse = self._apply_ramping_pulse(pulse)
        if self.apply_bound:
            pulse = self._apply_bounds(pulse)
        return pulse





class PulseGen(object):
    """
    Pulse generator
    Base class for all Pulse generators
    The object can optionally be instantiated with a Dynamics object,
    in which case the timeslots and amplitude scaling and offset
    are copied from that.
    Otherwise the class can be used independently by setting:
    tau (array of timeslot durations)
    or
    num_tslots and pulse_time for equally spaced timeslots

    Attributes
    ----------
    num_tslots : integer
        Number of timeslots, aka timeslices
        (copied from Dynamics if given)

    pulse_time : float
        total duration of the pulse
        (copied from Dynamics.evo_time if given)

    scaling : float
        linear scaling applied to the pulse
        (copied from Dynamics.initial_ctrl_scaling if given)

    offset : float
        linear offset applied to the pulse
        (copied from Dynamics.initial_ctrl_offset if given)

    tau : array[num_tslots] of float
        Duration of each timeslot
        (copied from Dynamics if given)

    lbound : float
        Lower boundary for the pulse amplitudes
        Note that the scaling and offset attributes can be used to fully
        bound the pulse for all generators except some of the random ones
        This bound (if set) may result in additional shifting / scaling
        Default is -Inf

    ubound : float
        Upper boundary for the pulse amplitudes
        Note that the scaling and offset attributes can be used to fully
        bound the pulse for all generators except some of the random ones
        This bound (if set) may result in additional shifting / scaling
        Default is Inf

    periodic : boolean
        True if the pulse generator produces periodic pulses

    random : boolean
        True if the pulse generator produces random pulses

    log_level : integer
        level of messaging output from the logger.
        Options are attributes of qutip.logging_utils,
        in decreasing levels of messaging, are:
        DEBUG_INTENSE, DEBUG_VERBOSE, DEBUG, INFO, WARN, ERROR, CRITICAL
        Anything WARN or above is effectively 'quiet' execution,
        assuming everything runs as expected.
        The default NOTSET implies that the level will be taken from
        the QuTiP settings file, which by default is WARN
    """
    def __init__(self, dyn=None, params=None):
        self.parent = dyn
        self.params = params
        self.reset()

    def reset(self):
        """
        reset attributes to default values
        """
        if isinstance(self.parent, dynamics.Dynamics):
            dyn = self.parent
            self.num_tslots = dyn.num_tslots
            self.pulse_time = dyn.evo_time
            self.scaling = dyn.initial_ctrl_scaling
            self.offset = dyn.initial_ctrl_offset
            self.tau = dyn.tau
            self.log_level = dyn.log_level
        else:
            self.num_tslots = 100
            self.pulse_time = 1.0
            self.scaling = 1.0
            self.tau = None
            self.offset = 0.0

        self._uses_time = False
        self.time = None
        self._pulse_initialised = False
        self.periodic = False
        self.random = False
        self.lbound = None
        self.ubound = None
        self.ramping_pulse = None

        self.apply_params()

    def apply_params(self, params=None):
        """
        Set object attributes based on the dictionary (if any) passed in the
        instantiation, or passed as a parameter
        This is called during the instantiation automatically.
        The key value pairs are the attribute name and value
        """
        if not params:
            params = self.params

        if isinstance(params, dict):
            self.params = params
            for key in params:
                setattr(self, key, params[key])

    @property
    def log_level(self):
        return logger.level

    @log_level.setter
    def log_level(self, lvl):
        """
        Set the log_level attribute and set the level of the logger
        that is call logger.setLevel(lvl)
        """
        logger.setLevel(lvl)

    def gen_pulse(self):
        """
        returns the pulse as an array of vales for each timeslot
        Must be implemented by subclass
        """
        # must be implemented by subclass
        raise errors.UsageError(
            "No method defined for generating a pulse. "
            " Suspect base class was used where sub class should have been")

    def init_pulse(self):
        """
        Initialise the pulse parameters
        """
        if self.tau is None:
            self.tau = np.ones(self.num_tslots, dtype='f') * \
                self.pulse_time/self.num_tslots

        if self._uses_time:
            self.time = np.zeros(self.num_tslots, dtype=float)
            for k in range(self.num_tslots-1):
                self.time[k+1] = self.time[k] + self.tau[k]

        self._pulse_initialised = True

        if not self.lbound is None:
            if np.isinf(self.lbound):
                self.lbound = None
        if not self.ubound is None:
            if np.isinf(self.ubound):
                self.ubound = None

        if not self.ubound is None and not self.lbound is None:
            if self.ubound < self.lbound:
                raise ValueError("ubound cannot be less the lbound")

    def _apply_bounds_and_offset(self, pulse):
        """
        Ensure that the randomly generated pulse fits within the bounds
        (after applying the offset)
        Assumes that pulses passed are centered around zero (on average)
        """
        if self.lbound is None and self.ubound is None:
            return pulse + self.offset

        max_amp = max(pulse)
        min_amp = min(pulse)
        if ((self.ubound is None or max_amp + self.offset <= self.ubound) and
            (self.lbound is None or min_amp + self.offset >= self.lbound)):
            return pulse + self.offset

        # Some shifting / scaling is required.
        if self.ubound is None or self.lbound is None:
            # One of the bounds is inf, so just shift the pulse
            if self.lbound is None:
                # max_amp + offset must exceed the ubound
                return pulse + self.ubound - max_amp
            else:
                # min_amp + offset must exceed the lbound
                return pulse + self.lbound - min_amp
        else:
            bound_range = self.ubound - self.lbound
            amp_range = max_amp - min_amp
            if max_amp - min_amp > bound_range:
                # pulse range is too high, it must be scaled
                pulse = pulse * bound_range / amp_range

            # otherwise the pulse should fit anyway
            return pulse + self.lbound - min(pulse)

    def _apply_ramping_pulse(self, pulse, ramping_pulse=None):
        if ramping_pulse is None:
            ramping_pulse = self.ramping_pulse
        if ramping_pulse is not None:
            pulse = pulse*ramping_pulse

        return pulse


class PulseGenCrab(PulseGen):
    """
    Base class for all CRAB pulse generators
    Note these are more involved in the optimisation process as they are
    used to produce piecewise control amplitudes each time new optimisation
    parameters are tried

    Attributes
    ----------
    num_coeffs : integer
        Number of coefficients used for each basis function

    num_basis_funcs : integer
        Number of basis functions
        In this case set at 2 and should not be changed

    coeffs : float array[num_coeffs, num_basis_funcs]
        The basis coefficient values

    randomize_coeffs : bool
        If True (default) then the coefficients are set to some random values
        when initialised, otherwise they will all be equal to self.scaling
    """
    def __init__(self, dyn=None, num_coeffs=None, params=None):
        self.parent = dyn
        self.num_coeffs = num_coeffs
        self.params = params
        self.reset()

    def reset(self):
        """
        reset attributes to default values
        """
        PulseGen.reset(self)
        self.NUM_COEFFS_WARN_LVL = 20
        self.DEF_NUM_COEFFS = 4
        self._BSC_ALL = 1
        self._BSC_GT_MEAN = 2
        self._BSC_LT_MEAN = 3

        self._uses_time = True
        self.time = None
        self.num_basis_funcs = 2
        self.num_optim_vars = 0
        self.coeffs = None
        self.randomize_coeffs = True
        self._num_coeffs_estimated = False
        self.guess_pulse_action = 'MODULATE'
        self.guess_pulse = None
        self.guess_pulse_func = None
        self.apply_params()

    def init_pulse(self, num_coeffs=None):
        """
        Set the initial freq and coefficient values
        """
        PulseGen.init_pulse(self)
        self.init_coeffs(num_coeffs=num_coeffs)

        if self.guess_pulse is not None:
            self.init_guess_pulse()
        self._init_bounds()

        if self.log_level <= logging.DEBUG and not self._num_coeffs_estimated:
            logger.debug(
                    "CRAB pulse initialised with {} coefficients per basis "
                    "function, which means a total of {} "
                    "optimisation variables for this pulse".format(
                            self.num_coeffs, self.num_optim_vars))

    def init_coeffs(self, num_coeffs=None):
        """
        Generate the initial ceofficent values.

        Parameters
        ----------
        num_coeffs : integer
            Number of coefficients used for each basis function
            If given this overides the default and sets the attribute
            of the same name.
        """
        if num_coeffs:
            self.num_coeffs = num_coeffs

        self._num_coeffs_estimated = False
        if not self.num_coeffs:
            if isinstance(self.parent, dynamics.Dynamics):
                dim = self.parent.get_drift_dim()
                self.num_coeffs = self.estimate_num_coeffs(dim)
                self._num_coeffs_estimated = True
            else:
                self.num_coeffs = self.DEF_NUM_COEFFS
        self.num_optim_vars = self.num_coeffs*self.num_basis_funcs

        if self._num_coeffs_estimated:
            if self.log_level <= logging.INFO:
                logger.info(
                    "The number of CRAB coefficients per basis function "
                    "has been estimated as {}, which means a total of {} "
                    "optimisation variables for this pulse. Based on the "
                    "dimension ({}) of the system".format(
                            self.num_coeffs, self.num_optim_vars, dim))
            # Issue warning if beyond the recommended level
            if self.log_level <= logging.WARN:
                if self.num_coeffs > self.NUM_COEFFS_WARN_LVL:
                    logger.warn(
                        "The estimated number of coefficients {} exceeds "
                        "the amount ({}) recommended for efficient "
                        "optimisation. You can set this level explicitly "
                        "to suppress this message.".format(
                            self.num_coeffs, self.NUM_COEFFS_WARN_LVL))

        if self.randomize_coeffs:
            r = np.random.random([self.num_coeffs, self.num_basis_funcs])
            self.coeffs = (2*r - 1.0) * self.scaling
        else:
            self.coeffs = np.ones([self.num_coeffs,
                                   self.num_basis_funcs])*self.scaling

    def estimate_num_coeffs(self, dim):
        """
        Estimate the number coefficients based on the dimensionality of the
        system.
        Returns
        -------
        num_coeffs : int
            estimated number of coefficients
        """
        num_coeffs = max(2, dim - 1)
        return num_coeffs

    def get_optim_var_vals(self):
        """
        Get the parameter values to be optimised
        Returns
        -------
        list (or 1d array) of floats
        """
        return self.coeffs.ravel().tolist()

    def set_optim_var_vals(self, param_vals):
        """
        Set the values of the any of the pulse generation parameters
        based on new values from the optimisation method
        Typically this will be the basis coefficients
        """
        # Type and size checking avoided here as this is in the
        # main optmisation call sequence
        self.set_coeffs(param_vals)

    def set_coeffs(self, param_vals):
        self.coeffs = param_vals.reshape(
                    [self.num_coeffs, self.num_basis_funcs])

    def init_guess_pulse(self):

        self.guess_pulse_func = None
        if not self.guess_pulse_action:
            logger.WARN("No guess pulse action given, hence ignored.")
        elif self.guess_pulse_action.upper() == 'MODULATE':
            self.guess_pulse_func = self.guess_pulse_modulate
        elif self.guess_pulse_action.upper() == 'ADD':
            self.guess_pulse_func = self.guess_pulse_add
        else:
            logger.WARN("No option for guess pulse action '{}' "
                        ", hence ignored.".format(self.guess_pulse_action))

    def guess_pulse_add(self, pulse):
        pulse = pulse + self.guess_pulse
        return pulse

    def guess_pulse_modulate(self, pulse):
        pulse = (1.0 + pulse)*self.guess_pulse
        return pulse

    def _init_bounds(self):
        add_guess_pulse_scale = False
        if self.lbound is None and self.ubound is None:
            # no bounds to apply
            self._bound_scale_cond = None
        elif self.lbound is None:
            # only upper bound
            if self.ubound > 0:
                self._bound_mean = 0.0
                self._bound_scale = self.ubound
            else:
                add_guess_pulse_scale = True
                self._bound_scale = self.scaling*self.num_coeffs + \
                            self.get_guess_pulse_scale()
                self._bound_mean = -abs(self._bound_scale) + self.ubound
            self._bound_scale_cond = self._BSC_GT_MEAN

        elif self.ubound is None:
            # only lower bound
            if self.lbound < 0:
                self._bound_mean = 0.0
                self._bound_scale = abs(self.lbound)
            else:
                self._bound_scale = self.scaling*self.num_coeffs + \
                            self.get_guess_pulse_scale()
                self._bound_mean = abs(self._bound_scale) + self.lbound
            self._bound_scale_cond = self._BSC_LT_MEAN

        else:
            # lower and upper bounds
            self._bound_mean = 0.5*(self.ubound + self.lbound)
            self._bound_scale = 0.5*(self.ubound - self.lbound)
            self._bound_scale_cond = self._BSC_ALL

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
        if self._bound_scale_cond == self._BSC_ALL:
            pulse = np.tanh(pulse)*self._bound_scale + self._bound_mean
            return pulse
        elif self._bound_scale_cond == self._BSC_GT_MEAN:
            scale_where = pulse > self._bound_mean
            pulse[scale_where] = (np.tanh(pulse[scale_where])*self._bound_scale
                                        + self._bound_mean)
            return pulse
        elif self._bound_scale_cond == self._BSC_LT_MEAN:
            scale_where = pulse < self._bound_mean
            pulse[scale_where] = (np.tanh(pulse[scale_where])*self._bound_scale
                                        + self._bound_mean)
            return pulse
        else:
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

    def reset(self):
        """
        reset attributes to default values
        """
        PulseGenCrab.reset(self)
        self.freqs = None
        self.randomize_freqs = True

    def init_pulse(self, num_coeffs=None):
        """
        Set the initial freq and coefficient values
        """
        PulseGenCrab.init_pulse(self)

        self.init_freqs()

    def init_freqs(self):
        """
        Generate the frequencies
        These are the Fourier harmonics with a uniformly distributed
        random offset
        """
        self.freqs = np.empty(self.num_coeffs)
        ff = 2*np.pi / self.pulse_time
        for i in range(self.num_coeffs):
            self.freqs[i] = ff*(i + 1)

        if self.randomize_freqs:
            self.freqs += np.random.random(self.num_coeffs) - 0.5

    def gen_pulse(self, coeffs=None):
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
        if coeffs:
            self.coeffs = coeffs

        if not self._pulse_initialised:
            self.init_pulse()

        pulse = np.zeros(self.num_tslots)

        for i in range(self.num_coeffs):
            phase = self.freqs[i]*self.time
    #            basis1comp = self.coeffs[i, 0]*np.sin(phase)
    #            basis2comp = self.coeffs[i, 1]*np.cos(phase)
    #            pulse += basis1comp + basis2comp
            pulse += self.coeffs[i, 0]*np.sin(phase) + \
                        self.coeffs[i, 1]*np.cos(phase)

        if self.guess_pulse_func:
            pulse = self.guess_pulse_func(pulse)
        if self.ramping_pulse is not None:
            pulse = self._apply_ramping_pulse(pulse)

        return self._apply_bounds(pulse)
