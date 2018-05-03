



















class filter:
    def __init__(self):
        self.

    def __call__(self, x):
        pass

    def reverse(self, gradient):
        pass

    def init_timeslots(self, times, tau, T, t_step, num_x, num_ctrl):
        pass



class pass_througth(filter):
    def __init__(self):
        pass

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
            #tau = np.diff(times)
        elif tau is not None:
            t_step = len(tau)
            T = np.sum(tau)
            time = np.cumsum(np.insert(tau,0,0))
        else:
            #tau = np.ones(t_step, dtype='f') * T/t_step
            time = np.linspace(0,T,t_step+1)
        #self.time = np.zeros(self._num_tslots+1, dtype=float)
        #for t in range(self._num_tslots):
        #    self.time[t+1] = self.time[t] + self._tau[t]

        return np.ndarray([t_step, num_ctrl]), time


class fourrier(filter):
    """
        Pulse described as fourrier modes
        u(t) = a(j)*sin(pi*j*t/T)
        Takes a linearly spaced time and num_x <= len(times)
        When num_x < t_step, only the num_x lower mode are kept.
        (low-pass filter)
    """
    def __init__(self):
        pass

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

        return np.array([self.t_step, self.num_ctrl]), \
            np.linspace(0, T, self.t_step+1)


class spline():
    """
    Should this be a special case of convolution?
    Zero padding first and last?
    """
    def __init__(self, overSampleRate=5):
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

        return np.array([self.num_x, self.num_ctrl]), \
            np.linspace(0, T, self.t_step+1)



class convolution(filter):
    """

    """
    def __init__(self, pattern, spread=1):
        pass

    def __call__(self, x):
        pass

    def reverse(self, gradient):
        pass

    def init_timeslots(self, times, tau, T, t_step, num_x, num_ctrl):
        pass
