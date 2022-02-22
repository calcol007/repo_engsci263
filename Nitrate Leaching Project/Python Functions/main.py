import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from scipy.integrate import solve_ivp
from scipy import optimize
from numpy.linalg import solve
from math import exp
import warnings



# functions related to first best-fit model
##################################################################################################################################################################

def readInData():
    '''
    This function reads in the data of the number of cows and nitrate levels over the years.
        
        Parameters:
        -----------
        None
        Returns:
        --------
        ts: array-like
            Years over which the cows were counted.
        stock: array-like
            Number of cows over the years.
        ts1: array-like
            Years over which the nitrate levels were counted.
        lvls: array-like
            Nitrate levels over the years.
    '''
    ts, stock = np.genfromtxt("nl_cows.txt", delimiter = ",", skip_header = 1, unpack = True)
    ts1, lvls = np.genfromtxt("nl_n.csv", delimiter = ",", skip_header = 1, unpack = True)

    return ts, stock, ts1, lvls



def fittingModel(t, nstock):
    '''
        This function takes two arrays for t and nstock and guesses parameters from an equation into it.
        
        Parameters:
        -----------
        t : array-like
            independent variable array.
        nstock : array-like
            dependent variable array.
        
        Returns:
        --------
        params : array-like
            array for the parameter guesses.
        covar : array-like
            array for the covariance guesses.
    '''
    
    return optimize.curve_fit(testFunction, t, nstock)



def testFunction(t, a, b, c, d):
    '''
        This function takes two arrays for t and nstock and guesses parameters from an equation into it.
        
        Parameters:
        -----------
        None
        
        Returns:
        --------
        array-like
            array of stock numbers over time, t.
    '''
    return a*t**3 + b*t**2 + c*t + d



def pressure(t, P0, b, tMAR, PMAR):
    ''' 
    Evaluates pressure values at given times, t.
        
        Parameters:
        -----------
        t : array-like
            Array of times at which to evaluate pressure.
        P0 : float
            Unknown initial pressure within the aquifer.
        b : float
            Unknown parameter.
        tMAR : float
            Year when MAR initiative is implemented.
        
        PMAR : float
            Change in pressure introduce from MAR initiative. 
        
        Returns:
        --------
        array-like
            Array of pressure values at given times, t.
        
        Notes
        -----
        depends on whether MAR initiative implemented or not
    '''
    if t <= tMAR:
        return P0 * exp(-2*b*t)  # analytical solution if MAR initiative not yet implemented
    return (P0-PMAR/4)*exp(-2*b*t) + PMAR/4  # analytical solution if MAR initiative has been implemented



def dCdt(t, c, pars, tau, tC, tMAR, b1, alpha, deltaPa, deltaPMAR, PSurf):
    '''
        This function calculates the delayed pressure driven surface infiltration.
        
        Parameters:
        -----------
        t : array like
            times.
        c : array like
            concentrations.
        pars : array like
            guesses for the parameters.
        tau : double
            time offset (the delay between the action and its effect).
        tC : double
            time where active carbon introduced.
        tMAR : double
            time where MAR is operating.
        b1 : double
            coefficient for infiltration.
        alpha : double 
            coefficient for effect on active carbon sink on infiltration.
        deltaPa : double
            pressure drop across aquifer.
        deltaPMAR : double
            pressure injected by MAR.
        PSurf : double
            pressure from the surface into the aquifer.
        
        Returns:
        --------
        dCdt : double
            the change in concentration at a finite point.
    '''
    # setting up the guesses as seperate parameters
    b, bc, P0, M0 = pars

    # greating the model for cow stock
    stock = readInData()[1]
    params = fittingModel(t, stock)[0]
    
    # getting the number of cows at this time
    nStock = testFunction(t, *params)

    # infiltration depending on the carbon sink
    if t < tC + tau: 
        bPrime = b1
    else: 
        bPrime = alpha * b1

    # pressure drop of aquifer depending on MAR operating
    if t < tMAR + tau: 
        deltaPaPrime = deltaPa
    else: 
        deltaPaPrime = deltaPa + deltaPMAR

    # depending on the time, the model for the cows will not be accurate, so it is better to set a fixed value of the stock until the model becomes a better approximation of the stock count.

    if t < 1991 + tau: # before this time there was no data on the cows so use the first number in the array of data received (37772)
        return (-37772 * bPrime * (pressure(t, P0, b, tMAR, deltaPMAR) - PSurf) + bc * (pressure(t, P0, b, tMAR, deltaPMAR) - deltaPaPrime/2) * c) / M0
    else:
        return (-nStock * bPrime * (pressure(t, P0, b, tMAR, deltaPMAR) - PSurf) + bc * (pressure(t, P0, b, tMAR, deltaPMAR) - deltaPaPrime/2) * c) / M0



def unitTest():

    # this function will perform unit tests with the function dCdt to ensure the function is working properly

    # when manually calculated results 0
    C = dCdt(0, 0, [0, 0, 0, 1], 0, 0, 0, 0, 1, 1, 1, 1)
    assert(C == 0)

    # when manually calculated results -75544
    C = dCdt(0, 0, [0, -1, -1, 1], 0, 0, 0, -1, 1, 1, 1, 1)
    assert(C == -75544)

    # when manually calculated results 1
    C = dCdt(0, 1, [0, 1, 1, 1], 0, 0, 0, 0, 1, 0, 0, 1)
    assert(C == 1)

    print("Passed all unit tests.")



def improved_euler_step(f, tk, yk, h, pars):
    '''
        This function calculates one step of the improved euler method.
        
        Parameters:
        -----------
        f : function
            derivative function that is to be solved.
        tk : double
            time to use as input into euler step.
        yk : double
            solution at time, tk, to use as input into euler step.
        h : double
            step-size.
        pars : double
            parameters to input into derivative function, f. 
        
        Returns:
        --------
        double
            next solution, y, at time, tk + h
    '''
    # calculating predictor gradient
    k1 = f(tk, yk, *pars)

    # perform Euler step
    y1 = yk + h * k1

    # calculating corrector gradient
    k2 = f(tk + h, y1, *pars)

    # returning corrected y,n+1
    return yk + h * (k1 + k2) / 2



def improved_euler_solve(f, t0, y0, t1, h, pars=[]):
    '''
        This function implements the improved_euler_step to numerically solve an ODE.
        
        Parameters:
        -----------
        f : function
            derivative function that is to be solved.
        t0 : double
            initial time.
        y0 : double
            initial solution at t0.
        h : double
            step-size.
        pars : double
            parameters to input into derivative function, f. 
        
        Returns:
        --------
        ts : array-like
            array of times.
        ys : array-like
            array of solutions at times, ts.
    '''
    # initialising variables
    nt = int(np.ceil((t1-t0)/h))  # compute number of improved Euler steps to take
    ts = t0+np.arange(nt+1)*h    # x array
    ys = 0.*ts  # array to store solution
    ys[0] = y0  # set initial value

    # loop to perform improved Euler's method 
    for i in range(nt):
        ys[i + 1] = improved_euler_step(f, ts[i], ys[i], h, pars)

    # returning arrays of the solution
    return ts, ys



# functions related to benchmarking
##################################################################################################################################################################

def simplified_ode_model(t, c, h, a, b, g, Psurf, Pa):
    ''' 
    Return the derivative dc/dt at time, t, for given parameters.
        Parameters:
        -----------
        t : float
            Independent variable - time in years.
        c : float
            Dependent variable - concentration of nitrate in the groundwater.
        a : float
            Pressure strength parameter.
        g : float
            (-nstock*b1/Mo) where nstock = the number of cattle stock, b1 = nitrate leaching strength parameter, Mo = initial mass of nitrate.
        h : float
            (bc/Mo) where bc =  high-pressure boundary strength parameter.
        b : float
            Recharge strength parameter.
        Psurf : float
            Surface pressure of aquifer.
        Pa : float
            Pressure across aquifer.
        Returns:
        --------
        dcdt : float
            Derivative of dependent variable with respect to independent variable.
    '''
    # derivative function to be used for numerically solving ODE
    dcdt = (-g*(a*(np.exp(-2*b*t)) - Psurf)) + (h*(a*(np.exp(-2*b*t)) - Pa/2))*c
    return dcdt



# functions related to interpolation using cubic splines
##################################################################################################################################################################

def spline_coefficient_matrix(xi):    
    ''' 
    Evaluates a polynomial.
        
        Parameters
        ----------
        xi : array-like
             Array of subinterval boundaries 
        Returns
        -------
        A : array-like
            Coefficient matrix 
        Notes
        -----
        Subinterval boundary points, xi, assumed to be in ascending order.
    '''
    # get number of subinterval boundary points
    N = len(xi)

    # initialise matrix of zeros
    A = np.zeros((4*(N-1), 4*(N-1)))

    # populating matrix with respective values
    for i in np.arange(0, (2*N-3), 2):
        A[i, 0 + 2*i] = 1
        
    for i in np.arange(1, (2*N-2), 2):
        A[i, 0 + 2*(i-1)] = 1
        A[i, 1 + 2*(i-1)] = (xi[int(0.5*(i-1) + 1)] - xi[int(0.5*(i-1))])
        A[i, 2 + 2*(i-1)] = (xi[int(0.5*(i-1) + 1)] - xi[int(0.5*(i-1))])**2
        A[i, 3 + 2*(i-1)] = (xi[int(0.5*(i-1) + 1)] - xi[int(0.5*(i-1))])**3

    # do not include derivative equations if only two boundaries
    if N != 2:
        for i in np.arange(2*N-2, (2*N-2) + 2*(N-3) + 1, 2):
            v = int(0.5*(i - 2*(N-1)))
            A[i, 1 + 2*(i - 2*(N-1))] = 1
            A[i, 2 + 2*(i - 2*(N-1))] = 2*(xi[v+1] - xi[v])
            A[i, 3 + 2*(i - 2*(N-1))] = 3*((xi[v+1] - xi[v])**2)
            A[i, 5 + 2*(i - 2*(N-1))] = -1
            
        for i in np.arange(2*N-1, (2*N-1) + 2*(N-3) + 1, 2):
            u = int(0.5*(i - 1 - 2*(N-1)))
            A[i, 2 + 2*(i - 1 - 2*(N-1))] = 2
            A[i, 3 + 2*(i - 1 - 2*(N-1))] = 6*(xi[u+1] - xi[u])
            A[i, 6 + 2*(i - 1 - 2*(N-1))] = -2
        
    A[4*(N-1)-2, 2] = 2
    A[4*(N-1)-1, 4*(N-1)-2] = 2
    A[4*(N-1)-1, 4*(N-1)-1] = 6*(xi[N-1] - xi[N-2])
        
    return A



def spline_rhs(xi, yi):
    ''' 
    Evaluates a polynomial.
        
        Parameters
        ----------
        xi : array-like
             Array of subinterval boundaries
        
        yi : array-like
             Array of dependent variable values at respective subinterval boundaries
        Returns
        -------
        rhs : array-like
              Vector of right-hand-side matrix equation
    
        Notes
        -----
        Subinterval boundary points, xi, assumed to be in ascending order.
    '''
    # get number of evaluated points
    N = len(yi)

    # initialise rhs vector
    rhs = np.zeros(4*(N-1))

    # populating the rhs vector
    rhs[:2*(N-1):2] = yi[:N-1]
    rhs[1:2*(N-1):2] = yi[1:]

    return rhs
    


def spline_interpolate(xj, xi, ak):
    ''' 
    Return a set of interpolated values.
        
        Parameters
        ----------
        xj : array-like
             Array of points for interpolation
        xi : array-like
             Array of subinterval boundaries 
        ak : array-like
             Vector of spline polynomial coefficients     
        
        Returns
        -------
        yj : array-like
             Array of interpolated values at the given xj (interpolating) points
    
        Notes
        -----
        Interpolation points, xj, assumed to be in ascending order.
    '''
    # get number of subinterval boundaries and interpolation points
    N = len(xi)
    M = len(xj)
    # initialise j counter to 0
    j = 0
    # initialise numpy array for yj
    yj = np.zeros(M)

    # loop over intervals
    for i in range(N-1):
        # get spline coefficients 
        coef = ak[4*i: 4*(i+1)]
        # while loop to compute dot product of correct values and coefficients
        while j < M and xj[j] >= xi[i] and xj[j] < xi[i+1]:
            yj[j] = np.dot(coef, [1, (xj[j] - xi[i]), (xj[j] - xi[i])**2, (xj[j] - xi[i])**3])
            j += 1
            
    return yj



def polyval(a,xi):
    ''' 
    Evaluates a polynomial.
        
        Parameters
        ----------
        a : np.array
            Vector of polynomial coefficients.
        xi : np.array
            Points at which to evaluate polynomial.
        
        Returns
        -------
        yi : np.array
            Evaluated polynomial.
            
        Notes
        -----
        Polynomial coefficients assumed to be increasing order, i.e.,
        
        yi = Sum_(i=0)^len(a) a[i]*xi**i
        
    '''
    # initialise output at correct length
    yi = 0.*xi
    
    # loop over polynomial coefficients
    for i,ai in enumerate(a):
        yi = yi + ai*xi**i
        
    return yi



# functions related to the improved best-fit model
##################################################################################################################################################################

def fittingModel2(t, nstock):
    '''
        This function takes two arrays for t and nstock and guesses parameters from an equation into it.
        
        Parameters:
        -----------
        t : array-like
            independent variable array.
        nstock : array-like
            dependent variable array.
        
        Returns:
        --------
        params : array-like
            array for the parameter guesses.
        covar : array-like
            array for the covariance guesses.
    '''
    
    return optimize.curve_fit(testFunction2, t, nstock)



def testFunction2(t, a, b):
    '''
        This function takes two arrays for t and nstock and guesses parameters from an equation into it.
        
        Parameters:
        -----------
        None
        
        Returns:
        --------
        array-like
            array of stock numbers over time, t.
    '''
    return a*t + b



def ode_model_concentration(t, c, pars, tau, ts, ak, tc, alpha, b1, t0, tMAR, PMAR, Psurf, Pa):
    '''     
    ODE model for concentration which is to be solved.
        Parameters:
        -----------
        t : array-like
            Array of times at which to evaluate concentration.
        c : array-like
            Array of concentrations.
        pars : array-like
            Array of parameters. Pars = [b, bc, P0, M0]
        tau : float
            Nitrate leaching time delay
        ts: array-like
            Years over which the cows were counted.
        ak : array-like
            Vector of spline polynomial coefficients 
        tc : float
            Year when carbon sink initiative implemented.
        alpha : float
            Parameter indicating successfulness of carbon sink initiative.
        b1 : float
            Parameter indicating strength of nitrate infiltration into groundwater.
        t0 : float
            Initial time of solution.
        tMAR : float
            Year when MAR initiative is implemented.
        PMAR : float
            Change in pressure introduce from MAR initiative. 
        Psurf : float
            Surface overpressure driving nitrate leaching 
        Pa : float
            Pressure drop across the aquifer.
       
        Returns:
        --------
        float
            Value of ode model
    '''
    b, bc, P0, M0 = pars
    if t <= tMAR:
        return (-n_stock(t-tau, ts, ak)*b_dash(t-tau, tc, alpha, b1)*(pressure(t, P0, b, tMAR, PMAR) - Psurf) + bc*(pressure(t, P0, b, tMAR, PMAR) - Pa/2)*c) / M0  # ODE model before tMAR
    return (-n_stock(t-tau, ts, ak)*b_dash(t-tau, tc, alpha, b1)*(pressure(t, P0, b, tMAR, PMAR) - Psurf) + bc*(pressure(t, P0, b, tMAR, PMAR) - (Pa+PMAR)/2)*c) / M0  # ODE model after tMAR



def future_cattle_model(t, a, b):
    '''
        This function returns the model associated with cattle stock growth in the future.
        
        Parameters:
        -----------
        a: float
            parameter for linear equation
        b: float
            parameter for linear equation
        
         Returns:
        --------
        array-like
            array for cow valuesa at given time values
        
    '''
    return  a*t + b



def b_dash(t, tc, alpha, b1):
    ''' 
    Evaluates b_dash array to be passed into ode_model_concentration.
        Parameters:
        -----------
        t : array-like
            Array of times at which to evaluate concentration.
        tc : float
            Year when carbon sink initiative implemented.
        alpha : float
            Parameter indicating successfulness of carbon sink initiative.
        b1 : float
            Parameter indicating strength of nitrate infiltration into groundwater.
        Returns:
        --------
        array-like
            Array of values of b_dash depending on carbon sink initiative implementation.
    '''
    if t < tc:
        return b1  # if carbon sink not yet installed
    return alpha * b1  # if carbon sink installed



def n_stock(t, ts, ak):
    ''' 
    Evaluates n_stock array to be passed into ode_model_concentration.
        Parameters:
        -----------
        t : array-like
            Array of times at which to evaluate concentration.
        ts: array-like
            Years over which the cows were counted.
        ak : array-like
            Vector of spline polynomial coefficients 
        Returns:
        --------
        array-like
            Array of cattle stock numbers at given times, t.
    '''
    stock = readInData()[1]
    if t <= ts[-1]:
        i = np.searchsorted(ts, t) - 1
        if i == -1:
            return 37772  # number of cattle stock asssumed to be present before 1990
        elif i == len(ts) - 1:
            t  = ts[-1]
        return polyval(ak[4*i:4*(i+1)], t-ts[i])  # cattle stock numbers using interpolation of data points
    else:
        return future_cattle_model(t, *fittingModel2(ts, stock)[0])  # future cattle stock model



def sum_of_squares(pars, ts1, lvls, c0, tau, ts, ak, tc, alpha, b1, PMAR, Psurf, Pa): 
    ''' 
    Find the sum of squares at each data point.
        Parameters:
        -----------
        pars : array-like
            Array of parameters. Pars = [b, bc, P0, M0]
        ts1: array-like
            Years over which the nitrate levels were counted.
        lvls : array-like
            Nitrate levels over the years.
        tau : float
            Nitrate leaching time delay
        ts: array-like
            Years over which the cows were counted.
        ak : array-like
            Vector of spline polynomial coefficients 
        tc : float
            Year when carbon sink initiative implemented.
        alpha : float
            Parameter indicating successfulness of carbon sink initiative.
        b1 : float
            Parameter indicating strength of nitrate infiltration into groundwater.
        PMAR : float
            Change in pressure introduce from MAR initiative. 
        Psurf : float
            Surface overpressure driving nitrate leaching 
        Pa : float
            Pressure drop across the aquifer.
        Returns:
        --------
        float
            value of the sum of squares
    '''
    # Solve ode using LSODA algorithm 
    c = solve_ivp(ode_model_concentration, [ts1[0], ts1[-1]], [c0], "LSODA", t_eval=ts1, args=[pars, tau, ts, ak, tc, alpha, b1, ts1[0], ts1[-1], PMAR, Psurf, Pa])
    return np.sum(np.square(c.y[0] - lvls))  # calculate and return sum of squares



# functions related to uncertainty
##################################################################################################################################################################

def grid_search():
    ''' This function implements a grid search to compute the posterior over b and bc.
		Returns:
		--------
		b : array-like
			Vector of 'b' parameter values.
		bc : array-like
			Vector of 'bc' parameter values.
		P : array-like
			Posterior probability distribution.
	'''    
    #load in the data
    ts, stock, ts1, lvls = readInData()
    A = spline_coefficient_matrix(ts)
    b = spline_rhs(ts,stock)
    ak = solve(A,b)
    bounds = [[24.01595224, 24.01595224], [3.12086543, 3.12086543], [414.17187084, 414.17187084], [0.89568477, 0.89568477]]  # bounds to search for parameter values
    args = [ts1, lvls, 0.2, 2, ts, ak, 2010, 0.3, 1e-4, 0, 0.05, 0.1]
    res = differential_evolution(sum_of_squares, bounds, args, seed = 20000, disp = False)  # get optimized parameter values
    pars = [res.x][0]
    t_range = [1980, 2019.5]
    t_preMAR = [1980, 2019.5]
    c = solve_ivp(ode_model_concentration, t_range, [args[2]], "LSODA", args=[res.x]+args[3:-3]+t_preMAR+args[-3:])
    for i in range(len(c.y[0])):
        var = lvls[i] - c.y[0][i]
	# define parameter ranges for the grid search
    b_best = pars[0]
    bc_best = pars[1]
	# number of values considered for each parameter within a given interval
    N = 50	

	# vectors of parameter values
    b = np.linspace(b_best/2,b_best*1.5, N)
    bc = np.linspace(bc_best/2,bc_best*1.5, N)

	# grid of parameter values: returns every possible combination of parameters in b and bc
    B, BC = np.meshgrid(b, bc, indexing='ij')

	# empty 2D matrix for objective function
    S = np.zeros(B.shape)

    # compute objective function for each combination of paramters b and bc
    for i in range(len(b)):
        for j in range(len(bc)):
            pars = [b[i], bc[j], pars[2], pars[3]]
            c = solve_ivp(ode_model_concentration, [ts1[0], ts1[-1]], [0.2], "LSODA", t_eval=ts1, args=[pars, 2, ts, ak, 2010, 0.3, 1e-4, ts1[0], ts1[-1], 0, 0.05, 0.1])
            S[i,j] = np.sum(np.square(c.y[0] - lvls))/var
	# compute the posterior
    P = np.exp(-S/2.)
    # normalize to a probability density function
    Pint = np.sum(P)*(b[1]-b[0])*(bc[1]-bc[0])
    P = P/Pint
    return b, bc, P



def fit_mvn(parspace, dist):
    """Finds the parameters of a multivariate normal distribution that best fits the data
    Parameters:
    -----------
        parspace : array-like
            list of meshgrid arrays spanning parameter space
        dist : array-like 
            PDF over parameter space
    Returns:
    --------
        mean : array-like
            distribution mean
        cov : array-like
            covariance matrix		
    """
    
    # dimensionality of parameter space
    N = len(parspace)
    
    # flatten arrays
    parspace = [p.flatten() for p in parspace]
    dist = dist.flatten()
    
    # compute means
    mean = [np.sum(dist*par)/np.sum(dist) for par in parspace]
    
    # compute covariance matrix
        # empty matrix
    cov = np.zeros((N,N))
        # loop over rows
    for i in range(0,N):
            # loop over upper triangle
        for j in range(i,N):
                # compute covariance
            cov[i,j] = np.sum(dist*(parspace[i] - mean[i])*(parspace[j] - mean[j]))/np.sum(dist)
                # assign to lower triangle
            if i != j: cov[j,i] = cov[i,j]
            
    return np.array(mean), np.array(cov)



def construct_samples(b,bc,P,N_samples):
	''' This function constructs samples from a multivariate normal distribution
	    fitted to the data.
		Parameters:
		-----------
		b : array-like
			Vector of 'b' parameter values.
		bc : array-like
			Vector of 'bc' parameter values.
		P : array-like
			Posterior probability distribution.
		N_samples : int
			Number of samples to take.
		Returns:
		--------
		samples : array-like
			parameter samples from the multivariate normal
	'''

	# fit the data using multivariate gaussain distribution
	B, BC = np.meshgrid(b,bc,indexing='ij')
	mean, covariance = fit_mvn([B,BC], P)

	# generate samples using multivariate normal function
	samples = np.random.multivariate_normal(mean, covariance, N_samples)

	return samples



def model_ensemble(samples):
    ''' Runs the model for given parameter samples and plots the results.
		Parameters:
		-----------
		samples : array-like
			parameter samples from the multivariate normal
	'''
    # read in data 
    ts, stock, ts1, lvls = readInData()
    A = spline_coefficient_matrix(ts)
    b = spline_rhs(ts,stock)
    ak = solve(A,b)
    bounds = [[24.01595224, 24.01595224], [3.12086543, 3.12086543], [414.17187084, 414.17187084], [0.89568477, 0.89568477]]  # bounds to search for parameter values
    args = [ts1, lvls, 0.2, 2, ts, ak, 2010, 0.3, 1e-4, 0, 0.05, 0.1]
    res = differential_evolution(sum_of_squares, bounds, args, seed = 20000, disp = False)  # get optimized parameter values

    t_range = [1980, 2019.5]
    t_preMAR = [1980, 2019.5]
    c = solve_ivp(ode_model_concentration, t_range, [args[2]], "LSODA", args=[res.x]+args[3:-3]+t_preMAR+args[-3:])
    # calculate variance in parameters (observational error)
    for i in range(len(c.y[0])):
        var = lvls[i] - c.y[0][i]

    params = [res.x][0]

	# plot of best fit with uncertainty
    f1, ax1 = plt.subplots(1, 1)
    ax2 = ax1.twinx()

	# for each random sample solve and plot the model
    for b, bc in samples:
        pars = [b, bc, params[2], params[3]]
        c = solve_ivp(ode_model_concentration, t_range, [args[2]], "LSODA", args=[pars]+args[3:-3]+t_preMAR+args[-3:])
        ax2.plot(c.t, c.y[0],'k-', lw=0.25,alpha=0.2)

    # plot the model forecast with uncertainty
    t_range = [2019.5, 2050]
    args[2] = c.y[0][-1]
   
    # for each random sample solve and plot the forward modelling with each predictor parameter value 
    for b, bc in samples:
        pars = [b, bc, params[2], params[3]]
        args[9] = 0.00
        c_0 = solve_ivp(ode_model_concentration, t_range, [args[2]], "LSODA", args=[pars]+args[3:-3]+t_preMAR+args[-3:]) # PMAR = 0.00
        ax2.plot(c_0.t, c_0.y[0],'c-', lw=0.25,alpha=0.2)

        args[9] = 0.02
        c_1 = solve_ivp(ode_model_concentration, t_range, [args[2]], "LSODA", args=[pars]+args[3:-3]+t_preMAR+args[-3:]) # PMAR = 0.02
        ax2.plot(c_1.t, c_1.y[0],'g-', lw=0.25,alpha=0.2)
        
        args[9] = 0.05
        c_2 = solve_ivp(ode_model_concentration, t_range, [args[2]], "LSODA", args=[pars]+args[3:-3]+t_preMAR+args[-3:]) # PMAR = 0.05
        ax2.plot(c_2.t, c_2.y[0],'r-', lw=0.25,alpha=0.2)
        
        args[9] = 0.075
        c_3 = solve_ivp(ode_model_concentration, t_range, [args[2]], "LSODA", args=[pars]+args[3:-3]+t_preMAR+args[-3:]) # PMAR = 0.075
        ax2.plot(c_3.t, c_3.y[0],'b-', lw=0.25,alpha=0.2)
        
        args[9] = 0.1
        c_4 = solve_ivp(ode_model_concentration, t_range, [args[2]], "LSODA", args=[pars]+args[3:-3]+t_preMAR+args[-3:]) # PMAR = 0.1
        ax2.plot(c_4.t, c_4.y[0],'m-', lw=0.25,alpha=0.2)
        

    ax2.plot(c.t, c.y[0], 'k-', label = "nitrate concentration best fit")
    ax2.plot(c_0.t, c_0.y[0], 'c-', label = "PMAR = 0.00")
    ax2.plot(c_1.t, c_1.y[0], 'g-', label = "PMAR = 0.02")
    ax2.plot(c_2.t, c_2.y[0], 'r-', label = "PMAR = 0.05")
    ax2.plot(c_3.t, c_3.y[0], 'b-', label = "PMAR = 0.075")
    ax2.plot(c_4.t, c_4.y[0], 'm-', label = "PMAR = 0.1")
    ax2.axhline(y=8.475, color='k', linestyle=':', label = "Drinking Water Standard (NZDWS)\n 75% of Nitrate-MAV (Maximum Allowable Value)")
    ax1.set_xlabel("time [years]")
    ax1.set_ylabel("no. cows")
    ax2.set_ylabel("nitrate concentration [mg/L]")
    plt.title("Forward Modelling with Various MAR Schemes")
    plt.title("Forward Modelling with Uncertainty for Various MAR Schemes")
    ax2.legend(loc='upper left')
    ax1.legend(loc='upper right')
    
    # plot error bars
    v = var
    ax2.errorbar(ts1,lvls,yerr=v,fmt='ro', label='data')
    ax2.legend()
    savePlot = False  # set to True to save the plot as 'uncertainty.png'
    if savePlot:
        plt.savefig('uncertainty.png')   
    else:
        plt.show()



# main function to run functions and output results
##################################################################################################################################################################

def main():

    # command to prevent unnecessary warnings regarding covariances and use of scipy functions e.g. LSODA
    warnings.filterwarnings('ignore')


    # testing that dcdt function is working correctly with unit test
    ##################################################################################################################################################################

    unitTest()


    # plotting the benchmark
    ##################################################################################################################################################################

    f1, ax1 = plt.subplots(1, 1)

    # plot numerical solution
    ts, cs = improved_euler_solve(f = simplified_ode_model, t0 = 0, y0 = np.exp(-0.5), t1 = 10, h = 0.1,  pars=[1, 1, 1, 0, 1, 1])
    ax1.plot(ts, cs, 'bx-', label = 'numerical solution')
    
    # plot analytical solution
    tanalytical = np.linspace(0, 10, 1000)
    canalytical = (np.exp(-0.5*np.exp(-2*tanalytical)))*(np.exp(-0.5*tanalytical))
    ax1.plot(tanalytical, canalytical, 'r-', label = 'analytical solution')

    # plot steady-state solution
    plt.axhline(y=0.0, color='k', linestyle=':', label = 'steady-state')

    # create title, labels, legend, xlimits
    ax1.legend(loc = "upper right")
    ax1.set_xlabel('time [yrs]')
    ax1.set_ylabel('nitrate concentration [g/L]')
    ax1.set_xlim([ts[0], ts[-1]])
    plt.title('benchmark: [h, a, b, g, Psurf, Pa] = [1, 1, 1, 0, 1, 1]')
    # show plot
    plt.show()


    # plotting the first best-fit model
    ##################################################################################################################################################################
    
    # reading in the data from the txt files
    ts, stock, ts1, lvls = readInData()

    # parameters for the cow ode
    params, covar = fittingModel(ts, stock)

    # plotting the graph related to the cow data
    tModel = np.linspace(ts[0], ts[-1], 1000)
    yModel = testFunction(tModel, *params)
    
    f, ax1 = plt.subplots(1,1)
    ax1.plot(ts, stock, 'ro', label = "data")
    ax1.plot(tModel, yModel, 'k-', label = "best fit")
    ax1.set_xlabel("time [years]")
    ax1.set_ylabel("no. cows")
    plt.title("Best-fit Polynomial Approximation of Cattle Stock Data")
    ax1.legend()
    
    savePlot = False
    if savePlot:
        plt.savefig('modelOfStock.png')   
    else:
        plt.show()
    

    # plotting the data
    f1, ax1 = plt.subplots(nrows = 1, ncols = 1)
    ax2 = ax1.twinx()
    ax1.plot(ts, stock, 'ro', label = "cattle data points")
    ax2.plot(ts1, lvls, 'ko', label = "concentration data points")
    ax1.set_xlabel("time [years]")
    ax1.set_ylabel("no. cows")
    ax2.set_ylabel("nitrate concentration [mg/L]")

    # creating graph from model

    # physical constants
    alpha = 0.3
    b1 = 0.0001
    tau = 2
    PSurf = 0.05
    deltaPa = 0.1

    # other constants
    tC = 2010
    tMAR = 2019.5
    deltaPMAR = 0.1

    '''
        inital guesses:
        bc = 1      # 
        M0 = 1      # initial mass
        P0 = 5      # initial pressure
        C0 = 0.2    # initial concentration
    '''
    
    '''
        constants after manual iteration:
        b = 24      
        bc = 3      
        P0 = 415    
        M0 = 1
        C0 = 0.2
    '''

    guesses = [24, 3, 410, 1.9]

    # using improved euler's method to plot the data
    tc, cs = improved_euler_solve(f = dCdt, t0 = ts1[0], y0 = 0.2, t1 = ts1[-1], h = 1, pars=[guesses, tau, tC, tMAR, b1, alpha, deltaPa, deltaPMAR, PSurf])

    # plotting graph onto existing plot
    ax2.plot(tc, cs, 'k-', label = "best fit")
    ax1.set_xlabel("time [years]")
    ax1.set_ylabel("no. cows")
    ax2.set_ylabel("nitrate concentration [mg/L]")
    plt.title("Solved Nitrate Concentration ODE Model and Data\n [b, bc, P0, M0] = [24, 3, 415, 1]")
    ax2.legend(loc='upper left')
    ax1.legend(loc='upper right')

    savePlot = False
    if savePlot:
        plt.savefig('modelOfConcentration.png')   
    else:
        plt.show()


    # plot the misfit
    c_misfit = []
    for i in range(0, len(ts1)):
        j = 0
        while tc[j] <= ts1[i]:
            j = j+1
        m = (cs[j]-cs[j-1])/(tc[j]-tc[j-1])
        c_misfit.append((m*ts1[j-1] + (cs[j-1]-(m*tc[j-1])))-lvls[i])
    average_misfit = np.sum(np.absolute(c_misfit))/len(ts1)
    
    label = "Average Misfit = " + str(average_misfit)
    f1, ax1 = plt.subplots(1, 1)
    ax1.plot(ts1, c_misfit, 'rx', label = "misfit")
    plt.axhline(y=0.0, color='k', linestyle=':')
    ax1.set_xlabel("time [years]")
    ax1.set_ylabel("concentration misfit [mg/L]")
    plt.title("Misfit for Solved Nitrate Concentration ODE Model")
    ax1.axhline(y=average_misfit, color='r', linestyle=':', label = label)
    ax1.legend()

    savePlot = False
    if savePlot:
        plt.savefig('firstMisfit.png')   
    else:
        plt.show()


    # plotting the improved best-fit model
    ##################################################################################################################################################################

    # plot the interpolated cattle stock data
    f1, ax1 = plt.subplots(1,1)
    ax1.plot(ts, stock, 'ro', label = "cattle data")
    A = spline_coefficient_matrix(ts)
    b = spline_rhs(ts, stock)
    ak = solve(A,b)
    label = "interpolated"
    stockArr = np.zeros(606)
    # run the interpolation functions and plotting
    for i, ti1, ti2 in zip(range(len(ts)-1), ts[:-1], ts[1:]):
        tv = np.linspace(ti1,ti2,101)
        stockv = polyval(ak[4*i:4*(i+1)], tv-ti1)
        ax1.plot(tv,stockv, 'r-', label=label)
        label = None
        for value in stockv:
            stockArr = np.append(stockArr, value)
    ax1.set_xlabel("time [years]")
    ax1.set_ylabel("no. cows")
    plt.title('Interpolation of Cattle Stock Data Points')
    ax1.legend()
    
    savePlot = False  # set to True to save the plot as 'interpolation.png'
    if savePlot:
        plt.savefig('interpolation.png')   
    else:
        plt.show()


    # plot the data with concentration best-fit
    f1, ax1 = plt.subplots(1, 1)
    ax2 = ax1.twinx()
    # bounds = [[0, 1000], [1, 5], [0, 1000], [0.75, 1.25]]  # bounds to search for parameter values
    bounds = [[24.01595224, 24.01595224], [3.12086543, 3.12086543], [414.17187084, 414.17187084], [0.89568477, 0.89568477]]  # actual optimized paramter values hardcoded to improve runtime
    args = [ts1, lvls, 0.2, 2, ts, ak, 2010, 0.3, 1e-4, 0, 0.05, 0.1]
    res = differential_evolution(sum_of_squares, bounds, args, seed = 20000, disp = False)  # get optimized parameter values

    t_range = [1980, 2019.5]
    t_preMAR = [1980, 2019.5]
    c = solve_ivp(ode_model_concentration, t_range, [args[2]], "LSODA", args=[res.x]+args[3:-3]+t_preMAR+args[-3:])

    title = "Solved Nitrate Concentration ODE Model and Data\n [b, bc, P0, M0] = " + str(res.x)

    ax1.plot(ts, stock, 'ro', label = "cattle data points")
    ax2.plot(ts1, lvls, 'ko', label = "concentration data points")
    ax2.plot(c.t, c.y[0], 'k-', label = "nitrate concentration best fit")
    ax1.set_xlabel("time [years]")
    ax1.set_ylabel("no. cows")
    ax2.set_ylabel("nitrate concentration [mg/L]")
    plt.title(title)
    ax2.legend(loc='upper left')
    ax1.legend(loc='upper right')

    savePlot = False  # set to True to save the plot as 'best_fit.png'
    if savePlot:
        plt.savefig('best_fit.png')   
    else:
        plt.show()
    

    # plot the misfit
    f1, ax1 = plt.subplots(1, 1)

    # linear interpolation to calculate misfit
    c_misfit = []
    for i in range(0, len(ts1)):
        j = 0
        while c.t[j] <= ts1[i]:
            j = j+1
        m = (c.y[0][j]-c.y[0][j-1])/(c.t[j]-c.t[j-1])
        c_misfit.append((m*ts1[j-1] + (c.y[0][j-1]-(m*c.t[j-1])))-lvls[i])
    average_misfit = np.sum(np.absolute(c_misfit))/len(ts1)

    label = "Average Misfit = " + str(average_misfit)
    ax1.plot(ts1, c_misfit, 'rx', label = "misfit")
    plt.axhline(y=0.0, color='k', linestyle=':')
    ax1.set_xlabel("time [years]")
    ax1.set_ylabel("concentration misfit [mg/L]")
    ax1.axhline(y=average_misfit, color='r', linestyle=':', label = label)
    plt.title("Misfit for Solved Nitrate Concentration ODE Model")
    plt.legend()
    
    savePlot = False  # set to True to save the plot as ''misfit.png'
    if savePlot:
        plt.savefig('misfit.png')   
    else:
        plt.show()


    # plotting the forecast modelling
    ##################################################################################################################################################################

    # plot the future cattle model
    f1, ax1 = plt.subplots(1, 1)
    ax2 = ax1.twinx()
    t = np.linspace(ts[0], 2050, 1000)
    ax1.plot(ts, stock, 'ro', label = "cattle data")
    ax1.plot(t, future_cattle_model(t, *fittingModel2(ts, stock)[0]), 'k-', label = "best fit")
    ax1.set_xlabel("time [years]")
    ax1.set_ylabel("no. cows")
    ax1.legend()
    plt.title("Future Cattle Stock Model")

    savePlot = False  # set to True to save the plot as 'best_fit.png'
    if savePlot:
        plt.savefig('future_cattle_model.png')   
    else:
        plt.show()

    
    # plot forward modelling
    f1, ax1 = plt.subplots(1, 1)
    ax2 = ax1.twinx()

    t_range = [2019.5, 2050]
    args[2] = c.y[0][-1]
    args[9] = 0.00
    c_0 = solve_ivp(ode_model_concentration, t_range, [args[2]], "LSODA", args=[res.x]+args[3:-3]+t_preMAR+args[-3:]) # PMAR = 0.00

    args[9] = 0.02
    c_1 = solve_ivp(ode_model_concentration, t_range, [args[2]], "LSODA", args=[res.x]+args[3:-3]+t_preMAR+args[-3:]) # PMAR = 0.02

    args[9] = 0.05
    c_2 = solve_ivp(ode_model_concentration, t_range, [args[2]], "LSODA", args=[res.x]+args[3:-3]+t_preMAR+args[-3:]) # PMAR = 0.05

    args[9] = 0.075
    c_3 = solve_ivp(ode_model_concentration, t_range, [args[2]], "LSODA", args=[res.x]+args[3:-3]+t_preMAR+args[-3:]) # PMAR = 0.075

    args[9] = 0.1
    c_4 = solve_ivp(ode_model_concentration, t_range, [args[2]], "LSODA", args=[res.x]+args[3:-3]+t_preMAR+args[-3:]) # PMAR = 0.1

    ax1.plot(ts, stock, 'ro', label = "cattle data points")
    ax2.plot(ts1, lvls, 'ko', label = "concentration data points")
    ax2.plot(c.t, c.y[0], 'k-', label = "nitrate concentration best fit")
    ax2.plot(c_0.t, c_0.y[0], 'c-', label = "PMAR = 0.00")
    ax2.plot(c_1.t, c_1.y[0], 'g-', label = "PMAR = 0.02")
    ax2.plot(c_2.t, c_2.y[0], 'r-', label = "PMAR = 0.05")
    ax2.plot(c_3.t, c_3.y[0], 'b-', label = "PMAR = 0.075")
    ax2.plot(c_4.t, c_4.y[0], 'm-', label = "PMAR = 0.1")
    ax2.axhline(y=8.475, color='k', linestyle=':', label = "Drinking Water Standard (NZDWS)\n 75% of Nitrate-MAV (Maximum Allowable Value)")
    ax1.set_xlabel("time [years]")
    ax1.set_ylabel("no. cows")
    ax2.set_ylabel("nitrate concentration [mg/L]")
    plt.title("Forward Modelling with Various MAR Schemes")
    ax2.legend(loc='upper left')
    ax1.legend(loc='upper right')

    savePlot = False  # set to True to save the plot as 'forward_modelling.png'
    if savePlot:
        plt.savefig('forward_modelling.png')   
    else:
        plt.show()


    # plotting the uncertainty 
    ################################################################################################################################################################## 

    # plot uncertainty and uncertainty in model forecasting
    b, bc, posterior = grid_search()
    N = 100
    samples = construct_samples(b, bc, posterior, N)
    model_ensemble(samples)



if __name__ == "__main__":
    main()