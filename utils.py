# -*- coding: utf-8 -*-
"""
Created on Mon May  2 15:27:20 2022

@author: korisnik
"""



import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import seaborn as sns
from copy import deepcopy
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats
from scipy.stats import norm
from scipy.optimize import curve_fit

# -------- Miscleanous -------- #
def reverse_z_axis(A):
    """
    make the depth of index iz becme the depth of index n - iz where n is the 
    length of the matix 

    Parameters
    ----------
    A : np.array

    Returns
    -------
    new_A : np.array
    reversed array

    """
    n, p = np.shape(A)
    new_A = np.ones((n, p))
    for i in range(n):
        new_A[i, :] = A[n-1 - i]
    return new_A


def cumulative_sum(arrZN):
    L = []
    S = 0
    for i, e in enumerate(arrZN):
        if not np.isnan(e):
            S += e
            L.append(S)
    return np.array(L)


def add_element_to_top_np1darray(H, elem=0):
    """
    adds an element in 1st position of a np.array, eauivalent of a 'list.append'
    but adds in 1st position an for a 1d array
    
    H : e.g. -1 * d.dep_1darr
    """
    n = len(H)
    H2 = elem*np.ones((n+1,))
    H2[1:] = H
    return H2


# -------- File management -------- #

def file_to_nparray(fname): 
    """
    Converts the file into np.array, replacing -9 by np.nan

    Parameters
    ----------
    fname : str
        text path

    Returns
    -------
    data : np.array
    
    attributes : list of str

    units : list of str

    """
    
    f = open(fname,"r")
    dat = f.readlines()
    attributes, units = dat[3].split(",")[:-1], dat[4].split(",")[:-1]
    raw_data = dat[5:]
    f.close()
    
    data = []
    for i in range(len(raw_data)):
        raw_line_i = raw_data[i].split(",")[:-1]
        line_i = [float(word) for word in raw_line_i]
        data.append(line_i)
    data = np.array(data)
    
    data = replace_9s_by_nan(data)
    
    return data, manage_blanks(attributes), units


def manage_blanks(word_lst):
    """
    removes the annoying blanks in the attributes names
    """
    new_word_lst = []
    
    for w in word_lst:
        w_new =  w.split()[0]
        
        for sub_word in w.split():
            if sub_word != " ":
                w_new = sub_word 
                
        new_word_lst.append(sub_word)
    return new_word_lst
        

def replace_9s_by_nan(data):
    """
    helper of file_to_array

    """
    n,p = np.shape(data)
    for i in range(n):
        for j in range(p):
            if data[i,j] == -9:
                data[i,j] = np.nan
    return data


def dates_to_datetime(rawdate):
    """
    Transform rawdate into datetime.datetime

    Parameters
    ----------
    rawdate : float
        YYMMDD, for instance 881031

    Returns
    -------
    datetime.date
        for instance 1988-10-31

    """
    rawdate = int(rawdate)
    YY = rawdate //10**4
    MM = (rawdate //10**2) %100
    DD = rawdate % 10**2
    
    if YY > 50:
        YY += 1900
    else:
        YY += 2000
    
    return datetime(YY, MM, DD)


def manage_dates(raw_dates_arr):
    """
    Transforms the raw date np.array into a list of datetime.datetime

    Parameters
    ----------
    dates : np.array
        
    Returns
    -------
    np.array of datetimes.date objects

    """
    
    n = len(raw_dates_arr)
    dt_dates_lst = []
    
    for i in range(n):
        newdate = dates_to_datetime(int(raw_dates_arr[i]))
        dt_dates_lst.append(newdate)
    
    return np.array(dt_dates_lst)



# -------- Functions for DataFrames -------- #

def warning_nint(d):
    if d.nint != 8 :
        print()
        print(9999999999999999999999999999999999999)
        print("please check that you have 8 depth intervals (8 reauired)")
        print()
        print()
        
        
def make_dataframe(d, i):
    """
    makes df for the depth zi
    helper of makes dataframe

    """
    data = d.dates_mpl_lst[i], d.dep_lst[i], d.chl_lst[i], d.l12_lst[i], d.PB_lst[i]
    df_zi = pd.DataFrame(data=data, index=d.attr)
    return df_zi


def is_in(elem, lst):
    if lst.count(elem) > 0:
        return True
    else:
        return False



# -------- Functions for seasonnal analysis -------- #


def make_data_arrays(d):
    """
    makes the array of Chl data for each month (evan if there is no mesurement)
    for the nint depths. Each row is the data at a specific depth. 
    Same thing for l12 and PB.

    Returns
    -------
    chl_array : (nb_of_months, nint)-np.array
        chl over time at each depth
    l12_array : same thing
    PB_array : same thing

    """
    # x
    ny = 2019-1988
    nm = 12
    y_start = 88
    
    # z
    nint = d.nint   
    
    # Arrays :
    chl_array = np.nan * np.ones((nint, nm*(ny + 1))).T
    l12_array = np.nan * np.ones((nint, nm*(ny + 1))).T
    PB_array  = np.nan * np.ones((nint, nm*(ny + 1))).T
    
    # Fill with the data we have for depth iz:
    for iz in range(nint):
        rawdates = d.rawdates_lst[iz]
        chl_data = d.chl_lst[iz]
        l12_data = d.l12_lst[iz]
        PB_data  = d.PB_lst[iz]
        
        
        for i, rd in enumerate(rawdates):
            yy, mm = int(rd//10**4), int((rd//10**2)%100)
            if yy < 50:
                yy += 100
            
            " indexes of the raw data we have in the of the array"
            index_years = yy - y_start
            index_months = mm - 1   # start at month 0, ends at 11
            index_time = 12* index_years + index_months
            
            chl_array[index_time, iz] = chl_data[i]
            l12_array[index_time, iz] = l12_data[i]
            PB_array[index_time, iz]  = PB_data[i]
            
    return chl_array, l12_array, PB_array


def make_data_array_hawaii(arr, rawdates):
    """
    makes the array of data array for each month (evan if there is no mesurement)
    for the nint depths. Each row is the data at a specific depth. 

    Returns
    -------
    data_filled array

    """
    # x
    ny = 2020-1988
    nm = 12
    y_start = 88

    # Array to fill :
    data_array = np.nan * np.ones((1, nm*(ny + 1))).T
    
    for i, rd in enumerate(rawdates):
        yy, mm = int(rd % 100), int((rd//10**4))
        if yy < 50:
            yy += 100
        
        " indexes of the raw data we have in the of the array"
        index_years = yy - y_start
        index_months = mm - 1   # start at month 0, ends at 11
        index_time = 12* index_years + index_months
        
        data_array[index_time] = arr[i]

    return data_array

    
def fill_gaps_inside(arr):
    A = deepcopy(arr)
    n, p = np.shape(arr)
    for i in range(1, n-1):
        for j in range(1, p-1):
            if np.isnan(A[i, j]) :
                A[i, j] = np.nanmean(A[i-1:i+2, j-1:j+2])
                
    return A

def fill_gaps_edges(arr):
    A = deepcopy(arr)
    n, p = np.shape(arr)
    # first fill the inside
    
    i = 0
    for j in range(1, p-1):
        if np.isnan(A[i, j]) :
            A[i, j] = np.nanmean(A[i:i+2, j-1:j+2])
    
    i = n-1
    for j in range(1, p-1):
        if np.isnan(A[i, j]) :
            A[i, j] = np.nanmean(A[i-1:i, j-1:j+2])
            
    j = 0
    for i in range(1, n-1):
        if np.isnan(A[i, j]) :
            A[i, j] = np.nanmean(A[i-1:i+2, j:j+2])
    
    j = p-1
    for i in range(1, n-1):
        if np.isnan(A[i, j]) :
            A[i, j] = np.nanmean(A[i-1:i+2, j-1:j])
            
    i, j = 0, 0
    A[i, j] = np.nanmean(A[i:i+2, j:j+2])
    
    i, j = n-1, 0
    A[i, j] = np.nanmean(A[i-1:i, j:j+2])
    
    i, j = n-1, p-1
    A[i, j] = np.nanmean(A[i-1:i, j-1:j])
    
    i, j = 0, p-1
    A[i, j] = np.nanmean(A[i:i+2, j-1:j])
            
    return A
        
        
def fill_gaps(arr):
    A = deepcopy(arr)
    A = fill_gaps_inside(A)
    A = fill_gaps_edges(A)
    A = fill_gaps_inside(A)
    A = fill_gaps_edges(A)
    return A


def dev_to_the_mean(d, seasonnal_data, iz=0, title="??", unit="unit ??", l_plot=0):
    
    seasonnal_data_mean = np.mean(seasonnal_data, axis=0)
    seasonnal_data_dev = seasonnal_data - seasonnal_data_mean
    
    if l_plot:
        fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)
        im1 = axes.pcolor(seasonnal_data_dev, cmap="bwr")
        
        divider = make_axes_locatable(axes)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im1, cax=cax, orientation='vertical')
        
        axes.set_xlabel("# month")
        axes.set_ylabel("# year from start")
        fig.suptitle("{0} deviation from the mean seasonnal cycle, depth={1}, {2}".format(title, d.depth_labels_lst[iz], unit))
        
        fig.show()
    
    return seasonnal_data_dev



def make_seasonnal_plot(d, data_array, title="??", unit="unit ??", iz=0, put_data_in_log=False, l_plot=0):
    """
    Plots an array that has the data of one year for each line, each column 
    is a month (january ... december) and the value is the data (or LOG (data))

    Parameters
    ----------
    data_array : (? , nint) np.array
        Month by month data array for each depth, for each quantity

    iz : int, optional
        index f the depth interval of the data. The default is 0.

    Returns
    -------
    seasonal_data : np array with 12 times less lines and 12 times more columns
        DESCRIPTION.

    """
        
    data_iz = data_array[:, iz]
    u = len(data_iz)
    
    if put_data_in_log:       
        data_iz_years = np.log(data_iz.reshape((u//12, 12)))   
        "line i=0 for year 0, etc. "
    else:
        data_iz_years = data_iz.reshape((u//12, 12)) 
    
    seasonal_data = fill_gaps(data_iz_years)
        
    if l_plot :
        fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)
        im1 = axes.pcolor(seasonal_data, cmap="viridis")
        
        divider = make_axes_locatable(axes)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im1, cax=cax, orientation='vertical')
        
        axes.set_xlabel("# month")
        axes.set_ylabel("# year from start")
        if put_data_in_log:
            unit = "  log" + unit
            fig.suptitle("{0} seasonnal cycle, depth={1}, {2}".format(title, d.depth_labels_lst[iz], unit))
        else:
            fig.suptitle("{0} seasonnal cycle, depth={1}, {2}".format(title, d.depth_labels_lst[iz], unit))
        fig.show()
        
    return seasonal_data


def normalised_std(d, seasonnal_data, iz, title="??", unit="unit ??", l_plot=0):
    """
    takes a np.array X and plots and returns (X - mean X) / std X  
    -------
    seasonnal_data_normalized :
        (n/12,12) np.array

    """
    seasonnal_data_mean = np.mean(seasonnal_data, axis=0)
    seasonnal_data_std = np.std(seasonnal_data, axis=0)
    seasonnal_data_normalized = (seasonnal_data - seasonnal_data_mean)/seasonnal_data_std
    
    if l_plot:
        fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)
        im1 = axes.pcolor(seasonnal_data_normalized, cmap="bwr")
        
        divider = make_axes_locatable(axes)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im1, cax=cax, orientation='vertical')
        
        axes.set_xlabel("# month")
        axes.set_ylabel("# year from start")
        fig.suptitle("{0} normalized std of the mean seasonnal cycle, depth={1}, {2}".format(title, d.depth_labels_lst[iz], unit))
        
        fig.show()
    
    return seasonnal_data_normalized



def normalised_std_lst(d, seasonnal_data_lst, title="??", unit="unit ??", l_plot=1):
    """
    Plots the seasonnal graph of the normalised and centered seasonnal graph of the data, depth by depth
    Returns the list of the np.arrays centered and normaliyed, in the same order as in the parameters
    The x-axis represent the 12 monts of the year and the y-axis are the years 

    """
    
    seasonnal_data_normalized = [normalised_std(d, seasonnal_data_lst[iz], iz, title="Chl", unit="( mg / m3 )", l_plot=0) for iz in range(8)]

    if l_plot:
         nrows=2
         ncols=4
         fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True)

         mini = min([np.min(seasonnal_data_normalized[i]) for i in range(8)])
         maxi = max([np.max(seasonnal_data_normalized[i]) for i in range(8)])
         abs_max = max(np.abs(mini), np.abs(maxi))

         for i in range(nrows):
             for j in range(ncols):    
                 im_ij = axes[i, j].pcolor(seasonnal_data_normalized[i+nrows*j], cmap="bwr", vmin = -abs_max, vmax = abs_max)
                 axes[i, j].set_xlabel(d.depth_labels_lst[i+2*j])
                 
                 if j==3:
                     divider = make_axes_locatable(axes[i,j])
                     cax = divider.append_axes('right', size='5%', pad=0.05)
                     fig.colorbar(im_ij, cax=cax, orientation='vertical')
                     
                     
         axes[0, 0].set_ylabel("# year from start")
         fig.suptitle("{0} normalized deviation to the seasonnal cycle ({1})".format(title, unit))
         
         fig.show()
         
    return seasonnal_data_normalized


def make_seasonnal_data_lst(d, seasonnal_data_array, title="??", unit="unit ??", l_plot=0, l_plot_in_1_graph=1, put_data_in_log=False):
    """
    computes and returns the list : [seasonnal matrices at depth zi for zi in depths]

    Parameters
    ----------
    seasonnal_data_array : (384, 8) np.array containing all temporal data for the 8 depths
    e.g.:
    
    Returns
    -------
    seasonnal_data_lst : List of the seasonnal matrices of data (e.g. Chl), 
    where the ith element of the list is for depth zi 

    """
    if not(l_plot_in_1_graph): 
        # then l_plot CAN be true (but don't need)
        seasonnal_data_lst = [make_seasonnal_plot(d, seasonnal_data_array, title="Chl", unit="mg / m3", iz=iz, put_data_in_log=put_data_in_log, l_plot=l_plot) for iz in range(8)]
        return seasonnal_data_lst   

        
    else:
        # then l_plot CAN * False in the following commmand
        seasonnal_data_lst = [make_seasonnal_plot(d, seasonnal_data_array, title="Chl", unit="mg / m3", iz=iz, put_data_in_log=put_data_in_log, l_plot=0) for iz in range(8)]
        
        if l_plot:
            nrows = 2
            ncols = 4
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True)
        
            mini = min([np.min(seasonnal_data_lst[i]) for i in range(8)])
            maxi = max([np.max(seasonnal_data_lst[i]) for i in range(8)])
        
            for i in range(nrows):
                for j in range(ncols):   
        
                    im_ij = axes[i, j].pcolor(seasonnal_data_lst[i+nrows*j], cmap="viridis", vmin=mini, vmax=maxi)
                    axes[i, j].set_xlabel(d.depth_labels_lst[i+2*j])
                    
                    if j==3:
                        divider = make_axes_locatable(axes[i,j])
                        cax = divider.append_axes('right', size='5%', pad=0.05)
                        fig.colorbar(im_ij, cax=cax, orientation='vertical')
                    
            axes[0, 0].set_ylabel("# year from start")
            
            
            if put_data_in_log:
                unit = "log " + unit
                fig.suptitle("{0} seasonnal cycle ({1})".format(title, unit))
            else:
                fig.suptitle("{0} seasonnal cycle ({1})".format(title, unit))
            
            fig.show()
            
    return seasonnal_data_lst
             


# -------- Functions for statistics and regressions -------- #

def correlation_matrix(Cov):
    """
    returns the correlation matrix, the correlation coefficeient of a 2x2 R 
    covariance matrix is R[0,1]


    """
    n, p = np.shape(Cov)
    R_m = np.ones((n, p))
    
    for i in range(n):
        for j in range(p):
            R_m[i, j] = Cov[i, j]/np.sqrt(Cov[i, i])/np.sqrt(Cov[j, j])
    return R_m


def r_squared(Cov):
    n, p = np.shape(Cov)
    
    if (n, p) != (2,2):
        raise ValueError("r_squared functionnrreauires a 2x2 Cov matrix !") 
        print("r_squared functionnrreauires a 2x2 Cov matrix !")
    
    else:
        R_m = correlation_matrix(Cov)
        return np.round(R_m[0,1]**2, 2)
    

def remove_nans(u):
    """
    returns the subarrays of u only for the index where u is not nan

    Parameters
    ----------
    u : (1, n)-np.array

    Returns
    -------
    u not-nan-subarray

    """

    u_b = ~np.isnan(u)
    bool_selection = u_b   
    
    idx = np.array([i for i in range(len(u))])

    return u[idx[bool_selection]]

def remove_nans2(u, v):
    """
    returns the subarrays of u and v only for the index where 
    u AND v are not nan

    Parameters
    ----------
    u : (1, n)-np.array
    v : (1, n)-np.array      ! must have the same size !

    Returns
    -------
    u, v not-nan-subarrays

    """

    u_b, v_b = ~np.isnan(u), ~np.isnan(v)
    bool_selection = u_b * v_b   # keep only the index where u AND v are not nan
    
    idx = np.array([i for i in range(len(u))])

    return u[idx[bool_selection]], v[idx[bool_selection]]



def reglin(arr):
    """
    linear regression of the data 
    
    to get the polynomial function from the coefficients,
    just do np.poly1d(coeffs)
    
    !! the unit of a is the unit of the arry/step, where step is the time
    difference between 2 steps of the index list v !!

    Parameters
    ----------
    arr : 1d ndarray

    Returns
    -------
    coeffs : coeffs of the 1st order polynomial function 
    R2 : correlation coefficient

    """
    u = arr
    v = np.array([i for i in range(len(u))])
    
    coeffs, Cov = np.polyfit(v, u, deg=1, cov=1)
    R2 = r_squared(Cov)
    
    return coeffs, R2



def plot_linear_regression_data_over_time(data_all_months, iz=0, l_plot=0):
    """
    Plots the monthly trend (linear regression) of the input time series, for 
    every month, at depth #iz

    Parameters
    ----------
    data_all_months : (nyears, 8)-np.array : time series for each depth
        example : d.chl_array
    iz : int
    # of the depth interval.
    
    Returns
    -------
    lst_of_polynomials: [(a_jan_z1, b_jan_z1), (a_fev_z1, b_fev_z1), ... ]
    or the equivalent for the other months

    """
    
    data_all_months_iz = data_all_months[:, iz]
    
    lst_of_polynomials = [] # list of {the slope, intercept at month it} for it
    
    for it in range(12):    # number of months in a year
        idx_month_i = [i for i in range(len(data_all_months_iz)) if i % 12 == it]
        data_month_i = data_all_months_iz[idx_month_i] 
        
        arr_month = data_month_i
        u = remove_nans(arr_month)
        v = np.array([i for i in range(len(u))])
        
        curve, Cov = np.polyfit(u, v, deg=1, cov=1)
        slope,  intercept = curve
        lst_of_polynomials.append((slope,  intercept))
    
        poly = np.poly1d(curve)
        
        R2 = r_squared(Cov)
        
        if l_plot:
            plt.title(f"month {it+1}, R2 = {R2}")
            sns.regplot(x=u, y=v)
            plt.plot(u, poly(u))
            plt.show()
    
    return lst_of_polynomials


def all_polynomials_reglin(d, data_all_months):
    """
    computes the list of {the list of polynomial for all months} for all depths

    Parameters
    ----------
    data_all_months : (nyears, 8)-np.array : time series for each depth
        example : d.chl_array

    Returns
    -------
    lst_lst_polynomials : list of list of tuples, of length 8 (8 depths)
        [[(a_jan_z1, b_jan_z1), (a_fev_z1, b_fev_z1), ... ],
         [(a_jan_z2, b_jan_z2), (a_fev_z2, b_fev_z2), ... ],
         .
         .
         .
         [(a_jan_z8, b_jan_z8), (a_fev_z8, b_fev_z8), ... ]] 

    """
    lst_lst_polynomials = []
    for iz in range(8):
        lst_of_polynomials_month_iz = plot_linear_regression_data_over_time(data_all_months, iz, l_plot=0)
        lst_lst_polynomials.append(lst_of_polynomials_month_iz)
        
    return lst_lst_polynomials



def make_filled_data_array(d, seasonnal_data_lst):
    """
    takes seasonnal filled data list and makes the arrays of flled data at 
    each depth, flattening the matrices at each depth 

    Parameters
    ----------
    seasonnal_data_lst 
    eg: seasonnal_chl_lst

    Returns
    -------
    Matix_all : (8, 384) data matrix.

    """
    Matix_all = np.ones((d.nint, 12*len(seasonnal_data_lst[0])))
    for iz in range(8):
        seasonnal_matrix = seasonnal_data_lst[iz]
        flat = seasonnal_matrix.flatten()
        Matix_all[iz] = flat
              
    return Matix_all


def slope_array(d, data):
    """
    returns the array of the time-slopes for each depth

    Parameters
    ----------

    data : (8, whatever) data time-series array 
        e.g : d.chl_array

    Returns
    -------
    slopes_arr : (8,)-array of the slopes

    """

    slopes_lst = []
    
    for iz in range(8):
        arr = data[iz]
        coeffs, R2 = reglin(arr)
        slope = coeffs[0]
        slopes_lst.append(slope)
        
    slopes_arr = np.array(slopes_lst)
    return slopes_arr


def select_data_from_month_i(d, data, it):
    """
    takes data from all months and selects the data of the desired month at 
    the desired depth

    Parameters
    ----------
    data : (8, whatever) data time-series array 
        e.g : d.chl_array
    
    iz : int
    depth index
    
    it : int
    month index

    Returns
    -------
    data_it_iz : (8, whatever // 12) np.array
        data subarray of data of the month i at depth zi
    years : TYPE
        DESCRIPTION.

    """
    n, p = np.shape(data)
    lst_month_i =  [i for i in range(p) if i % 12 == it]
    
    data_it = data[:, lst_month_i]      # data at all depths for the months i
    years = np.array([i for i in range(len(data_it))])
    return data_it, years



def make_slopes_matrix(d, data_array):
    """
    makes the matix of the slope of the seasonnal trend, for each depth
    The time unit is 1 year

    Parameters
    ----------
    data_array : e.g.: d.chl_array

    Returns
    -------
    Slopes : (8, 12)-np.array : first column is the slope of the linear trend 
    at all depths for januaries, the 2nd column for febuaries ...
    vertically : depths

    """
    Slopes = np.ones((8, 12))       # 8 depths, 12 months
    
    for it in range(12):
        data_it, years = select_data_from_month_i(d, data_array, it=it)
        slopes_all_z = slope_array(d, data_it)
        Slopes[:, it] = slopes_all_z
        
    return Slopes


def integrate_from_surface(A, H):
    """
    integrate along z

    Parameters
    ----------
    A : (n,p) data array
    H : positive depth array (positive depths)
        eg.: -1 * d.dep_1darr

    Returns
    -------
    integrated_A : (n-1 ,p) np.array, verically integrated with trapezoid rule
    """
    
    Dz = H[1:] - H[:-1] 
    n, p = np.shape(A)
    integrated_A = np.ones((n-1, p))
    
    for it in range(p):
        
        S = 0
        
        for iz in range(n-1):
            # iz : 0...n-2
            S += (A[iz, it] + A[iz+1, it])/2 *Dz[iz]
            integrated_A[iz, it] = S
            
    return integrated_A


def integrate_over_the_watercolumn_v1(A, H):
    """
    integrate along z over the entire watercolumn

    Parameters
    ----------
    A : (n ,p) data array
    H : positive depth array (positive depths)
        eg.: -1 * d.dep_1darr

    Returns
    -------
    integrated_A : (1 ,p) np.array, verically integrated with trapezoid rule
    """
    
    Dz = H[1:] - H[:-1] 
    n, p = np.shape(A)
    integrated_A = np.ones((n-1, p))
    
    for it in range(p):
        
        S = 0
        
        for iz in range(n-1):
            # iz : 0...n-2
            S += (A[iz, it] + A[iz+1, it])/2 *Dz[iz]
            integrated_A[iz, it] = S
            
    return integrated_A[-1]


def integrate_over_the_watercolumn_rectangles(A, H):
    """
    integrate along z over the entire watercolumn using rectangles

    Parameters
    ----------
    A : (n, p) data array 
        eg: A = d.chl_array
    
    H : positive depth array (positive depths)
        eg.: d.dep_1darr_0_positive

    Returns
    -------
    integrated_A : (1 ,p) np.array, verically integrated with trapezoid rule
    """
    Dz = H[1:] - H[:-1] 
    n, p = np.shape(A)
    integrated_A = np.ones((n, p))
    
    for it in range(p):
        
        S = 0
        
        for iz in range(n):
            # iz : 0...n-1
            S += A[iz, it] * Dz[iz]
            integrated_A[iz, it] = S
    return integrated_A[-1]

def integrate_over_the_watercolumn(A, H):
    """
    integrate along z over the entire watercolumn with trapezes

    Parameters
    ----------
    A : (n, p) data array 
        eg: A = d.chl_array
    
    H : positive depth array (positive depths)
        eg.: d.dep_1darr_0_positive

    Returns
    -------
    integrated_A : (1 ,p) np.array, verically integrated with trapezoid rule
    """
    Dz = H[1:] - H[:-1] 
    n, p = np.shape(A)
    integrated_A = np.ones((n, p))
    
    for it in range(p):
        
        S = 0
        av = A[0, it]
        for iz in range(n):
            # iz : 0...n-1
            if n == 0:
                S += A[0, it] * Dz[0]
            else:
                S += (A[iz, it] + A[iz-1, it])/2 * Dz[iz]
            
            integrated_A[iz, it] = S
    return integrated_A[-1]

        
# -------- Seasonal analysis : mean -------- #


def get_the_mean_for_month_i(data_arrZ, idx_month_i):
    L_values_month_i = []
    for it in range(len(data_arrZ)):
        if it % 12 == idx_month_i:
            L_values_month_i.append(data_arrZ[it])
            
    mean = sum(L_values_month_i)/len(L_values_month_i)
    return mean


def get_the_mean_for_all_months(data_arrZ):
    """
    returns the 1D array of all monthly means

    """
    mean_lst = [get_the_mean_for_month_i(data_arrZ, i_month) for i_month in range(12)]
    return np.array(mean_lst)
    

def make_mean_seasonnal_matrix(d, data_array):
    """
    return the (8, 12) matrix of the mean of the data (chl or other ... ) for
     the specified months and depths

    """
    M = np.zeros((8, 12))
    for it in range(12):
        for iz in range(8):
            M[iz, it] = get_the_mean_for_month_i(data_array[iz], it)
    return M



# -------- Gaussian parameters -------- #

def gauss(x, B0, H, x0, sigma):
    return B0 + H * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def params_gaussian_regression(depth, profile):
    """
    Gives the parameters of the gaussian that is best fitted

    Parameters
    ----------
    depth : d.dep_1darr.
    profile : d.chl_array[:, 0].

    """
    p0 = 0.1, 0.1, -100, 25    # starting parameters
    parameters, covariance = curve_fit(gauss, xdata=depth, ydata=profile, p0=p0, maxfev=5000)
    B0, H, zm, sigma = parameters
    
    return B0, H, zm, sigma


def parameters_array(d):
    """
    makes a (4, nb_months) array of the parameters of the gaussian fit for the
    (filled) data at month it for all it.
    
    each line is for a parameter:
        B0, H, zm, and sigma

    """
    Gaussian_param_arr = np.ones((4, d.p))
    
    for it in range(d.p):
        B0, H, zm, sigma = params_gaussian_regression(depth=d.dep_1darr, profile=d.chl_array[:, it])
        Gaussian_param_arr[0, it] = B0
        Gaussian_param_arr[1, it] = H
        Gaussian_param_arr[2, it] = zm
        Gaussian_param_arr[3, it] = sigma
        
    return Gaussian_param_arr[0, :], Gaussian_param_arr[1, :], -1*Gaussian_param_arr[2, :], Gaussian_param_arr[3, :]



def make_theoretical_B_array_all_data(d):
    """
    uses the parameters of the fitted gaussian to reconstruct the theoretical 
    values of B (which is CHL) at every depth, for every month

    Returns
    -------
    Gauss_t : (d.nint, d.p) nparray

    """
    Gauss_t = np.nan * np.ones((d.nint, d.p))
    
    for it in range(d.p):
        depths = d.dep_1darr * (-1) 
        B0it, Hit, zmit, sigmait = d.B0_arr[it], d.H_arr[it], d.zm_arr[it], d.sigma_arr[it]
        Btheo_it = gauss(depths, B0it, Hit, zmit, sigmait)
    
        Gauss_t[:, it] = Btheo_it
    return Gauss_t


def remove_outliers(data, qu=3, ql=3, l_negative=0, l_H=0):
    """
    Removes the outliers based on percentiles
    l_negative is to remove negative data (used for B0)
    qu is the upper quantile to trow away, ql is the lesser quantile to trow away
    
    """
    m, q3, q97 = np.median(data), np.percentile(data, ql), np.percentile(data, 100-qu)
    idx_of_removed_data = []
    filtered_data = deepcopy(data)
    
    
    for it, e in enumerate(data):
        cond1 = q97 < e 
        cond2 = q3 > e 
        if cond1 or cond2:
            filtered_data[it] = np.nan
            idx_of_removed_data.append(it)

        else:
            if l_negative:
                if e < 0 :
                    filtered_data[it] = np.nan
                    idx_of_removed_data.append(it)

            if l_H:
                if e < 0 :
                    filtered_data[it] = np.nan
                    idx_of_removed_data.append(it)

    return filtered_data, idx_of_removed_data




