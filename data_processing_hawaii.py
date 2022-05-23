# -*- coding: utf-8 -*-
"""
Created on Tue May  3 17:33:09 2022

@author: korisnik
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May  2 14:38:20 2022

@author: korisnik
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
from utils import *
from plots import *
import seaborn as sns
import pandas as pd
from copy import deepcopy
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit
from scipy.stats import norm


sns.set_style("darkgrid")


class Data:
    
    def __init__(self, nint=8, fname="C:/Users/korisnik/Internship/1988-2022_chl_light12_HOT-DOGS.txt"):
        
        self.fname = fname
        self.colors = ["blue", "green"]
        
        
        # -------- Loading data -------- #
        
        data, attr, units = file_to_nparray(fname)
        depths_all, chl_all, l12_all = data[:, 5], data[:, 6], data[:, 7]
        rawdates_all = data[:, 1]
        dates = manage_dates(data[:, 1])

        self.PB = l12_all/chl_all
        self.attr = ["date", "depth", "chl", "l12", "PB"]
        self.labels_months = ["january","february", "mars", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
        self.labels_months_short = ["jan", "feb", "mars", "apr", "may", "june", "july", "aug", "sept", "oct", "nov", "dec"]
        
        
        # -------- Data selection for each depth range -------- #
        
        self.nint = nint
        
        # boolean arrays used to select data in each depth interval:
        n, p = np.shape(data)
        Z, O = np.zeros((n,)), np.ones((n,))
        l_dep_lst = self.nint*[0]
        l_dep_lst[0] = (np.where(0 <= depths_all, O, Z)   * np.where(depths_all < 10,  O, Z)).astype(bool)
        l_dep_lst[1] = (np.where(10 <= depths_all, O, Z)  * np.where(depths_all < 35,  O, Z)).astype(bool)
        l_dep_lst[2] = (np.where(35 <= depths_all, O, Z)  * np.where(depths_all < 60,  O, Z)).astype(bool)
        l_dep_lst[3] = (np.where(60 <= depths_all, O, Z)  * np.where(depths_all < 87,  O, Z)).astype(bool)
        l_dep_lst[4] = (np.where(87 <= depths_all, O, Z)  * np.where(depths_all < 113, O, Z)).astype(bool)
        l_dep_lst[5] = (np.where(113 <= depths_all, O, Z) * np.where(depths_all < 138, O, Z)).astype(bool)
        l_dep_lst[6] = (np.where(138 <= depths_all, O, Z) * np.where(depths_all < 163, O, Z)).astype(bool)
        l_dep_lst[7] = (np.where(163 <= depths_all, O, Z)).astype(bool)

        # label of each depth interval :
        self.depth_labels_lst = ["5 m", "25 m", "45 m", "75 m", "100 m", "125 m", "150 m", "175 m"]  
        self.depth_labels_lst_reversed = ["175 m", "150 m", "125 m", "100 m", "75 m", "45 m", "25 m", "5 m"] 
        
        # indexes for each depth interval : 
        self.idx_lst = [np.arange(n)[l_dep_lst[i]] for i in range(self.nint)]
        
        
        # -------- Data list for each depth -------- #
        
        # times for each depth : 
        self.dates_mpl_lst = [dates[self.idx_lst[i]] for i in range(self.nint)]
        self.rawdates_lst = [rawdates_all[self.idx_lst[i]] for i in range(self.nint)]
       
        # depths for each depth interval :
        self.dep_lst = [depths_all[self.idx_lst[i]] for i in range(self.nint)]
        
        # chl for each depth interval :
        self.chl_lst = [chl_all[self.idx_lst[i]] for i in range(self.nint)]
        
        # l12 for each depth interval :
        self.l12_lst = [l12_all[self.idx_lst[i]] for i in range(self.nint)]
        
        # PB for each depth interval : 
        self.PB_lst = [self.PB[self.idx_lst[i]] for i in range(self.nint)]
        
        
        # -------- Month by month data array for each depth, for each quantity -------- #
        self.chl_array_sparse, self.l12_array_sparse, self.PB_array_sparse = make_data_arrays(self)
        self.dep_1darr = -1 *np.array([5 , 25, 45, 75, 100, 125, 150, 175])
        self.dep_1darr_0_positive = np.array([0, 5 , 25, 45, 75, 100, 125, 150, 175])
        self.dep_nb_arr = np.array([i for i in range(nint)])
        
        
        # -------- Filled data -------- #
        
        seasonnal_chl_lst = make_seasonnal_data_lst(self, self.chl_array_sparse, title="Chl", unit="mg / m3", put_data_in_log=False, l_plot=0)
        seasonnal_l12_lst = make_seasonnal_data_lst(self, self.l12_array_sparse, title="l12", unit=" mg C / m3 ", put_data_in_log=False, l_plot=0)
        seasonnal_PB_lst = make_seasonnal_data_lst(self, self.PB_array_sparse, title="PB", unit="mg C / mg chl", put_data_in_log=False, l_plot=0)
        
        self.chl_array = make_filled_data_array(self, seasonnal_chl_lst)
        self.l12_array = make_filled_data_array(self, seasonnal_l12_lst)
        self.PB_array  = make_filled_data_array(self, seasonnal_PB_lst )
        self.n, self.p = np.shape(self.chl_array)
        
        
        # -------- Integration over depth of the time_series -------- #
        
        self.chlZ_rect = integrate_over_the_watercolumn_rectangles(self.chl_array, self.dep_1darr_0_positive)
        self.chlZ = integrate_over_the_watercolumn(self.chl_array, self.dep_1darr_0_positive)
        self.chlZN = (self.chlZ - np.mean(self.chlZ)) / np.std(self.chlZ)
        self.chlZNT = cumulative_sum(self.chlZN)

        self.l12Z = integrate_over_the_watercolumn(self.l12_array, self.dep_1darr_0_positive)
        self.l12ZN = (self.l12Z - np.mean(self.l12Z)) / np.std(self.l12Z)
        self.l12ZNT = cumulative_sum(self.l12ZN)

        self.PBZ = integrate_over_the_watercolumn(self.PB_array, self.dep_1darr_0_positive)
        self.PBZN = (self.PBZ - np.mean(self.PBZ)) / np.std(self.PBZ)
        self.PBZNT = cumulative_sum(self.PBZN)
        
        
        # -------- Comparaison with Hawaii website data -------- #
        
        fname = "C:/Users/korisnik/Internship/1988-2022_integrated_chl_pandas"
        df = pd.read_csv(fname, sep=",", names=["crn", "date", "julian", "chl"])
        self.sparse_chl_hawaii = make_data_array_hawaii(df["chl"], df["date"])
        
        
        # -------- Mean matrices --------  #
        
        self.chl_mean_seasonnal_matrix = make_mean_seasonnal_matrix(self, self.chl_array)
        self.l12_mean_seasonnal_matrix = make_mean_seasonnal_matrix(self, self.l12_array)
        self.PB_mean_seasonnal_matrix = make_mean_seasonnal_matrix(self, self.PB_array)
        
        
        # -------- Seasonnal means : blooming period --------  #
        
        self.chl_idxmax = (np.argmax(self.chl_mean_seasonnal_matrix, axis=1)+11) % 12
        self.l12_idxmax = (np.argmax(self.l12_mean_seasonnal_matrix, axis=1)+11) % 12
        self.PB_idxmax  = (np.argmax(self.PB_mean_seasonnal_matrix , axis=1)+11) % 12


        # -------- Fitting a gaussian to the biomass (chl) profile -------- #
        
        self.B0_arr, self.H_arr, self.zm_arr, self.sigma_arr = parameters_array(self)
        
        self.B0f, self.idx_of_removed_B0 = remove_outliers(self.B0_arr, l_negative=1, qu=3, ql=13)
        self.Hf, self.idx_of_removed_H = remove_outliers(self.H_arr, l_negative=0, qu=10, ql=3)
        self.zmf, self.idx_of_removed_zm = remove_outliers(self.zm_arr, l_negative=0, qu=1, ql=1)
        self.sigmaf, self.idx_of_removed_sigma = remove_outliers(self.sigma_arr, l_negative=0, qu=10, ql=5)
        
        # cumulative sums of the normalized deviations #
        self.B0N = (self.B0f-np.nanmean(self.B0f))/np.nanstd(self.B0f)
        self.B0NT = cumulative_sum(self.B0N)

        self.HN = (self.Hf-np.nanmean(self.Hf))/np.nanstd(self.Hf)
        self.HNT = cumulative_sum(self.HN)
        
        self.zmN = (self.zmf-np.nanmean(self.zmf))/np.nanstd(self.zmf)
        self.zmNT = cumulative_sum(self.zmN)
        
        self.sigmaN = (self.sigmaf-np.nanmean(self.sigmaf))/np.nanstd(self.sigmaf)
        self.sigmaNT = cumulative_sum(self.sigmaN)
        

        
d = Data()


#%% -------- Plots for each depth range -------- #

plot_l12(d, l_plot=0)
plot_chl(d, l_plot=0)
plot_PB (d, l_plot=0)



# -------- Distribution plots for each depth range -------- #

distribution_plots(d, d.chl_lst, title="chl", l_plot=0)
distribution_plots(d, d.l12_lst, title="l12", l_plot=0)
distribution_plots(d, d.PB_lst , title="PB" , l_plot=0)
        


# -------- Box plots -------- #

box_plot1(d, d.chl_lst, title="Chl", xlabel="chl ($mg / m^3$)", ylabel="depth", l_plot=0)
box_plot1(d, d.l12_lst, title="L12", xlabel="l12 (mg C / m3)", ylabel="depth", l_plot=0)
box_plot1(d, d.PB_lst , title="PB", xlabel="PB (mg C / mg chl)", ylabel="depth", l_plot=0)



# -------- Seasonnal deviations -------- #

plot_seasonnal_data_chl(d, plot_linear=0, plot_log=0, plot_nsdt=0)
plot_seasonnal_data_l12(d, plot_linear=0, plot_log=0, plot_nsdt=0)
plot_seasonnal_data_PB (d, plot_linear=0, plot_log=0, plot_nsdt=0)



# -------- Getting trends and slopes -------- #


# fit a straight line to the data at depth iz #
plot_reglin_chl(d, iz=0, l_plot=0)   
       

# plot the evolution of its slope over depth #
plot_slopes_chl(d, l_plot=0)
plot_slopes_l12(d, l_plot=0)
plot_slopes_PB (d, l_plot=0)


# same thing with a seasonnal analysis #
plot_reglin_chl_seasonnal(d, iz=3, it=11, l_plot=0)
plot_reglin_l12_seasonnal(d, iz=3, it=11, l_plot=0)
plot_reglin_PB_seasonnal (d, iz=3, it=11, l_plot=0)


# plot the evolution of their slopes over depth #
plot_seasonnal_time_slopes_chl(d, l_plot=0)
plot_seasonnal_time_slopes_l12(d, l_plot=0)
plot_seasonnal_time_slopes_PB (d, l_plot=0)


# seasonnal tendency over 10 yrs : slope matrix of the trends #
plot_chl_time_integrated_trend(d, l_plot=0)
plot_l12_time_integrated_trend(d, l_plot=0)
plot_PB_time_integrated_trend (d, l_plot=0)




#%% -------- Integration over depth of the time-series -------- #

# plot of the vertically integrated time-series #
plot_chlZ (d.chlZ, l_plot=0)
plot_l12Z (d.l12Z, l_plot=0)
plot_PBZ  (d.PBZ , l_plot=0)


# compararison to Hawaii integrated data #
plot_comparison_with_HOT_rect(d.sparse_chl_hawaii, d.chlZ_rect, l_plot=0)
plot_comparison_with_HOT(d.sparse_chl_hawaii, d.chlZ, l_plot=0)


# cumulative sums #
plot_chlZNT(d.chlZN, d.chlZNT, l_plot=0)
plot_l12ZNT(d.l12ZN, d.l12ZNT, l_plot=0)
plot_PBZNT (d.PBZN , d.PBZNT , l_plot=0)


# plot of the mean matrices #
plot_chl_mean_seasonnal_matrix(d, d.chl_mean_seasonnal_matrix, l_plot=0)
plot_l12_mean_seasonnal_matrix(d, d.l12_mean_seasonnal_matrix, l_plot=0)
plot_PB_mean_seasonnal_matrix (d, d.PB_mean_seasonnal_matrix , l_plot=0)


# get the mean of the integrated data for each month #
plot_monthly_mean_chlZ(d, d.chlZ, l_plot=0)
plot_monthly_mean_l12Z(d, d.l12Z, l_plot=0)
plot_monthly_mean_PBZ (d, d.PBZ , l_plot=0)



#%%  -------- Seasonnal means -------- #

# plot the surface mean 'sine like' chl curve (i.e. index 0) #
plot_monthly_mean_chl_surf(d, (d.chl_array)[0], l_plot=0)
plot_monthly_mean_l12_surf(d, (d.l12_array)[0], l_plot=0)
plot_monthly_mean_PB_surf (d, (d.PB_array )[0], l_plot=0)


# plot the mean chl curve for all depths #
plot_chl_mean_depth_seasonal_curve(d, l_plot=0)
plot_l12_mean_depth_seasonal_curve(d, l_plot=0)
plot_PB_mean_depth_seasonal_curve (d, l_plot=0)
        
        
# Bloom period : idx of the max of the means matrix at zi #
plot_bloom_chl(d, l_plot=0)
plot_bloom_l12(d, l_plot=0)
plot_bloom_PB (d, l_plot=0)
        

#%% -------- Fitting a gaussian to the biomass (chl) profile -------- #

# mean profiles #

plot_chl_profile_for_all_yrs(d, l_plot=0)
plot_chl_profile_for_some_yrs(d, L_yrs_to_plot=[0, 10 , 20, 30], l_plot=0)


# regressions #

plot_gaussian_regression_month_i(d, 87, l_plot=0)
  

# histograms raw data #
distribution_plot2(d, d.B0_arr, title="B0", l_plot=0)
distribution_plot2(d, d.H_arr, title="H", l_plot=0)
distribution_plot2(d, d.zm_arr, title="zm", l_plot=0)
distribution_plot2(d, d.sigma_arr, title="sigma", l_plot=0)


# histograms filtered data #

distribution_plot2(d, d.B0f, title="B0 filtered", xlabel= "B0 (mg Chl/m3)", bins=30, kde=True, l_plot=0)
distribution_plot2(d, d.Hf, title="H filtered", xlabel="H (mg Chl/m3)" , bins=30, kde=True, l_plot=0)
distribution_plot2(d, d.zmf, title="zm filtered", xlabel="zm (m)" , bins=30, kde=True, l_plot=0)
distribution_plot2(d, d.sigmaf, title="sigma filtered", xlabel="sigma (m)", bins=30, kde=True, l_plot=0)


# linear regressions #

plot_reglin_B0(d, d.B0f, l_plot=0)
plot_reglin_H(d, d.Hf, l_plot=0)
plot_reglin_zm(d, d.zmf, l_plot=0)
plot_reglin_sigma(d, d.sigmaf, l_plot=0)


# cumulative sums of the normalized deviations #
d.B0N = (d.B0f-np.nanmean(d.B0f))/np.nanstd(d.B0f)
d.B0NT = cumulative_sum(d.B0N)
plot_cumulative_sum_B0N(d.B0N, l_plot=0)

d.HN = (d.Hf-np.nanmean(d.Hf))/np.nanstd(d.Hf)
d.HNT = cumulative_sum(d.HN)
plot_cumulative_sum_HN(d.HN, l_plot=0)

d.zmN = (d.zmf-np.nanmean(d.zmf))/np.nanstd(d.zmf)
d.zmNT = cumulative_sum(d.zmN)
plot_cumulative_sum_zmN(d.zmN, l_plot=0)

d.sigmaN = (d.sigmaf-np.nanmean(d.sigmaf))/np.nanstd(d.sigmaf)
d.sigmaNT = cumulative_sum(d.sigmaN)
plot_cumulative_sum_sigmaN(d.sigmaN, l_plot=0)




#%% -------- Compare the theoretical gaussian values to mesurements -------- #

# load all data #
Btheo_all = make_theoretical_B_array_all_data(d)
B_mes_all = d.chl_array

# isolate removed data #
idx_of_removed_B0 = d.idx_of_removed_B0 
Btheo_discarded = Btheo_all[:, idx_of_removed_B0]
B_mes_discarded = B_mes_all[:, idx_of_removed_B0]

# isolate filtered data #
Btheo_f = np.array([Btheo_all[:, i] for i in range(d.p) if ~np.isin(i,idx_of_removed_B0)]).T
B_mes_f = np.array([B_mes_all[:, i] for i in range(d.p) if ~np.isin(i,idx_of_removed_B0)]).T



if 0:
        
    # Linear regression of all data #
    
    coeffs, Cov = np.polyfit(B_mes_all.flatten(), Btheo_all.flatten(), deg=1, cov=1)
    R2 = r_squared(Cov)
    
    poly = np.poly1d(coeffs)
    
    plt.plot(B_mes_all.flatten(), Btheo_all.flatten(), ".")
    plt.plot(B_mes_all.flatten(), poly(B_mes_all.flatten()), label=f"a {round(coeffs[0], 6)}   b {round(coeffs[1], 1)}   R2 {round(R2, 2)}", c="C1")
    
    plt.title(f"B0 gauss Vs B0 mesured : {round(100*coeffs[0],1)} % slope")
    plt.legend()
    plt.xlabel("# months")
    plt.ylabel("B0 (mg Chl/m3)")
    plt.show()
    
    
    
    # Linear regression of filtered data #
    
    coeffs, Cov = np.polyfit(B_mes_f.flatten(), Btheo_f.flatten(), deg=1, cov=1)
    R2 = r_squared(Cov)
    
    poly = np.poly1d(coeffs)
    
    plt.plot(B_mes_f.flatten(), Btheo_f.flatten(), ".")
    plt.plot(B_mes_f.flatten(), poly(B_mes_f.flatten()), label=f"a {round(coeffs[0], 4)}   b {round(coeffs[1], 1)}   R2 {round(R2, 2)}", c="C1")
    
    plt.title(f"B0 gauss (no outliers) Vs B0 mesured : {round(100*coeffs[0],1)} % slope")
    plt.legend()
    plt.xlabel("# months")
    plt.ylabel("B0 (mg Chl/m3)")
    plt.show()
    
    
    
    # Linear regression of outliers #
    
    coeffs, Cov = np.polyfit(B_mes_discarded.flatten(), Btheo_discarded.flatten(), deg=1, cov=1)
    R2 = r_squared(Cov)
    
    poly = np.poly1d(coeffs)
    
    plt.plot(B_mes_discarded.flatten(), Btheo_discarded.flatten(), ".")
    plt.plot(B_mes_discarded.flatten(), poly(B_mes_discarded.flatten()), label=f"a {round(coeffs[0], 4)}   b {round(coeffs[1], 1)}   R2 {round(R2, 2)}", c="C1")
    
    plt.title(f"B0 gauss (only outliers) Vs B0 mesured : {round(100*coeffs[0],1)} % slope")
    plt.legend()
    plt.xlabel("# months")
    plt.ylabel("B0 (mg Chl/m3)")
    plt.show()



def plot_Bmesured_VS_Bgauss_all_data(B_mes_all, Btheo_all, l_plot=0):
    if l_plot:
        # Linear regression of all data #
        
        coeffs, Cov = np.polyfit(B_mes_all.flatten(), Btheo_all.flatten(), deg=1, cov=1)
        R2 = r_squared(Cov)
        
        poly = np.poly1d(coeffs)
        
        plt.plot(B_mes_all.flatten(), Btheo_all.flatten(), ".")
        plt.plot(B_mes_all.flatten(), poly(B_mes_all.flatten()), label=f"a {round(coeffs[0], 6)}   b {round(coeffs[1], 1)}   R2 {round(R2, 2)}", c="C1")
        
        plt.title(f"B0 gauss Vs B0 mesured : {round(100*coeffs[0],1)} % slope")
        plt.legend()
        plt.xlabel("B0 mesured")
        plt.ylabel("B0 gauss")
        plt.show()



def plot_Bmesured_VS_Bgauss_filtered_and_outliers(B_mes_discarded, Btheo_discarded, B_mes_f, Btheo_f, l_plot=0):
    if l_plot:
        # Linear regression of outliers #
        fig, ax = plt.subplots()
        coeffs, Cov = np.polyfit(B_mes_discarded.flatten(), B_mes_discarded.flatten(), deg=1, cov=1)
        R2 = r_squared(Cov)
        
        poly = np.poly1d(coeffs)
        labl1, labl2 = "outliers",  f"a {round(coeffs[0], 2)}   b {round(coeffs[1], 1)}   R2 {round(R2, 2)}"
        
    
        isortd = np.argsort(B_mes_discarded.flatten())   
        ax.loglog(B_mes_discarded.flatten()[isortd], Btheo_discarded.flatten()[isortd], ".", color="r", label=labl1)
        ax.loglog(B_mes_discarded.flatten()[isortd], poly(B_mes_discarded.flatten()[isortd]), label=labl2, color="C1")
    
    
        
        # Linear regression of filtered data #
        
        coeffs, Cov = np.polyfit(B_mes_f.flatten(), Btheo_f.flatten(), deg=1, cov=1)
        R2 = r_squared(Cov)
        
        poly = np.poly1d(coeffs)
        labl3, labl4 = "filtered data",  f"a {round(coeffs[0], 2)}   b {round(coeffs[1], 1)}   R2 {round(R2, 2)}"
    
    
        isortd = np.argsort(B_mes_f.flatten())
        ax.loglog(B_mes_f.flatten()[isortd], Btheo_f.flatten()[isortd], ".", color="C0", label=labl3)
        ax.loglog(B_mes_f.flatten()[isortd], poly(B_mes_f.flatten()[isortd]), color="C2", label=labl4)
        
        
        
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [2, 3, 0, 1]
        ax.set_title(f"B0 gauss Vs B0 mesured")
        ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
        
        
        ax.set_xlabel("B0 mesured")
        ax.set_ylabel("B0 gauss")
        plt.show()
   
        
plot_Bmesured_VS_Bgauss_all_data(B_mes_all, Btheo_all, l_plot=0)

plot_Bmesured_VS_Bgauss_filtered_and_outliers(B_mes_discarded, Btheo_discarded, B_mes_f, Btheo_f, l_plot=1)