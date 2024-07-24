#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 23:09:41 2024

@authors: Vaibhava Srivastava, Claus Kadelka
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numba import njit
 
plt.rcParams['text.usetex'] = False

@njit
def RK4(func, X0, ts, beta_H, beta_W, h, replacement_rate, kappa, R_H, R_W, CFR_H, CFR_W, mu, tau, resource_capacity, r_I):
    """
    Runge Kutta 4 solver.
    """
    dt = ts[1] - ts[0]
    nt = len(ts)
    X  = np.zeros((nt, len(X0)),dtype=np.float64)
    X[0,:] = X0
    p_F = X0[0] + X0[3]
    p_HM = X0[1] + X0[4]
    p_WM = X0[2] + X0[5]
    rho = 2 * p_F * p_HM / (p_F**2 + p_HM**2) #this is scale invariant, i.e., it does not matter what p_F + p_HM is
    for i in range(nt-1):
        k1 = func(X[i], ts[i],beta_H, beta_W, h, replacement_rate, kappa, R_H, R_W, CFR_H, CFR_W, mu, tau, resource_capacity, rho, p_F, p_HM, p_WM, r_I)
        k2 = func(X[i] + dt/2. * k1, ts[i] + dt/2.,beta_H, beta_W, h, replacement_rate, kappa, R_H, R_W, CFR_H, CFR_W, mu, tau, resource_capacity, rho, p_F, p_HM, p_WM, r_I)
        k3 = func(X[i] + dt/2. * k2, ts[i] + dt/2.,beta_H, beta_W, h, replacement_rate, kappa, R_H, R_W, CFR_H, CFR_W, mu, tau, resource_capacity, rho, p_F, p_HM, p_WM, r_I)
        k4 = func(X[i] + dt    * k3, ts[i] + dt,beta_H, beta_W, h, replacement_rate, kappa, R_H, R_W, CFR_H, CFR_W, mu, tau, resource_capacity, rho, p_F, p_HM, p_WM, r_I)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    return X
 
@njit
def ODE(x,t,beta_H, beta_W, h, replacement_rate, kappa, R_H, R_W, CFR_H, CFR_W, mu, tau, resource_capacity, rho, p_F, p_HM, p_WM, r_I):
    #in the manuscript, the relative level of replacement of infected individuals is fixed at zero, that is r_I = 0
    S_F, S_HM, S_WM, I_F, I_HM, I_WM, R_F, R_HM, R_WM, D_F, D_HM, D_WM = x
    N_F = S_F+I_F+R_F+D_F
    N_HM = S_HM+I_HM+R_HM+D_HM
    if I_F + I_HM <= tau * resource_capacity:
        phi_H = 1
    else:
        phi_H = tau * resource_capacity / (I_F + I_HM)
    if I_WM <= (1-tau) * resource_capacity:
        phi_W = 1
    else:
        phi_W = (1-tau) * resource_capacity / I_WM
        
    dx = np.zeros(12,dtype=np.float64)
    dx[0] = -beta_H * S_F * ((1+h*N_HM/N_F) * I_F + (1-h) * I_HM) / (N_F+N_HM);
    dx[1] = -beta_H * S_HM * ((1-h) * I_F + (1+h*N_F/N_HM) * I_HM) / (N_F+N_HM) - replacement_rate * S_HM + kappa * replacement_rate * S_WM;
    dx[2] = -beta_W * S_WM * I_WM/p_WM + replacement_rate * S_HM - kappa * replacement_rate * S_WM;
    dx[3] = beta_H * S_F * ((1+h*N_HM/N_F) * I_F + (1-h) * I_HM) / (N_F+N_HM) - mu  * phi_H * I_F - R_H * I_F;
    dx[4] = beta_H * S_HM * ((1-h) * I_F + (1+h*N_F/N_HM) * I_HM) / (N_F+N_HM) - mu  * phi_H * I_HM - R_H * I_HM - r_I * replacement_rate * I_HM + r_I * kappa * replacement_rate * I_WM;
    dx[5] = beta_W * S_WM * I_WM/p_WM - mu * phi_W * I_WM - R_W * I_WM + replacement_rate * I_HM + r_I * replacement_rate * I_HM - r_I * kappa * replacement_rate * I_WM;
    dx[6] = mu  * phi_H * I_F + R_H * (1 - CFR_H) * I_F;
    dx[7] = mu  * phi_H * I_HM + R_H * (1 - CFR_H) * I_HM - replacement_rate * R_HM + kappa * replacement_rate * R_WM;
    dx[8] = mu *  phi_W * I_WM + R_W * (1 - CFR_W) * I_WM + replacement_rate * R_HM - kappa * replacement_rate * R_WM;
    dx[9] = R_H * CFR_H * I_F;
    dx[10] = R_H * CFR_H * I_HM;
    dx[11] = R_W * CFR_W * I_WM;
    return dx
 

def infer_index_given_min_max_number(value,min_value,max_value,number):
    dx = (number-1)/(max_value-min_value)
    return (value-min_value) * dx
 
def infer_ticks(ticks,parameter_values):
    min_value = min(parameter_values)
    max_value = max(parameter_values)
    number = len(parameter_values)
    return np.array([infer_index_given_min_max_number(el,min_value,max_value,number) for el in ticks])

def find_total_deaths_extreme_cases(ODE,initial_conditions,ts,beta_H, beta_W, h, replacement_rate, kappa, R_H, R_W, CFR_H, CFR_W, mu, r_I):
    tau_values = np.linspace(0, 1, 101);
    res = []
    for j,tau in enumerate(tau_values):
        sol = RK4(ODE,initial_conditions,ts,beta_H, beta_W, h, replacement_rate, kappa, R_H, R_W, CFR_H, CFR_W, mu, tau, resource_capacity=1e6, r_I=r_I)
        total_deaths = sum(sol[-1,-3:])
        res.append(total_deaths)
    total_deaths_optimal_treatment_both = min(res)
    total_deaths_only_treat_war = res[0]
    total_deaths_only_treat_home = res[-1] 
    sol = RK4(ODE,initial_conditions,ts,beta_H, beta_W, h, replacement_rate, kappa, R_H, R_W, CFR_H, CFR_W, mu=0, tau=0.5, resource_capacity=0, r_I=r_I)
    total_deaths_no_treatment = sum(sol[-1,-3:])
    return total_deaths_optimal_treatment_both,total_deaths_only_treat_war,total_deaths_only_treat_home,total_deaths_no_treatment

def find_FES_extreme_cases(ODE,initial_conditions,ts,beta_H, beta_W, h, replacement_rate, kappa, R_H, R_W, CFR_H, CFR_W, mu, r_I):
    tau_values = np.linspace(0, 1, 101);
    res = []
    for j,tau in enumerate(tau_values):
        sol = RK4(ODE,initial_conditions,ts,beta_H, beta_W, h, replacement_rate, kappa, R_H, R_W, CFR_H, CFR_W, mu, tau, resource_capacity=1e6, r_I=r_I)
        FES = 1-sum(sol[-1,:3])
        res.append(FES)
    FES_optimal_treatment_both = min(res)
    sol = RK4(ODE,initial_conditions,ts,beta_H, beta_W, h, replacement_rate, kappa, R_H, R_W, CFR_H, CFR_W, mu=0, tau=0.5, resource_capacity=0, r_I=r_I)
    FES_no_treatment = 1-sum(sol[-1,:3])
    return FES_optimal_treatment_both,FES_no_treatment


def gss(ODE,initial_conditions,ts, beta_H, beta_W, h, replacement_rate, kappa, R_H, R_W, CFR_H, CFR_W, mu, resource_capacity,r_I, tolerance=1e-3,tau_min = 0,tau_max = 1):
    """
    adapted from https://en.wikipedia.org/wiki/Golden-section_search
    """
    invphi = (np.sqrt(5) - 1) / 2  # 1 / phi
    while abs(tau_max - tau_min) > tolerance:
        potentially_new_tau_min = tau_max - (tau_max - tau_min) * invphi
        potentially_new_tau_max = tau_min + (tau_max - tau_min) * invphi
        
        sol = RK4(ODE,initial_conditions,ts,beta_H, beta_W, h, replacement_rate, kappa, R_H, R_W, CFR_H, CFR_W, mu, potentially_new_tau_min, resource_capacity,r_I)
        total_deaths_at_potentially_new_tau_min = sum(sol[-1,-3:])
        sol = RK4(ODE,initial_conditions,ts,beta_H, beta_W, h, replacement_rate, kappa, R_H, R_W, CFR_H, CFR_W, mu, potentially_new_tau_max, resource_capacity,r_I)
        total_deaths_at_potentially_new_tau_max = sum(sol[-1,-3:])
        
        if total_deaths_at_potentially_new_tau_min < total_deaths_at_potentially_new_tau_max:
            tau_max = potentially_new_tau_max
        else:  # f(c) > f(d) to find the maximum
            tau_min = potentially_new_tau_min

    return (tau_min + tau_max) / 2,(total_deaths_at_potentially_new_tau_min+total_deaths_at_potentially_new_tau_max)/2

# Define parameters
beta_H = 0.6;
beta_W = 0.9;
h = 0.;
replacement_rate = 0.05;
kappa = 1;
R_H = 0.2;
R_W = R_H;
CFR_H = 0.05;
CFR_W = CFR_H;
mu = 0.2;
p_m = 0.5; #proportion male

#set time vector
dt = 0.25
T = 100#300
ts = np.linspace(0, T, int(T/dt)+1);

#set initial conditions
eps = 1e-04; #proportion initially infected
initial_conditions = np.array([(1-p_m)*(1-eps),kappa*p_m/(1+kappa)*(1-eps),p_m/(1+kappa)*(1-eps), (1-p_m)*eps,kappa*p_m/(1+kappa)*eps,p_m/(1+kappa)*eps, 0, 0, 0, 0, 0, 0],dtype=np.float64);




#explanatory plot
r_I=0
resource_capacity = 0.05
#generate an intution by looking at equi-distantly space tau values for a fixed treatment capacity (resource_capacity)
for tau in np.linspace(0.01,0.99,20):
    colors = ['r','b']
    lss = ['--','-']
    #beta_W = 0.6
    sol = RK4(ODE,initial_conditions,ts,beta_H, beta_W, h, replacement_rate, kappa, R_H, R_W, CFR_H, CFR_W, mu, tau, resource_capacity,r_I)
    f,ax = plt.subplots(figsize=(4,3))
    line1, = ax.plot([-10,T],[tau*resource_capacity,tau*resource_capacity],color=colors[0],ls=lss[0],label='at home')
    line2, = ax.plot([-10,T],[(1-tau)*resource_capacity,(1-tau)*resource_capacity],color=colors[1],ls=lss[0],label='at war')
    
    line3, = ax.semilogy(ts,sol[:,3]+sol[:,4],color=colors[0],ls=lss[1],label='at home')
    line4, = ax.semilogy(ts,sol[:,5],color=colors[1],ls=lss[1],label='at war')
    legend1 = ax.legend(handles=[line1,line2],frameon=False,title='treatment capacity  ',loc='center',bbox_to_anchor=[1.22,0.3])
    ax.add_artist(legend1)
    legend2 = ax.legend(handles=[line3,line4],frameon=False,title='infected',loc='center',bbox_to_anchor=[1.22,0.65])
    ax.set_ylim([1e-5,1])
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_ylabel('Infected (in % of total population)')
    ax.set_xlabel('Time')
    ax.set_xlim([-5,T+5])
    total_deaths = np.round(sum(sol[-1,-3:]*100),2)
    ax.text(T*0.48,0.9,r'$\tau$'+' ='+'\ndeaths ='+'\nFES =',ha='right',va='center')
    ax.text(T*0.52,0.9,str(tau)+'\n'+str(total_deaths)+'%\n'+str(np.round(100*(1-sum(sol[-1,:3])),2))+'%',ha='left',va='center')
    #plt.savefig('explanatory_plot_bH%s_bW%s_tau%s_c%s.pdf' % (str(beta_H),str(beta_W),str(tau),str(resource_capacity)),bbox_inches = "tight")   


#generate the same plot for specific tau values for a fixed treatment capacity (resource_capacity)
r_I=0
resource_capacity = 0.05
tau_values = [0.01,0.2,0.66,0.99]
n_tau_values = len(tau_values)
f,ax = plt.subplots(2,n_tau_values,figsize=(8,5),sharey='row',sharex=True,height_ratios=[2,1])
beta_W = 3*beta_H
assert R_H==R_W
for ii,tau in enumerate(tau_values):
    colors = ['r','b']
    lss = ['--','-']
    sol = RK4(ODE,initial_conditions,ts,beta_H, beta_W, h, replacement_rate, kappa, R_H, R_W, CFR_H, CFR_W, mu, tau, resource_capacity, r_I)
    line1, = ax[0,ii].plot([-10,T],[tau*resource_capacity,tau*resource_capacity],color=colors[0],ls=lss[0],label='at home')
    line2, = ax[0,ii].plot([-10,T],[(1-tau)*resource_capacity,(1-tau)*resource_capacity],color=colors[1],ls=lss[0],label='at war')
    
    line3, = ax[0,ii].semilogy(ts,sol[:,3]+sol[:,4],color=colors[0],ls=lss[1],label='at home')
    line4, = ax[0,ii].semilogy(ts,sol[:,5],color=colors[1],ls=lss[1],label='at war')

    argmax_home = (sol[:,3]+sol[:,4]).argmax()
    argmax_war = (sol[:,5]).argmax()
    line5, = ax[0,ii].plot([ts[argmax_home]],[7e-6],'ro',markerfacecolor='none',label='at home',clip_on=False)
    line6, = ax[0,ii].plot([ts[argmax_war]],[7e-6],'bx',label='at war',clip_on=False) 
    
    if ii==n_tau_values-1:
        legend1 = ax[0,ii].legend(handles=[line1,line2],frameon=False,title='Treatment capacity  ',loc='center',bbox_to_anchor=[1.5,0.8])
        ax[0,ii].add_artist(legend1)
        legend2 = ax[0,ii].legend(handles=[line3,line4],frameon=False,title='Infected',loc='center',bbox_to_anchor=[1.5,0.5])
        ax[0,ii].add_artist(legend2)
        legend3 = ax[0,ii].legend(handles=[line5,line6],frameon=False,title='Argmax infected',loc='center',bbox_to_anchor=[1.5,0.2])

    ax[0,ii].set_ylim([1e-5,1])
    ax[0,ii].spines[['right', 'top']].set_visible(False)
    if ii==0:
        ax[0,ii].set_ylabel('Infected (as proportion of total population)')
    ax[0,ii].set_xlim([-5,T+5])
    total_deaths = np.round(sum(sol[-1,-3:]*100),2)
    deaths_at_home = np.round(sum(sol[-1,-3:-1]*100),2)
    deaths_at_war = np.round((sol[-1,-1]*100),2)
    ax[0,ii].text(T*0.58,0.4,r'$\tau$'+' ='+'\nFES =\ndeaths =\nat home =\nat war =',ha='right',va='bottom')
    ax[0,ii].text(T*0.58,0.4,str(tau)+'\n'+str(np.round(100*(1-sum(sol[-1,:3])),2))+'%\n'+str(total_deaths)+'%\n'+str(deaths_at_home)+'%\n'+str(deaths_at_war)+'%',ha='left',va='bottom')

    treatment_efficiency_1 = np.minimum(1,tau * resource_capacity / (sol[:,3]+sol[:,4]))
    prob_natural_recovery_before_treatment_1 = R_H/(mu*treatment_efficiency_1 + R_H)
    CFR_H_eff = prob_natural_recovery_before_treatment_1 * CFR_H
    line1, = ax[1,ii].plot(ts,CFR_H_eff,'r:',lw=2,label='at home') 
    treatment_efficiency_2 = np.minimum(1,(1-tau) * resource_capacity / (sol[:,5]))
    prob_natural_recovery_before_treatment_2 = R_W/(mu*treatment_efficiency_2 + R_W)
    CFR_W_eff = prob_natural_recovery_before_treatment_2 * CFR_W
    line2, = ax[1,ii].plot(ts,CFR_W_eff,'b',ls='-.',label='at war') 
    if ii==n_tau_values-1:
        legend1 = ax[1,ii].legend(handles=[line1,line2],frameon=False,title='effective CFR',loc='center',bbox_to_anchor=[1.5,0.5])
    ax[1,ii].spines[['right', 'top']].set_visible(False)
    if ii==0:
        ax[1,ii].set_ylabel('Effective CFR')    
    ax[1,ii].set_xlabel('Time')

plt.savefig('explanatory_plot_CFR_bH%s_bW%s_h%s_r%s_kappa%s_tauall_c%s.pdf' % (str(beta_H),str(beta_W),str(h),str(R_H),str(kappa),str(resource_capacity)),bbox_inches = "tight")   

#generate the same plot for another set of specific tau values for a fixed treatment capacity (resource_capacity), this time for beta_W = 3*beta_H
r_I=0
resource_capacity = 0.05
tau_values = [0.01,0.79,0.88,0.99]
n_tau_values = len(tau_values)
f,ax = plt.subplots(2,n_tau_values,figsize=(8,5),sharey='row',sharex=True,height_ratios=[2,1])
beta_W = beta_H
assert R_H==R_W
for ii,tau in enumerate(tau_values):
    colors = ['r','b']
    lss = ['--','-']
    sol = RK4(ODE,initial_conditions,ts,beta_H, beta_W, h, replacement_rate, kappa, R_H, R_W, CFR_H, CFR_W, mu, tau, resource_capacity, r_I)
    line1, = ax[0,ii].plot([-10,T],[tau*resource_capacity,tau*resource_capacity],color=colors[0],ls=lss[0],label='at home')
    line2, = ax[0,ii].plot([-10,T],[(1-tau)*resource_capacity,(1-tau)*resource_capacity],color=colors[1],ls=lss[0],label='at war')
    
    line3, = ax[0,ii].semilogy(ts,sol[:,3]+sol[:,4],color=colors[0],ls=lss[1],label='at home')
    line4, = ax[0,ii].semilogy(ts,sol[:,5],color=colors[1],ls=lss[1],label='at war')

    argmax_home = (sol[:,3]+sol[:,4]).argmax()
    argmax_war = (sol[:,5]).argmax()
    line5, = ax[0,ii].plot([ts[argmax_home]],[7e-6],'ro',markerfacecolor='none',label='at home',clip_on=False)
    line6, = ax[0,ii].plot([ts[argmax_war]],[7e-6],'bx',label='at war',clip_on=False)
    
    if ii==n_tau_values-1:
        legend1 = ax[0,ii].legend(handles=[line1,line2],frameon=False,title='Treatment capacity  ',loc='center',bbox_to_anchor=[1.5,0.8])
        ax[0,ii].add_artist(legend1)
        legend2 = ax[0,ii].legend(handles=[line3,line4],frameon=False,title='Infected',loc='center',bbox_to_anchor=[1.5,0.5])
        ax[0,ii].add_artist(legend2)
        legend3 = ax[0,ii].legend(handles=[line5,line6],frameon=False,title='Argmax infected',loc='center',bbox_to_anchor=[1.5,0.2])

    ax[0,ii].set_ylim([1e-5,1])
    ax[0,ii].spines[['right', 'top']].set_visible(False)
    if ii==0:
        ax[0,ii].set_ylabel('Infected (as proportion of total population)')
    ax[0,ii].set_xlim([-5,T+5])
    total_deaths = np.round(sum(sol[-1,-3:]*100),2)
    deaths_at_home = np.round(sum(sol[-1,-3:-1]*100),2)
    deaths_at_war = np.round((sol[-1,-1]*100),2)
    ax[0,ii].text(T*0.58,0.4,r'$\tau$'+' ='+'\nFES =\ndeaths =\nat home =\nat war =',ha='right',va='bottom')
    ax[0,ii].text(T*0.58,0.4,str(tau)+'\n'+str(np.round(100*(1-sum(sol[-1,:3])),2))+'%\n'+str(total_deaths)+'%\n'+str(deaths_at_home)+'%\n'+str(deaths_at_war)+'%',ha='left',va='bottom')

    treatment_efficiency_1 = np.minimum(1,tau * resource_capacity / (sol[:,3]+sol[:,4]))
    prob_natural_recovery_before_treatment_1 = R_H/(mu*treatment_efficiency_1 + R_H)
    CFR_H_eff = prob_natural_recovery_before_treatment_1 * CFR_H
    line1, = ax[1,ii].plot(ts,CFR_H_eff,'r:',lw=2,label='at home') 
    treatment_efficiency_2 = np.minimum(1,(1-tau) * resource_capacity / (sol[:,5]))
    prob_natural_recovery_before_treatment_2 = R_W/(mu*treatment_efficiency_2 + R_W)
    CFR_W_eff = prob_natural_recovery_before_treatment_2 * CFR_W
    line2, = ax[1,ii].plot(ts,CFR_W_eff,'b',ls='-.',label='at war') 
    if ii==n_tau_values-1:
        legend1 = ax[1,ii].legend(handles=[line1,line2],frameon=False,title='effective CFR',loc='center',bbox_to_anchor=[1.5,0.5])
    ax[1,ii].spines[['right', 'top']].set_visible(False)
    if ii==0:
        ax[1,ii].set_ylabel('Effective CFR')    
    ax[1,ii].set_xlabel('Time')

plt.savefig('explanatory_plot_CFR_bH%s_bW%s_h%s_r%s_kappa%s_tauall_c%s.pdf' % (str(beta_H),str(beta_W),str(h),str(R_H),str(kappa),str(resource_capacity)),bbox_inches = "tight")   








#explanatory plot for the finding that tau^star shifts so much for h=1
r_I=0
h = 1.
resource_capacity_values_total = [0.0002,0.005,0.025,0.15]
tau_values = [.21,0.4,0.9,0.99]
for resource_capacity in resource_capacity_values_total:
    resource_capacity_values = [resource_capacity]*4
    n_tau_values = len(tau_values)
    f,ax = plt.subplots(2,n_tau_values,figsize=(8,5),sharey='row',sharex=True,height_ratios=[2,1])
    beta_W = 3*beta_H
    assert R_H==R_W
    for ii,(tau,resource_capacity) in enumerate(zip(tau_values,resource_capacity_values)):
        colors = ['r','b','']
        lss = ['--','-','-.',':']
        sol = RK4(ODE,initial_conditions,ts,beta_H, beta_W, h, replacement_rate, kappa, R_H, R_W, CFR_H, CFR_W, mu, tau, resource_capacity, r_I)
        line1, = ax[0,ii].plot([-10,T],[tau*resource_capacity,tau*resource_capacity],color=colors[0],ls=lss[0],label='at home')
        line2, = ax[0,ii].plot([-10,T],[(1-tau)*resource_capacity,(1-tau)*resource_capacity],color=colors[1],ls=lss[0],label='at war')
        
    
        #line3, = ax[0,ii].semilogy(ts,sol[:,3] + sol[:,4],color=colors[0],ls=lss[1],label='at home')
        line4, = ax[0,ii].semilogy(ts,sol[:,5],color=colors[1],ls=lss[3],lw=2,label='male at war')
        line7, = ax[0,ii].semilogy(ts,sol[:,3],color=colors[0],ls=lss[2],label='female at home')
        line8, = ax[0,ii].semilogy(ts,sol[:,4],color=colors[0],ls=lss[1],label='male at home')
    
        argmax_home_female = (sol[:,3]).argmax()
        argmax_home_male = (sol[:,4]).argmax()
        argmax_war = (sol[:,5]).argmax()
        line5, = ax[0,ii].plot([ts[argmax_home_female]],[7e-6],'ro',markerfacecolor='none',label='female at home',clip_on=False)
        line6, = ax[0,ii].plot([ts[argmax_war]],[7e-6],'bx',label='male at war',clip_on=False) 
        line9, = ax[0,ii].plot([ts[argmax_home_male]],[7e-6],'rx',label='male at home',clip_on=False)
        
        if ii==n_tau_values-1:
            legend1 = ax[0,ii].legend(handles=[line1,line2],frameon=False,title='Treatment capacity  ',loc='center',bbox_to_anchor=[1.65,0.9])
            ax[0,ii].add_artist(legend1)
            legend2 = ax[0,ii].legend(handles=[line7,line8,line4],frameon=False,title='Infected',loc='center',bbox_to_anchor=[1.65,0.52])
            ax[0,ii].add_artist(legend2)
            legend3 = ax[0,ii].legend(handles=[line5,line9,line6],frameon=False,title='Argmax infected',loc='center',bbox_to_anchor=[1.65,0.1])
    
        ax[0,ii].set_ylim([1e-5,1])
        ax[0,ii].spines[['right', 'top']].set_visible(False)
        if ii==0:
            ax[0,ii].set_ylabel('Infected (as proportion of total population)')
        ax[0,ii].set_xlim([-5,T+5])
        total_deaths = np.round(sum(sol[-1,-3:]*100),2)
        deaths_at_home = np.round(sum(sol[-1,-3:-1]*100),2)
        deaths_at_war = np.round((sol[-1,-1]*100),2)
        ax[0,ii].text(T*0.58,0.4,r'$\tau$'+' ='+'\nc =\nFES =\ndeaths =\nat home =\nat war =',ha='right',va='bottom')
        ax[0,ii].text(T*0.58,0.4,str(tau)+'\n'+str(resource_capacity)+'\n'+str(np.round(100*(1-sum(sol[-1,:3])),2))+'%\n'+str(total_deaths)+'%\n'+str(deaths_at_home)+'%\n'+str(deaths_at_war)+'%',ha='left',va='bottom')
    
        treatment_efficiency_1 = np.minimum(1,tau * resource_capacity / (sol[:,3]+sol[:,4]))
        prob_natural_recovery_before_treatment_1 = R_H/(mu*treatment_efficiency_1 + R_H)
        CFR_H_eff = prob_natural_recovery_before_treatment_1 * CFR_H
        line1, = ax[1,ii].plot(ts,CFR_H_eff,'r:',lw=2,label='at home') 
        treatment_efficiency_2 = np.minimum(1,(1-tau) * resource_capacity / (sol[:,5]))
        prob_natural_recovery_before_treatment_2 = R_W/(mu*treatment_efficiency_2 + R_W)
        CFR_W_eff = prob_natural_recovery_before_treatment_2 * CFR_W
        line2, = ax[1,ii].plot(ts,CFR_W_eff,'b',ls='-.',label='at war') 
        if ii==n_tau_values-1:
            legend1 = ax[1,ii].legend(handles=[line1,line2],frameon=False,title='effective CFR',loc='center',bbox_to_anchor=[1.65,0.5])
        ax[1,ii].spines[['right', 'top']].set_visible(False)
        if ii==0:
            ax[1,ii].set_ylabel('Effective CFR')    
        ax[1,ii].set_xlabel('Time')
    
    plt.savefig('explanatory_plot_fig4_bH%s_bW%s_h%s_r%s_kappa%s_tauselected_c%s.pdf' % (str(beta_H),str(beta_W),str(h),str(R_H),str(kappa),str(resource_capacity)),bbox_inches = "tight")   
    









#2 subplots, mortality, main figure
r_I = 0


tau_values = np.linspace(0, 1, 301);
beta_H = 0.6
beta_W = 3*beta_H
kappa=1.
resource_capacity_values = [0,0.05,0.0002,0.15,0.001,0.3,0.005,0.01,0.02,0.03]; #resorted for nice legend appearance
R_H = 0.2
R_W = 0.2

f,ax = plt.subplots(1,2,sharey=True,figsize=(8,3.5))
for ii,h in enumerate([0.,1.]):
    res = []
    for i,resource_capacity in enumerate(resource_capacity_values):
        res.append( [] )
        for j,tau in enumerate(tau_values):
            sol = RK4(ODE,initial_conditions,ts,beta_H, beta_W, h, replacement_rate, kappa, R_H, R_W, CFR_H, CFR_W, mu, tau, resource_capacity,r_I)
            total_deaths = sum(sol[-1,-3:])
            res[-1].append(total_deaths)
    res = 100*np.array(res)
    
    resource_capacity_values_full = np.logspace(-5,0,1000)
    res_full = []
    res_full_max_values = []
    accuracy=1e-5
    for i,resource_capacity in enumerate(resource_capacity_values_full):
        argmax,max_value = gss(ODE,initial_conditions,ts, beta_H, beta_W, h, replacement_rate, kappa, R_H, R_W, CFR_H, CFR_W, mu, resource_capacity,r_I, tolerance=accuracy)
        if i>0 and max_value==res_full_max_values[-1]:
            break
        res_full.append( argmax)
        res_full_max_values.append(max_value)
    res_full = np.array(res_full)
    res_full_max_values = 100*np.array(res_full_max_values)
    optimal_mortality_infinite_resources = 100*find_total_deaths_extreme_cases(ODE,initial_conditions,ts,beta_H, beta_W, h, replacement_rate, kappa, R_H, R_W, CFR_H, CFR_W, mu,r_I)[0]
    
    
    
    lines = []
    for i in range(len(resource_capacity_values)):
        lines.append(ax[ii].plot(tau_values,res[i],label=list(map(str,resource_capacity_values))[i],color = cm.tab10(i%10),ls='-' if i<10 else '-.')[0])
    ax[ii].plot(tau_values[np.argmin(res,1)][np.array([1,2,3,4,6,7,8,9])],np.min(res,1)[np.array([1,2,3,4,6,7,8,9])],'ko')
    ax[ii].plot(res_full,res_full_max_values,'k-',lw=2)
    line = ax[ii].plot([tau_values[np.argmin(res,1)][1]],[np.min(res,1)[1]],'ko-',label=r'$\tau^* =$ argmin $\tau$ at a fixed treatment capacity $c$')[0]
    if ii==0:
        ax[ii].set_ylabel('Total Mortality (in % of total population)')
    ax[ii].set_xlabel(r'Treatment resource allocation $\tau$')
    ax[ii].spines[['right', 'top']].set_visible(False)
    if ii==1:
        legend1 = ax[ii].legend(handles = lines,frameon=False,bbox_to_anchor = [-0.17,1.28],loc='upper center',ncol=7,title=r'Treatment capacity ($c$)')          
        ax[ii].add_artist(legend1)
        legend2 = ax[ii].legend(handles=[line],frameon=False,loc='upper center',bbox_to_anchor = [0.376,1.128])
        ax[ii].add_artist(legend2)
    ax[ii].set_ylim([1.2,4.9])
    xticklabels = ax[ii].get_xticklabels()
    xticklabels[list(ax[ii].get_xticks()).index(0)].set_text('0\nall at war')
    xticklabels[list(map(lambda el: round(el,2),ax[ii].get_xticks())).index(1.)].set_text('1\nall at home')
    ax[ii].set_xticklabels(xticklabels)
plt.savefig('fig5_b2b1_%s_hbothextremes_k_%s_gamma_%s.pdf' % (str(np.round(beta_W/beta_H,2)),str(kappa),str(R_H)),bbox_extra_artists=(legend1,legend2,),bbox_inches = "tight")   





#2 subplots, FES, main figure
r_I = 0


tau_values = np.linspace(0, 1, 301);
beta_H = 0.6
beta_W = 3*beta_H
kappa=1.
resource_capacity_values = [0,0.05,0.0002,0.15,0.001,0.3,0.005,0.01,0.02,0.03]; #resorted for nice legend appearance
R_H = 0.2
R_W = 0.2

f,ax = plt.subplots(1,2,sharey=True,figsize=(8,3.5))
for ii,h in enumerate([0.,1.]):
    res = []
    for i,resource_capacity in enumerate(resource_capacity_values):
        res.append( [] )
        for j,tau in enumerate(tau_values):
            #sol = integrate.odeint(ODE, initial_conditions, ts, args=(beta_H, beta_W, h, replacement_rate, kappa, R_H, R_W, CFR_H, CFR_W, mu, tau, resource_capacity))
            sol = RK4(ODE,initial_conditions,ts,beta_H, beta_W, h, replacement_rate, kappa, R_H, R_W, CFR_H, CFR_W, mu, tau, resource_capacity,r_I)
            FES = 1-sum(sol[-1,:3])
            res[-1].append(FES)
    res = 100*np.array(res)
    
    resource_capacity_values_full = np.logspace(-5,0,200)
    res_full = []
    for i,resource_capacity in enumerate(resource_capacity_values_full):
        res_full.append( [] )
        for j,tau in enumerate(tau_values):
            #sol = integrate.odeint(ODE, initial_conditions, ts, args=(beta_H, beta_W, h, replacement_rate, kappa, R_H, R_W, CFR_H, CFR_W, mu, tau, resource_capacity))
            sol = RK4(ODE,initial_conditions,ts,beta_H, beta_W, h, replacement_rate, kappa, R_H, R_W, CFR_H, CFR_W, mu, tau, resource_capacity,r_I)
            FES = 1-sum(sol[-1,:3])
            res_full[-1].append(FES)
    res_full = 100*np.array(res_full)
    optimal_FES_infinite_resources = 100*find_FES_extreme_cases(ODE,initial_conditions,ts,beta_H, beta_W, h, replacement_rate, kappa, R_H, R_W, CFR_H, CFR_W, mu,r_I)[0]
    which = np.min(res_full,1) - optimal_FES_infinite_resources > 1e-7
    which = np.arange(list(np.min(res_full,1)[:-1]-np.min(res_full,1)[1:]<1e-9).index(True))
    which = resource_capacity_values_full<=(0.02 if ii==0 else 0.05)
    
    
    lines = []
    for i in range(len(resource_capacity_values)):
        lines.append(ax[ii].plot(tau_values,res[i],label=list(map(str,resource_capacity_values))[i],color = cm.tab10(i%10),ls='-' if i<10 else '-.')[0])
    which2 = [1,2,4,6,7,8,9] if ii==1 else [2,4,6,7,8,9]
    ax[ii].plot(tau_values[np.argmin(res,1)][np.array(which2)],np.min(res,1)[np.array(which2)],'ko')
    ax[ii].plot(tau_values[np.argmin(res_full,1)][which],np.min(res_full,1)[which],'k-',lw=2)
    line = ax[ii].plot([tau_values[np.argmin(res,1)][2]],[np.min(res,1)[2]],'ko-',label=r'$\tau^* =$ argmin $\tau$ at a fixed treatment capacity $c$')[0]
    if ii==0:
        ax[ii].set_ylabel('Final epidemic size (in %)')
    ax[ii].set_xlabel(r'Treatment resource allocation $\tau$')
    ax[ii].spines[['right', 'top']].set_visible(False)
    if ii==1:
        legend1 = ax[ii].legend(handles = lines,frameon=False,bbox_to_anchor = [-0.17,1.3],loc='upper center',ncol=7,title=r'Treatment capacity ($c$)')          
        ax[ii].add_artist(legend1)
        legend2 = ax[ii].legend(handles=[line],frameon=False,loc='upper center',bbox_to_anchor = [0.376,1.148])
        ax[ii].add_artist(legend2)
    ax[ii].set_ylim([50,100])
    xticklabels = ax[ii].get_xticklabels()
    xticklabels[list(ax[ii].get_xticks()).index(0)].set_text('0\nall at war')
    xticklabels[list(map(lambda el: round(el,2),ax[ii].get_xticks())).index(1.)].set_text('1\nall at home')
    ax[ii].set_xticklabels(xticklabels)
plt.savefig('fig5_b2b1_FES_%s_hbothextremes_k_%s_gamma_%s.pdf' % (str(np.round(beta_W/beta_H,2)),str(kappa),str(R_H)),bbox_extra_artists=(legend1,legend2,),bbox_inches = "tight")   











#2 subplots, mortality, different CFRs
r_I = 0


tau_values = np.linspace(0, 1, 301);
beta_H = 0.6
beta_W = 3*beta_H
kappa=1.
resource_capacity_values = [0,0.0002,0.001,0.005,0.01,0.02,0.03,0.05,0.15,0.3];
resource_capacity_values = [0,0.05,0.0002,0.15,0.001,0.3,0.005,0.01,0.02,0.03];
R_H = 0.2
R_W = 0.2

CFR_values = [0.05,0.1]

colors = ['green','purple']
lss = ['-',':']
lws = [1,2]

f,ax = plt.subplots(1,2,sharey=True,figsize=(8,3.5))
for ii,h in enumerate([0.,1.]):
    resource_capacity_values_full = np.logspace(-5,0,100)
    for iii,CFR_H in enumerate(CFR_values):
        for jjj,CFR_W in enumerate(CFR_values):
            res_full = []
            res_full_max_values = []
            accuracy=1e-5
            for i,resource_capacity in enumerate(resource_capacity_values_full):
                argmax,max_value = gss(ODE,initial_conditions,ts, beta_H, beta_W, h, replacement_rate, kappa, R_H, R_W, CFR_H, CFR_W, mu, resource_capacity,r_I, tolerance=accuracy)
                if i>0 and max_value==res_full_max_values[-1]:
                    break
                res_full.append( argmax)
                res_full_max_values.append(max_value)
            res_full = np.array(res_full)
            res_full_max_values = 100*np.array(res_full_max_values)
            
    
            ax[ii].plot(res_full,res_full_max_values,color=colors[iii],ls=lss[jjj],lw=2)
            #line = ax[ii].plot([tau_values[np.argmin(res,1)][1]],[np.min(res,1)[1]],'ko-',label=r'$\tau^* =$ argmin $\tau$ at a fixed treatment capacity $c$')[0]
    if ii==0:
        ax[ii].set_ylabel('Total Mortality (in % of total population)')
    ax[ii].set_xlabel(r'Optimal treatment resource allocation $\tau^*$')
    ax[ii].spines[['right', 'top']].set_visible(False)            
    if ii==0:
        line1 = ax[ii].plot([0.5],[5],color=colors[0],ls='-',label=str(int(CFR_values[0]*100))+'%')[0]
        line2 = ax[ii].plot([0.5],[5],color=colors[1],ls='-',label=str(int(CFR_values[1]*100))+'%')[0]
        legend1 = ax[ii].legend(handles = [line1,line2],frameon=False,bbox_to_anchor = [0.25,1],loc='upper center',ncol=1,title=r'CFR${}_H$')          
        ax[ii].add_artist(legend1)
        line3 = ax[ii].plot([0.5],[5],color='k',ls=lss[0],lw=lws[0],label=str(int(CFR_values[0]*100))+'%')[0]
        line4 = ax[ii].plot([0.5],[5],color='k',ls=lss[1],lw=lws[1],label=str(int(CFR_values[1]*100))+'%')[0]
        legend2 = ax[ii].legend(handles=[line3,line4],frameon=False,loc='upper center',bbox_to_anchor = [0.6,1],ncol=1,title=r'CFR${}_W$')
        ax[ii].add_artist(legend2)
    xticklabels = ax[ii].get_xticklabels()
    xticklabels[list(ax[ii].get_xticks()).index(0)].set_text('0\nall at war')
    xticklabels[list(map(lambda el: round(el,2),ax[ii].get_xticks())).index(1.)].set_text('1\nall at home')
    ax[ii].set_xticklabels(xticklabels)
    #ax.set_title(r'$\beta_W/\beta_H =$ '+str(np.round(beta_W/beta_H,2)) +r', $h =$ '+str(h)+r', $\kappa =$ '+str(kappa)+', $\gamma =$ '+str(R_H))
plt.savefig('diffCFR_b2b1_%s_hbothextremes_k_%s_gamma_%s.pdf' % (str(np.round(beta_W/beta_H,2)),str(kappa),str(R_H)),bbox_extra_artists=(legend1,legend2,),bbox_inches = "tight")   
CFR_H = 0.05
CFR_W = 0.05



#2 subplots, FES, different CFRs
r_I = 0


tau_values = np.linspace(0, 1, 301);
beta_H = 0.6
beta_W = 3*beta_H
kappa=1.
resource_capacity_values = [0,0.0002,0.001,0.005,0.01,0.02,0.03,0.05,0.15,0.3];
resource_capacity_values = [0,0.05,0.0002,0.15,0.001,0.3,0.005,0.01,0.02,0.03];
R_H = 0.2
R_W = 0.2

CFR_values = [0.05,0.1]

colors = ['green','purple']
lss = ['-',':']
lws = [2,2]

f,ax = plt.subplots(1,2,sharey=True,figsize=(8,3.5))
for ii,h in enumerate([0.,1.]):
    resource_capacity_values_full = np.logspace(-5,0,100)
    for iii,CFR_H in enumerate(CFR_values):
        for jjj,CFR_W in enumerate(CFR_values):
            res_full = []
            res_full_max_values = []
            accuracy=1e-5
            for i,resource_capacity in enumerate(resource_capacity_values_full):
                dummy = []
                for j,tau in enumerate(tau_values):
                    sol = RK4(ODE,initial_conditions,ts,beta_H, beta_W, h, replacement_rate, kappa, R_H, R_W, CFR_H, CFR_W, mu, tau, resource_capacity,r_I)
                    FES = 1-sum(sol[-1,:3])
                    dummy.append(FES)
                argmin = np.argmin(dummy)
                min_value = np.min(dummy)
                if resource_capacity>0.01 and np.abs(min_value-np.min(res_full_max_values))<1e-4:
                    break
                res_full.append( argmin)
                res_full_max_values.append(min_value)
            res_full = np.array(res_full)/(len(tau_values)-1)
            res_full_max_values = 100*np.array(res_full_max_values)
            

            ax[ii].plot(res_full,res_full_max_values,color=colors[iii],ls=lss[jjj],lw=lws[iii])
            #line = ax[ii].plot([tau_values[np.argmin(res,1)][1]],[np.min(res,1)[1]],'ko-',label=r'$\tau^* =$ argmin $\tau$ at a fixed treatment capacity $c$')[0]
    if ii==0:
        ax[ii].set_ylabel('Final epidemic size (in %)')
    ax[ii].set_xlabel(r'Optimal treatment resource allocation $\tau^*$')
    ax[ii].spines[['right', 'top']].set_visible(False)    
    ax[ii].set_xlim([-0.05,1.05])        
    if ii==0:
        line1 = ax[ii].plot([0.5],[80],color=colors[0],ls='-',lw=lws[0],label=str(int(CFR_values[0]*100))+'%')[0]
        line2 = ax[ii].plot([0.5],[80],color=colors[1],ls='-',label=str(int(CFR_values[1]*100))+'%')[0]
        legend1 = ax[ii].legend(handles = [line1,line2],frameon=False,bbox_to_anchor = [0.25,1],loc='upper center',ncol=1,title=r'CFR${}_H$')          
        ax[ii].add_artist(legend1)
        line3 = ax[ii].plot([0.5],[80],color='k',ls=lss[0],lw=lws[0],label=str(int(CFR_values[0]*100))+'%')[0]
        line4 = ax[ii].plot([0.5],[80],color='k',ls=lss[1],lw=lws[1],label=str(int(CFR_values[1]*100))+'%')[0]
        legend2 = ax[ii].legend(handles=[line3,line4],frameon=False,loc='upper center',bbox_to_anchor = [0.6,1],ncol=1,title=r'CFR${}_W$')
        ax[ii].add_artist(legend2)
    xticklabels = ax[ii].get_xticklabels()
    xticklabels[list(ax[ii].get_xticks()).index(0)].set_text('0\nall at war')
    xticklabels[list(map(lambda el: round(el,2),ax[ii].get_xticks())).index(1.)].set_text('1\nall at home')
    ax[ii].set_xticklabels(xticklabels)
    #ax.set_title(r'$\beta_W/\beta_H =$ '+str(np.round(beta_W/beta_H,2)) +r', $h =$ '+str(h)+r', $\kappa =$ '+str(kappa)+', $\gamma =$ '+str(R_H))
plt.savefig('diffCFR_FES_b2b1_%s_hbothextremes_k_%s_gamma_%s.pdf' % (str(np.round(beta_W/beta_H,2)),str(kappa),str(R_H)),bbox_extra_artists=(legend1,legend2,),bbox_inches = "tight")   
CFR_H = 0.05
CFR_W = 0.05




#2 subplots, mortality, different treatment rates
r_I = 0

tau_values = np.linspace(0, 1, 301);
beta_H = 0.6
beta_W = 3*beta_H
kappa=1.
resource_capacity_values = [0,0.0002,0.001,0.005,0.01,0.02,0.03,0.05,0.15,0.3];
resource_capacity_values = [0,0.05,0.0002,0.15,0.001,0.3,0.005,0.01,0.02,0.03];
R_H = 0.2
R_W = 0.2

mu_values = [0.1,0.2,0.3]

f,ax = plt.subplots(1,2,sharey=True,figsize=(8,3.5))
for ii,h in enumerate([0.,1.]):
    resource_capacity_values_full = np.logspace(-5,0,1000)
    for iii,mu in enumerate(mu_values):
        res_full = []
        res_full_max_values = []
        accuracy=1e-5
        for i,resource_capacity in enumerate(resource_capacity_values_full):
            argmax,max_value = gss(ODE,initial_conditions,ts, beta_H, beta_W, h, replacement_rate, kappa, R_H, R_W, CFR_H, CFR_W, mu, resource_capacity,r_I, tolerance=accuracy)
            if i>0 and max_value==res_full_max_values[-1]:
                break
            res_full.append( argmax)
            res_full_max_values.append(max_value)
        res_full = np.array(res_full)
        res_full_max_values = 100*np.array(res_full_max_values)
        

        ax[ii].plot(res_full,res_full_max_values,label=str(mu))
        #line = ax[ii].plot([tau_values[np.argmin(res,1)][1]],[np.min(res,1)[1]],'ko-',label=r'$\tau^* =$ argmin $\tau$ at a fixed treatment capacity $c$')[0]
    if ii==0:
        ax[ii].set_ylabel('Total Mortality (in % of total population)')
    ax[ii].set_xlabel(r'Optimal treatment resource allocation $\tau^*$')
    ax[ii].spines[['right', 'top']].set_visible(False)            
    if ii==0:
        ax[ii].legend(frameon=False,bbox_to_anchor = [0.5,1],loc='upper center',ncol=2,title=r'treatment rate ($\mu$)') 
    xticklabels = ax[ii].get_xticklabels()
    xticklabels[list(ax[ii].get_xticks()).index(0)].set_text('0\nall at war')
    xticklabels[list(map(lambda el: round(el,2),ax[ii].get_xticks())).index(1.)].set_text('1\nall at home')
    ax[ii].set_xticklabels(xticklabels)
    #ax.set_title(r'$\beta_W/\beta_H =$ '+str(np.round(beta_W/beta_H,2)) +r', $h =$ '+str(h)+r', $\kappa =$ '+str(kappa)+', $\gamma =$ '+str(R_H))
plt.savefig('diffmu_b2b1_%s_hbothextremes_k_%s_gamma_%s.pdf' % (str(np.round(beta_W/beta_H,2)),str(kappa),str(R_H)),bbox_extra_artists=(legend1,legend2,),bbox_inches = "tight")   
mu = 0.2




#2 subplots, FES, different treatment rates
r_I = 0


tau_values = np.linspace(0, 1, 301);
beta_H = 0.6
beta_W = 3*beta_H
kappa=1.
resource_capacity_values = [0,0.0002,0.001,0.005,0.01,0.02,0.03,0.05,0.15,0.3];
resource_capacity_values = [0,0.05,0.0002,0.15,0.001,0.3,0.005,0.01,0.02,0.03];
R_H = 0.2
R_W = 0.2

mu_values = [0.1,0.2,0.3]

colors = ['green','purple']
lss = ['-',':']
lws = [2,2]

f,ax = plt.subplots(1,2,sharey=True,figsize=(8,3.5))
for ii,h in enumerate([0.,1.]):
    resource_capacity_values_full = np.logspace(-5,0,1000)
    for iii,mu in enumerate(mu_values):
        res_full = []
        res_full_max_values = []
        accuracy=1e-5
        for i,resource_capacity in enumerate(resource_capacity_values_full):
            dummy = []
            for j,tau in enumerate(tau_values):
                sol = RK4(ODE,initial_conditions,ts,beta_H, beta_W, h, replacement_rate, kappa, R_H, R_W, CFR_H, CFR_W, mu, tau, resource_capacity,r_I)
                FES = 1-sum(sol[-1,:3])
                dummy.append(FES)
            argmin = np.argmin(dummy)
            min_value = np.min(dummy)
            if resource_capacity>0.01 and np.abs(min_value-np.min(res_full_max_values))<1e-5:
                break
            res_full.append( argmin)
            res_full_max_values.append(min_value)
        res_full = np.array(res_full)/(len(tau_values)-1)
        res_full_max_values = 100*np.array(res_full_max_values)
        

        ax[ii].plot(res_full,res_full_max_values,label=str(mu))
        #line = ax[ii].plot([tau_values[np.argmin(res,1)][1]],[np.min(res,1)[1]],'ko-',label=r'$\tau^* =$ argmin $\tau$ at a fixed treatment capacity $c$')[0]
    if ii==0:
        ax[ii].set_ylabel('Final epidemic size (in %)')
    ax[ii].set_xlabel(r'Optimal treatment resource allocation $\tau^*$')
    ax[ii].spines[['right', 'top']].set_visible(False)    
    ax[ii].set_xlim([-0.05,1.05])        
    if ii==0:
        ax[ii].legend(frameon=False,bbox_to_anchor = [0.5,1],loc='upper center',ncol=2,title=r'treatment rate ($\mu$)') 
    xticklabels = ax[ii].get_xticklabels()
    xticklabels[list(ax[ii].get_xticks()).index(0)].set_text('0\nall at war')
    xticklabels[list(map(lambda el: round(el,2),ax[ii].get_xticks())).index(1.)].set_text('1\nall at home')
    ax[ii].set_xticklabels(xticklabels)
    #ax.set_title(r'$\beta_W/\beta_H =$ '+str(np.round(beta_W/beta_H,2)) +r', $h =$ '+str(h)+r', $\kappa =$ '+str(kappa)+', $\gamma =$ '+str(R_H))
plt.savefig('diffmu_FES_b2b1_%s_hbothextremes_k_%s_gamma_%s.pdf' % (str(np.round(beta_W/beta_H,2)),str(kappa),str(R_H)),bbox_extra_artists=(legend1,legend2,),bbox_inches = "tight")   
mu=0.2









#2 subplots, mortality, different gender homophily
r_I = 0

tau_values = np.linspace(0, 1, 301);
beta_H = 0.6
beta_W = 3*beta_H
kappa=1.
resource_capacity_values = [0,0.0002,0.001,0.005,0.01,0.02,0.03,0.05,0.15,0.3];
resource_capacity_values = [0,0.05,0.0002,0.15,0.001,0.3,0.005,0.01,0.02,0.03];
R_H = 0.2
R_W = 0.2

h_values = [0.,0.33,0.67,1.]

f,ax = plt.subplots(1,2,sharey=True,figsize=(8,3.5))
for ii,h in enumerate([0.,1.]):
    resource_capacity_values_full = np.logspace(-5,0,500)
    for iii,h in enumerate(h_values):
        res_full = []
        res_full_max_values = []
        accuracy=1e-4
        max_value_total = gss(ODE,initial_conditions,ts, beta_H, beta_W, h, replacement_rate, kappa, R_H, R_W, CFR_H, CFR_W, mu, 100,r_I, tolerance=accuracy)[1]

        for i,resource_capacity in enumerate(resource_capacity_values_full):
            argmax,max_value = gss(ODE,initial_conditions,ts, beta_H, beta_W, h, replacement_rate, kappa, R_H, R_W, CFR_H, CFR_W, mu, resource_capacity,r_I, tolerance=accuracy)
            if i>0 and max_value<max_value_total+accuracy:
                break
            res_full.append( argmax)
            res_full_max_values.append(max_value)
        res_full = np.array(res_full)
        res_full_max_values = 100*np.array(res_full_max_values)
        

        ax[ii].plot(res_full,res_full_max_values,label=str(h))
        #line = ax[ii].plot([tau_values[np.argmin(res,1)][1]],[np.min(res,1)[1]],'ko-',label=r'$\tau^* =$ argmin $\tau$ at a fixed treatment capacity $c$')[0]
    if ii==0:
        ax[ii].set_ylabel('Total Mortality (in % of total population)')
    ax[ii].set_xlabel(r'Optimal treatment resource allocation $\tau^*$')
    ax[ii].spines[['right', 'top']].set_visible(False)            
    if ii==0:
        legend1 = ax[ii].legend(frameon=False,bbox_to_anchor = [0.5,1.2],loc='upper center',ncol=2,title=r'gender homophily ($h$)') 
    xticklabels = ax[ii].get_xticklabels()
    xticklabels[list(ax[ii].get_xticks()).index(0)].set_text('0\nall at war')
    xticklabels[list(map(lambda el: round(el,2),ax[ii].get_xticks())).index(1.)].set_text('1\nall at home')
    ax[ii].set_xticklabels(xticklabels)
    #ax.set_title(r'$\beta_W/\beta_H =$ '+str(np.round(beta_W/beta_H,2)) +r', $h =$ '+str(h)+r', $\kappa =$ '+str(kappa)+', $\gamma =$ '+str(R_H))
plt.savefig('diffh_b2b1_%s_hbothextremes_k_%s_gamma_%s.pdf' % (str(np.round(beta_W/beta_H,2)),str(kappa),str(R_H)),bbox_extra_artists=(legend1,legend2,),bbox_inches = "tight")   
h  = 0.



#2 subplots, FES, different gender homophily
r_I = 0


tau_values = np.linspace(0, 1, 301);
beta_H = 0.6
beta_W = 3*beta_H
kappa=1.
resource_capacity_values = [0,0.0002,0.001,0.005,0.01,0.02,0.03,0.05,0.15,0.3];
resource_capacity_values = [0,0.05,0.0002,0.15,0.001,0.3,0.005,0.01,0.02,0.03];
R_H = 0.2
R_W = 0.2

h_values = [0.,0.33,0.67,1.]


f,ax = plt.subplots(1,2,sharey=True,figsize=(8,3.5))
for ii,h in enumerate([0.,1.]):
    resource_capacity_values_full = np.logspace(-5,0,500)
    for iii,h in enumerate(h_values):
        res_full = []
        res_full_max_values = []
        accuracy=1e-5
        for i,resource_capacity in enumerate(resource_capacity_values_full):
            dummy = []
            for j,tau in enumerate(tau_values):
                sol = RK4(ODE,initial_conditions,ts,beta_H, beta_W, h, replacement_rate, kappa, R_H, R_W, CFR_H, CFR_W, mu, tau, resource_capacity,r_I)
                FES = 1-sum(sol[-1,:3])
                dummy.append(FES)
            argmin = np.argmin(dummy)
            min_value = np.min(dummy)
            if resource_capacity>0.01 and np.abs(min_value-np.min(res_full_max_values))<1e-5:
                break
            res_full.append( argmin)
            res_full_max_values.append(min_value)
        res_full = np.array(res_full)/(len(tau_values)-1)
        res_full_max_values = 100*np.array(res_full_max_values)
        

        ax[ii].plot(res_full,res_full_max_values,label=str(h))
        #line = ax[ii].plot([tau_values[np.argmin(res,1)][1]],[np.min(res,1)[1]],'ko-',label=r'$\tau^* =$ argmin $\tau$ at a fixed treatment capacity $c$')[0]
    if ii==0:
        ax[ii].set_ylabel('Final epidemic size (in %)')
    ax[ii].set_xlabel(r'Optimal treatment resource allocation $\tau^*$')
    ax[ii].spines[['right', 'top']].set_visible(False)    
    ax[ii].set_xlim([-0.05,1.05])        
    if ii==0:
        ax[ii].legend(frameon=False,bbox_to_anchor = [0.5,1.2],loc='upper center',ncol=2,title=r'gender homophily ($h$)') 
    xticklabels = ax[ii].get_xticklabels()
    xticklabels[list(ax[ii].get_xticks()).index(0)].set_text('0\nall at war')
    xticklabels[list(map(lambda el: round(el,2),ax[ii].get_xticks())).index(1.)].set_text('1\nall at home')
    ax[ii].set_xticklabels(xticklabels)
    #ax.set_title(r'$\beta_W/\beta_H =$ '+str(np.round(beta_W/beta_H,2)) +r', $h =$ '+str(h)+r', $\kappa =$ '+str(kappa)+', $\gamma =$ '+str(R_H))
plt.savefig('diffh_FES_b2b1_%s_hbothextremes_k_%s_gamma_%s.pdf' % (str(np.round(beta_W/beta_H,2)),str(kappa),str(R_H)),bbox_extra_artists=(legend1,legend2,),bbox_inches = "tight")   
mu=0.2






#2 subplots, mortality, different replacement rate
r_I = 0

tau_values = np.linspace(0, 1, 301);
beta_H = 0.6
beta_W = 3*beta_H
kappa=1.
R_H = 0.2
R_W = 0.2

rwh_values = [0.01,0.025,0.05,0.1]

f,ax = plt.subplots(1,2,sharey=True,figsize=(8,3.5))
for ii,h in enumerate([0.,1.]):
    resource_capacity_values_full = np.logspace(-5,0,300)
    for iii,replacement_rate in enumerate(rwh_values):
        res_full = []
        res_full_max_values = []
        accuracy=1e-5
        max_value_total = gss(ODE,initial_conditions,ts, beta_H, beta_W, h, replacement_rate, kappa, R_H, R_W, CFR_H, CFR_W, mu, 100,r_I, tolerance=accuracy)[1]

        for i,resource_capacity in enumerate(resource_capacity_values_full):
            argmax,max_value = gss(ODE,initial_conditions,ts, beta_H, beta_W, h, replacement_rate, kappa, R_H, R_W, CFR_H, CFR_W, mu, resource_capacity,r_I, tolerance=accuracy)
            if i>0 and max_value<max_value_total+accuracy:
                break
            res_full.append( argmax)
            res_full_max_values.append(max_value)
        res_full = np.array(res_full)
        res_full_max_values = 100*np.array(res_full_max_values)
        

        ax[ii].plot(res_full,res_full_max_values,label=str(replacement_rate))
        #line = ax[ii].plot([tau_values[np.argmin(res,1)][1]],[np.min(res,1)[1]],'ko-',label=r'$\tau^* =$ argmin $\tau$ at a fixed treatment capacity $c$')[0]
    if ii==0:
        ax[ii].set_ylabel('Total Mortality (in % of total population)')
    ax[ii].set_xlabel(r'Optimal treatment resource allocation $\tau^*$')
    ax[ii].spines[['right', 'top']].set_visible(False)            
    if ii==0:
        legend1 = ax[ii].legend(frameon=False,bbox_to_anchor = [0.45,0.7],loc='upper center',ncol=2,title=r'soldier replacement rate ($\eta$)') 
    xticklabels = ax[ii].get_xticklabels()
    xticklabels[list(ax[ii].get_xticks()).index(0)].set_text('0\nall at war')
    xticklabels[list(map(lambda el: round(el,2),ax[ii].get_xticks())).index(1.)].set_text('1\nall at home')
    ax[ii].set_xticklabels(xticklabels)
    #ax.set_title(r'$\beta_W/\beta_H =$ '+str(np.round(beta_W/beta_H,2)) +r', $h =$ '+str(h)+r', $\kappa =$ '+str(kappa)+', $\gamma =$ '+str(R_H))
plt.savefig('diffreplrate_b2b1_%s_hbothextremes_k_%s_gamma_%s.pdf' % (str(np.round(beta_W/beta_H,2)),str(kappa),str(R_H)),bbox_extra_artists=(legend1,),bbox_inches = "tight")   
replacement_rate = 0.05



#2 subplots, FES, different replacement rate
r_I = 0


tau_values = np.linspace(0, 1, 301);
beta_H = 0.6
beta_W = 3*beta_H
kappa=1.
R_H = 0.2
R_W = 0.2

rwh_values = [0.01,0.025,0.05,0.1]


f,ax = plt.subplots(1,2,sharey=True,figsize=(8,3.5))
for ii,h in enumerate([0.,1.]):
    resource_capacity_values_full = np.logspace(-5,0,500)
    for iii,replacement_rate in enumerate(rwh_values):
        res_full = []
        res_full_max_values = []
        accuracy=1e-5
        for i,resource_capacity in enumerate(resource_capacity_values_full):
            dummy = []
            for j,tau in enumerate(tau_values):
                sol = RK4(ODE,initial_conditions,ts,beta_H, beta_W, h, replacement_rate, kappa, R_H, R_W, CFR_H, CFR_W, mu, tau, resource_capacity,r_I)
                FES = 1-sum(sol[-1,:3])
                dummy.append(FES)
            argmin = np.argmin(dummy)
            min_value = np.min(dummy)
            if resource_capacity>0.01 and np.abs(min_value-np.min(res_full_max_values))<1e-5:
                break
            res_full.append( argmin)
            res_full_max_values.append(min_value)
        res_full = np.array(res_full)/(len(tau_values)-1)
        res_full_max_values = 100*np.array(res_full_max_values)
        

        ax[ii].plot(res_full,res_full_max_values,label=str(replacement_rate))
        #line = ax[ii].plot([tau_values[np.argmin(res,1)][1]],[np.min(res,1)[1]],'ko-',label=r'$\tau^* =$ argmin $\tau$ at a fixed treatment capacity $c$')[0]
    if ii==0:
        ax[ii].set_ylabel('Final epidemic size (in %)')
    ax[ii].set_xlabel(r'Optimal treatment resource allocation $\tau^*$')
    ax[ii].spines[['right', 'top']].set_visible(False)    
    ax[ii].set_xlim([-0.05,1.05])        
    if ii==0:
        legend1 = ax[ii].legend(frameon=False,bbox_to_anchor = [0.45,0.7],loc='upper center',ncol=2,title=r'soldier replacement rate ($\eta$)') 
    xticklabels = ax[ii].get_xticklabels()
    xticklabels[list(ax[ii].get_xticks()).index(0)].set_text('0\nall at war')
    xticklabels[list(map(lambda el: round(el,2),ax[ii].get_xticks())).index(1.)].set_text('1\nall at home')
    ax[ii].set_xticklabels(xticklabels)
    #ax.set_title(r'$\beta_W/\beta_H =$ '+str(np.round(beta_W/beta_H,2)) +r', $h =$ '+str(h)+r', $\kappa =$ '+str(kappa)+', $\gamma =$ '+str(R_H))
plt.savefig('diffreplrate_FES_b2b1_%s_hbothextremes_k_%s_gamma_%s.pdf' % (str(np.round(beta_W/beta_H,2)),str(kappa),str(R_H)),bbox_extra_artists=(legend1,),bbox_inches = "tight")   
replacement_rate = 0.05






























#new style of plot
r_I = 0

for R in [0.2,0.1]:
    R_H = R
    R_W = R
    for h in [0.,1.]:
        for beta_ratio in [1,1.5,2,3]:
            beta_W = beta_H*beta_ratio
            resource_capacity_values = np.logspace(-4,0,1000)
            res = []
            for i,resource_capacity in enumerate(resource_capacity_values):
                res.append( [] )
                for j,tau in enumerate(tau_values):
                    #sol = integrate.odeint(ODE, initial_conditions, ts, args=(beta_H, beta_W, h, replacement_rate, kappa, R_H, R_W, CFR_H, CFR_W, mu, tau, resource_capacity))
                    sol = RK4(ODE,initial_conditions,ts,beta_H, beta_W, h, replacement_rate, kappa, R_H, R_W, CFR_H, CFR_W, mu, tau, resource_capacity,r_I)
                    total_deaths = sum(sol[-1,-3:])
                    res[-1].append(total_deaths)
            res = np.array(res)
            optimal_mortality_infinite_resources = find_total_deaths_extreme_cases(ODE,initial_conditions,ts,beta_H, beta_W, h, replacement_rate, kappa, R_H, R_W, CFR_H, CFR_W, mu,r_I)[0]
            
            f,ax = plt.subplots()
            which = np.min(res,1) - optimal_mortality_infinite_resources > 1e-6
            im = ax.scatter(tau_values[np.argmin(res,1)][which],np.min(res,1)[which],c = np.log10(resource_capacity_values[which]))
            ax.set_ylabel('Total Mortality')
            ax.set_xlabel(r'optimal resource allocation ($\tau^*$)')
            ax.legend(frameon=False,bbox_to_anchor = [0.5,1.35],loc='upper center',ncol=4,title=r'Treatment capacity ($c$)')          
            ax.set_title(r'$\beta_W/\beta_H =$ '+str(np.round(beta_W/beta_H,2)) +r', $h =$ '+str(h)+r', $\kappa =$ '+str(kappa)+', $\gamma =$ '+str(R_H))
            ax.set_xlim([0,1])
            # Create a divider for the existing axes instance
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
               
            # Create the colorbar with a title
            cbar = f.colorbar(im, cax=cax)
            cbar.set_label('resource capacity (c)')
            cbar.set_ticks(np.array([-4,-3,-2,-1]))
            cbar.set_ticklabels([r'$10^{-4}$','$10^{-3}$',r'$0.01$',r'$0.1$'])
            
            plt.savefig('fig5new_b2b1_%s_h_%s_k_%s_gamma_%s_rI_%s.pdf' % (str(np.round(beta_W/beta_H,2)),str(h),str(kappa),str(R_H),str(r_I)),bbox_inches = "tight")   
             


h = 0.
kappa = 1
R_H = 0.2
R_W = 0.2
beta_H = 0.6
resource_capacity = 0.02
eps = 1e-04; #proportion initially infected
replacement_rate =0.05
n_beta_ratios = 11
initial_conditions = np.array([(1-p_m)*(1-eps),kappa*p_m/(1+kappa)*(1-eps),p_m/(1+kappa)*(1-eps), (1-p_m)*eps,kappa*p_m/(1+kappa)*eps,p_m/(1+kappa)*eps, 0, 0, 0, 0, 0, 0],dtype=np.float64);
beta_ratios = np.logspace(-np.log10(3),np.log10(3),n_beta_ratios)#np.linspace(-,3, accuracy);

assert R_H==R_W
res = []
fes = []
deaths_war = []
sols = []
tau_values = np.linspace(0, 1, 501);
for ii,beta_ratio in enumerate(beta_ratios):
    beta_W = beta_H*beta_ratio
    res.append( [] )
    fes.append( [] )
    deaths_war.append( [] )
    sols.append( [])
    for j,tau in enumerate(tau_values):
        #sol = integrate.odeint(ODE, initial_conditions, ts, args=(beta_H, beta_W, h, replacement_rate, kappa, R_H, R_W, CFR_H, CFR_W, mu, tau, resource_capacity))
        sol = RK4(ODE,initial_conditions,ts,beta_H, beta_W, h, replacement_rate, kappa, R_H, R_W, CFR_H, CFR_W, mu, tau, resource_capacity, r_I)
        total_deaths = sum(sol[-1,-3:])
        res[-1].append(total_deaths)
        deaths_war[-1].append(sol[-1,-1])
        fes[-1].append(1-sum(sol[-1,:3]))
        sols[-1].append(sol.copy())
res  = np.array(res)
fes  = np.array(fes)
deaths_war = np.array(deaths_war)
sols = np.array(sols)

colors=['r','b']
f,ax = plt.subplots(2,3,sharex=True,sharey=True,figsize=(8,6))
for a,ii in enumerate([5,-1]):
    for b,j in enumerate([0,int(len(tau_values)/2),-1]):
        ax[a,b].plot(ts,100*(sols[ii,j,:,3]+sols[ii,j,:,4]),'r',label='at home')
        ax[a,b].plot(ts,100*sols[ii,j,:,5],'b',label='at war')
        #ax[a,b].set_title(r'$\beta_W/\beta_H =$ '+str(round(beta_ratios[ii],2)) + r', $\tau = $'+str(tau_values[j]))
        if b==0:
            ax[a,b].set_ylabel('Infected [% of total population]')
        if a==1:
            ax[a,b].set_xlabel('Time')

ymin,ymax = ax[0,0].get_ylim()
xmin,xmax = ax[0,0].get_xlim()
ydiff = ymax-ymin
for a,ii in enumerate([5,-1]):
    for b,j in enumerate([0,int(len(tau_values)/2),-1]):
        argmax_home = (sols[ii,j,:,3]+sols[ii,j,:,4]).argmax()
        argmax_war = sols[ii,j,:,5].argmax()
        ax[a,b].plot([ts[argmax_home]],[ymin+0.015*(ymax-ymin)],'ro',label='argmax')
        ax[a,b].plot([ts[argmax_war]],[ymin+0.015*(ymax-ymin)],'bx',label='argmax')
        ax[a,b].legend(loc='best',frameon=0,title='Infected')

        if b==0:
            line = ax[a,b].plot([-40,-40],[ymin,ymax],'k-',lw=0.5,clip_on=False)[0]
            line = ax[a,b].text(-45,(ymin+ymax)/2,r'$\beta_W = 3\beta_H = 1.8$' if a==1 else r'$\beta_W = \beta_H = 0.6$',va='center',ha='center',rotation=90)
        if a==0:
            line2 = ax[a,b].plot([xmin,xmax],[ymax*1.05,ymax*1.05],'k-',lw=0.5,clip_on=False)[0]
            line2 = ax[a,b].text((xmin+xmax)/2,ymax*1.1,r'$\tau = $'+str(round(tau_values[j],1)),va='center',ha='center',rotation=0)

        ax[a,b].text(80,ymin+0.9*ydiff,'deaths = ',va='center',ha='right')
        ax[a,b].text(80,ymin+0.8*ydiff,'at home = ',va='center',ha='right')
        #ax[a,b].text(80,ymin+0.7*ydiff,'at war = ',va='center',ha='right')
        ax[a,b].text(80,ymin+0.9*ydiff,str(round(100*res[ii,j],2))+'%',va='center',ha='left')
        ax[a,b].text(80,ymin+0.8*ydiff,str(round(100*(res-deaths_war)[ii,j],2))+'%',va='center',ha='left')
        #ax[a,b].text(80,ymin+0.7*ydiff,str(round(100*deaths_war[ii,j],2))+'%',va='center',ha='left')
        ax[a,b].set_xlim([xmin,xmax])
        ax[a,b].set_ylim([ymin,ymax])
        ax[a,b].spines[['right', 'top']].set_visible(False)

plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.savefig('explanation_strange_beta_H_observation_2x3_kappa%f_eta%f_h%f_gamma%f_betaH%f_C%f_rI%f.pdf' % (kappa,replacement_rate,h,R_H,beta_H,resource_capacity,r_I),bbox_extra_artists=(line,line2),bbox_inches='tight')



#4-dimensional sensitivity analysis, generate optimal resource allocation plots and total mortality at the same time
accuracy = 150 #ideally choose an odd number because np.linspace includes both endpoints.
replacement_rate = 0.05
for SCALE_BETAS in [0]: 
    dummy_values = [0.15,0.3] if SCALE_BETAS else [0.2,0.1] #'low','high'
    
    R_H = 0.1
    R_W = 0.1
    beta_H = 0.6
    
    resource_capacity_values = np.linspace(0,0.003, accuracy);
    
    for h in [0.,1.]:      
        f, ax = plt.subplots(2, 2, figsize=(5, 5), sharex='col', sharey='row')
        g, gax = plt.subplots(2, 2, figsize=(5, 5), sharex='col', sharey='row')
        for iiii,kappa in enumerate([1.,4.]):
            for jjjj,dummy in enumerate(dummy_values):
                if h==0 or jjjj==0:
                    resource_capacity_values = np.linspace(0,0.08, accuracy);
                else:
                    resource_capacity_values = np.linspace(0,0.08, accuracy); 
                beta_ratios = np.logspace(-np.log10(3),np.log10(3),accuracy)#np.linspace(-,3, accuracy);

                if SCALE_BETAS: #SCALE_BETAS determines if R0 should be scaled by modifying transmission rates or recovery rates (default, SCALE_BETAS==0)
                    beta_H = dummy
                else:
                    R_H = dummy
                    R_W = dummy
    
                #Recompute initial conditions which depend, among others, on kappa
                initial_conditions = np.array([(1-p_m)*(1-eps),kappa*p_m/(1+kappa)*(1-eps),p_m/(1+kappa)*(1-eps), (1-p_m)*eps,kappa*p_m/(1+kappa)*eps,p_m/(1+kappa)*eps, 0, 0, 0, 0, 0, 0],dtype=np.float64);
    
                res = []
                for i,resource_capacity in enumerate(resource_capacity_values):
                    res.append( [] )
                    for ii,beta_ratio in enumerate(beta_ratios):
                        beta_W = beta_H*beta_ratio
                        res[-1].append( [] )
                        for j,tau in enumerate(tau_values):
                            #sol = integrate.odeint(ODE, initial_conditions, ts, args=(beta_H, beta_W, h, replacement_rate, kappa, R_H, R_W, CFR_H, CFR_W, mu, tau, resource_capacity))
                            sol = RK4(ODE,initial_conditions,ts,beta_H, beta_W, h, replacement_rate, kappa, R_H, R_W, CFR_H, CFR_W, mu, tau, resource_capacity, r_I)
                            total_deaths = sum(sol[-1,-3:])
                            res[-1][-1].append(total_deaths)
                res = np.array(res)
                
                optimal_mortality_infinite_resources = np.array([find_total_deaths_extreme_cases(ODE,initial_conditions,ts,beta_H, beta_H * beta_ratio, h, replacement_rate, kappa, R_H, R_W, CFR_H, CFR_W, mu, r_I)[0] for beta_ratio in beta_ratios])
                optimal_mortality_infinite_resources_as_tilematrix = np.tile(optimal_mortality_infinite_resources[:,np.newaxis], len(optimal_mortality_infinite_resources)).T
               
                
                data  = res.argmin(2)/(len(tau_values)-1)
                data[0,:] = data[1,:]
                masked_array = np.ma.masked_where(res.min(2)-optimal_mortality_infinite_resources_as_tilematrix < 1e-7, data)
                gdata = res.min(2)
                cmap = cm.rainbow  # Can be any colormap that you want after the cm
                cmap.set_bad(color='gray')
                gcmap = cm.Purples
                print(kappa,beta_H)
                #plt.imshow(masked_array,cmap=cmap,origin='lower',vmin=0,vmax=1)
                images=[]
                images.append(ax[jjjj,iiii].imshow(masked_array,cmap=cmap,origin='lower',vmin=0,vmax=1))
                gimages = []
                gimages.append(gax[jjjj,iiii].imshow(gdata,cmap=gcmap,origin='lower',vmin=0,vmax=max(CFR_H,CFR_W)))
                
                
                #resource_capacity_values_ticks = [0,0.002,0.004,0.006,0.008]
                if max(resource_capacity_values)<0.005:
                    resource_capacity_values_ticks = [0,0.001,0.002,0.003,0.004]
                elif max(resource_capacity_values)<0.009:
                    resource_capacity_values_ticks = [0,0.002,0.004,0.006,0.008]
                elif max(resource_capacity_values)<0.016:
                    resource_capacity_values_ticks = [0,0.005,0.01,0.015]
                elif max(resource_capacity_values)<0.025:
                    resource_capacity_values_ticks = [0,0.005,0.01,0.015,0.02]
                else:
                    resource_capacity_values_ticks = [0,0.02,0.04,0.06,0.08]
                    
                    
                #resource_capacity_values_ticks = [0,0.02,0.04,0.06,0.08]
                ax[jjjj,iiii].set_yticks(infer_ticks(resource_capacity_values_ticks,resource_capacity_values))
                gax[jjjj,iiii].set_yticks(infer_ticks(resource_capacity_values_ticks,resource_capacity_values))
                if iiii==0:
                    ax[jjjj,iiii].set_yticklabels(list(map(str,resource_capacity_values_ticks)))
                    gax[jjjj,iiii].set_yticklabels(list(map(str,resource_capacity_values_ticks)))
                    ax[jjjj,iiii].set_ylabel(r'treatment capacity ($c$)')
                    gax[jjjj,iiii].set_ylabel(r'treatment capacity ($c$)')
                    
               
                beta_ratios_ticks = [1,1.5,2,2.5,3]
                beta_ratios_ticks = np.logspace(-np.log10(3),np.log10(3),3)
                xticks = [0,(accuracy-1)/2,accuracy-1]
                xlabels = list(map(str,beta_ratios_ticks))
                xlabels = ['1/3','1','3']
                ax[jjjj,iiii].set_xticks(xticks)
                gax[jjjj,iiii].set_xticks(xticks)
                if jjjj==1:
                    ax[jjjj,iiii].set_xticklabels(xlabels)
                    gax[jjjj,iiii].set_xticklabels(xlabels)
                    ax[jjjj,iiii].set_xlabel(r'$\beta_W/\beta_H$')
                    gax[jjjj,iiii].set_xlabel(r'$\beta_W/\beta_H$')
                           
        # Get the position of the first and last subplots
        pos1 = ax[0, 0].get_position()
        pos2 = ax[1, 1].get_position()
        
        # Set the colorbar position based on the combined height of the subplots
        cbar_ax = f.add_axes([pos2.x1 + 0.02, pos2.y0, 0.02, pos1.y1 - pos2.y0])
        gcbar_ax = g.add_axes([pos2.x1 + 0.02, pos2.y0, 0.02, pos1.y1 - pos2.y0])
        
        # Add the colorbar
        cbar = f.colorbar(images[0], cax=cbar_ax)
        cbar.set_label(r'optimal $\tau$ value')
        cbar_ax.text(5,0,'focus\nat war',va='center',ha='left')
        cbar_ax.text(5,1,'focus\nat home',va='center',ha='left')

        gcbar = g.colorbar(gimages[0], cax=gcbar_ax)
        gcbar.set_label(r'mortality at optimal $\tau$ value [% of total population]')
    
    
        ax_left = f.add_axes([pos1.x0 - 0.18, pos2.y0, 0.02, pos1.y1 - pos2.y0])
        ax_left.set_xticks([])
        ax_left.set_yticks([])
        for spine in ax_left.spines.values():
            spine.set_visible(False)
        ax_left.plot([0,0],[0,1],'k-',lw=0.5)
        ax_left.set_ylim([0,1])
        ax_left.text(-0.28,0.5,'outbreak severity',ha='center',va='center',rotation=90)    
        if SCALE_BETAS:
            ax_left.text(-0.1,0.25,'high ('+r'$\beta_H =$ '+str(dummy_values[1])+')',ha='center',va='center',rotation=90)    
            ax_left.text(-0.1,0.75,'low ('+r'$\beta_H =$ '+str(dummy_values[0])+')',ha='center',va='center',rotation=90)    
        else:
            ax_left.text(-0.1,0.25,'high ('+r'$\gamma_H = \gamma_W =$ '+str(dummy_values[1])+')',ha='center',va='center',rotation=90)    
            ax_left.text(-0.1,0.75,'low ('+r'$\gamma_H = \gamma_W =$ '+str(dummy_values[0])+')',ha='center',va='center',rotation=90)    
    
        ax_top = f.add_axes([pos1.x0, pos1.y1 + 0.01, pos2.x1 - pos1.x0, 0.02])
        ax_top.set_xticks([])
        ax_top.set_yticks([])
        for spine in ax_top.spines.values():
            spine.set_visible(False)
        ax_top.plot([0,1],[0,0],'k-',lw=0.5)
        ax_top.set_xlim([0,1])
        ax_top.text(0.5,0.28,'severity of war',ha='center',va='center')    
        ax_top.text(0.25,0.1,'high ('+r'$\kappa =$ '+str(1)+')',ha='center',va='center',rotation=0)    
        ax_top.text(0.75,0.1,'low ('+r'$\kappa =$ '+str(4)+')',ha='center',va='center',rotation=0)    

        gax_left = g.add_axes([pos1.x0 - 0.18, pos2.y0, 0.02, pos1.y1 - pos2.y0])
        gax_left.set_xticks([])
        gax_left.set_yticks([])
        for spine in gax_left.spines.values():
            spine.set_visible(False)
        gax_left.plot([0,0],[0,1],'k-',lw=0.5)
        gax_left.set_ylim([0,1])
        gax_left.text(-0.28,0.5,'outbreak severity',ha='center',va='center',rotation=90)    
        if SCALE_BETAS:
            gax_left.text(-0.1,0.25,'high ('+r'$\beta_H =$ '+str(dummy_values[1])+')',ha='center',va='center',rotation=90)    
            gax_left.text(-0.1,0.75,'low ('+r'$\beta_H =$ '+str(dummy_values[0])+')',ha='center',va='center',rotation=90)    
        else:
            gax_left.text(-0.1,0.25,'high ('+r'$\gamma_H = \gamma_W =$ '+str(dummy_values[1])+')',ha='center',va='center',rotation=90)    
            gax_left.text(-0.1,0.75,'low ('+r'$\gamma_H = \gamma_W =$ '+str(dummy_values[0])+')',ha='center',va='center',rotation=90)    
    
        gax_top = g.add_axes([pos1.x0, pos1.y1 + 0.01, pos2.x1 - pos1.x0, 0.02])
        gax_top.set_xticks([])
        gax_top.set_yticks([])
        for spine in gax_top.spines.values():
            spine.set_visible(False)
        gax_top.plot([0,1],[0,0],'k-',lw=0.5)
        gax_top.set_xlim([0,1])
        gax_top.text(0.5,0.28,'severity of war',ha='center',va='center')    
        gax_top.text(0.25,0.1,'high ('+r'$\kappa =$ '+str(1)+')',ha='center',va='center',rotation=0)    
        gax_top.text(0.75,0.1,'low ('+r'$\kappa =$ '+str(4)+')',ha='center',va='center',rotation=0)    
    

        # Adjust the spacing between subplots
        f.subplots_adjust(wspace=0.1, hspace=0.1)
        f.savefig('fig6_h_%s_k_%s_%s_%s_maxc%s_rI%s_accuracy%s.pdf' % (str(h),'both','beta' if SCALE_BETAS else 'gamma','both',str(np.round(max(resource_capacity_values),4)),str(r_I),str(accuracy)),bbox_inches = "tight")   
        g.subplots_adjust(wspace=0.1, hspace=0.1)
        g.savefig('fig6_h_%s_k_%s_%s_%s_maxc%s_rI%s_accuracy%s_deaths.pdf' % (str(h),'both','beta' if SCALE_BETAS else 'gamma','both',str(np.round(max(resource_capacity_values),4)),str(r_I),str(accuracy)),bbox_inches = "tight")   





#compute the effective reproductive number for variable treatment capacities (Fig 1)
beta_H = 0.3
beta_W = 0.4
kappa = 1

def phi(infected, resource):
    return np.minimum(1, resource / infected)

def lambda_2(tau, S_F, S_HM, N_F, N_HM, h, rho, beta_H, gamma_H, mu, I_H, C):
    return beta_H * (S_F + S_HM) / ((N_F + N_HM) * (mu * phi(I_H, tau * C) + gamma_H))


def lambda_3(tau, S_WM, N_WM, beta_W, mu, I_WM, C, gamma_W):
    return beta_W * S_WM / (N_WM * (mu * phi(I_WM, (1 - tau) * C) + gamma_W))

Cs = [0,0.0005,0.002,0.006,0.01]
f, ax = plt.subplots(nrows=1,ncols=len(Cs),figsize=(9,2.6),sharey=True)
h=0
for iii,C in enumerate(Cs):  # Parameter h
    gamma_H = 0.2
    gamma_W = 0.2
    mu = 0.2

    p_m = 0.5
    p_S = 1 - 1e-2
    p_I = 1e-2
    p_R = 0

    if p_S + p_I + p_R != 1:  # rescale
        sum_ = p_S + p_I + p_R
        p_S /= sum_
        p_I /= sum_
        p_R /= sum_

    N_F, N_HM, N_WM = (1 - p_m), kappa * p_m / (1 + kappa), p_m / (1 + kappa)
    S_F, S_HM, S_WM = N_F * p_S, N_HM * p_S, N_WM * p_S
    I_F, I_HM, I_WM = N_F * p_I, N_HM * p_I, N_WM * p_I
    I_H = I_F + I_HM

    rho = 2 * N_F * N_HM / (N_F**2 + N_HM**2)

    # Define the range for tau
    tau = np.linspace(0, 1, 100)  # Change the range as needed

    # Plot lambda_2 and lambda_3 with respect to tau
    l2 = lambda_2(tau, S_F, S_HM, N_F, N_HM, h, rho, beta_H, gamma_H, mu, I_H, C)
    l3 = lambda_3(tau, S_WM, N_WM, beta_W, mu, I_WM, C, gamma_W)
    R_eff = np.maximum(l2,l3)
    argmin_R_eff = np.where(np.abs(R_eff-np.min(R_eff))<1e-3)[0]
    ax[iii].plot(tau, R_eff , '-',color='#AAAAAA' ,lw=3,label=r'$R_{\text{eff}}(\tau)$')
    ax[iii].plot(tau, l2, 'r--', label=r'$R_{\text{eff,home}}(\tau)$')
    ax[iii].plot(tau, l3, 'b--', label=r'$R_{\text{eff,war}}(\tau)$')
    ax[iii].plot(tau[argmin_R_eff], [0.6]*len(argmin_R_eff), 'go', label=r'$\tau^*=$argmin $R_{\text{eff}}(\tau)$')
    if iii==2:
        ax[iii].set_xlabel(r'Treatment resource allocation $\tau$ ($\tau = 0$: all at war, $\tau=1$: all at home)')
    #ax[iii].set_title('S = %s%%, I = %s%%, R = %s%%\nbeta_H = %s, beta_W = %s, h = %s, C = %s' % (
#        str(round(p_S * 100, 2)), str(round(p_I * 100, 2)), str(round(p_R * 100, 2)), str(round(beta_H, 2)), str(round(beta_W, 2)), str(round(h, 2)), str(round(C, 4))))
    ax[iii].set_title('c = '+str(round(C, 4)))
    #ax[iii].set_ylim([0, ax[iii].get_ylim()[1]])
    ax[iii].set_ylim([0.57, ax[iii].get_ylim()[1]])
    ax[iii].set_xlim([0,1])
    if iii==len(Cs)-1:
        ax[iii].legend(loc='center', frameon=0,bbox_to_anchor=[1.8,0.5])
    ax[iii].set_yticks([0.75,1,1.25,1.5,1.75,2])
    ax[iii].set_xticklabels(['0','0.5','1'])    
    if iii==0:
        ax[iii].set_ylabel('effective reproduction number')

plt.savefig('R_eff_S%s_I%s_R%s_betaH%s_betaW%s_h%s_C%s.pdf' % (str(round(p_S * 100, 2)), str(round(p_I * 100, 2)), str(round(p_R * 100, 2)), str(round(beta_H, 2)), str(round(beta_W, 2)), str(round(h, 2)), str(round(C, 4))),bbox_inches = "tight")   
 





