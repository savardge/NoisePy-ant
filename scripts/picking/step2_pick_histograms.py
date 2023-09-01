"""
Script to plot an histogram of all dispersion curve picks for different cross-component and different filters.
Takes as input the large, merged table with all station pair picks (.csv file)
Written by Genevieve Savard @UniGe (2023)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


def get_mean(inst_periods, group_velocity):
    """ Get mean and standard deviation of group velocity picks along periods"""
    inst_periods_uniq = np.unique(inst_periods)
    gv_moy = np.zeros(shape=(len(inst_periods_uniq),))
    gv_std = np.zeros(shape=(len(inst_periods_uniq),))
    for iper, per in enumerate(inst_periods_uniq):
        gv_moy[iper] = np.mean(group_velocity[inst_periods==per])
        gv_std[iper] = np.std(group_velocity[inst_periods==per])
    return inst_periods_uniq, gv_moy, gv_std


def plot_picks(picks, ax=None, dmax=None, bins=100, title="Pick density"):
    """ Plotting function """
    picks2 = picks.copy()

    inst_periods_uniq, gv_moy, gv_std = get_mean(picks2.inst_period.values, picks2.group_velocity.values)
    
    heatmap, xedges, yedges = np.histogram2d(picks2.inst_period, picks2.group_velocity, bins=bins)
    
    # Cut first column and first row
    heatmap = heatmap[1:,10:]
    xedges = xedges[1:]
    yedges = yedges[10:]
    
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    print(np.min(heatmap), np.max(heatmap))
    
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(14,10))
        if dmax:
            im = ax.imshow(heatmap.T, extent=extent, origin='lower', cmap='nipy_spectral', vmin=0, vmax=dmax)
        else:
            im = ax.imshow(heatmap.T, extent=extent, origin='lower', cmap='nipy_spectral')
        ax.set_title(title)
        plt.tight_layout()
        cb = plt.colorbar(im, shrink=0.5, label="# picks")
        plt.show()
        plt.close()
    else:
        if dmax:
            im = ax.imshow(heatmap.T, extent=extent, origin='lower', cmap='nipy_spectral', vmin=0, vmax=dmax)
        else:
            im = ax.imshow(heatmap.T, extent=extent, origin='lower', cmap='nipy_spectral')
        ax.plot(inst_periods_uniq, gv_moy, c="w", ls="--", lw=2)
        ax.plot(inst_periods_uniq, gv_moy+2*gv_std, c="w", ls=":", lw=1)
        ax.plot(inst_periods_uniq, gv_moy-2*gv_std, c="w", ls=":", lw=1)
        ax.set_title(title)        
        ax.set(xlim=(xedges[0], xedges[-1]), ylim=(yedges[0], yedges[-1]))
        ax.xaxis.set_minor_locator(AutoMinorLocator(10))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.tick_params(which='both', width=1)
        ax.tick_params(which='major', length=9)
        ax.tick_params(which='minor', length=5, color='k')        
    return heatmap, xedges, yedges, im


# Load pick table
picks = pd.read_csv("picks_merged_coh.csv")

# Define some plotting parameters
Tmin = 0.2 # minimum period
Tmax = 5.5 # maximum period
dT = 0.1 # period interval (see parameters used in disperion picking script)
vmin = 0.5 # minimum group velocity for plotting
vmax = 3.5 # maximun group velocity
dvel = 0.01 # velocity interval (see parameters used in disperion picking script)
vel = np.arange(vmin, vmax, dvel)
bins = [np.arange(Tmin,Tmax,dT),np.arange(vmin,vmax,dvel)]

# Choose picking method (argmax,topology), type of lag (positive,negative,symmetric) and stacking method (linear,pws)
pick_method = "argmax" # topology, argmax
lag = "sym" # pos,neg,sym
stack_method = "pws" # linear, pws
score_thresh = 0.5 # score threshold for the topology method 

# Plot: 1 subplot per component (change code below if you don't want all the components in one figure)

fig, axs = plt.subplots(4,2,figsize=(12,12))
ax = axs[0][0]
picks_select = picks.loc[(picks.pick_method==pick_method) & 
                         (picks.score >= score_thresh) & 
                         (picks.component=="ZZ") & 
                         (picks.lag==lag) & 
                         (picks.stack_method==stack_method) , ["inst_period", "group_velocity"]]
heatmap, xedges, yedges, im = plot_picks(picks_select, ax=ax, bins=bins, title="")
ax.text(Tmin+0.2, vmax-0.3, "ZZ", fontsize=12, BackgroundColor="w")
fig.colorbar(im, ax=ax, shrink=0.5, label="# picks")

ax = axs[0][1]
picks_select = picks.loc[(picks.pick_method==pick_method) & 
                         (picks.score >= score_thresh) & 
                         (picks.component=="ZR") & 
                         (picks.lag==lag) & 
                         (picks.stack_method==stack_method) , ["inst_period", "group_velocity"]]
heatmap, xedges, yedges, im = plot_picks(picks_select, ax=ax, bins=bins, title="")
ax.text(Tmin+0.2, vmax-0.3, "ZR", fontsize=12, BackgroundColor="w")
fig.colorbar(im, ax=ax, shrink=0.5, label="# picks")

ax = axs[1][0]
picks_select = picks.loc[(picks.pick_method==pick_method) & 
                         (picks.score >= score_thresh) & 
                         (picks.component=="RZ") & 
                         (picks.lag==lag) & 
                         (picks.stack_method==stack_method) , ["inst_period", "group_velocity"]]
heatmap, xedges, yedges, im = plot_picks(picks_select, ax=ax, bins=bins, title="")
ax.text(Tmin+0.2, vmax-0.3, "RZ", fontsize=12, BackgroundColor="w")
fig.colorbar(im, ax=ax, shrink=0.5, label="# picks")

ax = axs[1][1]
picks_select = picks.loc[(picks.pick_method==pick_method) & 
                         (picks.score >= score_thresh) & 
                         (picks.component=="RR") & 
                         (picks.lag==lag) & 
                         (picks.stack_method==stack_method) , ["inst_period", "group_velocity"]]
heatmap, xedges, yedges, im = plot_picks(picks_select, ax=ax, bins=bins, title="")
ax.text(Tmin+0.2, vmax-0.3, "RR", fontsize=12, BackgroundColor="w")
fig.colorbar(im, ax=ax, shrink=0.5, label="# picks")

ax = axs[2][0]
picks_select = picks.loc[(picks.pick_method==pick_method) & 
                         (picks.score >= score_thresh) & 
                         (picks.component=="all4"), ["inst_period", "group_velocity"]]
heatmap, xedges, yedges, im = plot_picks(picks_select, ax=ax, bins=bins, title="")
ax.text(Tmin+0.2, vmax-0.3, "ZZ*ZR*RR*RZ", fontsize=12, BackgroundColor="w")
fig.colorbar(im, ax=ax, shrink=0.5, label="# picks")
ax.set_ylabel("Group velocity (km/s)")
ax.set_xlabel("Period (s)")

ax = axs[2][1]
picks_select = picks.loc[(picks.pick_method==pick_method) & 
                         (picks.score >= score_thresh) & 
                         (picks.component=="ZZ-ZR") & 
                         (picks.stack_method==stack_method) , ["inst_period", "group_velocity"]]
heatmap, xedges, yedges, im = plot_picks(picks_select, ax=ax, bins=bins, title="")
ax.text(Tmin+0.2, vmax-0.3, "ZZ*ZR", fontsize=12, BackgroundColor="w")
fig.colorbar(im, ax=ax, shrink=0.5, label="# picks")

ax = axs[3][0]
picks_select = picks.loc[(picks.pick_method==pick_method) & 
                         (picks.score >= score_thresh) & 
                         (picks.component=="RR-RZ") & 
                         (picks.stack_method==stack_method) , ["inst_period", "group_velocity"]]
heatmap, xedges, yedges, im = plot_picks(picks_select, ax=ax, bins=bins, title="")
ax.text(Tmin+0.2, vmax-0.3, "RR*RZ", fontsize=12, BackgroundColor="w")
fig.colorbar(im, ax=ax, shrink=0.5, label="# picks")

ax = axs[3][1]
picks_select = picks.loc[(picks.pick_method==pick_method) & 
                         (picks.score >= score_thresh) & 
                         (picks.component=="TT") & 
                         (picks.lag==lag) & 
                         (picks.stack_method==stack_method) , ["inst_period", "group_velocity"]]
heatmap, xedges, yedges, im = plot_picks(picks_select, ax=ax, bins=bins, title="")
ax.text(Tmin+0.2, vmax-0.3, "TT", fontsize=12, BackgroundColor="w")
fig.colorbar(im, ax=ax, shrink=0.5, label="# picks")

for ax in axs.ravel():
    ax.set(ylabel="Group velocity [km/s]", xlabel="Period [s]")
    
fig.tight_layout()
plt.show()
plt.close()