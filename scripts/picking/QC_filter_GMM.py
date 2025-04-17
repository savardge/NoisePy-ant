"""
Filter dispersion picks at each period T by fitting a Gaussian Mixture Model (GMM) to the histogram of group velocities, and exclude outliers outside a defined standard deviation window. Then merge the filtered picks into a single file and visualize the pick distribution.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.mixture import GaussianMixture
from findpeaks import findpeaks
from matplotlib.ticker import AutoMinorLocator
import matplotlib as mpl

# Matplotlib aesthetics
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams.update({'font.size': 14})


def get_mean(inst_periods, group_velocity):
    inst_periods_uniq = np.unique(inst_periods)
    gv_moy = np.zeros(len(inst_periods_uniq))
    gv_std = np.zeros(len(inst_periods_uniq))
    for i, per in enumerate(inst_periods_uniq):
        mask = inst_periods == per
        gv_moy[i] = np.mean(group_velocity[mask])
        gv_std[i] = np.std(group_velocity[mask])
    return inst_periods_uniq, gv_moy, gv_std


def plot_picks(picks, ax=None, dmax=None, bins=100, title="Pick density"):
    picks2 = picks.copy()
    inst_periods_uniq, gv_moy, gv_std = get_mean(
        picks2.inst_period.values, picks2.group_velocity_mean.values
    )
    heatmap, xedges, yedges = np.histogram2d(
        picks2.inst_period, picks2.group_velocity_mean, bins=bins
    )
    heatmap = heatmap[1:, 10:]
    xedges = xedges[1:]
    yedges = yedges[10:]
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 10))
        im = ax.imshow(heatmap.T, extent=extent, origin='lower', cmap='nipy_spectral',
                       vmin=0 if dmax is None else 0, vmax=dmax)
        ax.set_title(title)
        plt.tight_layout()
        plt.colorbar(im, shrink=0.5, label="# picks")
        plt.show()
        plt.close()
    else:
        im = ax.imshow(heatmap.T, extent=extent, origin='lower', cmap='nipy_spectral',
                       vmin=0 if dmax is None else 0, vmax=dmax)
        ax.plot(inst_periods_uniq, gv_moy, 'w--', lw=2)
        ax.plot(inst_periods_uniq, gv_moy + 2 * gv_std, 'w:', lw=1)
        ax.plot(inst_periods_uniq, gv_moy - 2 * gv_std, 'w:', lw=1)
        ax.set_title(title)
        ax.set_xlim(xedges[0], xedges[-1])
        ax.set_ylim(yedges[0], yedges[-1])
        ax.xaxis.set_minor_locator(AutoMinorLocator(10))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.tick_params(which='both', width=1)
        ax.tick_params(which='major', length=9)
        ax.tick_params(which='minor', length=5, color='k')
    return heatmap, xedges, yedges, im


def process_dispersion_picks(input_folder, output_fname, T_array, wave, score_thresh, snr_thresh, margin, multiple,
                             vave, bins):
    dflist_picks_all = []
    for T in T_array:
        print(f"Processing T = {T:.1f}s")
        fname = os.path.join(input_folder, f"picks_merged_V3_rma2_normZ_lambda1.5_SNR5.0_T{T:.1f}.csv")
        picks = pd.read_csv(fname)
        picks["station_pair"] = picks["stasrc"] + "_" + picks["starcv"]

        if wave == "love":
            picks_select = picks.query(
                "score >= @score_thresh and component == 'TT' and snr_nbG >= @snr_thresh"
            )
        elif wave == "rayleigh":
            valid_comps = ['ZZ', 'all4', 'ZZ-ZR', 'ZR', 'RZ']
            picks_select = picks.query(
                "score >= @score_thresh and component in @valid_comps and snr_nbG >= @snr_thresh"
            )
        else:
            continue

        X = picks_select.group_velocity.values
        counts, edges = np.histogram(X, bins=bins)
        counts = savgol_filter(counts, 20, 1, mode='nearest')

        fp = findpeaks(method='topology', verbose=0, limit=10)
        results = fp.fit(counts)
        imax, scores = results["persistence"]["y"], results["persistence"]["score"]

        peaks = [(edges[pos], score) for pos, score in zip(imax, scores) if 20 <= pos < len(counts) - 50]
        if not peaks:
            continue
        peak_vg = max(peaks, key=lambda x: x[1])[0]

        x = X[(X > peak_vg - margin) & (X < peak_vg + margin)].reshape(-1, 1)
        gmm = GaussianMixture(n_components=1, max_iter=1000, random_state=10, covariance_type='spherical')
        gmm.fit(x)

        gauss_mean = float(gmm.means_[0][0])
        gauss_std = np.sqrt(float(gmm.covariances_[0]))

        picks_keep = picks_select.query(
            "@gauss_mean - @multiple * @gauss_std < group_velocity < @gauss_mean + @multiple * @gauss_std"
        )

        picks_by_pair = picks_keep.groupby('station_pair').agg(
            group_velocity_mean=('group_velocity', 'mean'),
            group_velocity_median=('group_velocity', 'median'),
            group_velocity_std=('group_velocity', 'std'),
            group_velocity_min=('group_velocity', 'min'),
            group_velocity_max=('group_velocity', 'max'),
            group_velocity_count=('group_velocity', 'count')
        ).reset_index()

        picks_by_pair['std_percent'] = (
                picks_by_pair['group_velocity_std'] / picks_by_pair['group_velocity_mean'] * 100
        )
        picks_by_pair['inst_period'] = T

        pair_info = picks_select.drop_duplicates(subset='station_pair')[
            ['station_pair', 'azimuth', 'backazimuth', 'distance', 'stasrc', 'starcv']
        ]
        picks_by_pair = picks_by_pair.merge(pair_info, on='station_pair', how='left')
        picks_by_pair['ratio_d_lambda'] = (
                picks_by_pair['distance'] / (picks_by_pair['inst_period'] * vave)
        )

        dflist_picks_all.append(picks_by_pair)

    picks_all_T = pd.concat(dflist_picks_all).drop_duplicates()
    print(f"Saving to {output_fname}")
    picks_all_T.to_csv(output_fname, index=False)
    return picks_all_T


def main():
    # Parameters
    vmin, vmax, dvel = 0.2, 3.5, 0.01
    bins = np.arange(vmin, vmax, dvel)
    score_thresh = 0.9
    snr_thresh = 5
    margin = 0.5
    multiple = 3
    vave = 3.0
    T_array = np.arange(0.2, 6.0, 0.1)
    wave = "rayleigh"
    output_folder = "/home/users/s/savardg/aargau_ant/dispersion-curves/"
    input_folder = os.path.join(output_folder, "picks_merged_V3_rma2_normZ_lambda1.5_SNR5.0_by_T")
    output_fname = os.path.join(
        output_folder,
        f"picks_V3_GaussFiltPeriod_rma2_normZ_{wave}_lambda1.5_SNR{snr_thresh:.1f}_margin{margin:.1f}_multiple{multiple}_chatgpt.csv"
    )
    output_plot = os.path.join(output_folder, f"picks_plot_{wave}.png")

    picks_all_T = process_dispersion_picks(
        input_folder, output_fname, T_array, wave,
        score_thresh, snr_thresh, margin, multiple, vave, bins
    )

    picks_select = picks_all_T.loc[
        (picks_all_T.ratio_d_lambda > 1.5) &
        (picks_all_T.std_percent < 10) &
        (picks_all_T.group_velocity_count > 6)
        ]

    Tmin, Tmax, dT = 0.201, 6.501, 0.1
    vmin_plot, vmax_plot = 0.501, 3.501
    bins_plot = [np.arange(Tmin, Tmax, dT), np.arange(vmin_plot, vmax_plot, dvel)]

    fig, ax = plt.subplots(figsize=(12, 8))
    heatmap, xedges, yedges, im = plot_picks(picks_select, ax=ax, bins=bins_plot, title="Mean")
    fig.colorbar(im, ax=ax, shrink=0.5, label="# picks")
    ax.set(title=f"SNR >= {snr_thresh}")
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    print(f"Saved plot to {output_plot}")
    plt.close()


# if __name__ == "__main__":

main()
