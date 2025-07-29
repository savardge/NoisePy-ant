import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def plot_map(V_map, V_dat, stats, mask, stations, x_grid, y_grid, fignum, period, sigma, LC):
    dx_grid = x_grid[1] - x_grid[0]
    dy_grid = y_grid[1] - y_grid[0]
    x_grid_eff = x_grid + dx_grid / 2
    y_grid_eff = y_grid + dy_grid / 2

    V_map = np.where((V_map > 3.5) | (V_map < 0.5), np.nan, V_map)
    V_map = np.where(np.isnan(mask), np.nan, V_map)

    fig = plt.figure(fignum, figsize=(15, 8))
    fig.clf()
    fig.patch.set_facecolor('white')

    ax1 = plt.subplot(2, 4, (1, 2, 5, 6))
    ax1.set_aspect('equal')
    im = ax1.pcolormesh(x_grid_eff, y_grid_eff, V_map.T, shading='auto')
    im.set_alpha(np.nan_to_num(mask.T, nan=0))

    # Optional background map overlay
    try:
        background_file = 'data/hautesorne_background_terrain_faults.png'
        bg = mpimg.imread(background_file)
        ax1.imshow(bg, extent=[0, x_grid[-1], 0, y_grid[-1]], origin='upper', alpha=1.0)
    except FileNotFoundError:
        pass

    ax1.set_xlim([x_grid[0], x_grid[-1]])
    ax1.set_ylim([y_grid[0], y_grid[-1]])
    hb = plt.colorbar(im, ax=ax1)
    hb.set_label('Group velocity (km/s)', fontsize=14)

    ax1.plot(stations['xstat'], stations['ystat'], 'vk', markersize=3, label='Stations')
    ax1.set_title(f'T = {period:.1f} s, σ = {sigma:.2e}, LC = {LC:.2e}, dx=dy={dx_grid:.1f} km\n{len(V_dat)} picks used')
    ax1.set_xlabel('Easting (km)', fontsize=12)
    ax1.set_ylabel('Northing (km)', fontsize=12)
    ax1.grid(True)

    ax2 = plt.subplot(2, 4, 3)
    ax2.hist(V_dat, bins=100, color='g')
    ax2.axvline(np.mean(V_dat), color='r', linestyle='--')
    ax2.set_title('Dispersion picks')
    ax2.set_xlabel('Group velocity (km/s)')

    ax3 = plt.subplot(2, 4, 7)
    ax3.hist(V_map[~np.isnan(V_map)], bins=100, color='g')
    ax3.axvline(np.mean(V_dat), color='r', linestyle='--')
    ax3.set_title('Inverted group velocities')
    ax3.set_xlabel('Group velocity (km/s)')

    ax4 = plt.subplot(2, 4, 4)
    ax4.hist(stats['misfit_prior'], bins=100, color='skyblue')
    ax4.axvline(np.mean(stats['misfit_prior']), color='r')
    ax4.axvline(np.mean(stats['misfit_prior']) - 2 * np.std(stats['misfit_prior']), color='g')
    ax4.axvline(np.mean(stats['misfit_prior']) + 2 * np.std(stats['misfit_prior']), color='g')
    ax4.set_title(f"Misfit prior model: {stats['restit_prior']:.2f}%")
    ax4.set_xlabel('Misfit [s]')
    ax4.set_ylabel('# measurements')

    ax5 = plt.subplot(2, 4, 8)
    ax5.hist(stats['misfit_post'], bins=100, color='skyblue')
    ax5.axvline(np.mean(stats['misfit_post']), color='r')
    ax5.axvline(np.mean(stats['misfit_post']) - 2 * np.std(stats['misfit_post']), color='g')
    ax5.axvline(np.mean(stats['misfit_post']) + 2 * np.std(stats['misfit_post']), color='g')
    ax5.set_title(f"Misfit after inversion: {stats['restit_post']:.2f}%\nVariance reduction: {stats['var_red'] * 100:.1f}%")
    ax5.set_xlabel('Misfit [s]')
    ax5.set_ylabel('# measurements')

    plt.tight_layout()
    plt.show()
