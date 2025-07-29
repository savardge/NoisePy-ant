% datadir = '../../data-riehen/run3_dcV2_mul2_g200m'
datadir = '../../data-aargau/run3_dcV2_mul2_g500m'
comp = 'ZZ'

output_path = [datadir, '/vg-maps/ray-path-density-plots'];
if ~exist(output_path,"dir")
    mkdir(output_path)
end

load([datadir '/dist_stat.mat'], 'DIST_mat', 'stat_list', 'net_list')
kernel_dir = [datadir '/vg-maps/data_kern_' comp ]
grid_file = [datadir '/grid/stat_grid.mat']

mapfile = [datadir, '/grid/map_matrix_terrain_wFaults.mat']

%% Load
load(grid_file, 'x_grid', 'y_grid', 'dx_grid', 'dy_grid')
load([datadir '/dist_stat.mat'], 'DIST_mat', 'stat_list', 'net_list', 'x_stat', 'y_stat')
load(mapfile, 'map', 'x_map', 'y_map')

%% Loop over periods
Tc_list = 0.2:0.1:5.5

for ind_Tc=1:length(Tc_list)

    Tc = Tc_list(ind_Tc);
    disp(['Period = ', num2str(Tc)])
    figname = [output_path '/density_map_' comp '_T' sprintf('%03.1f', Tc) '.png']
    load([kernel_dir '/data_and_kern_T' sprintf('%3.1f',Tc) '_' comp '.mat'], '-mat', 'v_moy', 'TAU', 'G_mat'); % G = kernel

    figure(1); set(gcf,'color','w','Position',[1921 77 1920 880]); clf
    plot_ray_density(G_mat, x_grid, y_grid, dx_grid, dy_grid, x_stat, y_stat, 1, x_map, y_map, map)
    
    export_fig(figname)
end