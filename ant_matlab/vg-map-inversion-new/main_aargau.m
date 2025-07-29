function []=main_aargau(period) %period, sigma, LC)
%
%
%
tstart = tic;

LC_vec = [0.2 0.5 1 1.5 2 5];
sigma_vec = [0.0 0.05 0.1 0.5 1]

wave = 'rayleigh';

output_root = '/home/users/s/savardg/scratch/matlab-swant/data-aargau/run6_dcV3_mul3_g200m/vg-maps-new'

save_output = true;


%% Parameters
% STATIONS
station_file = '/home/users/s/savardg/aargau_ant/aargau_stations_AA_NEZ.csv'

% GRID
min_lat = 47.3680;
max_lat = 47.6236;
min_lon = 8.0432;
max_lon = 8.4041;
dx_grid = 0.2; % in km
dy_grid = dx_grid;
% dx_grid2 = 0.25; % in km
% dy_grid2 = dx_grid2;

% PICKS
% V3
pick_file_csv = ['/home/users/s/savardg/aargau_ant/dispersion-curves/picks_V3_GaussFiltPeriod_rma2_normZ_' wave '_lambda1.5_SNR5.0_margin0.5_multiple3.csv'];
max_std_percent = 20;
min_count = 3;

% V2
% pick_file_csv = '/media/genevieve/sandisk4TB/riehen-data/vg-maps/picks_V2_topology_pws/all_picks_ZZ_lamb1.5_mul2.csv'
% min_snr_nbG = 10;


%% Get stations and grid extent
[grid_origin, extent, stations] = get_stations_extent([min_lat, max_lat, min_lon, max_lon], station_file);

%% Read picks
data = readtable(pick_file_csv);

% Get specific period
data.inst_period = round(data.inst_period,2);
data = data(data.inst_period == period, :);

% Filter by quality
% V3
data = data(data.std_percent <= max_std_percent & data.count >= min_count,:);
data = renamevars(data,['mean'],['group_velocity']);

% V2
% data = renamevars(data,['pair'],['station_pair']);
% Filter by quality
% data = data(data.snr_nbG >= min_snr_nbG,:);

% Filter for picks with stations that are exclusively in station list
data = data(ismember(data.stasrc,stations.id) & ismember(data.starcv,stations.id), :);
data = sortrows(data, 'station_pair');

sprintf('Number of picks/ray paths: %d', size(data,1))

%% Set up data vector

% Data vector (group arrival time)
V_dat = data.group_velocity;
TAU = data.distance ./ V_dat;

% V_std = data.std;
% TAU_std = V_std ./ (V_dat.^2); % error for 1/x is (1/x^2)*sigmax

%% Inversion grid and kernel
% Define grid
x_grid = 0:dx_grid:ceil(extent(1));
y_grid = 0:dy_grid:ceil(extent(2));

% Get data kernel
[G, mask, G_sum] = get_data_kernel(x_grid, y_grid, data, stations);

% Prior model (uniform starting model with average velocity )
N_m = length(x_grid) * length(y_grid); % number of model cells
v_moy = mean(V_dat);
v_prior = v_moy * ones(N_m,1);

%% Inversions

for k=1:length(LC_vec)
    LC = LC_vec(k);

    for n=1:length(sigma_vec)
        sigma = sigma_vec(n);
        

        % Output paths
        output_folder = [output_root '/inv_TV_' wave '_sigma' num2str(sigma) '_LC' num2str(LC)]
        if ~exist(output_folder, 'dir')
            mkdir(output_folder)
        end
        output_fname = [output_folder, '/inv_TV_' wave '_sigma' num2str(sigma) '_LC' num2str(LC) '_T' sprintf('%.1f', period) '.mat']
        output_fig = [output_folder, '/inv_TV_' wave '_sigma' num2str(sigma) '_LC' num2str(LC) '_T' sprintf('%.1f', period) '.png']

        % Tarantola-Valette inversion 
        fprintf('Launching inversion with sigma = %.2e LC=%.2e\n', sigma, LC)
        tic
         [V_map, stats] = TV_inversion_2step(x_grid, y_grid, sigma, LC, TAU, v_prior, G);        
%        [V_map, stats] = TV_inversion(x_grid, y_grid, sigma, LC, TAU, v_prior, G);
        toc
        
        % Save
        if save_output
            pick_params = struct( ...
                'pick_file_csv', pick_file_csv, ...
                'max_std_percent', max_std_percent, ...
                'min_count', min_count, ...
                'station_file', station_file, ...
                'period', period);
            grid_params = struct( ...
                'grid_origin', grid_origin, ...
                'extent', extent, ...
                'dx_grid', dx_grid, ...
                'dy_grid', dy_grid, ...
                'x_grid', x_grid, ...
                'y_grid', y_grid);
            
            save(output_fname, 'grid_params', 'pick_params', 'V_map', 'stats', 'mask', 'sigma', 'LC', 'G_sum')
        end

        % Plot
        fignum = 1;
        plot_map(V_map, V_dat, stats, mask, stations, x_grid, y_grid, fignum, period, sigma, LC)
        saveas(gcf,output_fig)
        
    
    end
end

fprintf('Done.')
toc(tstart)


end

%% FUNCTIONS

function []=plot_map(V_map, V_dat, stats, mask, stations, x_grid, y_grid, fignum, period, sigma, LC)

%% Plot
% cmap = brewermap(256,'RdBu');
% cmap = brewermap(256,'RdBu');
cmap = flipud(jet);

% grid
dx_grid = x_grid(2) - x_grid(1);
dy_grid = y_grid(2) - y_grid(1);
x_grid_eff = x_grid + dx_grid/2; 
y_grid_eff = y_grid + dy_grid/2; 

% Background
%background_file = '/media/genevieve/sandisk4TB/matlab-swant/data-riehen/run1_dcV1_g500m/riehen-background-terrain.png';
background_file = '/home/users/s/savardg/scratch/matlab-swant/data-riehen/run1_dcV1_g500m/riehen-background-terrain.png';
map = imread(background_file);

% % Remove outliers
V_map(V_map>4.0) = nan;
V_map(V_map<0.5) = nan;
V_map(isnan(mask)) = nan;

figure(fignum); 
set(gcf,'color','w');clf

subplot(2,4,[1 2 5 6])
hold on
im = pcolor(x_grid_eff,y_grid_eff,V_map');
set(im,'facealpha','flat','alphadata',mask')
shading('interp');
imagesc(linspace(0,x_grid(end),size(map,2)),linspace(y_grid(end),0,size(map,1)),map,'alphadata',1.0);
axis equal
axis tight
set(gca,'xlim',[x_grid(1) x_grid(end)],'ylim',[y_grid(1) y_grid(end)]);
chi=get(gca,'Children');set(gca,'Children',flipud(chi))
colormap(cmap);
hb=colorbar;
ylabel(hb,'Group velocity (km/s)','fontsize',14)    
% set(gca,'CLim', [min(V_map(:)),max(V_map(:))])
% set(gca,'CLim', [0.5 3.5])
box on

% Plot stations
plot(stations.xstat,stations.ystat,'vk','linewidth',1.5,'markersize',2,'markerfacecolor','k')

% Plot wells
min_depth_m = 1000;
%load("/media/genevieve/sandisk4TB/matlab-swant/data-swisstopo/swisstopo-mat/deep_wells_RiehenGrid.mat")
load("/home/users/s/savardg/scratch/matlab-swant/data-swisstopo/swisstopo-mat/deep_wells_RiehenGrid.mat")
idx = find(deepwells.xgrid > x_grid(1) & deepwells.xgrid < x_grid(end) & deepwells.ygrid > y_grid(1) & deepwells.ygrid < y_grid(end) & deepwells.depth > min_depth_m); % index in map view
hwells = scatter(deepwells.xgrid(idx), deepwells.ygrid(idx), 80, "red", 'filled', 'hexagram', 'MarkerEdgeColor', 'k', 'LineWidth',2);
 
set(gca,'linewidth',1.5,'fontsize',14) %,'layer','top')   
xlabel('Easting (km)', 'FontSize',14); 
ylabel('Northing (km)','FontSize',14);
title(sprintf('T = %.1f s, sigma = %.2f, LC = %.2f, dx=dy= %.1f km\n%d picks used',[period, sigma, LC, dx_grid]))

% Plot Vg distribution
v_moy = mean(V_dat);
h2 = subplot(2,4,3); cla; hold on
histogram(V_dat,100, FaceColor="g")
vline(v_moy,'r--')
xlabel('Group velocity (km/s)'); title('Dispersion picks')
h3 = subplot(2,4,7); cla; hold on
histogram(V_map(:),100, FaceColor="g")
vline(v_moy,'r--')
linkaxes([h2,h3],'x')
xlabel('Group velocity (km/s)'); title('Inverted group velocities')

% Plot misfit distribution
h4 = subplot(2,4,4);  cla;
histogram(stats.misfit_prior,100); 
vline(mean(stats.misfit_prior), "r"); 
vline(mean(stats.misfit_prior)-2*std(stats.misfit_prior), "g");
vline(mean(stats.misfit_prior)+2*std(stats.misfit_prior), "g");
title(['Misfit prior model: ' sprintf('%5.2f',stats.restit_prior) '%']); 
ylabel('# measurements'); xlabel('Misfit [s]')
h5 = subplot(2,4,8);  cla;
histogram(stats.misfit_post,100); 
vline(mean(stats.misfit_post), "r"); 
vline(mean(stats.misfit_post)-2*std(stats.misfit_post), "g");
vline(mean(stats.misfit_post)+2*std(stats.misfit_post), "g");
title(sprintf('Misfit after inversion: %5.2f %%,\n variance reduction: %5.1f %%',stats.restit_post, stats.var_red*100)); 
ylabel('# measurements'); xlabel('Misfit [s]')
linkaxes([h4,h5],'x')


end


function [G, mask, G_sum] = get_data_kernel(x_grid, y_grid, data, stations)
% Make G matrix (data kernel)
% extent: grid extent in km from SW origin
% x_grid, y_grid: grid array in x and y in km
% data: table of picks with columns: stasrc, starcv
% stations: table of stations with columns id (NET.STA), xstat, ystat 
dx_grid = x_grid(2) - x_grid(1);
dy_grid = y_grid(2) - y_grid(1);

X_GRID = repmat(x_grid',[1, length(y_grid)]);
% Y_GRID = repmat(y_grid,[length(x_grid), 1]);

nb_cell = numel(X_GRID); % number of grid cells
nb_ray = size(data, 1);  % Number of ray paths

dl = 1e-3; % steps in distance for discrete ray path
G = zeros(nb_ray,nb_cell);
IND_LIN_GRID = reshape(1:numel(X_GRID),size(X_GRID)); % Linear grid index

for iray = 1:nb_ray

    % Station pair and coordinates
    stasrc = data{iray, 'stasrc'};       
    xstat_src = stations{strcmp(stations.id, stasrc),'xstat'};
    ystat_src = stations{strcmp(stations.id, stasrc),'ystat'};    
    starcv = data{iray, 'starcv'}; 
    xstat_rcv = stations{strcmp(stations.id, starcv),'xstat'};
    ystat_rcv = stations{strcmp(stations.id, starcv),'ystat'};

    % Get ray path coordinates
    delta_x = xstat_rcv - xstat_src;
    delta_y = ystat_rcv - ystat_src;    
    dist = sqrt(delta_x^2 + delta_y^2);            
    ux_ray = delta_x/dist; uy_ray = delta_y/dist;            
    x_ray_vec = xstat_src + (0:dl:dist)' * ux_ray;
    y_ray_vec = ystat_src + (0:dl:dist)' * uy_ray;    
    x_ind = floor((x_ray_vec - x_grid(1)) / dx_grid) + 1;    % x ind of cell it falls on (if always positive values?)
    y_ind = floor((y_ray_vec - y_grid(1)) / dy_grid) + 1;    % y ind of cell it falls on (if always positive values?)

    % Convert subscripts to linear indices
    linear_indices = sub2ind(size(IND_LIN_GRID), x_ind, y_ind);

    % Accumulate values in G using linear indexing
    G(iray, :) = accumarray(linear_indices, dl, [nb_cell, 1])' + G(iray, :);
end

% Get density mask
min_density = 3; % Minimum number of ray paths crossing cell
thres_dist = 10e-3; % Minimum distance being traveled in cell in km
G3D = reshape(G',[length(x_grid) length(y_grid) size(G',2)]);
G_count = zeros(size(G3D));
ind_G_ray = G3D(:) > thres_dist; % count ray if >100m in cell
G_count(ind_G_ray) = 1;
G_sum = sum(G_count,3);
mask = nan(size(G_sum));
mask(G_sum > min_density) = 1.0;

end

function [V_map, stats] = TV_inversion_2step(x_grid, y_grid, sigma, LC, TAU, v_prior, G)
% Tarantola-Valette inversion
% x_grid
%

%% Prep inversion

dx_grid = x_grid(2) - x_grid(1);
dy_grid = y_grid(2) - y_grid(1);
X_GRID = repmat(x_grid',[1, length(y_grid)]);
Y_GRID = repmat(y_grid,[length(x_grid), 1]);
L0 = sqrt(dx_grid^2 + dy_grid^2); % size of model cells
N_m = length(x_grid) * length(y_grid); % number of model cells
x_cell = reshape(X_GRID,[N_m, 1]); y_cell = reshape(Y_GRID,[N_m, 1]);
X_CELL = repmat(x_cell,[1 N_m]); Y_CELL = repmat(y_cell,[1 N_m]);
DIST_CELL = sqrt((X_CELL-X_CELL').^2+(Y_CELL-Y_CELL').^2);

%% define prior covariance matrices

d = TAU;
% Cd_vec = (TAU_std).^2;
rel_err = 10/100; % relative error on data
Cd_vec = (rel_err * TAU).^2;
% CD1 = diag(Cd_vec);
CD_inv1 = diag(1./Cd_vec);
% CD_inv1 = spdiags(1./Cd_vec1, length(Cd_vec1),length(Cd_vec1));
CM = (sigma * L0 / LC)^2 * exp(-1/LC*DIST_CELL);
% CM = (sigma)^2 * exp(-1/LC*DIST_CELL);
CM_inv = inv(CM);

%% inversion 1
m_prior1 = ones(N_m,1) ./ v_prior;
d_prior1 = G * m_prior1;

% Inversion
m_est1 = m_prior1 + (G' * CD_inv1 * G + CM_inv ) \ G' * CD_inv1 * (d - d_prior1);
d_post1 = G * m_est1;

clear CD_inv1 m_prior1

% Misfit
misfit1 =  d - d_post1;

% Update CD to downweight outliers
ioutliers = abs(misfit1-mean(misfit1)) > 2 * std(misfit1);
Cd_vec(ioutliers) = sqrt((Cd_vec(ioutliers).^2).*exp((abs(misfit1(ioutliers))./(2*std(misfit1)))-1)); % c.f. Eqn 6 in Liu, C., & Yao, H. (2017). Surface Wave Tomography with Spatially Varying Smoothing Based on Continuous Model Regionalization. Pure and Applied Geophysics, 174(3), 937–953. https://doi.org/10.1007/s00024-016-1434-5
% % CD2 = diag(Cd_vec);
CD_inv2 = diag(1./Cd_vec);
% CD_inv2 = spdiags(1./Cd_vec, length(Cd_vec),length(Cd_vec));

% Update m_prior and d_prior
m_prior2 = m_est1;
d_prior2 = G * m_prior2;

% Inversion 2
m_est2 = m_prior2 + (G' * CD_inv2 * G + CM_inv ) \ G' * CD_inv2 * (d - d_prior2);
d_post2 = G * m_est2;

clear CD_inv2 CM_inv G m_prior2

% Calculate misfit for prior model
var_prior = var(d-d_prior1); % corresponding variance of travel-times residuals
restit_prior = sqrt(mean(((d-d_prior1)./d).^2))*100; % in percent

% Calculate fit
var_post = var(d-d_post2); % variance of travel-time residuals after inversion
var_red = 1 - var_post/var_prior; % variation reduction
restit_post = sqrt(mean(((d-d_post2)./d).^2))*100; % in percent

% Misfit
misfit_prior =  d - d_prior1;
misfit_post =  d - d_post2;

% Reshape
S_map2 = reshape(m_est2,[length(x_grid), length(y_grid)]);
V_map = 1 ./ S_map2; % * 1000;

stats = struct( ...
    'var_prior', var_prior, ...
    'var_post', var_post, ...
    'var_red', var_red, ...
    'restit_prior', restit_prior, ...
    'restit_post', restit_post, ...
    'misfit_post', misfit_post, ...
    'misfit_prior',misfit_prior...
    );

end

function [V_map, stats] = TV_inversion(x_grid, y_grid, sigma, LC, TAU, v_prior, G)
% Tarantola-Valette inversion
% x_grid
%

%% Prep inversion

dx_grid = x_grid(2) - x_grid(1);
dy_grid = y_grid(2) - y_grid(1);
X_GRID = repmat(x_grid',[1, length(y_grid)]);
Y_GRID = repmat(y_grid,[length(x_grid), 1]);
L0 = sqrt(dx_grid^2 + dy_grid^2); % size of model cells
N_m = length(x_grid) * length(y_grid); % number of model cells
x_cell = reshape(X_GRID,[N_m, 1]); y_cell = reshape(Y_GRID,[N_m, 1]);
X_CELL = repmat(x_cell,[1 N_m]); Y_CELL = repmat(y_cell,[1 N_m]);
DIST_CELL = sqrt((X_CELL-X_CELL').^2+(Y_CELL-Y_CELL').^2);

%% define prior covariance matrices

d = TAU;
% Cd_vec = (TAU_std).^2;
rel_err = 10/100; % relative error on data
Cd_vec = (rel_err * TAU).^2;
% CD1 = diag(Cd_vec);
CD_inv = diag(1./Cd_vec);
% CD_inv = spdiags(1./Cd_vec, length(Cd_vec),length(Cd_vec));
CM = (sigma * L0 / LC)^2 * exp(-1/LC*DIST_CELL);
% CM = (sigma)^2 * exp(-1/LC*DIST_CELL);
CM_inv = inv(CM);

%% inversion 
% s_prior = 1./v_prior; % prior homogeneous slowness
m_prior = ones(N_m,1) ./ v_prior;
d_prior = G * m_prior;

% Inversion
m_est = m_prior + (G' * CD_inv * G + CM_inv ) \ G' * CD_inv * (d - d_prior);
d_post = G * m_est;

% Reshape
S_map = reshape(m_est,[length(x_grid), length(y_grid)]);
V_map = 1 ./ S_map; % * 1000;

% Calculate misfit for prior model
var_prior = var(d-d_prior); % corresponding variance of travel-times residuals
restit_prior = sqrt(mean(((d-d_prior)./d).^2))*100; % in percent

% Calculate fit
var_post = var(d-d_post); % variance of travel-time residuals after inversion
var_red = 1 - var_post/var_prior; % variation reduction
restit_post = sqrt(mean(((d-d_post)./d).^2))*100; % in percent

% Misfit
misfit_prior =  d - d_prior;
misfit_post =  d - d_post;

stats = struct( ...
    'var_prior', var_prior, ...
    'var_post', var_post, ...
    'var_red', var_red, ...
    'restit_prior', restit_prior, ...
    'restit_post', restit_post, ...
    'misfit_post', misfit_post, ...
    'misfit_prior',misfit_prior...
    );

end

function [grid_origin, extent, stations] = get_stations_extent(bounds, station_file)
%get_stations_extent: get grid extent
%   bounds: [min_lat, max_lat, min_lon, max_lon]
%   station_file: CSV file with columns station, latitude, longitude,
% Returns: [grid_origin, extent, stations]

%% Specify extent
min_lat = bounds(1); %47.4813;
max_lat = bounds(2); %47.6630;
min_lon = bounds(3); %7.5296;
max_lon = bounds(4); %7.8139;
grid_origin = [min_lat, min_lon]; % SW corner

% Grid extent
[xmax, ymax] = ll2xy_sphericalproj(max_lat, max_lon, grid_origin);
extent = [xmax, ymax];
sprintf('Grid origin (SW corner) lat,long: ( %.4f, %.4f )\nGrid extent: %.2f km by %.2f km', min_lat, min_lon, xmax, ymax)

%% Get stations
stations = readtable(station_file);
if find(ismember(stations.Properties.VariableNames,'channel'))
    stations = stations(contains(stations.channel, 'Z'),:);
end
nsta0 = height(stations);

% Keep stations inside grid
[xstat, ystat] = ll2xy_sphericalproj(stations.latitude, stations.longitude, grid_origin);
stations.xstat = xstat;
stations.ystat = ystat;
stations = stations(xstat > 0 & xstat < xmax & ystat > 0 & ystat < ymax,:);
stations.id = strcat(stations.network, '.', stations.station);
nsta = height(stations);
sprintf('Number of stations inside grid: %d (%d removed).', nsta, nsta0-nsta)

end

function [x, y] = ll2xy_sphericalproj(lat, lon, grid_origin)
%ll2xy_sphericalproj: : Spherical coordinate projection
%   lat: vector of latitudes
%   lon: vector of longitudes
%   grid_origin: SW corner of grid: [min_lat, min_lon]
% Returns: [x, y] coordinates in km

R_earth = 6371;

ref_lat = grid_origin(1)*ones(size(lat));  % south west corner chosen as grid origin
ref_lon = grid_origin(2)*ones(size(lon));

x = R_earth * cos((lat + ref_lat) / 2 * pi/180).*(lon - ref_lon) * pi/180;
y = R_earth * (lat - ref_lat) * pi/180;

end
