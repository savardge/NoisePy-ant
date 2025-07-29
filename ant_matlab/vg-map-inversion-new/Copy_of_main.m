function []=main()
%
%
%

%% Parameters

% STATIONS
station_file = '/media/genevieve/sandisk4TB/riehen-data/vg-maps/stations_nodes_noisepy.csv';

% GRID
min_lat = 47.4813;
max_lat = 47.6630;
min_lon = 7.5296;
max_lon = 7.8139;

% PICKS

%% Get stations and grid extent
[grid_origin, extent, stations] = get_stations_extent([min_lat, max_lat, min_lon, max_lon], station_file);

%% Read picks
% Get specific period with awk
% cmd = ['head -30 ' fname ' | awk -F, ''{ if ($2 == 2) {print $2,$3} }'' ']
% col_time = 9;
% cmd = ['cat ' fname ' | awk -F, ''{ if (substr($' num2str(col_time) ',1,6) == ' period ') {print $0} }'' ']
% [status, data] = unix(cmd)
% textscan(data, '%f %f')
% datacell = textscan(data, '%f %f')
% table(datacell{1},datacell{2}, 'VariableNames', {'T','Vg'})

pick_file_csv = '/media/genevieve/sandisk4TB/riehen-data/picks_V3_GaussFiltPeriod_rma2_normZ_rayleigh_lambda1.5_SNR5.0_margin0.5_multiple3.csv';
data = readtable(pick_file_csv);

% Filter by quality
max_std_percent = 15;
min_count = 5;
data = data(data.std_percent <= max_std_percent & data.count >= min_count,:);


% pick_file_csv = '/media/genevieve/sandisk4TB/riehen-data/vg-maps/picks_V2_topology_pws/all_picks_ZZ_lamb1.5_mul2.csv'
% data = readtable(pick_file_csv);
% data = renamevars(data,['pair'],['station_pair']);
% Filter by quality
% data = data(data.snr_nbG >= 100,:);

% Get specific period
period = 4.0;
data.inst_period = round(data.inst_period,2);
data = data(data.inst_period == period, :);

%% Set up data vector

% Filter for stations that are in station list
data = data(ismember(data.stasrc,stations.id) & ismember(data.starcv,stations.id), :);
data = sortrows(data, 'station_pair');

% Data vector (group arrival time)
V_dat = data.mean;
V_std = data.std;
% V_dat = data.group_velocity;
% V_std = data.score;

TAU_std = V_std ./ (V_dat.^2); % error for 1/x is (1/x^2)*sigmax
TAU = data.distance ./ V_dat;

%% Define grid

x_max = extent(1);
y_max = extent(2);

% Grid spacing
dx_grid = 0.15; % in km
dy_grid = dx_grid;

x_grid = 0:dx_grid:ceil(x_max);
y_grid = 0:dy_grid:ceil(y_max);
X_GRID = repmat(x_grid',[1, length(y_grid)]);
Y_GRID = repmat(y_grid,[length(x_grid), 1]);

%% Make G matrix (data kernel)

nb_cell = numel(X_GRID);
nb_ray = size(data, 1);

dl = 1e-3; % steps in distance for discrete ray path
G = zeros(nb_ray,nb_cell);
IND_LIN_GRID = reshape(1:numel(X_GRID),size(X_GRID));

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


% Average velocity (uniform starting model)
v_moy = mean(V_dat);

%% Prep inversion
L0 = sqrt(dx_grid^2 + dy_grid^2); % size of model cells
N_m = length(x_grid) * length(y_grid); % number of model cells
x_cell = reshape(X_GRID,[N_m, 1]); y_cell = reshape(Y_GRID,[N_m, 1]);
X_CELL = repmat(x_cell,[1 N_m]); Y_CELL = repmat(y_cell,[1 N_m]);
DIST_CELL = sqrt((X_CELL-X_CELL').^2+(Y_CELL-Y_CELL').^2);

%% define prior covariance matrices
% sigma = 0.2;
% sigma = 0.05;
% LC = 0.8;
sigma = 0.2;
LC = 0.5;

d = TAU;
Cd_vec1 = (TAU_std).^2;
rel_err = 20/100; % relative error on data
Cd_vec1 = (rel_err * TAU).^2;
CD1 = diag(Cd_vec1);
CD_inv1 = diag(1./Cd_vec1);
CM = (sigma * L0 / LC)^2 * exp(-1/LC*DIST_CELL);
% CM = (sigma)^2 * exp(-1/LC*DIST_CELL);
CM_inv = inv(CM);

%% inversion 
s_prior = 1/v_moy; % prior homogeneous slowness
m_prior1 = s_prior * ones(N_m,1);
d_prior1 = G * m_prior1;

% Calculate misfit for prior homogeneous model
misfit0 =  d - d_prior1;
misfit_mean0 = mean(misfit0);
misfit_std0 = std(misfit0);
var_homo1 = var(d-d_prior1); % corresponding variance of travel-times residuals
restit0 = sqrt(mean(((d-d_prior1)./d).^2))*100; % in percent

% Inversion
m_est1 = m_prior1 + (G' * CD_inv1 * G + CM_inv ) \ G' * CD_inv1 * (d - d_prior1);
d_post1 = G * m_est1;

% Calculate fit
var_post1 = var(d-d_post1); % variance of travel-time residuals after inversion
var_red1 = 1 - var_post1/var_homo1; % variation reduction
restit1 = sqrt(mean(((d-d_post1)./d).^2))*100; % in percent

% Reshape
S_map1 = reshape(m_est1,[length(x_grid), length(y_grid)]);
V_map = 1 ./ S_map1; % * 1000;

% Misfit
misfit1 =  d - d_post1;
misfit_mean1 = mean(misfit1);
misfit_std1 = std(misfit1);


%% Plot

% Get density mask
min_density = 3;
thres_dist = 10e-3; % km
G3D = reshape(G',[length(x_grid) length(y_grid) size(G',2)]);
G_count = zeros(size(G3D));
ind_G_ray = G3D(:) > thres_dist; % count ray if >100m in cell
G_count(ind_G_ray) = 1;
G_sum = sum(G_count,3);
mask = nan(size(G_sum));
mask(G_sum > min_density) = 1.0;

% grid
x_grid_eff = x_grid + dx_grid/2; 
y_grid_eff = y_grid + dy_grid/2; 

% Background
background_file = '/media/genevieve/sandisk4TB/matlab-swant/data-riehen/run1_dcV1_g500m/riehen-background-terrain.png'
map = imread(background_file);

% % Remove outliers
V_map(V_map>3.5) = nan;
V_map(V_map<0.5) = nan;
V_map(isnan(mask)) = nan;

figure(1); 
set(gcf,'color','w');clf
hold on
im = pcolor(x_grid_eff,y_grid_eff,V_map');
set(im,'facealpha','flat','alphadata',mask')
shading('interp');
imagesc(linspace(0,x_max,size(map,2)),linspace(y_max,0,size(map,1)),map,'alphadata',1.0);
axis equal
axis tight
set(gca,'xlim',[0 extent(1)],'ylim',[0 extent(2)]);
chi=get(gca,'Children');set(gca,'Children',flipud(chi))
colormap(flipud(jet));
hb=colorbar;
ylabel(hb,'Group velocity (km/s)','fontsize',14)    
set(gca,'CLim', [min(V_map(:)),max(V_map(:))])
% set(gca,'CLim', [0.5 3.5])
box on

plot(stations.xstat,stations.ystat,'vk','linewidth',1.5,'markersize',2,'markerfacecolor','k')
load("/media/genevieve/sandisk4TB/matlab-swant/data-swisstopo/swisstopo-mat/deep_wells_RiehenGrid.mat")
idx = find(deepwells.xgrid > 0 & deepwells.xgrid < x_max & deepwells.ygrid > 0 & deepwells.ygrid < y_max & deepwells.depth > 1000 ); % index in map view
hwells = scatter(deepwells.xgrid(idx), deepwells.ygrid(idx), 80, "red", 'filled', 'hexagram', 'MarkerEdgeColor', 'k', 'LineWidth',2);
 
set(gca,'linewidth',1.5,'fontsize',14) %,'layer','top')   
xlabel('Easting (km)', 'FontSize',14); 
ylabel('Northing (km)','FontSize',14);
title(sprintf('T = %.1f s, sigma = %.2f, LC = %.2f, dx=dy= %.1f km',[period, sigma, LC, dx_grid]))

%% Plot distribution
figure(2);set(gcf,'color','w');
clf
h1=subplot(2,1,1); hold on
hist(V_dat,100)
vline(v_moy,'r--')
xlabel('Group velocity (km/s)'); title('Dispersion picks')
h2=subplot(2,1,2); hold on
hist(V_map(:),100)
vline(v_moy,'r--')
linkaxes([h1,h2],'x')
xlabel('Group velocity (km/s)'); title('Inverted group velocities')

end

%% FUNCTIONS

function [grid_origin, extent, stations] = get_stations_extent(bounds, station_file)
%step1_stations_extent: get grid extent
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