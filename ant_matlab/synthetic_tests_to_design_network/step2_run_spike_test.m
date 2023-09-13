%% Load grid and kernel data 
% (produced by "create_grid_and_kernel_from_stations.m" script)

folder_name = 'NANT_stations_example_D20km_N18'; % Folder where .mat files are
load([folder_name, '/kernel.mat'])
load([folder_name, '/stat_grid.mat'])
load([folder_name, '/map_matrix.mat'])

%% Define some parameters

% Synthetic model parameters
v_ref = 2.5; % Reference velocity in km/s
v_perturb = 10/100; % Velocity perturbation of the anomalies for checkerboard model 
nb_el_x = 2;  % number of elements defining the size of the spike anomalies
nb_el_y = 2;  % number of elements defining the size of the spike anomalies
% if nb_el = 2 and grid spacing is 500m, then anomalies are squares of 1 km x 1 km
ind_x_spikes = [20 10 30]; % x index on grid of the spikes
ind_y_spikes = [20 10 30]; % x index on grid of the spikes

% Period at which to measure
T = 4; % period in seconds
% Threshold to exclude station pairs too close to measure at period T
% Station pairs for which their inter-station distance is smaller than
% mul_lambda_thresh multiples of the wavelength apart are excluded (nb
% wavelength is phase velocity times period)
mul_lambda_thresh = 1.5; 

% Inversion parameters
rel_err = 5/100; % relative error on data. keep whatever, 5%, since it's not real data...
LC = 0.1; % correlation distance. higher values cause more spatial smoothing
sigma = 5; % trade-off parameter between data fit and model complexity. Higher values fit data more.

%% Cut G to station pairs that satisfy distance threshold for given period (multiples of wavelengths)

vs_ave = 3.0; % Average Vs for the area under study in the upper crust
lambda = vs_ave * T; % wavelength = velocity * period
ind_keep = find(interstation_distance > mul_lambda_thresh*lambda);
G = G_mat(ind_keep, :);
disp(['Number of ray paths that are respect ' num2str(mul_lambda_thresh) '*wavelength distance threshold: ' num2str(size(G,1)) ' out of ' num2str(nb_ray)])

%% Make synthetic checkerboard model

V_2D = ones(size(X_GRID)) * v_ref; % initialize synthetic velocity model
v_up = (1 + v_perturb) * v_ref; % velocity of positive anomalies
v_down = (1 - v_perturb) * v_ref; % Velocity of negative anomalies
ind_x = 1 : length(x_grid); 
ind_y = 1 : length(y_grid);

% MODIFY LINES BELOW FOR CUSTOM SPIKE SHAPES!
for k=1:length(ind_x_spikes)
    V_2D(ind_x_spikes(k):ind_x_spikes(k)+nb_el_x,ind_y_spikes(k):ind_y_spikes(k)+nb_el_y) = v_up; % here it puts positive anomalies. Can switch to negative
end

S_2D = 1./V_2D; % slowness synthetic model
S_lin = reshape(S_2D, [numel(X_GRID), 1]); % slowness synthetic model as a vector

%% Calculate synthetic group travel times
TAU = G * S_lin; % TAU contains the group traveltimes between each station pair

%% Perform inversion

d = TAU;  % d = travel time data
N_d = length(d); % number of data points
N_m = size(G,2); % number of model cells

% Define data prior covariance matrix
Cd_vec = (rel_err * d).^2;
CD = diag(Cd_vec);
CD_inv = diag(1 ./ Cd_vec);

% Define model prior
s_prior = 1 / v_ref; % prior homogeneous slowness
m_prior = s_prior * ones(N_m,1);

% Define model prior covariance
L0 = sqrt(dx_grid^2 + dy_grid^2); % size of model cells
x_cell = reshape(X_GRID,[N_m, 1]); y_cell=reshape(Y_GRID,[N_m, 1]);
X_CELL = repmat(x_cell,[1 N_m]); Y_CELL=repmat(y_cell,[1 N_m]);
DIST_CELL = sqrt((X_CELL-X_CELL').^2+(Y_CELL-Y_CELL').^2);
CM = (sigma * L0 / LC)^2 * exp(-1 / LC * DIST_CELL);
CM_inv = inv(CM);

% Perform Tarantola-Valette inversion
m_est = m_prior + (G' * CD_inv * G + CM_inv) \ G' * CD_inv * (d - G * m_prior);
S_map = reshape(m_est,[length(x_grid), length(y_grid)]); % recovered slowness map
V_map = 1./S_map; % recovered velocity map

%% Make a mask for plotting, to hide cells where no ray is sampling
min_density = 1; % require at least 3 ray paths to cross the cell to keep it
thres_dist = 0.01; % distance threshold in km to count a cell as being crossed by ray
G3D = reshape(G_mat',[length(x_grid) length(y_grid) size(G_mat',2)]);
G_count = zeros(size(G3D));
ind_G_ray = G3D(:) > thres_dist; % count ray if >100m in cell
G_count(ind_G_ray) = 1;
G_sum = sum(G_count,3);
mask = zeros(size(G_sum));
mask(G_sum > min_density) = 0.4; % define level of transparency

%% Plot

% in kernel, grid nodes defined at bottom left of cell (should dblcheck); 
% in imagesc, node is at center of cell; these new effective axes compensate for this
x_grid_eff = x_grid + dx_grid/2; 
y_grid_eff = y_grid + dy_grid/2; 

figure(1); 
clf; 
set(gcf,'color','w'); 
colormap('jet')
% Plot recovered model
subplot(1,2,1); cla;
set(gca,'linewidth',1.5,'fontsize',14,'layer','top')
hold on ;box on
imagesc(x_map,y_map,map) % Plot background
axis equal
set(gca,'xlim',[min(x_grid), max(x_grid)],'ylim',[min(y_grid), max(y_grid)]);
im = pcolor(x_grid_eff,y_grid_eff,V_map'); % plot data
set(im,'facealpha','flat','alphadata',mask') % apply mask
shading('flat');
plot(x_stat,y_stat,'vw','linewidth',0.5,'markersize',5, 'MarkerFaceColor','k')
cb = colorbar;
ylabel(cb, 'Velocity [km/s]')
title('Recovered model')
% plot synthetic model
subplot(1,2,2); cla;
set(gca,'linewidth',1.5,'fontsize',14,'layer','top')
hold on ;box on
imagesc(x_map,y_map,map) % Plot background
axis equal
set(gca,'xlim',[min(x_grid), max(x_grid)],'ylim',[min(y_grid), max(y_grid)]);
im = pcolor(x_grid_eff,y_grid_eff,V_2D'); % plot data
set(im,'facealpha','flat','alphadata',mask') % apply mask
shading('flat');
plot(x_stat,y_stat,'vw','linewidth',0.5,'markersize',5, 'MarkerFaceColor','k')
cb = colorbar;
ylabel(cb, 'Velocity [km/s]')
title('Synthetic model')
sgtitle(['Checkerboard test at T = ' num2str(T) ' s'])

% Save
fig_fname = [folder_name, '/spike_T' num2str(T) '_mul' num2str(mul_lambda_thresh) '_' num2str(nb_el) 'x' num2str(nb_el) '.png'];
saveas(gcf, fig_fname)
disp(['Saved figure to: ' fig_fname])