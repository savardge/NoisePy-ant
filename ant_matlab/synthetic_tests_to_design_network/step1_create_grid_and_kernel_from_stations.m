clear all; close all;
clc

%% Define grid parameters and input files

% CSV file with station coordinates (Name,longitude,latitude)
% csv_file_stations = 'NANT_stations_example_D20km_N18.csv';

% Image file of map background where the limits of the image corresponds
% exactly to the min,max latitude and longitude below
map_background_file = 'tempest_background_terrain.png';

% Define min and max latitude and longitude of grid
min_lat = 42.035733;
max_lat = 44.314715;
min_lon =  9.174113;
max_lon = 12.316203;

% Define the grid spacing (in km)
dx_grid = 2.5; 
dy_grid = 2.5; 

%% Load station coordinates 

% Read CSV file
% csv_file_stations = 'NANT_stations_example_D20km_N18.csv';
% [filepath, name, ext] = fileparts(csv_file_stations);

% Define output directory. Here it uses the name of the csv file without
% the extension
% output_folder = name; % fullfile(filepath, name);
% if ~exist(output_folder, 'dir')
%     mkdir(output_folder)
% end
% % Import coordinates, write to ant_matlab file
% [stat_list, stat_lon, stat_lat] = read_csv_station_file(csv_file_stations);
% mat_file_stations = [output_folder '/stat_list.mat'];
% save(mat_file_stations, 'stat_list','stat_lon', 'stat_lat')
% disp(['Saved station coordinates to ' mat_file_stations])


output_folder = 'TEMPEST'
load('TEMPEST/stat_list.mat')
nb_stat = length(stat_list); % Number of stations

%% Convert coordinates from lat,long to x,y on grid using spherical projection
R_earth = 6371; % radius of Earth, don't change

SW_corner = [min_lat min_lon];
SE_corner = [min_lat max_lon];
NW_corner = [max_lat min_lon];
NE_corner = [max_lat max_lon];

ref_lat_glob = min_lat*ones(size(stat_lat));  % south west corner chosen as grid origin
ref_lon_glob = min_lon*ones(size(stat_lon));

% Max X and Y coordinates
x_max = R_earth*cos((max_lat+min_lat)/2*pi/180).*(max_lon-min_lon)*pi/180; % nb: by definition xmin=0 and ymin=0
y_max = R_earth*(max_lat-min_lat)*pi/180;

% Station coordinates to X,Y
x_stat = R_earth * cos((stat_lat+ref_lat_glob) / 2*pi / 180) .* (stat_lon-ref_lon_glob) * pi/180;
y_stat = R_earth * (stat_lat - ref_lat_glob) * pi/180;

%% Calculate inter-station distances

DIST_mat = zeros(nb_stat,nb_stat);
for ind_stat1=1:nb_stat
    for ind_stat2=1:nb_stat
        dx = x_stat(ind_stat2) - x_stat(ind_stat1);
        dy = y_stat(ind_stat2) - y_stat(ind_stat1);       
        DIST_mat(ind_stat1,ind_stat2) = sqrt(dx^2 + dy^2); % distance between stations in km        
    end
end
dist_stat_fname = [output_folder, '/dist_stat.mat'];
save(dist_stat_fname, 'DIST_mat','x_stat','y_stat','stat_list','x_max','y_max','SW_corner','SE_corner','NW_corner','NE_corner','-mat')
disp(['Saved matrix of inter-stations distances to: ' dist_stat_fname])

%% Make grid

x_grid = 0:dx_grid:ceil(x_max);
y_grid = 0:dy_grid:ceil(y_max);
X_GRID = repmat(x_grid',[1, length(y_grid)]);
Y_GRID = repmat(y_grid,[length(x_grid), 1]);

stat_grid_fname = [output_folder, '/stat_grid.mat'];
save(stat_grid_fname, 'X_GRID', 'Y_GRID', 'x_stat', 'y_stat', 'x_grid', 'y_grid', 'dx_grid', 'dy_grid','x_max','y_max');
disp(['Saved grid and station x,y coordinates in: ' stat_grid_fname])

% plot
figure(1); set(gcf,'color','w');clf
plot(x_stat,y_stat,'vr', 'MarkerFaceColor','r');
hold on
plot(X_GRID,Y_GRID,'+k');
axis([0 x_max 0 y_max])
axis equal
title(['Grid and station locations: ' num2str(length(x_grid)) 'x' num2str(length(y_grid)) ' nodes.'])
xlim([min(x_grid), max(x_grid)])
ylim([min(y_grid), max(y_grid)])
fig_fname = [output_folder, '/station_map_grid_simple.png'];
saveas(gcf, fig_fname, 'png')
disp(['Saved station map and grid to: ' fig_fname])

%% Create kernel G

nb_cell = numel(X_GRID);
nb_ray = nb_stat * (nb_stat-1)/2;

interstation_distance = zeros(nb_ray,1);
ray_mat = cell(nb_stat,nb_stat);
dl = 0.1; %0.01;  % Plot ray path interpolated every dl km

iray = 1;
for s1 = 1:nb_stat-1
    for s2 = s1+1:nb_stat
        
        % Get inter-station distance
        delta_x = x_stat(s2) - x_stat(s1); 
        delta_y = y_stat(s2) - y_stat(s1);
        dist = sqrt(delta_x^2 + delta_y^2); % Distance between stations
        interstation_distance(iray) = dist;
       
        % get ray path x and y coordinates
        ux_ray = delta_x / dist;
        uy_ray = delta_y / dist;
        ray_x = x_stat(s1) + (0:dl:dist) * ux_ray;
        ray_y = y_stat(s1) + (0:dl:dist) * uy_ray;
        ray_mat{s1,s2} = [ray_x' ray_y'];       
        
        iray = iray + 1;
    end
end

% G_ij=distance traveled by ray i in cell j
G_mat = zeros(nb_ray, nb_cell);
IND_LIN_GRID = reshape(1:numel(X_GRID),[length(x_grid), length(y_grid)]);
IND_S1 = zeros(nb_ray,1); % to retrieve station from ray index
IND_S2 = zeros(nb_ray,1);
ind_ray = 0;
for s1 = 1:nb_stat-1
    for s2 = s1+1:nb_stat

        ind_ray = ind_ray + 1;
        IND_S1(ind_ray) = s1; IND_S2(ind_ray) = s2;  % to retrieve station from ray index

        x_ray_vec = ray_mat{s1,s2}(:,1);
        y_ray_vec = ray_mat{s1,s2}(:,2);

        x_ind = floor((x_ray_vec - x_grid(1)) / dx_grid) + 1;    % x ind of cell it falls on (if always positive values?)
        y_ind = floor((y_ray_vec - y_grid(1)) / dy_grid) + 1;    % y ind of cell it falls on (if always positive values?)

        for rr = 1:length(x_ray_vec) % points along the ray

            G_mat(ind_ray, IND_LIN_GRID(x_ind(rr), y_ind(rr))) = G_mat(ind_ray, IND_LIN_GRID(x_ind(rr), y_ind(rr))) + dl;

        end

    end
end

disp(['Number of ray paths in G_mat: ',num2str(ind_ray)])
% Save
kernel_fname = [output_folder '/kernel.mat'];
save(kernel_fname,'G_mat','dx_grid', 'dy_grid', 'X_GRID', 'Y_GRID', 'x_grid', 'y_grid', 'x_stat', 'y_stat', 'interstation_distance', '-v7.3');
disp(['Saved kernel G in: ' kernel_fname])

%% Create map background for plotting

map = imread(map_background_file);

x_map_0 = 0; 
x_map_1 = x_max;
y_map_0 = y_max; 
y_map_1 = 0; %reversed
x_map = linspace(x_map_0,x_map_1,size(map,2));
y_map = linspace(y_map_0,y_map_1,size(map,1));

% Plot just stations and background
figure(2); clf; set(gcf, 'color', 'w')
set(gca,'linewidth',1.5,'fontsize',14,'layer','top')
hold on ;box on
imagesc(x_map,y_map,map)
plot(x_stat,y_stat,'vk','linewidth',1.5,'markersize',5, 'MarkerFaceColor','r')
axis equal
set(gca,'xlim',[0 x_max],'ylim',[0 y_max]);
xlabel('Easting (km)'); ylabel('Northing (km)');
title('Station map')
saveas(gcf,[output_folder, '/station_map_with_background.png'])

% Plot stations, grid and background
figure(3); clf; set(gcf, 'color', 'w')
set(gca,'linewidth',1.5,'fontsize',14,'layer','top')
hold on ;box on
imagesc(x_map,y_map,map)
plot(X_GRID,Y_GRID,'+k');
plot(x_stat,y_stat,'vk','linewidth',1.5,'markersize',5, 'MarkerFaceColor','r')
axis equal
set(gca,'xlim',[0 x_max],'ylim',[0 y_max]);
xlabel('Easting (km)'); ylabel('Northing (km)');
title('Station map')
saveas(gcf,[output_folder, '/grid_map_with_background.png'])

% Save
map_matrix_fname = [output_folder, '/map_matrix.mat'];
save(map_matrix_fname, 'map', 'x_map', 'y_map')
disp(['Save map background matrix to: ' map_matrix_fname])
