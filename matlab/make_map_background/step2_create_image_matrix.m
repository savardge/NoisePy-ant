% Once a basemap image has been created with its limits corresponding to
% the 4 corners of the 2D inverson grid, an output file can be created to
% facilitated plotting of the group velocity maps and map-view slices of
% the Vs model.
%

clear all; close all;
clc

% Define paths of inputs and outputs (USER-DEFINED)
stat_grid_file = ['../../grid/stat_grid.mat']
output_file = replace(stat_grid_file, 'stat_grid', 'map_matrix_terrain')
background_file = [datadir, '/background-aargau-terrain.png'] % Name of image file with map background

%% Load grid info 

load(stat_grid_file, 'x_grid', 'y_grid', 'x_max', 'y_max', 'x_stat', 'y_stat')

%% Read background image and define x,y coordinates

map_temp = imread(background_file);

% if need to adjust
% ind_x_keep=760:3500;  
% ind_y_keep=400:2200;  % careful; axis reversed
% map=map_temp(ind_y_keep,ind_x_keep,:);

% if not
map = map_temp; 
x_map_0 = 0; 
x_map_1 = x_max;
y_map_0 = y_max; 
y_map_1 = 0; %reversed
x_map = linspace(x_map_0,x_map_1,size(map,2));
y_map = linspace(y_map_0,y_map_1,size(map,1));

% Save to output file
save(output_file, 'map', 'x_map', 'y_map')
disp(['Background image saved to :' output_file])

%% Plot
figure('position',get(0,'screensize'))
set(gca,'linewidth',1.5,'fontsize',14,'layer','top')
hold on
box on
imagesc(x_map,y_map,map)
plot(x_stat,y_stat,'vk','linewidth',1.5,'markersize',5)
axis equal
set(gca,'xlim',[0 x_max],'ylim',[0 y_max]);
xlabel('Easting (km)'); ylabel('Northing (km)');

% Save station map image
saveas(gcf,'map_stat.png')

