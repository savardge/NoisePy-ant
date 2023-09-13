% This script creates a KML file with the 4 placemarkers corresponding to
% the 4 corners of the 2D inversion grid. The latitude/longitude
% coordinates of the 4 grid corners are taken from file dist_stat.mat.
% 
% Using this KML file, one can open GoogleEarth of ArcGIS Online and take a
% custom-area screenshot of a map background (with e.g. topography, place names,
% etc.) to use as image background for map-view plots of the inverted velocity models.
% The image limits must match the 4 corners in dist_stat.mat.
%

clear all; close all;
clc

% KML output filename
name_out = 'inversion_2Dgrid.kml'; % USER-DEFINED

% Load dist_stat.mat that contains the 2D inversion grid corner coordinates (SW, NW, NE, NW)
load ../../data/D_grid_and_ray_kernel/dist_stat.mat *corner % USER-DEFINED
coord_mat = [...
    SW_corner(2) SW_corner(1); ...
    NW_corner(2) NW_corner(1); ...
    NE_corner(2) NE_corner(1); ...
    SE_corner(2) SE_corner(1)];

%% Write coordinates to output KML file

% Read template KML file to modify
fid = fopen('./KML_template.kml','rt');
temp_str = textscan(fid,'%s','delimiter','\n');
fclose(fid);

% Open output file and write grid corner coordinates
fid = fopen(name_out,'wt+');

for ind_line=1:3    
    str=temp_str{1}{ind_line};
    fprintf(fid,'%s \n',str);
end
str = ['<name>' name_out '</name>'];
fprintf(fid,'%s \n',str);

for ind_line=5:7
    str=temp_str{1}{ind_line};
    fprintf(fid,'%s \n',str);
end

for ind_mark=1:4
    fprintf(fid,'%s \n','<Placemark>');
    str1=['<name>' num2str(ind_mark) '</name>'];
    fprintf(fid,'%s \n',str1);
    fprintf(fid,'%s \n','<Point>');
    str2=['<coordinates>' num2str(coord_mat(ind_mark,1)) ',' num2str(coord_mat(ind_mark,2)) ',0</coordinates>'];
    fprintf(fid,'%s \n',str2);
    fprintf(fid,'%s \n','</Point>');
    fprintf(fid,'%s \n','</Placemark>');
end

for ind_line=14:16
    str=temp_str{1}{ind_line};
    fprintf(fid,'%s \n',str);
end

fclose(fid);