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