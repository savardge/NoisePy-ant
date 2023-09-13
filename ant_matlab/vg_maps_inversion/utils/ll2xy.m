function [x,y] = ll2xy(lat,long, SW_corner)
% Convert lat,long to X,Y using spherical coord projection
R_earth = 6371;
% SW_corner = [min_lat min_lon]
ref_lat_glob = SW_corner(1) * ones(size(lat));  % south west corner chosen as grid origin
ref_lon_glob = SW_corner(2) * ones(size(long));
x = R_earth * cos((lat + ref_lat_glob)/2*pi/180).*(long - ref_lon_glob) *pi/180;
y = R_earth * (lat - ref_lat_glob) * pi/180;

end