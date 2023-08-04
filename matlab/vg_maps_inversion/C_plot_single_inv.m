% This script plots the inverted group velocity map result by script B for
% a given parameter combination (sigma, LC).
%
% Written by Thomas Planes (2019)
% Modified by Genevieve Savard (2023)

clear all; %close all;
%clc

%% USER INPUTS (see also paths under plot_single_inv()
sigma = 10
LC = 0.5;
% datadir = '../../data-aargau/run3_dcV2_g500m_mul2'
datadir = '../../data-riehen/run3_dcV2_mul2_g200m'
comp = 'ZZ'

%% Plot results for a period range
Tc_list = 2.5 %[0.5 1.0 1.5 2.0 2.5]
for ii=1:length(Tc_list)
    plot_single_inv(sigma, LC, Tc_list(ii), datadir, comp)
end

%%
function []=plot_single_inv(sigma, LC, Tc, datadir, comp)

%% USER-DEFINED PATHS
load([datadir, '/grid/map_matrix_terrain_wFaults.mat'], 'map', 'x_map', 'y_map')
load([datadir,'/grid/stat_grid.mat'], 'x_grid', 'y_grid', 'x_stat', 'y_stat', 'dx_grid', 'dy_grid')
load([datadir '/vg-maps/inv_T' sprintf('%3.1f',Tc) '_full_params_set_' comp '.mat'], 'm_est_struc', 'd', 'd_post_struc', 'm_prior', 'sigma_vec', 'LC_vec','G'); 

%%
[~, ind_sigma] = min(abs(sigma_vec-sigma));
[~, ind_LC] = min(abs(LC_vec-LC));

fprintf('LC = %3.1f, sigma = %5.2f\n', LC_vec(ind_LC), sigma_vec(ind_sigma))

data_obs=d; clear d;
data_est=d_post_struc{ind_sigma,ind_LC};

restit = sqrt(mean(((data_obs-data_est)./data_obs).^2))*100; % in percent

m_est = m_est_struc{ind_sigma,ind_LC};
%v_moy=1/mean(m_prior(:));

d_homo = G*m_prior; % data generated with homogeneous medium with mean picked group velocity
var_homo = var(data_obs-d_homo); % corresponding variance of travel-times residuals
var_post = var(data_obs-data_est); % variance of travel-time residuals after inversion
var_red = 1-var_post/var_homo; % variation reduction

% in kernel, grid nodes defined at bottom left of cell (should dblcheck); 
% in imagesc, node is at center of cell; these new effective axes compensate for this
x_grid_eff = x_grid + dx_grid/2; 
y_grid_eff = y_grid + dy_grid/2; 

% Reshape
S_map = reshape(m_est,[length(x_grid), length(y_grid)]);
V_map = 1./S_map*1000;

% Get density mask
min_density = 3;
thres_dist = 0.01; % km
G3D = reshape(G',[length(x_grid) length(y_grid) size(G',2)]);
% count ray if dist travelled in cell above threshold of ~100m?
G_count = zeros(size(G3D));
ind_G_ray = G3D(:) > thres_dist; % count ray if >100m in cell
G_count(ind_G_ray) = 1;
G_sum = sum(G_count,3);
mask = zeros(size(G_sum));
mask(G_sum > min_density) = 1;

% Plot
figure('position',get(0,'screensize'), 'color','w')
set(gca,'linewidth',1.5,'fontsize',16,'layer','top')
hold on
box on
im = pcolor(x_grid_eff,y_grid_eff,V_map');
set(im,'facealpha','flat','alphadata',mask')
%, 'alphadata', mask');
% alpha(mask)
% set(im, 'facealpha', mask);
% pcolor(x_grid_eff,y_grid_eff,S_map');
shading('interp');
%imagesc(x_grid_eff,y_grid_eff,V_map');
imagesc(x_map,y_map,map,'alphadata',0.6);
axis equal
axis tight
% set(gca,'xlim',[x_map(1) x_map(end)],'ylim',[y_map(end) y_map(1)]);
colormap(flipud(jet))
hb=colorbar;
ylabel(hb,'Group velocity (m/s)','fontsize',16)
plot(x_stat,y_stat,'vk','linewidth',1.5,'markersize',10,'markerfacecolor','k')
title({['T=' num2str(Tc) ' s  (sigma=' num2str(sigma) ', Lc=' num2str(LC) ')'];['misfit data=' num2str(restit,'%.1f') ' %'];['VarRed=' num2str(var_red*100,'%.1f') ' %']});
xlabel('Easting (km)'); ylabel('Northing (km)');
%saveas(gcf,['T' num2str(Tc) 's.png'])
% set(gca, 'CLim', [1.0, 4.5])

outdir = [datadir, '/vg-maps/full_params_set']
if ~exist(outdir,'dir')
    mkdir(outdir)
end
pause(0.1)
figfile = [outdir, '/inv_T' sprintf('%3.1f',Tc) 's_sigma' num2str(sigma) '_LC' num2str(LC) comp '.png']
saveas(gcf,figfile)

end