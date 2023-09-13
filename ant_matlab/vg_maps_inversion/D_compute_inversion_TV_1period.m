% This script does the group velocity map inversion for a given period Tc
% and regulaization parameters sigma, LC.
% G.S. added consecutive inversions (up to 5 total), downweighting outliers at
% each step.
% 
% Written by Thomas Planes (2019) and Genevieve Savard (2023)

clear all; %close all

%% USER INPUTS
Tc = 1.5

datadir = '/media/savardg/sandisk4TB/matlab-swant/data-riehen/run3_dcV2_mul2_g200m'
comp = 'ZZ'
kernel_dir = [datadir '/vg-maps/data_kern_' comp ]
kernel_file = [datadir '/grid/kernel_200mX200m.mat']
map_file =[datadir '/grid/map_matrix_terrain_wFaults_200mX200m.mat'];

sigma = 8; % sigma (trade-off factor) 
LC = 0.3;  % LC (correlation distance) to choose
rel_err = 5/100; % relative error on data

%% Load stuff
load([kernel_dir '/data_and_kern_T' sprintf('%3.1f',Tc) '_' comp '.mat'], '-mat', 'v_moy', 'TAU', 'G_mat'); % G = kernel
d = TAU;
G = G_mat;
N_d = length(d); % number of data points

load(kernel_file, 'dx_grid', 'dy_grid', 'X_GRID', 'Y_GRID', 'x_grid', 'y_grid', 'x_stat', 'y_stat')
L0 = sqrt(dx_grid^2 + dy_grid^2); % size of model cells
N_m = length(x_grid) * length(y_grid); % number of model cells
x_cell = reshape(X_GRID,[N_m, 1]); y_cell = reshape(Y_GRID,[N_m, 1]);
X_CELL = repmat(x_cell,[1 N_m]); Y_CELL = repmat(y_cell,[1 N_m]);
DIST_CELL = sqrt((X_CELL-X_CELL').^2+(Y_CELL-Y_CELL').^2);

% get map background for plotting:
load(map_file, 'x_map', 'y_map', 'map')

% Get density mask
min_density = 3;
thres_dist = 0.01; % km
G3D = reshape(G_mat',[length(x_grid) length(y_grid) size(G_mat',2)]);
% count ray if dist travelled in cell above threshold of ~100m?
G_count = zeros(size(G3D));
ind_G_ray = G3D(:) > thres_dist; % count ray if >100m in cell
G_count(ind_G_ray) = 1;
G_sum = sum(G_count,3);
mask = zeros(size(G_sum));
mask(G_sum > min_density) = 0.4;

%% define prior covariance matrices

Cd_vec1 = (rel_err*d).^2;
CD1 = diag(Cd_vec1);
CD_inv1 = diag(1./Cd_vec1);

CM = (sigma * L0 / LC)^2 * exp(-1/LC*DIST_CELL);
% CM = (sigma)^2 * exp(-1/LC*DIST_CELL);
CM_inv = inv(CM);
    
%% inversion step 1
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
V_map1 = 1 ./ S_map1 * 1000;

%% inversion 2
% Misfit
misfit1 =  d - d_post1;
misfit_mean1 = mean(misfit1);
misfit_std1 = std(misfit1);

% Change CD
Cd_vec2 = Cd_vec1;
ioutliers = abs(misfit1-misfit_mean1)>2*misfit_std1;
Cd_vec2(ioutliers) = Cd_vec2(ioutliers).*exp((abs(misfit1(ioutliers))./(2*misfit_std1)-1)); % c.f. Liu and Yao 2016
CD2 = diag(Cd_vec2);
CD_inv2 = diag(1./Cd_vec2);

% Update m_prior and d_prior
m_prior2 = m_est1;
d_prior2 = G * m_prior2;
m_est2 = m_prior2 + (G' * CD_inv2 * G + CM_inv ) \ G' * CD_inv2 * (d - d_prior2);
d_post2 = G * m_est2;

% New misfit
misfit2 =  d - d_post2;
misfit_mean2 = mean(misfit2);
misfit_std2 = std(misfit2);

var_post2 = var(d-d_post2);
var_red2 = 1 - var_post2/var_homo1; % variation reduction
restit2 = sqrt(mean(((d-d_post2)./d).^2))*100; % in percent

% Reshape
S_map2 = reshape(m_est2,[length(x_grid), length(y_grid)]);
V_map2 = 1 ./ S_map2 * 1000;

% figure(101); 
% h1=subplot(2,1,1); 
% histogram(misfit1,100); vline(misfit_mean1, "r"); 
% vline(misfit_mean1-2*misfit_std1, "g");vline(misfit_mean1+2*misfit_std1, "g");
% h2=subplot(2,1,2); 
% histogram(misfit2,100); vline(misfit_mean2, "r"); 
% vline(misfit_mean2-2*misfit_std2, "g");vline(misfit_mean2+2*misfit_std2, "g");
% linkaxes([h1,h2],'xy')

%% inversion 3

% Change CD
Cd_vec3 = Cd_vec2;
ioutliers = abs(misfit2-misfit_mean2)>2*misfit_std2;
Cd_vec3(ioutliers) = Cd_vec3(ioutliers).*exp((abs(misfit2(ioutliers))./(2*misfit_std2)-1)); % c.f. Liu and Yao 2016
CD3 = diag(Cd_vec3);
CD_inv3 = diag(1./Cd_vec3);

% Update m_prior and d_prior
m_prior3 = m_est2;
d_prior3 = G * m_prior3;
m_est3 = m_prior3 + (G' * CD_inv3 * G + CM_inv ) \ G' * CD_inv3 * (d - d_prior3);
d_post3 = G * m_est3;

% New misfit
misfit3 =  d - d_post3;
misfit_mean3 = mean(misfit3);
misfit_std3 = std(misfit3);

var_post3 = var(d-d_post3);
var_red3 = 1 - var_post3/var_homo1; % variation reduction
restit3 = sqrt(mean(((d-d_post3)./d).^2))*100; % in percent

% Reshape
S_map3 = reshape(m_est3,[length(x_grid), length(y_grid)]);
V_map3 = 1 ./ S_map3 * 1000;

% figure(101); 
% h1=subplot(2,1,1); 
% histogram(misfit1,100); vline(misfit_mean1, "r"); 
% vline(misfit_mean1-2*misfit_std1, "g");vline(misfit_mean1+2*misfit_std1, "g");
% h2=subplot(2,1,2); 
% histogram(misfit3,100); vline(misfit_mean3, "r"); 
% vline(misfit_mean3-2*misfit_std3, "g");vline(misfit_mean3+2*misfit_std3, "g");
% linkaxes([h1,h2],'xy')

%% inversion 4

% % Change CD
% Cd_vec4 = Cd_vec3;
% ioutliers = abs(misfit3-misfit_mean3)>2*misfit_std3;
% Cd_vec4(ioutliers) = Cd_vec4(ioutliers).*exp((abs(misfit3(ioutliers))./(2*misfit_std3)-1)); % c.f. Liu and Yao 2016
% CD4 = diag(Cd_vec4);
% CD_inv4 = diag(1./Cd_vec4);
% 
% % Update m_prior and d_prior
% m_prior4 = m_est3;
% d_prior4 = G * m_prior4;
% m_est4 = m_prior4 + (G' * CD_inv4 * G + CM_inv ) \ G' * CD_inv4 * (d - d_prior4);
% d_post4 = G * m_est4;
% 
% % New misfit
% misfit4 =  d - d_post4;
% misfit_mean4 = mean(misfit4);
% misfit_std4 = std(misfit4);

% var_post4 = var(d-d_post4);
% var_red4 = 1 - var_post4/var_homo1; % variation reduction
% restit4 = sqrt(mean(((d-d_post4)./d).^2))*100; % in percent

% reshape
% S_map4 = reshape(m_est4,[length(x_grid), length(y_grid)]);
% V_map4 = 1 ./ S_map4 * 1000;
%% inversion 5

% % Change CD
% Cd_vec5 = Cd_vec4;
% ioutliers = abs(misfit4-misfit_mean4)>2*misfit_std4;
% Cd_vec5(ioutliers) = Cd_vec5(ioutliers).*exp((abs(misfit4(ioutliers))./(2*misfit_std4)-1)); % c.f. Liu and Yao 2016
% CD5 = diag(Cd_vec5);
% CD_inv5 = diag(1./Cd_vec5);
% 
% % Update m_prior and d_prior
% m_prior5 = m_est4;
% d_prior5 = G * m_prior5;
% m_est5 = m_prior5 + (G' * CD_inv5 * G + CM_inv ) \ G' * CD_inv5 * (d - d_prior5);
% d_post5 = G * m_est5;
% 
% % New misfit
% misfit5 =  d - d_post5;
% misfit_mean5 = mean(misfit5);
% misfit_std5 = std(misfit5);

% Calculate fit
% var_post5 = var(d-d_post5);
% var_red5 = 1 - var_post5/var_homo1; % variation reduction
% restit5 = sqrt(mean(((d-d_post5)./d).^2))*100; % in percent

% Reshape
% S_map5 = reshape(m_est5,[length(x_grid), length(y_grid)]);
% V_map5 = 1 ./ S_map5 * 1000;
%% save


%% Plot maps
% in kernel, grid nodes defined at bottom left of cell (should dblcheck); 
% in imagesc, node is at center of cell; these new effective axes compensate for this
x_grid_eff = x_grid + dx_grid/2; 
y_grid_eff = y_grid + dy_grid/2; 

figure('color','w');clf
subtightplot(1,3,1)
set(gca,'linewidth',1.5,'fontsize',12,'layer','top')
hold on
box on
im = pcolor(x_grid_eff,y_grid_eff,V_map1');
set(im,'facealpha','flat','alphadata',mask')
shading('interp');
imagesc(x_map,y_map,map,'alphadata',0.6);
axis equal
axis tight
% set(gca,'xlim',[x_map(1) x_map(end)],'ylim',[y_map(end) y_map(1)]);
colormap(flipud(jet))
% hb=colorbar;
% ylabel(hb,'Group velocity (m/s)','fontsize',14)
plot(x_stat,y_stat,'vk','linewidth',1.5,'markersize',2,'markerfacecolor','k')
% title({['T=' num2str(Tc) ' s  (sigma=' num2str(sigma) ', Lc=' num2str(LC) ')'];['misfit data=' num2str(restit1,'%.1f') ' %'];['VarRed=' num2str(var_red1*100,'%.1f') ' %']});
title({['Model after 1st inversion'];['misfit data=' num2str(restit1,'%.1f') ' %'];['Variance reduction=' num2str(var_red1*100,'%.1f') ' %']});
xlabel('Easting (km)'); ylabel('Northing (km)');
set(gca,'CLim', [min(V_map3(:)),max(V_map3(:))])
% yticklabels([])

subtightplot(1,3,2)
set(gca,'linewidth',1.5,'fontsize',12,'layer','top')
hold on
box on
im = pcolor(x_grid_eff,y_grid_eff,V_map2');
set(im,'facealpha','flat','alphadata',mask')
shading('interp');
imagesc(x_map,y_map,map,'alphadata',0.6);
axis equal
axis tight
% set(gca,'xlim',[x_map(1) x_map(end)],'ylim',[y_map(end) y_map(1)]);
colormap(flipud(jet))
% hb=colorbar;
% ylabel(hb,'Group velocity (m/s)','fontsize',14)
plot(x_stat,y_stat,'vk','linewidth',1.5,'markersize',2,'markerfacecolor','k')
% title({['T=' num2str(Tc) ' s  (sigma=' num2str(sigma) ', Lc=' num2str(LC) ')']; ...
%     ['misfit data=' num2str(restit2,'%.1f') ' %']; ...
%     ['VarRed=' num2str(var_red2*100,'%.1f') ' %']});
title({['Model after 2nd inversion']; ...
    ['misfit data=' num2str(restit2,'%.1f') ' %']; ...
    ['Variance reduction=' num2str(var_red2*100,'%.1f') ' %']});
xlabel('Easting (km)'); %ylabel('Northing (km)');
% set(gca,'CLim', [min(V_map2(:)),max(V_map2(:))])
yticklabels([])
set(gca,'CLim', [min(V_map3(:)),max(V_map3(:))])

subtightplot(1,3,3)
set(gca,'linewidth',1.5,'fontsize',12,'layer','top')
hold on
box on
im = pcolor(x_grid_eff,y_grid_eff,V_map3');
set(im,'facealpha','flat','alphadata',mask')
shading('interp');
imagesc(x_map,y_map,map,'alphadata',0.6);
axis equal
axis tight
% set(gca,'xlim',[x_map(1) x_map(end)],'ylim',[y_map(end) y_map(1)]);
colormap(flipud(jet))
hb=colorbar('Location', 'south');
set(hb,'position',[0.3640    0.1230    0.2817    0.0232])
ylabel(hb,'Group velocity (km/s)','fontsize',14)
plot(x_stat,y_stat,'vk','linewidth',1.5,'markersize',2,'markerfacecolor','k')
% title({['T=' num2str(Tc) ' s  (sigma=' num2str(sigma) ', Lc=' num2str(LC) ')']; ...
%     ['misfit data=' num2str(restit3,'%.1f') ' %']; ...
%     ['VarRed=' num2str(var_red3*100,'%.1f') ' %']});
title({['Model after 3rd inversion']; ...
    ['misfit data=' num2str(restit3,'%.1f') ' %']; ...
    ['Variance reduction=' num2str(var_red3*100,'%.1f') ' %']});
xlabel('Easting (km)'); %ylabel('Northing (km)');
yticklabels([])
set(gca,'CLim', [min(V_map3(:)),max(V_map3(:))])


%% Plot misfit distributions 

figure('color','w'); clf
h0=subplot(4,1,1); 
histogram(misfit0.*1e-3,100); 
vline(misfit_mean0.*1e-3, "r"); 
vline(misfit_mean0.*1e-3-2*misfit_std0.*1e-3, "g");
vline(misfit_mean0.*1e-3+2*misfit_std0.*1e-3, "g");
title(['Data misfit relative to constant velocity prior model: ' sprintf('%5.2f',restit0) '%']); ylabel('# measurements'); xlabel('Misfit [s]')
h1=subplot(4,1,2); 
histogram(misfit1.*1e-3,100); 
vline(misfit_mean1.*1e-3, "r"); 
vline(misfit_mean1.*1e-3-2*misfit_std1.*1e-3, "g");
vline(misfit_mean1.*1e-3+2*misfit_std1.*1e-3, "g");
title(['Data misfit after 1st inversion: ' sprintf('%5.2f',restit1) '%, variance reduction: ' num2str(var_red1*100,'%.1f') '%']); ylabel('# measurements'); xlabel('Misfit [s]')
h2=subplot(4,1,3); 
histogram(misfit2.*1e-3,100); 
vline(misfit_mean2.*1e-3, "r"); 
vline(misfit_mean2.*1e-3-2*misfit_std2.*1e-3, "g");
vline(misfit_mean2.*1e-3+2*misfit_std2.*1e-3, "g");
title(['Data misfit after 2nd inversion: ' sprintf('%5.2f',restit2) '%, variance reduction: ' num2str(var_red2*100,'%.1f') '%']); ylabel('# measurements'); xlabel('Misfit [s]')
h3=subplot(4,1,4); 
histogram(misfit3.*1e-3,100); 
vline(misfit_mean3.*1e-3, "r"); 
vline(misfit_mean3.*1e-3-2*misfit_std3.*1e-3, "g");
vline(misfit_mean3.*1e-3+2*misfit_std3.*1e-3, "g");
title(['Data misfit after 3rd inversion: ' sprintf('%5.2f',restit3) '%, variance reduction: ' num2str(var_red3*100,'%.1f') '%']); ylabel('# measurements'); xlabel('Misfit [s]')
linkaxes([h0,h1,h2,h3],'xy')



