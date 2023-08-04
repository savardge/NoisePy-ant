
% datafiles = {
%     '../../data-riehen/run3_dcV2_mul2_g200m/vs-model/data_picked_all_data_LC0.3_sigma8_ZZ.mat', ...
%     '../../data-riehen/run3_dcV2_mul2_g200m/vs-model/data_picked_all_data_LC0.3_sigma8_ZZ-ZR.mat', ...
%     '../../data-riehen/run3_dcV2_mul2_g200m/vs-model/data_picked_all_data_LC0.3_sigma8_ZR.mat', ...
%     '../../data-riehen/run3_dcV2_mul2_g200m/vs-model/data_picked_all_data_LC0.3_sigma8_RR-RZ.mat', ...
%     '../../data-riehen/run3_dcV2_mul2_g200m/vs-model/data_picked_all_data_LC0.3_sigma8_RZ.mat', ...
%     '../../data-riehen/run3_dcV2_mul2_g200m/vs-model/data_picked_all_data_LC0.3_sigma8_RR.mat', ...
%     }
% output_file = '../../data-riehen/run3_dcV2_mul2_g200m/vs-model/data_picked_all_data_LC0.3_sigma8_RayleighMean.mat'

% datafiles = {
%     '../../data-riehen/run4_dcV2_mul3_g200m/vs-model/data_picked_all_data_LC0.3_sigma8_ZZ.mat', ...
%     '../../data-riehen/run4_dcV2_mul3_g200m/vs-model/data_picked_all_data_LC0.3_sigma8_ZZ-ZR.mat', ...
%     '../../data-riehen/run4_dcV2_mul3_g200m/vs-model/data_picked_all_data_LC0.3_sigma8_ZR.mat'}
% output_file = '../../data-riehen/run4_dcV2_mul3_g200m/vs-model/data_picked_all_data_LC0.3_sigma8_RayleighMean.mat'

% datafiles = {
%     '../../data-aargau/run4_dcV2_mul3_g500m/vs-model/data_picked_all_data_LC0.8_sigma4_ZZ.mat', ...
%     '../../data-aargau/run4_dcV2_mul3_g500m/vs-model/data_picked_all_data_LC0.8_sigma4_ZR.mat', ...
%     '../../data-aargau/run4_dcV2_mul3_g500m/vs-model/data_picked_all_data_LC0.8_sigma4_ZZ-ZR.mat'}%, ...
% %     '../../data-aargau/run4_dcV2_mul3_g500m/vs-model/data_picked_all_data_LC0.8_sigma4_RZ.mat'}    
% output_file = '../../data-aargau/run4_dcV2_mul3_g500m/vs-model/data_picked_all_data_LC0.8_sigma4_RayleighMeanNOR.mat'

% datafiles = {
%     '../../data-aargau/run3_dcV2_mul2_g500m/vs-model/data_picked_all_data_LC0.8_sigma4_ZZ.mat', ...
%     '../../data-aargau/run3_dcV2_mul2_g500m/vs-model/data_picked_all_data_LC0.8_sigma4_ZR.mat', ...
%     '../../data-aargau/run3_dcV2_mul2_g500m/vs-model/data_picked_all_data_LC0.8_sigma4_ZZ-ZR.mat'}%, ...
% %     '../../data-aargau/run3_dcV2_mul2_g500m/vs-model/data_picked_all_data_LC0.8_sigma4_RZ.mat', ...
% %     '../../data-aargau/run3_dcV2_mul2_g500m/vs-model/data_picked_all_data_LC0.8_sigma4_RR-RZ.mat', ...
% %     '../../data-aargau/run3_dcV2_mul2_g500m/vs-model/data_picked_all_data_LC0.8_sigma4_RR.mat'}
%  output_file = '../../data-aargau/run3_dcV2_mul2_g500m/vs-model/data_picked_all_data_LC0.8_sigma4_RayleighMeanNOR.mat'

load(datafiles{1})
raycount_sum = zeros(size(raycount_total(:)));
vg_pick_mat_mean = zeros(size(vg_pick_mat));
vg_pick_mat_all = zeros([size(vg_pick_mat) length(datafiles)]);
raycount_mat_all = zeros([size(vg_pick_mat) length(datafiles)]);
for k=1:length(datafiles)
    datafile = datafiles{k};
    load(datafile, 'vg_pick_mat','raycount_total')
    raycount_weight = repmat(reshape(raycount_total,[1 length(ind_x_range)*length(ind_y_range)]),[size(vg_pick_mat,1) 1]);
    vg_pick_mat_mean = vg_pick_mat_mean + raycount_weight.*vg_pick_mat;
    raycount_sum = raycount_sum + raycount_total(:);    

    vg_pick_mat_all(:,:,k) = vg_pick_mat;
    raycount_mat_all(:,:,k) = raycount_weight;
end
raycount_normalize = repmat(reshape(raycount_sum,[1 length(ind_x_range)*length(ind_y_range)]),[size(vg_pick_mat,1) 1]);
vg_pick_mat_mean = vg_pick_mat_mean ./ raycount_normalize;

vg_pick_mat_wmean = zeros(size(vg_pick_mat_mean));
vg_pick_mat_wstd = zeros(size(vg_pick_mat_mean));
for i=1:size(vg_pick_mat_wmean,1)
    for j=1:size(vg_pick_mat_wmean,2)
        if sum(squeeze(raycount_mat_all(i,j,:))) == 0; continue;end
        weighted_std = std(squeeze(vg_pick_mat_all(i,j,:)),squeeze(raycount_mat_all(i,j,:)));
        weighted_mean = sum(squeeze(vg_pick_mat_all(i,j,:)).*squeeze(raycount_mat_all(i,j,:)))/sum(squeeze(raycount_mat_all(i,j,:)));
        vg_pick_mat_wmean(i,j) = weighted_mean;
        vg_pick_mat_wstd(i,j) = weighted_std;
    end
end

vg_pick_mat = vg_pick_mat_wmean;
save(output_file, 'T_pick', 'vg_pick_mat', ...
    'datafiles','vg_pick_mat_wstd', 'raycount_mat_all', 'vg_pick_mat_all', ...
    'ind_x_range', 'ind_y_range', ...
    'x_grid', 'y_grid', 'dx_grid', ...
    'dy_grid', 'sigma', 'LC')    

%% Plot curves
% figure
% plot(T_pick,vg_pick_mat);
% xlabel('Period (s)')
% ylabel('Group velocity (m/s)')
% title('Picked dispersion curves')
figure;
clf;
xbinedges = min(T_pick)-0.05:0.1:max(T_pick)+0.05;
ybinedges = 500:100:3500;

subplot(1,2,1)
hold on;
T_pick_mat = repmat(T_pick, 1, size(vg_pick_mat,2));
h2 = histogram2(T_pick_mat(:),vg_pick_mat_mean(:), ...
    'XBinEdges', xbinedges,'YBinEdges', ybinedges, 'DisplayStyle','tile','ShowEmptyBins','off');
set(h2,'LineStyle','none')
set(gca,'XTick',min(T_pick):max(T_pick), ...
    'xlim',[min(T_pick),max(T_pick)], ...
    'ylim', [500 4500]);
xlabel('Period (s)')
ylabel('Vg (m/s)')
title([comp ':' datadir])
cb2 = colorbar;
ylabel(cb2, 'pick density');

subplot(1,2,2)
hold on;
h2 = histogram2(T_pick_mat(:),vg_pick_mat_wmean(:), ...
    'XBinEdges', xbinedges,'YBinEdges', ybinedges, 'DisplayStyle','tile','ShowEmptyBins','off');
set(h2,'LineStyle','none')
set(gca,'XTick',min(T_pick):max(T_pick), ...
    'xlim',[min(T_pick),max(T_pick)], ...
    'ylim', [500 4500]);
xlabel('Period (s)')
ylabel('Vg (m/s)')
title([comp ':' datadir])
cb2 = colorbar;
ylabel(cb2, 'pick density');

%% Plot mean model
datadir = '../../data-aargau/run3_dcV2_mul2_g500m';
load([datadir, '/grid/map_matrix_terrain.mat'], 'map', 'x_map', 'y_map')
load([datadir, '/grid/kernel.mat'], 'x_grid','y_grid','x_stat','y_stat','dx_grid','dy_grid')
k = 25;
T = T_pick(k);
V_map = reshape(vg_pick_mat(k,:),[length(x_grid) length(y_grid)]);
std_map = reshape(vg_pick_mat_wstd(k,:),[length(x_grid) length(y_grid)]);
% in kernel, grid nodes defined at bottom left of cell (should dblcheck); 
% in imagesc, node is at center of cell; these new effective axes compensate for this
x_grid_eff = x_grid + dx_grid/2; 
y_grid_eff = y_grid + dy_grid/2; 

minvg = 100;
mask = nan(size(V_map));
mask(V_map > minvg ) = 1;
V = V_map(V_map>minvg ); vmoy = mean(V(:));
V_map_anom = V_map - vmoy;

figure(1); clf; 
% set(gcf, 'Position', [2479 58 1118 809])
set(gcf, 'color','w')    

subplot(1,2,1);cla
hold on
% im = pcolor(x_grid_eff,y_grid_eff,V_map'); % model
im = pcolor(x_grid_eff,y_grid_eff,V_map_anom'); % model
set(im,'facealpha','flat','alphadata',mask')
shading('interp');
imagesc(x_map,y_map,map,'alphadata',1.0); % background
xlim([min(x_map) max(x_map)])
ylim([min(y_map) max(y_map)])
chi=get(gca,'Children');set(gca,'Children',flipud(chi))
% colorbar
colormap(flipud(jet));
hb=colorbar;
% ylabel(hb,'Group velocity (m/s)','fontsize',12)
% set(gca,'CLim',[min(V) max(V)])
ylabel(hb,'Group velocity anomaly (m/s)','fontsize',12)
set(gca,'CLim',[min(V-vmoy) max(V-vmoy)])
% Plot stations
plot(x_stat,y_stat,'vk','linewidth',1.5,'markersize',2,'markerfacecolor','k')
set(gca,'linewidth',1.5,'fontsize',14,'layer','top')
box on
axis equal
axis tight
title('Mean model')

subplot(1,2,2);cla
hold on
im = pcolor(x_grid_eff,y_grid_eff,std_map'); % model
set(im,'facealpha','flat','alphadata',mask')
shading('interp');
imagesc(x_map,y_map,map,'alphadata',1.0); % background
xlim([min(x_map) max(x_map)])
ylim([min(y_map) max(y_map)])
chi=get(gca,'Children');set(gca,'Children',flipud(chi))
% colorbar
colormap(flipud(jet));
hb=colorbar;
ylabel(hb,'Group velocity deviation (m/s)','fontsize',12)
set(gca,'CLim',[min(std_map(V_map>minvg )) max(std_map(V_map>minvg ))])
% Plot stations
plot(x_stat,y_stat,'vk','linewidth',1.5,'markersize',2,'markerfacecolor','k')
set(gca,'linewidth',1.5,'fontsize',14,'layer','top')
box on
axis equal
axis tight
title('Weighted standard deviation')