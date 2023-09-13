function [] = A_extract_disp_curves(datadir,comp,sigma,LC)
% This script extract the local dispersion curves at each grid cell
% location from the file containing the merged Vg maps (last script in vg
% map inversion directory)
% example arguments:
% comp = 'TT'
% datadir = '../../data-riehen/run4_dcV2_mul3_g200m'
% sigma = 8;  LC = 0.3; % put chosen parameters


tic;

%% USER INPUTS
data_file = [datadir, '/vg-maps/all_data_LC' num2str(LC) '_sigma' num2str(sigma) '_' comp '.mat']
[~, name, ~] = fileparts(data_file);
load(data_file, 'DATA_V_all1','DATA_V_all2','DATA_V_all3', 'Tc_vec', 'x_grid', 'y_grid','dx_grid','dy_grid');
% DATA_V_all1 = vg maps after 1st inversion, DATA_V_all2 = vg maps after 2n
% inversion, etc.

%%
Tc_vec = double(round(Tc_vec,1)); % prevent some weird behaviour with tiny numerical precision difference
ind_x_range = 1:length(x_grid); 
ind_y_range = 1:length(y_grid);

DATA_temp1 = permute(DATA_V_all1(ind_x_range,ind_y_range,:),[3 1 2]); % put disp curve in first dimension
vg_pick_mat1 = reshape(DATA_temp1,[length(Tc_vec) length(ind_x_range)*length(ind_y_range)])*1000; % in m/s (2D matrix)
DATA_temp2 = permute(DATA_V_all2(ind_x_range,ind_y_range,:),[3 1 2]); % put disp curve in first dimension
vg_pick_mat2 = reshape(DATA_temp2,[length(Tc_vec) length(ind_x_range)*length(ind_y_range)])*1000; % in m/s (2D matrix)
DATA_temp3 = permute(DATA_V_all3(ind_x_range,ind_y_range,:),[3 1 2]); % put disp curve in first dimension
vg_pick_mat3 = reshape(DATA_temp3,[length(Tc_vec) length(ind_x_range)*length(ind_y_range)])*1000; % in m/s (2D matrix)
vg_pick_mat = vg_pick_mat3;
T_pick = Tc_vec';
T_pick_mat = repmat(T_pick, 1, size(vg_pick_mat3,2));

%% Get number of rays crossing each cell for all Vg maps

kernel_dir = [datadir '/vg-maps/data_kern_' comp ];
raycount_all = zeros([length(Tc_vec) length(x_grid) length(y_grid)]);
for ind_Tc=1:length(Tc_vec)
    Tc = Tc_vec(ind_Tc);
    % Get density mask
    load([kernel_dir '/data_and_kern_T' sprintf('%3.1f',Tc) '_' comp '.mat'], '-mat', 'G_mat');
    thres_dist = 0.01; % km
    G3D = reshape(G_mat',[length(x_grid) length(y_grid) size(G_mat',2)]);
    % count ray if dist travelled in cell above threshold of ~100m?
    G_count = zeros(size(G3D));
    ind_G_ray = G3D(:) > thres_dist; % count ray if >10m in cell
    G_count(ind_G_ray) = 1;
    G_sum = sum(G_count,3);
    raycount_all(ind_Tc,:) = reshape(G_sum, [1 length(ind_x_range)*length(ind_y_range)]);
end
raycount_total = squeeze(sum(raycount_all,1));

%% Save data with one file per index

% Get vector of indices with data
% min_density = 3;
% mask = zeros(size(raycount_total));
% mask(raycount_total > min_density) = 1;
% pcolor(x_grid,y_grid,mask')
% ind_with_data = reshape(mask, [1 length(ind_x_range)*length(ind_y_range)]);

% Output folder:
% output_dir = [datadir, '/vs-model/' name '_bycell']
% if ~exist(output_dir,'dir')
%     mkdir(output_dir);
% end

% Loop:
% for ind_lin=1:size(vg_pick_mat,2)
%     if mod(ind_lin,100) == 0
%         disp(ind_lin)
%     end
%     if ind_with_data(ind_lin) == 1 % Only write cells with data in them (according to density mask)        
%         vg_pick = vg_pick_mat(:,ind_lin);
%         vg_pick1 = vg_pick_mat1(:,ind_lin);
%         vg_pick2 = vg_pick_mat2(:,ind_lin);
%         vg_pick3 = vg_pick_mat3(:,ind_lin);
%         ray_count = raycount_all(:,ind_lin);
%         output_fname = [output_dir '/input_ind_lin_' sprintf('%d', ind_lin) '.mat'];
%         save(output_fname, 'T_pick', 'vg_pick', 'ray_count','vg_pick1','vg_pick2','vg_pick3')
%     end
% end

%% Save data with one file for all 2D grid indices
outputfile = [datadir, '/vs-model/data_picked_' name '.mat']
save(outputfile, 'T_pick', 'vg_pick_mat', ...
    'vg_pick_mat1', 'vg_pick_mat2', 'vg_pick_mat3', ...
    'ind_x_range', 'ind_y_range', ...
    'x_grid', 'y_grid', 'dx_grid', ...
    'dy_grid', 'sigma', 'LC', ...
    'raycount_all','raycount_total', ...
    'comp', 'data_file')

%% Plot curves
% figure
% plot(T_pick,vg_pick_mat);
% xlabel('Period (s)')
% ylabel('Group velocity (m/s)')
% title('Picked dispersion curves')
figure;
clf;
xbinedges = min(T_pick)-0.05:0.1:max(T_pick)+0.05;
hold on;
h2 = histogram2(T_pick_mat(:),vg_pick_mat(:),'XBinEdges', xbinedges, 'DisplayStyle','tile','ShowEmptyBins','off');
set(h2,'LineStyle','none')
set(gca,'XTick',min(T_pick):max(T_pick), ...
    'xlim',[min(T_pick),max(T_pick)], ...
    'ylim', [500 4500]);
xlabel('Period (s)')
ylabel('Vg (m/s)')
% title(['Inverted group velocity dispersion curves'])
title([comp ':' datadir])
cb2 = colorbar;
ylabel(cb2, 'pick density');

toc