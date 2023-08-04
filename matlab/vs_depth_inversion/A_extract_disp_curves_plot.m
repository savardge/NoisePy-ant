% Plot the extracted local dispersion curves derived from group velocity
% maps along with the raw picked dispersion curves picked between station
% pairs.
% Genevieve Savard (2023)

clear all;% close all;
clc

%% USER-DEFINED INPUTS
comp = 'ZZ'

sigma = 8;  LC = 0.3; % put chosen regularization parameters
datadir = '../../data-riehen/run3_dcV2_mul2_g200m'
data_file = [datadir, '/vg-maps/all_data_LC' num2str(LC) '_sigma' num2str(sigma) '_' comp '.mat']
% [~, name, ~] = fileparts(data_file);
load(data_file, 'DATA_V_all1','DATA_V_all2','DATA_V_all3', 'Tc_vec', 'x_grid', 'y_grid','dx_grid','dy_grid');
DATA_V_all = DATA_V_all3; % Take the results from the 3rd Vg map inversion step

%% 
Tc_vec = double(round(Tc_vec,1)); % prevent some weird behaviour with tiny numerical precision difference
ind_x_range = 1:length(x_grid); 
ind_y_range = 1:length(y_grid);
DATA_temp = permute(DATA_V_all(ind_x_range,ind_y_range,:),[3 1 2]); % put disp curve in first dimension

%% Ray path density mask
% kernel_dir = [datadir '/vg-maps/data_kern_' comp ]
% raycount_all = zeros([length(Tc_vec) length(x_grid) length(y_grid)]);
% for ind_Tc=1:length(Tc_vec)
%     Tc = Tc_vec(ind_Tc)
%     % Get density mask
%     load([kernel_dir '/data_and_kern_T' sprintf('%3.1f',Tc) '_' comp '.mat'], '-mat', 'G_mat');
%     thres_dist = 0.01; % km
%     G3D = reshape(G_mat',[length(x_grid) length(y_grid) size(G_mat',2)]);
%     % count ray if dist travelled in cell above threshold of ~100m?
%     G_count = zeros(size(G3D));
%     ind_G_ray = G3D(:) > thres_dist; % count ray if >10m in cell
%     G_count(ind_G_ray) = 1;
%     G_sum = sum(G_count,3);
%     raycount_all(ind_Tc,:) = reshape(G_sum, [1 length(ind_x_range)*length(ind_y_range)]);
% end
% min_density = 3;
% raycount_total = squeeze(sum(raycount_all,1));
% mask = zeros(size(raycount_total));
% mask(raycount_total > min_density) = 1;
% % pcolor(x_grid,y_grid,mask')
% ind_with_data = reshape(mask, [1 length(ind_x_range)*length(ind_y_range)]);

load([datadir '/grid/kernel.mat' ],'-mat','G_mat')
thres_dist = 0.01; % km
G3D = reshape(G_mat',[length(x_grid) length(y_grid) size(G_mat',2)]);
G_count = zeros(size(G3D));
ind_G_ray = G3D(:) > thres_dist; % count ray if >10m in cell
G_count(ind_G_ray) = 1;
G_sum = sum(G_count,3);
raycount = reshape(G_sum, [1 length(ind_x_range)*length(ind_y_range)]);
mask = zeros(size(raycount));
min_density = 3;
mask(raycount > min_density) = 1;

% Keep data with picks
vg_pick_mat = reshape(DATA_temp,[length(Tc_vec) length(ind_x_range)*length(ind_y_range)])*1000; % in m/s (2D matrix)
vg_pick_mat(:,~mask) = [];
T_pick = Tc_vec';
T_pick_mat = repmat(T_pick, 1, size(vg_pick_mat,2));

%% Save data separately for each cell

% output_dir = [datadir, '/vg-maps/' name '_bycell']
% if ~exist(output_dir,'dir')
%     mkdir(output_dir);
% end
% 
% for ind_lin=1:size(vg_pick_mat,2)
%     if mod(ind_lin,100) == 0
%         disp(ind_lin)
%     end
%     if ind_with_data(ind_lin) == 1 % Only write cells with data in them (according to density mask)        
%         vg_pick = vg_pick_mat(:,ind_lin);
%         ray_count = raycount_all(:,ind_lin);
%         output_fname = [output_dir '/input_ind_lin_' sprintf('%d', ind_lin) '.mat'];
%         save(output_fname, 'T_pick', 'vg_pick', 'ray_count')
%     end
% end
% 
% %save data_picked.mat T_pick vg_pick_mat ind_x_range ind_y_range x_grid y_grid dx_grid dy_grid sigma LC

%% Simple plot of extracted local dispersion curves
% figure
% plot(T_pick,vg_pick_mat);
% xlabel('Period (s)')
% ylabel('Group velocity (m/s)')
% title('Picked dispersion curves')

%% Plot with comparison to inter-station dispersion picks

% Riehen
[~,out] = unix(['ls ' datadir '/picks/*_' comp '_*']);
pickfile = strip(out)
load(pickfile)
load([datadir '/dist_stat.mat'], 'DIST_mat', 'stat_list', 'net_list')

min_T = 0.5; 
max_T = 5; 
dT = 0.1;
nb_stat = length(fieldnames(PICK_CELL)); % 2023 version
% nb_stat=size(PICK_CELL,1); % 2022
T_pick_all = [];
vg_pick_all = [];
for ss=1:nb_stat-1

    sta_src = [net_list{ss} '_' stat_list{ss}];  % 2023 version

    for rr=ss+1:nb_stat

        % 2022 version
%         pick_temp = PICK_CELL{ss,rr};
%         if ~isempty(pick_temp)
%             T_pick_all = [T_pick_all; pick_temp(:,1)];
%             vg_pick_all = [vg_pick_all; pick_temp(:,2) * 1000];
%         end

        % 2023 version
        sta_rcv = [net_list{rr} '_' stat_list{rr}];
        try
            pick_temp = getfield(PICK_CELL, sta_src, sta_rcv)';
                    
            if ~isempty(pick_temp)
            
                dist_plot=DIST_mat(ss,rr);

%                 color_plot=[0 0 0];
%                 plot(pick_temp(:,1),pick_temp(:,2),'-','linewidth',1.5,'color',color_plot);
                T_pick_all = [T_pick_all; double(round(pick_temp(:,1),1))];
                vg_pick_all = [vg_pick_all; double(pick_temp(:,2)) * 1000];
            end
        catch
            %disp(['Data not found for pair ', sta_src, '-', sta_rcv])
        end
        
    end
end

% Plot
figure;
clf;
xbinedges = min_T-0.05:dT:max_T+0.05;
% Plot density of picks
ax1 = subplot(1,2,1); hold on
ind = find(T_pick_all >= min_T & T_pick_all <= max_T);
vg_pick_all = vg_pick_all(ind);
T_pick_all = T_pick_all(ind);
h1 = histogram2(T_pick_all,vg_pick_all,'XBinEdges',xbinedges,'DisplayStyle','tile','ShowEmptyBins','off');

% Plot vg_moy at each period
T_uniq = round(unique(T_pick_all),1);
vg_moy = zeros(length(T_uniq),1);
vg_mode = zeros(length(T_uniq),1);
for k=1:length(T_uniq)
    vg_moy(k) = mean(vg_pick_all(T_pick_all == T_uniq(k)));
    vg_mode(k) = mode(vg_pick_all(T_pick_all == T_uniq(k)));
end
hold on;
plot(T_uniq, vg_moy, 'k--', 'LineWidth',2)
plot(T_uniq, vg_mode, 'r--', 'LineWidth',2)

% set(gca,'ylim',[1500 3500]); 
set(gca,'XTick',min_T:max_T,'xlim',[min_T,max_T]);
xlabel('Period (s)');
ylabel('Vg (m/s)');
title('Picked group velocity curves')
cb1 = colorbar;
ylabel(cb1, 'pick density');

ax2 = subplot(1,2,2); cla; hold on;
h2 = histogram2(T_pick_mat(:),vg_pick_mat(:),'XBinEdges', xbinedges, 'DisplayStyle','tile','ShowEmptyBins','off');
set(h2,'LineStyle','none')
set(gca,'XTick',min_T:max_T,'xlim',[min_T,max_T]);
xlabel('Period (s)');
ylabel('Vg (m/s)');
title('Inverted group velocity curves')
cb2 = colorbar;
ylabel(cb2, 'pick density');
hold on;
plot(T_uniq, vg_moy, 'k--', 'LineWidth',2)
% ylims = get(ax2,'ylim');
% set(ax1,'ylim',ylims)
set([ax1,ax2],'ylim',[500 3000])