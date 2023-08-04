% Template script for plotting group velocity maps. Customize as needed.
% 
% Written by Genevieve Savard (2023)

clear all; %close all;
clc

%% USER INPUTS

Tc_vec = 0.2:0.1:5.5; % range of periods to plot

datadir = '../../data-riehen'
comp = 'TT'

% Load inversion results
sigma = 8
LC = 0.3;
load([datadir '/F_inv_2D_vg_slices/grid200mX200m/all_inv_TV_sigma' num2str(sigma) '_LC' num2str(LC) '_' comp '.mat']);%,'m_est_struc','d_struc','d_post_struc','m_prior_struc','d_prior_struc','Tc_vec','x_grid','y_grid','x_stat','y_stat','dx_grid','dy_grid');

% Output dir for figures
figdir = [datadir '/F_inv_2D_vg_slices/grid200mX200m/inv_LC' sprintf('%3.1f',LC) '_sigma' num2str(sigma) '_' comp];
if ~exist(figdir,'dir')
    mkdir(figdir)
end

% Load data kernel
kernel_dir = [datadir '/F_inv_2D_vg_slices/grid200mX200m/data_kern_' comp ]

% Load map background
load([datadir, '/D_grid_and_ray_kernel/grid_200x200m/map_matrix_terrain.mat'], 'map', 'x_map', 'y_map')

% in kernel, grid nodes defined at bottom left of cell (should dblcheck); 
% in imagesc, node is at center of cell; these new effective axes compensate for this
x_grid_eff = x_grid + dx_grid/2; 
y_grid_eff = y_grid + dy_grid/2; 

%% Plot
figure(1); clf; 
set(gcf,'position',get(0,'screensize'), 'color','w')
for ind_Tc_list=1:length(Tc_vec)
    
    Tc = Tc_vec(ind_Tc_list)
    ind_Tc = find(Tc_vec==Tc);

    % output fig name
    figname = [figdir '/inv_T' num2str(Tc) 's_LC' num2str(LC) '_sigma' num2str(sigma) '_' comp '.png']

    try
        isempty(d_struc{ind_Tc})
    catch
        continue                                                                        
    end
    data_obs = d_struc{ind_Tc};
    data_est = d_post3_struc{ind_Tc};
    data_prior = d_prior1_struc{ind_Tc};
   
    restit=sqrt(mean(((data_obs-data_est)./data_obs).^2))*100; % in percent
    
    m_est = m_est3_struc{ind_Tc};
    m_prior = m_prior1_struc{ind_Tc};
    %v_moy=1/mean(m_prior(:));
    
    var_homo = var(data_obs-data_prior); % homogeneous vs observed data variance of travel-times residuals
    var_post = var(data_obs-data_est); % variance of travel-time residuals after inversion
    var_red = 1 - var_post/var_homo; % variation reduction
    
    S_map = reshape(m_est,[length(x_grid), length(y_grid)]);
    V_map = 1./S_map*1000;
        
    % Get density mask
    load([kernel_dir '/data_and_kern_T' sprintf('%3.1f',Tc) '_' comp '.mat'], '-mat', 'G_mat');
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

    %%
    clf;
    set(gca,'linewidth',1.5,'fontsize',16,'layer','top')
    hold on
    box on

    % Plot Vg map
    im = pcolor(x_grid_eff,y_grid_eff,V_map');
    set(im,'facealpha','flat','alphadata',mask')
    shading('interp');
    colormap(flipud(jet))
    hb=colorbar;
    ylabel(hb,'Group velocity (m/s)','fontsize',16)
    
    % Plot background
    imagesc(x_map,y_map,map,'alphadata',0.6);
%     set(gca,'xlim',[x_map(1) x_map(end)],'ylim',[y_map(end) y_map(1)]);
    
    % Plot stations
    plot(x_stat,y_stat,'vk','linewidth',1.5,'markersize',6,'markerfacecolor','k')

    % Plot wells 
    load("/home/savardg/research/swisstopo/Deep_wells/deep_wells_RiehenGrid.mat")
    idx = find(deepwells.xgrid > 0 & deepwells.xgrid < max(x_map) & deepwells.ygrid > 0 & deepwells.ygrid < max(y_map) ); % index in map view
    hwells = scatter(deepwells.xgrid(idx), deepwells.ygrid(idx), 80, "red", 'filled', 'hexagram', 'MarkerEdgeColor', 'k', 'LineWidth',2);
    idx_riehen2 = find(deepwells.name == "Riehen-2");
    scatter(deepwells.xgrid(idx_riehen2), deepwells.ygrid(idx_riehen2), 120, "green", 'filled', 'hexagram', 'MarkerEdgeColor', 'k', 'LineWidth',2);
    text(deepwells.xgrid(idx_riehen2)-1, deepwells.ygrid(idx_riehen2)+.6, "Riehen-2", "fontsize", 12, "FontWeight","bold") %"BackgroundColor","w")

    % Setup
    axis equal
    axis tight
    %title({['T=' num2str(Tc) ' s  (sigma=' num2str(sigma) ', Lc=' num2str(LC) ')'];['misfit data=' num2str(restit,'%.1f') ' %'];['VarRed=' num2str(var_red*100,'%.1f') ' %']});
    title(['Love wave group velocity at T = ' num2str(Tc) ' s']);
    xlabel('Easting (km)'); ylabel('Northing (km)');
    
%     set(gca, 'CLim', [0.5, 4.5])
    %% Save
    pause(0.2)    
    export_fig(figname)
%     saveas(gcf,[figdir '/inv_T' num2str(Tc) 's_LC' num2str(LC) '_sigma' num2str(sigma) '_tectocont.png'])
    %pause
%     clf
    
end


