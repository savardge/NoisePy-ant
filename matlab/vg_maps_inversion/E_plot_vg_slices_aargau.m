% Ploting script for Aargau, that also plots other data on maps like faults, heatflow, etc.
% Use as inspiration if you wish...
%
% Genevieve Savard (2023)

clear all; %close all;
clc

%%
% invtype = 'single'
invtype = 'three'
% datadir = '../../data-aargau/run4_dcV2_mul3_g500m'
datadir = '../../data-aargau/run3_dcV2_mul2_g500m'
% datadir = '../../data-aargau/run0_dc2022_g1km'
comp = 'RR-RZ'
load([datadir '/dist_stat.mat'], 'SW_corner')
load("../../data-swisstopo/swisstopo-mat/deep_wells_AargauGrid.mat")
swisstopodir_faults = '../../data-swisstopo/swisstopo-mat/Accident_tecto_aargau/';
swisstopodir_heat = '../../data-swisstopo/swisstopo-mat/geothermie_aargau/';
 
sigma = 4; %8
LC = 0.8; %0.3;
alldata = false % if alldata_*.mat exists
figdir = [datadir '/vg-maps/all_inv_TV_sigma' num2str(sigma) '_LC' num2str(LC)];
if ~exist(figdir,'dir')
    mkdir(figdir)
end

kernel_dir = [datadir '/vg-maps/data_kern_' comp ]
load([datadir, '/grid/map_matrix_terrain.mat'], 'map', 'x_map', 'y_map')
% load([datadir, '/grid/map_matrix.mat'], 'map', 'x_map', 'y_map')
if alldata
    load([datadir '/vg-maps/all_inv_TV_sigma' num2str(sigma) '_LC' num2str(LC) '/all_inv_TV_sigma' num2str(sigma) '_LC' num2str(LC) '_' comp '.mat']);%,'m_est_struc','d_struc','d_post_struc','m_prior_struc','d_prior_struc','Tc_vec','x_grid','y_grid','x_stat','y_stat','dx_grid','dy_grid');
else
    load([datadir, '/grid/kernel.mat'], 'x_grid','y_grid','x_stat','y_stat','dx_grid','dy_grid')
end


% in kernel, grid nodes defined at bottom left of cell (should dblcheck); 
% in imagesc, node is at center of cell; these new effective axes compensate for this
x_grid_eff = x_grid + dx_grid/2; 
y_grid_eff = y_grid + dy_grid/2; 

Tc_vec_choose = [0.2:0.1:6.0] %5.5:0.1:6.5 %0.2:0.1:5.5 %[2.5 3.0 3.5 4.0]; 
%Tc_vec_choose = 5.0:0.1:6.5 %[2.5 3.0 3.5 4.0]; 

for ind_Tc_list=1:length(Tc_vec_choose) 

    Tc = Tc_vec_choose(ind_Tc_list)
    
    % Load data
    fname = [datadir '/vg-maps/all_inv_TV_sigma' num2str(sigma) '_LC' num2str(LC) '/all_inv_TV_sigma' num2str(sigma) '_LC' num2str(LC) '_' comp '_T' sprintf('%3.1f',Tc) '.mat'];
    if ~exist(fname,'file')
        continue
    end
    load(fname);
    

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
    mask = nan(size(G_sum));
    mask(G_sum > min_density) = 1.0;

    if alldata
        data_obs = d_struc{ind_Tc};
        ind_Tc = find(abs(Tc_vec-Tc) < 1e-8);
        switch invtype
            case 'single'
                data_est = d_post_struc{ind_Tc};            
                m_est = m_est_struc{ind_Tc};
                m_prior = m_prior_struc{ind_Tc};
    %             data_prior = d_prior_struc{ind_Tc};
                data_prior = G_mat * m_prior;
            case 'three'            
    %             data_est = d_post1_struc{ind_Tc};
    %             m_est = m_est1_struc{ind_Tc};
    %             m_prior = m_prior1_struc{ind_Tc};
                data_est = d_post3_struc{ind_Tc};
                m_est = m_est3_struc{ind_Tc};
                m_prior = m_prior1_struc{ind_Tc};
    %             data_prior = d_prior1_struc{ind_Tc};
                data_prior = G_mat * m_prior;
        end
    else
        data_obs = d;
        data_est = d_post3;
        m_est = m_est3;
        m_prior = m_prior1;
        data_prior = G_mat * m_prior;
    end
    restit = sqrt(mean(((data_obs-data_est)./data_obs).^2))*100; % in percent    
    
    %v_moy=1/mean(m_prior(:));
    
    var_homo = var(data_obs-data_prior); % homogeneous vs observed data variance of travel-times residuals
    var_post = var(data_obs-data_est); % variance of travel-time residuals after inversion
    var_red = 1 - var_post/var_homo; % variation reduction
    
    S_map = reshape(m_est,[length(x_grid), length(y_grid)]);
    V_map = 1./S_map * 1000;
        
    if isnan(sum(V_map(:)))
        continue
    end
    %% Plot
    % Remove outliers
    V_map(V_map > 4) = nan;
    V_map(V_map < 0.5) = nan;
    
    % plot
    figure(1); clf; 
    set(gcf, 'Position', [2479 58 1118 809])
    set(gcf, 'color','w')    
    hold on
    im = pcolor(x_grid_eff,y_grid_eff,V_map'); % model
    set(im,'facealpha','flat','alphadata',mask')
    shading('interp');
    imagesc(x_map,y_map,map,'alphadata',1.0); % background
    xlim([min(x_map) max(x_map)])
    ylim([min(y_map) max(y_map)])
    chi=get(gca,'Children');set(gca,'Children',flipud(chi))

    % colorbar
    colormap(flipud(jet));
    hb=colorbar;
    ylabel(hb,'Group velocity (km/s)','fontsize',12)
    
    % Plot stations
    plot(x_stat,y_stat,'vk','linewidth',1.5,'markersize',4,'markerfacecolor','k')

    % Plot wells 
    
%     idx = find(deepwells.xgrid > 0 & deepwells.xgrid < max(x_map) & deepwells.ygrid > 0 & deepwells.ygrid < max(y_map) ); % index in map view
    idx = find(deepwells.xgrid > 0 & deepwells.xgrid < max(x_map) & deepwells.ygrid > 0 & deepwells.ygrid < max(y_map) & deepwells.depth > 1500 ); % index in map view
    hwells = scatter(deepwells.xgrid(idx), deepwells.ygrid(idx), 80, "red", 'filled', 'hexagram', 'MarkerEdgeColor', 'k', 'LineWidth',2);


   % Plot faults
    colthrust = "k";
    flist = dir(fullfile(swisstopodir_faults,'*'));
    count_tf = 0; count_uf = 0;
    for k=1:length(flist)
        file = flist(k).name;
        if file == '.'; continue;end
        load(fullfile(swisstopodir_faults,file))
        if contains(file, 'thrust')
            if count_tf == 0
                tf = plot(x,y,"Color",colthrust,"LineStyle","-","LineWidth",1, 'DisplayName','Thrust fault');                
            else
                plot(x,y,"Color",colthrust,"LineStyle","-","LineWidth",1)
            end
            count_tf = count_tf + 1;
        elseif contains(file, 'fault')
            if count_uf == 0
                uf = plot(x,y,"Color",[0.4 0.4 0.4],"LineStyle","-","LineWidth",1, 'DisplayName','Fault');
            else
                plot(x,y,"Color",[0.4 0.4 0.4],"LineStyle","-","LineWidth",1)
            end
            count_uf = count_uf + 1;
        end
    end 
    
        %% Plot heat flow    
    colheat = "r";
    flist = dir(fullfile(swisstopodir_heat,'*'));
    count = 0; 
    for k=1:length(flist)
        file = flist(k).name;
        if file == '.'; continue;end
        load(fullfile(swisstopodir_heat,file))

        if count == 0
            tf = plot(x,y,"Color",colheat,"LineStyle","--","LineWidth",1, 'DisplayName',['heat flux']);                
        else
            plot(x,y,"Color",colheat,"LineStyle","--","LineWidth",1)
        end
        count = count + 1;
        hold on
    end  

    %% Plot hot springs
    % Thermalbad Schinzach
    schinzach = [47.458850, 8.165973];%[8.165973‎, 47.458850];
    baden = [47.480781, 8.313892];
    zurzach = [47.589238, 8.290181]; %[8.290181‎, 47.589238];
    hotsprings = [schinzach;baden;zurzach];
    [hs_x,hs_y] = ll2xy(hotsprings(:,1),hotsprings(:,2), SW_corner);
    hhot = scatter(hs_x,hs_y, 80, 'yellow', 'filled', 'diamond', 'MarkerEdgeColor','k','LineWidth',1.5);


    % Setup
    set(gca,'linewidth',1.5,'fontsize',14,'layer','top')
    box on
    axis equal
    axis tight
    %title({['T=' num2str(Tc) ' s  (sigma=' num2str(sigma) ', Lc=' num2str(LC) ')'];['misfit data=' num2str(restit,'%.1f') ' %'];['VarRed=' num2str(var_red*100,'%.1f') ' %']});
    text(0.5, max(y_map)-1, ['T = ' num2str(Tc) ' s'],'fontsize',14,'BackgroundColor','w','EdgeColor','k','LineStyle','-');
    ylabel('Northing (km)');
    xlabel('Easting (km)'); 
    xlim([min(x_map) max(x_map)])
    ylim([min(y_map) max(y_map)])
    title(comp)
    pause(0.2)

    if strncmp(comp,'TT',2)
        fname = [figdir, '/vgmap_love' comp '_T' sprintf('%3.1f', Tc) 's.png']
    else
        fname = [figdir, '/vgmap_rayleigh' comp '_T' sprintf('%3.1f', Tc) 's.png']
    end
    export_fig(fname)
%     set(gca, 'CLim', [0.5, 4.5])
    
    
end


