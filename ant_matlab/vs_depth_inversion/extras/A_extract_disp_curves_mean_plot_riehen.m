clear all
%% Swisstopo
load("../../data-swisstopo/swisstopo-mat/deep_wells_RiehenGrid.mat")
swisstopodir_faults = '../../data-swisstopo/swisstopo-mat/Accident_tecto_riehen/';
swisstopodir_heat = '../../data-swisstopo/swisstopo-mat/geothermie_riehen/';

%% Load data
datadir = '../../data-riehen/run3_dcV2_mul2_g200m';
% datadir = '../../data-riehen/run4_dcV2_mul3_g200m';
input_file = [datadir, '/vs-model/data_picked_all_data_LC0.3_sigma8_RayleighMean.mat']
load(input_file)
load([datadir '/dist_stat.mat'], 'SW_corner')
load([datadir, '/grid/map_matrix_terrain.mat'], 'map', 'x_map', 'y_map')
load([datadir, '/grid/kernel.mat'], 'x_grid','y_grid','x_stat','y_stat','dx_grid','dy_grid')
% in kernel, grid nodes defined at bottom left of cell (should dblcheck); 
% in imagesc, node is at center of cell; these new effective axes compensate for this
x_grid_eff = x_grid + dx_grid/2; 
y_grid_eff = y_grid + dy_grid/2; 
minvg = 0.100;

%% Output path
fig_dir = [datadir '/vg-maps/mean-model-figs-abs']
if ~exist(fig_dir,'dir')
    mkdir(fig_dir)
end
%% Loop over periods
for k=1:length(T_pick)

    T = T_pick(k);
    V_map = reshape(vg_pick_mat(k,:),[length(x_grid) length(y_grid)]);
    std_map = reshape(vg_pick_mat_wstd(k,:),[length(x_grid) length(y_grid)]);
    V_map = V_map./1000;
    std_map = std_map/1000;
    % Mask no-data regions
    mask = nan(size(V_map));
    mask(V_map > minvg ) = 1;
    V = V_map(V_map>minvg ); vmoy = mean(V(:));
    V_map_anom = V_map - vmoy;
    
    %% Plot
    
    figure(1); clf; 
    % set(gcf, 'Position', [2479 58 1118 809])
    set(gcf, 'color','w')    
    
    %% Mean model
    subplot(1,2,1);cla
    hold on
    im = pcolor(x_grid_eff,y_grid_eff,V_map'); % model
%     im = pcolor(x_grid_eff,y_grid_eff,V_map_anom'); % model
    set(im,'facealpha','flat','alphadata',mask')
    shading('interp');
    imagesc(x_map,y_map,map,'alphadata',1.0); % background
    chi=get(gca,'Children');set(gca,'Children',flipud(chi))
    % colorbar
    colormap(flipud(jet));
    hb=colorbar;
    ylabel(hb,'Group velocity (km/s)','fontsize',12)
    set(gca,'CLim',[min(V) max(V)])
%     ylabel(hb,'Group velocity anomaly (m/s)','fontsize',12)
%     set(gca,'CLim',[min(V-vmoy) max(V-vmoy)])
    % Plot stations
    plot(x_stat,y_stat,'vk','linewidth',1.5,'markersize',2,'markerfacecolor','k')
    % Plot wells 
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
    % Plot hot springs
    % Thermalbad Schinzach
    schinzach = [47.458850, 8.165973];%[8.165973‎, 47.458850];
    baden = [47.480781, 8.313892];
    zurzach = [47.589238, 8.290181]; %[8.290181‎, 47.589238];
    hotsprings = [schinzach;baden;zurzach];
    [hs_x,hs_y] = ll2xy(hotsprings(:,1),hotsprings(:,2), SW_corner);
    hhot = scatter(hs_x,hs_y, 80, 'yellow', 'filled', 'diamond', 'MarkerEdgeColor','k','LineWidth',1.5);
    % Limits
    set(gca,'linewidth',1.5,'fontsize',14,'layer','top')
    box on
    axis equal
    axis tight
    title(['Mean model at T = ' sprintf('%3.1f', T) ' s'])
    xlim([min(x_map) max(x_map)])
    ylim([min(y_map) max(y_map)])
    
    %% Plot weighted standard deviation
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
    ylabel(hb,'Group velocity deviation (km/s)','fontsize',12)
    set(gca,'CLim',[min(std_map(V_map>minvg )) max(std_map(V_map>minvg ))])
    % Plot stations
    plot(x_stat,y_stat,'vk','linewidth',1.5,'markersize',2,'markerfacecolor','k')
    % Plot wells 
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
    
    % Limits
    set(gca,'linewidth',1.5,'fontsize',14,'layer','top')
    box on
    axis equal
    axis tight
    title('Weighted standard deviation')
    xlim([min(x_map) max(x_map)])
    ylim([min(y_map) max(y_map)])

    pause(0.1)
    fname = [fig_dir '/vgmap_wmean_T' sprintf('%3.1f',T) 's.png']
    export_fig(fname)
end