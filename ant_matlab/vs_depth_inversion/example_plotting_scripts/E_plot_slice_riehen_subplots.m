function [] = E_plot_slice_riehen_subplots()

datadir = '../../data-riehen/run3_dcV2_mul2_g200m'
dirinv = [datadir '/vs-model/run0_dv60_dz50m_N100_10L']
fname = [dirinv '/combined_result.mat']
load("../../data-swisstopo/swisstopo-mat/deep_wells_RiehenGrid.mat", 'deepwells')
swisstopodir = '../../data-swisstopo/swisstopo-mat/Accident_tecto_riehen/';
load(fname, 'Vs_min_mat', 'Vs_sol_mat', 'x_profile', 'y_profile', 'z_grid_eff')

output_fold = [dirinv '/plots']
if ~isfolder([output_fold])
    mkdir([output_fold])
end

load([datadir, '/grid/map_matrix_terrain.mat'], 'map', 'x_map', 'y_map')
load([datadir, '/grid/kernel.mat'], 'x_stat', 'y_stat')

%% Define slices
yslices = [7.70 10.7, 12.5, 14.9];
xslices = [5 9 12 15];
allslices = [yslices, xslices];
slicetypes = [repmat('y',[1 length(yslices)]),repmat('x',[1 length(xslices)])];

%% Plot params

dist_tol = 0.25 %5 %0.4; % tolerance for nearby stations/faults for depth sections
dist_tol_f = 0.25 %0.5 %0.1;
zmax = 5
colorlims = nan %[1500 3500]
amplitude_type = 'subtract_constant' % absolute, subtract_1D, subtract_constant

%% Colormaps
cmap_pm = cbrewer2('div','RdYlBu',50,'linear');
cmap_abs = flipud(jet(50));

%% Index without data
ind_nodata = find(sum(Vs_sol_mat,3)==0);
Vs_sol_mat(ind_nodata) = nan;

% From G_mat
load([datadir, '/grid/kernel.mat'], 'G_mat','x_grid','y_grid')
thres_dist = 0.01; % km
min_density = 1;
G3D = reshape(G_mat',[length(x_grid) length(y_grid) size(G_mat',2)]);
G_count = zeros(size(G3D));
ind_G_ray = G3D(:) > thres_dist; % count ray if >10m in cell
G_count(ind_G_ray) = 1;
G_sum = sum(G_count,3);
mask2D = nan(size(G_sum));
mask2D(G_sum >= min_density) = 1;
ind_no_data = find(G_sum < min_density);
mask3D = repmat(mask2D, [1 1 size(Vs_sol_mat,3)]);
Vs_sol_mat = Vs_sol_mat .* mask3D;

%% smooth Vs cube
nx_smooth = 3; 
ny_smooth = 3;
nz_smooth = 3;
Vs_smooth = smooth3(Vs_sol_mat,'box',[nx_smooth ny_smooth nz_smooth]);
% Vs_smooth=smooth3(Vs_min_mat,'box',[nx_smooth ny_smooth nz_smooth]);
% Vs_smooth = Vs_sol_mat;
Vs_smooth(ind_nodata) = nan;

%% Mean Vs 1D profile
mean_profile = zeros([length(z_grid_eff), 1]);
count = 0;
for ix=1:length(x_grid)
    for iy=1:length(y_grid)
        if G_sum(ix,iy) > 3 %~isnan(mask2D(ix,iy))
            mean_profile = mean_profile + squeeze(Vs_sol_mat(ix,iy,:));
            count = count + 1;
        end
    end
end
mean_profile = mean_profile./count;

%% Plot map figure showing cross-sections and XY slice
zdepth = 2000;
[~, zz] = min(abs(z_grid_eff - zdepth));

figure(1); clf; set(gcf,'position',[1945         135         949         710],'color','w')    
hold on
box on  
set(gca,'linewidth',1.5,'fontsize',12,'layer','top')

% Plot velocity model
Vs_slice = squeeze(Vs_smooth(:,:,zz));
Vs_slice_ref = repmat(mean_profile(zz), [size(Vs_slice,1) size(Vs_slice,2)]);
Vs_slice_anom = (Vs_slice - Vs_slice_ref) ./ Vs_slice_ref * 100;        
mask = nan(size(Vs_slice));
mask(Vs_slice > 0) = 1;
        
switch amplitude_type
    case {'subtract_1D', 'subtract_constant'}
        im = pcolor(x_profile,y_profile,Vs_slice_anom');
        set(im,'facealpha','flat','alphadata',mask')
        shading('interp');
        imagesc(x_map,y_map,map,'alphadata',1);              
        chi=get(gca,'Children');set(gca,'Children',flipud(chi))
        colormap(cmap_pm);
        hb=colorbar;
        ylabel(hb,'Vs anomaly (%)','fontsize',14)
        hold on; contour(x_profile,y_profile,Vs_slice_anom','color',[1 1 1].*0.5,'LineWidth',0.5,'LineStyle',':');
        set(gca,'Clim',[-17,10])
    case 'absolute'
        im = pcolor(x_profile,y_profile,Vs_slice');
        set(im,'facealpha','flat','alphadata',mask')
        shading('interp');
        imagesc(x_map,y_map,map,'alphadata',1.0); % background    
        chi=get(gca,'Children');set(gca,'Children',flipud(chi))
        colormap(cmap_abs);
        hb=colorbar;
        ylabel(hb,'Vs (m/s)','fontsize',12)
end

% Add cross-sections
for k=1:length(yslices)
    yslice = yslices(k);
    h = hline(yslice, "r"); set(h, 'Linewidth',2);
    text(0.5,yslice, ['H' num2str(k)], 'fontsize',14,'color','r','BackgroundColor','w','LineStyle','-')
    text(20,yslice, ['H' num2str(k) ''''], 'fontsize',14,'color','r','BackgroundColor','w','LineStyle','-')
end
for k=1:length(xslices)
    xslice = xslices(k);
    h = vline(xslice, "r"); set(h, 'Linewidth',2);
    text(xslice, 0.5, ['V' num2str(k)], 'fontsize',14,'color','r','BackgroundColor','w','LineStyle','-')
    text(xslice, 19.5, ['V' num2str(k) ''''], 'fontsize',14,'color','r','BackgroundColor','w','LineStyle','-')
end

% Add faults
flist = dir(fullfile(swisstopodir,'*'));
count_tf = 0; count_uf = 0;
for k=1:length(flist)
    file = flist(k).name;
    if file == '.'; continue;end
    load(fullfile(swisstopodir,file))
    if contains(file, 'thrust')
        if count_tf == 0
            tf = plot(x,y,"Color","k","LineStyle","-","LineWidth",2, 'DisplayName','Thrust fault');
        else
            plot(x,y,"Color","k","LineStyle","-","LineWidth",2)
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
% Plot wells 
idx = find(deepwells.xgrid > 0 & deepwells.xgrid < max(x_map) & deepwells.ygrid > 0 & deepwells.ygrid < max(y_map) & deepwells.depth > 1200 ); % index in map view
hwells = scatter(deepwells.xgrid(idx), deepwells.ygrid(idx), 80, "red", 'filled', 'hexagram', 'MarkerEdgeColor', 'k', 'LineWidth',2);        
% Add stations
hsta = plot(x_stat,y_stat,'vk','linewidth',1.5,'markersize',2,'markerfacecolor','k', 'DisplayName', 'Stations');    
% Setup
xlabel('distance along Easting (km)'); 
ylabel('distance along Northing (km)'); 
axis equal
axis tight
set(gca,'xlim',[x_map(1) x_map(end)],'ylim',[y_map(end) y_map(1)]);

% Add custom slices Guerler 1987
load([datadir '/dist_stat.mat'], 'SW_corner')
% P5 
pointll0 = [47.599152, 7.555517]; % West as in true P5
[x0,y0] = ll2xy(pointll0(1),pointll0(2), SW_corner);
pointll1 = [47.584044, 7.548651]; % West as in true P5
[x1,y1] = ll2xy(pointll1(1),pointll1(2), SW_corner);
pointll2 = [47.581265, 7.617573]; % break point P5
[x2,y2] = ll2xy(pointll2(1),pointll2(2), SW_corner);
pointll3 = [47.556074, 7.701086]; % East
[x3,y3] = ll2xy(pointll3(1),pointll3(2), SW_corner);
plot([x1,x2],[y1,y2],'--',color="g",LineWidth=2)
plot([x0,x3],[y0,y3],'-',color="g",LineWidth=2)
text(x0,y0,'P5','fontsize',14,'color','g','BackgroundColor','w','LineStyle','-')
text(x3,y3,'P5''','fontsize',14,'color','g','BackgroundColor','w','LineStyle','-')

% P3
pointll1 = [47.610536, 7.577393]; % S
[x1,y1] = ll2xy(pointll1(1),pointll1(2), SW_corner);
pointll2 = [47.616351, 7.706536]; % N
[x2,y2] = ll2xy(pointll2(1),pointll2(2), SW_corner);
plot([x1,x2],[y1,y2],'-',color="g",LineWidth=2)
text(x1,y1,'P3','fontsize',14,'color','g','BackgroundColor','w','LineStyle','-')
text(x2,y2,'P3''','fontsize',14,'color','g','BackgroundColor','w','LineStyle','-')


figname = [output_fold '/report_map_xsections.png']
export_fig(figname)

%% Cross-sections

%% Subplot y slices
slicetype = 'y'
% Initialize figure
figure(2); clf;
set(gcf,'position',[769 14 1042 974],'color','w')
hold on
box on  
set(gca,'linewidth',1.5,'fontsize',12,'layer','top')
haxes = [];
for k=1:length(yslices)
    k
    ax = subplot(length(yslices),1,k);    
    haxes = [haxes, ax];
    hold on
    box on
    levels = nan;
    distance = add_slice(slicetype, yslices(k), Vs_smooth, amplitude_type, x_profile, y_profile, z_grid_eff, mean_profile, cmap_pm, cmap_abs, levels);
    if ~isnan(colorlims)
        set(gca, 'CLim', colorlims)
    end        
    axis equal
    axis tight
    set(gca,'ydir','reverse','fontsize',12,'TickDir','out')
    add_stations(slicetype, yslices(k), dist_tol, x_stat, y_stat)
    add_faults(slicetype, yslices(k), swisstopodir, dist_tol_f)
    add_wells(slicetype, yslices(k), deepwells, dist_tol)   
    if k == length(yslices)
        xlabel('distance along Easting (km)','fontsize',12); 
    end
    ylabel('Depth (km)','fontsize',12);    
%     text(min(distance), -.5, ['H' num2str(k)], 'fontsize',14,'BackgroundColor','w','linestyle','-')
%     text(max(distance)-1, -.5, ['H' num2str(k) ''''], 'fontsize',14,'BackgroundColor','w','linestyle','-')
    text(1.5, -.5, ['H' num2str(k)], 'fontsize',14,'BackgroundColor','w','linestyle','-')
    text(20.1-1, -.5, ['H' num2str(k) ''''], 'fontsize',14,'BackgroundColor','w','linestyle','-')

    xlim([min(distance), max(distance)])
    ylim([0, zmax])
end
linkaxes(haxes,'xy')
namestr = '';
for k=1:length(yslices)
    namestr = [namestr strtrim(sprintf('%4.1f',yslices(k))) '_'];
end

handles = get(gcf,'Children');
for h=1:length(handles)    
    if ~isempty(handles(h).Tag) % == 'ColorBar'
        continue
    end
    handles(h).Tag
    set(handles(h),'Xlim',[1.5 20.1])
end

figname = [output_fold  '/report_yslices_' namestr(1:end-1) '.png']
export_fig(figname)

%% Subplot x slices
slicetype = 'x'
% Initialize figure
figure(3); clf;
set(gcf,'position',[769 14 1042 974],'color','w')
hold on
box on  
set(gca,'linewidth',1.5,'fontsize',12,'layer','top')
haxes = [];
for k=1:length(xslices)
    k;
    ax = subplot(length(xslices),1,k);    
    haxes = [haxes, ax];
    hold on
    box on
    levels = nan;
    distance = add_slice(slicetype, xslices(k), Vs_smooth, amplitude_type, x_profile, y_profile, z_grid_eff, mean_profile, cmap_pm, cmap_abs, levels);
    if ~isnan(colorlims)
        set(gca, 'CLim', colorlims)
    end        
    axis equal
    axis tight
    set(gca,'ydir','reverse','fontsize',12,'TickDir','out')
    add_stations(slicetype, xslices(k), dist_tol, x_stat, y_stat)
    add_faults(slicetype, xslices(k), swisstopodir, dist_tol_f)
    add_wells(slicetype, xslices(k), deepwells, dist_tol)   
    if k == length(xslices)
        xlabel('distance along Northing (km)','fontsize',12); 
    end
    ylabel('Depth (km)','fontsize',12);    
%     text(min(distance), -.5, ['V' num2str(k)], 'fontsize',14,'BackgroundColor','w','linestyle','-')
%     text(max(distance)-1, -.5, ['V' num2str(k) ''''], 'fontsize',14,'BackgroundColor','w','linestyle','-')
    text(1, -.5, ['V' num2str(k)], 'fontsize',14,'BackgroundColor','w','linestyle','-')
    text(20-1, -.5, ['V' num2str(k) ''''], 'fontsize',14,'BackgroundColor','w','linestyle','-')
    xlim([min(distance), max(distance)])
    ylim([0, zmax])
end
linkaxes(haxes,'xy')
namestr = '';
for k=1:length(xslices)
    namestr = [namestr strtrim(sprintf('%4.1f',xslices(k))) '_'];
end

handles = get(gcf,'Children');
for h=1:length(handles)    
    if  ~isempty(handles(h).Tag) % == 'ColorBar'
        continue
    end
    handles(h).Tag
    set(handles(h),'Xlim',[1 20])
end

figname = [output_fold  '/report_xslices_' namestr(1:end-1) '.png']
export_fig(figname)

end

function [distance]=add_slice(slicetype, profile_value, Vs_smooth, amplitude_type, x_profile, y_profile, z_grid_eff, mean_profile, cmap_pm, cmap_abs, levels)
% Get slice and plot it
    depth = z_grid_eff/1000;
    if slicetype == 'y'
        yy = find(abs(y_profile - profile_value) < 0.1,1,"first");
        Vs_slice = squeeze(Vs_smooth(:,yy,:));
        distance = x_profile;
    elseif slicetype == 'x'
        xx = find(abs(x_profile - profile_value) < 0.1,1,"first");
        Vs_slice=squeeze(Vs_smooth(xx,:,:));
        distance = y_profile;
    end
    mask = zeros(size(Vs_slice));
    mask(Vs_slice > 0) = 1;
    
    contourcolor = [1 1 1].*0.5;
    contourwidth = 0.5;
    contourstyle = ':';
    labelfontsize = 12;
    
    switch amplitude_type
        case 'subtract_1D'                
            % Get slice
            Vs_slice_ref = repmat(mean_profile',[size(Vs_slice,1), 1]);
            Vs_slice_anom = (Vs_slice - Vs_slice_ref) ./ Vs_slice_ref * 100;
            im = pcolor(distance,depth,Vs_slice_anom');
            set(im, 'facealpha','flat','alphadata', mask')
            shading('interp');
            colormap(cmap_pm);
            hb=colorbar;
            ylabel(hb,'Vs anomaly (%)','fontsize',labelfontsize)
            max_anom = nanmax(abs(Vs_slice_anom(:)));
            if isfinite(max_anom) && max_anom > 0
                set(gca, 'CLim', [-max_anom, max_anom]);
            end
            hold on;
            if ~isnan(levels)
                contour(distance,depth,Vs_slice_anom',levels,'color',contourcolor,'LineWidth',contourwidth,'LineStyle',contourstyle);
            else
                contour(distance,depth,Vs_slice_anom','color',contourcolor,'LineWidth',contourwidth,'LineStyle',contourstyle);
            end
        case 'subtract_constant' 
            mean_Vs = nanmean(Vs_smooth(:));
            Vs_slice_ref = ones(size(Vs_slice)) .* mean_Vs;
            Vs_slice_anom = (Vs_slice - Vs_slice_ref) ./ Vs_slice_ref * 100;
            im = pcolor(distance,depth,Vs_slice_anom');
            set(im, 'facealpha','flat','alphadata', mask')
            shading('interp');
            colormap(cmap_pm);
            hb=colorbar;
            ylabel(hb,'Vs anomaly (%)','fontsize',labelfontsize)
            max_anom = nanmax(abs(Vs_slice_anom(:)));
            if isfinite(max_anom) && max_anom > 0
                set(gca, 'CLim', [-max_anom, max_anom]);
            end
            hold on;
            if ~isnan(levels)
                contour(distance,depth,Vs_slice_anom',levels,'color',contourcolor,'LineWidth',contourwidth,'LineStyle',contourstyle);
            else
                contour(distance,depth,Vs_slice_anom','color',contourcolor,'LineWidth',contourwidth,'LineStyle',contourstyle);
            end
        case 'absolute'
            im = pcolor(distance,depth,Vs_slice');
            set(im, 'facealpha','flat','alphadata', mask')
            shading('interp');
            colormap(cmap_abs);            
            hb=colorbar;
            ylabel(hb,'Vs (m/s)','fontsize',labelfontsize)
    end     
end

function [] = add_stations(slicetype, profile_value, dist_tol, x_stat, y_stat)
% Add stations nearby
    markerfacecolor = [1 1 1]*0.5;
    markersize = 8;
    markeredgecolor = 'k';
    linewidth = 1.5;    
    if slicetype == 'y'
        ind = find(abs(y_stat - profile_value) < dist_tol);
        if ~isempty(ind)
            plot(x_stat(ind), zeros(size(ind)), 'vk','linewidth',linewidth,'markersize',markersize,'markerfacecolor',markerfacecolor,'MarkerEdgeColor',markeredgecolor, 'DisplayName', 'Stations');
        end
    elseif slicetype == 'x'
        ind = find(abs(x_stat - profile_value) < dist_tol);
        if ~isempty(ind)
            plot(y_stat(ind), zeros(size(ind)), 'vk','linewidth',linewidth,'markersize',markersize,'markerfacecolor',markerfacecolor,'MarkerEdgeColor',markeredgecolor, 'DisplayName', 'Stations');
        end

    end

end

function []=add_faults(slicetype, profile_value, swisstopodir, dist_tol_f)
% Add intersecting fault lines        

    flist = dir(fullfile(swisstopodir,'*'));
    for k=1:length(flist)
        file = flist(k).name;
        if file == '.'; continue;end
        load(fullfile(swisstopodir,file))

        if contains(file, 'thrust')
            symbol = '|k';
            linewidth = 3;
            markersize = 10;
            markerfacecolor = 'k';
        else
            symbol = '|k';
            linewidth = 3;
            markersize = 10;
            markerfacecolor = 'k';
        end
          
%         if slicetype == 'y'
%             ind = find(abs(y - profile_value) < dist_tol_f);
%             if ~isempty(ind)
%                 plot(x(ind), zeros(size(ind)), symbol,'linewidth',linewidth,'markersize',markersize,'markerfacecolor',markerfacecolor);
%             end
%         elseif slicetype == 'x'
%             ind = find(abs(x - profile_value) < dist_tol_f);
%             if ~isempty(ind)
%                 plot(y(ind), zeros(size(ind)), symbol,'linewidth',linewidth,'markersize',markersize,'markerfacecolor',markerfacecolor);
%             end 
%         end
        Lfault = [x;y];
        
        if slicetype == 'y'
            Lslice = [0,25;profile_value,profile_value];  
            P = InterX(Lfault,Lslice);
            if ~isempty(P)
                distcross = P(1);
                plot(distcross,0,symbol, 'MarkerSize',markersize, 'MarkerFaceColor',markerfacecolor,'LineWidth',linewidth)
            end
        elseif slicetype == 'x'
            Lslice = [profile_value,profile_value;0,25];            
            P = InterX(Lfault,Lslice);
            if ~isempty(P)
                distcross = P(2);
                plot(distcross,0,symbol, 'MarkerSize',markersize, 'MarkerFaceColor',markerfacecolor,'LineWidth',linewidth)
            end
        end

    end
end

function []=add_wells(slicetype, profile_value, deepwells, dist_tol)
% Add wells if close to slice
    markerfacecolor = 'r';
    marker = 'hexagram';
    markeredgecolor = 'k';
    linewidth = 2;  
    min_depth = 1000;  
    markersize = 80;
    if slicetype == 'y'       
        ind = find(abs(deepwells.ygrid - profile_value) < dist_tol & deepwells.depth >= min_depth);
        if ~isempty(ind)
            scatter(deepwells.xgrid(ind), zeros(size(ind)), markersize, markerfacecolor, 'filled', marker, 'MarkerEdgeColor', markeredgecolor, 'LineWidth',linewidth);
            plot([deepwells.xgrid(ind),deepwells.xgrid(ind)], [0, deepwells.depth(ind)*1e-3],LineStyle="--",LineWidth=1,color='k')
        end
    elseif slicetype == 'x'
        ind = find(abs(deepwells.xgrid - profile_value) < dist_tol & deepwells.depth >= min_depth);
        if ~isempty(ind)
            scatter(deepwells.ygrid(ind), zeros(size(ind)), markersize, markerfacecolor, 'filled', marker, 'MarkerEdgeColor', markeredgecolor, 'LineWidth',linewidth);
            plot([deepwells.ygrid(ind),deepwells.ygrid(ind)], [0, deepwells.depth(ind)*1e-3],LineStyle="--",LineWidth=1,color='k')
        end

    end
end