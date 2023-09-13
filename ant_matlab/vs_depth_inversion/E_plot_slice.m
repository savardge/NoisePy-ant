% This script plot slices in the xy, xz, and yz directions, at specified
% intervals. Must modify this code to your needs. See also example_plotting_scripts for
% more advance plotting examples (e.g. arbitrary slices, plotting
% additional data like faults, etc.)
%

close all
clear all

%% User define paths

% project directory
datadir = '../../data-aargau/run4_dcV2_mul3_g500m'

% directory with inversion results
dirinv = [datadir '/vs-model/run2_dv30_dz50m_N100_14L_wLVZ_ZZ'];
% Name of file with all 1D profiles merged
fname = [dirinv '/combined_result.mat'];
disp(['Reading Vs model from: ' fname])
load(fname, 'Vs_min_mat', 'Vs_sol_mat', 'x_profile', 'y_profile', 'z_grid_eff')

output_fold = [dirinv '/plots']
if ~isfolder([output_fold])
    mkdir([output_fold])
end

load([datadir, '/grid/map_matrix_terrain.mat'], 'map', 'x_map', 'y_map') % map background
kernel_file = [datadir, '/grid/kernel.mat'];
load(kernel_file, 'x_stat', 'y_stat')

%% Plot params

dist_tol = 0.25 %On depth slices, plot stations within dist_tol km from the slice
zmax = 5 % Max depth on depth sections

% Set up colorbar limits, nan is none
colorlims = nan %[1500 3500]

% whether to plot xy, yz, xz slices
plotxz = false
plotyz = true%true
plotxy = false %true

% Type of amplitude shown: 
% absolute Vs, subtract mean Vs, subtract mean 1D Vs
amplitude_type = 'subtract_constant' % absolute, subtract_constant, subtract_1D

%% Colormaps (download cbrewer from Matlab FileExchange)
cmap_pm = cbrewer2('div','RdYlBu',50,'linear');
cmap_abs = flipud(jet(50));

%% Index without data
ind_nodata = find(sum(Vs_sol_mat,3)==0);
Vs_sol_mat(ind_nodata) = nan;

% From G_mat
load(kernel_file, 'G_mat','x_grid','y_grid')
thres_dist = 0.01; % km
min_density = 1;
G3D = reshape(G_mat',[length(x_grid) length(y_grid) size(G_mat',2)]);
% count ray if dist travelled in cell above threshold of ~100m?
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
% figure;plot(mean_profile, -z_grid_eff, 'o-')
mean_Vs = nanmean(Vs_sol_mat(:));

%% Interpolate 3D Vs for visualization (might not be needed)
% Vs_smooth, x_profile, y_profile, z_grid_eff
% dx_new = 0.25; % km
% dy_new = 0.25; % km
% x_profile_new = x_profile(1):dx_new:x_profile(end);
% y_profile_new = y_profile(1):dy_new:y_profile(end);
% [Xq,Yq,Zq] = ndgrid(x_profile_new, y_profile_new, z_grid_eff);
% Vs_smooth_new = interpn(x_profile, y_profile, z_grid_eff, Vs_smooth, Xq,Yq,Zq);

%% plot one xz slices
if plotxz
    
    for yy=1:1:length(y_profile)
        
        yslice = y_profile(yy)
        
        Vs_slice = squeeze(Vs_smooth(:,yy,:));
        if isempty(find(~isnan(Vs_slice(:))))
            continue
        end
        mask = zeros(size(Vs_slice));
        mask(Vs_slice > 0) = 1;

        figure(1); clf; set(gcf,'position',[1945         423        1893         422],'color','w')
        set(gca,'linewidth',1.5,'fontsize',16,'layer','top')

        % Map view
        subplot(1,4,1); cla; hold on
        imagesc(x_map,y_map,map,'alphadata',1.0);               
        h = hline(yslice, "r"); set(h, 'Linewidth',2)
        xlabel('distance along Easting (km)'); 
        ylabel('distance along Northing (km)'); 
        flist = dir(fullfile(swisstopodir_faults,'*'));
        count_tf = 0; count_uf = 0;
        for k=1:length(flist)
            file = flist(k).name;
            if file == '.'; continue;end
            load(fullfile(swisstopodir_faults,file))
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
          
        % Stations
        hsta = plot(x_stat,y_stat,'vk','linewidth',1.5,'markersize',2,'markerfacecolor','k', 'DisplayName', 'Stations');    
        axis equal
        axis tight
        set(gca,'xlim',[x_map(1) x_map(end)],'ylim',[y_map(end) y_map(1)]);

        % Depth cross-section
        subplot(1,4,[2 3 4]); cla
        hold on
        box on
        switch amplitude_type
            case 'subtract_1D'       % subtract the average 1D profile, plot as high-low anomalies  
%                 mean_profile_slice = nanmean(Vs_slice,1);                
                Vs_slice_ref = repmat(mean_profile',[size(Vs_slice,1), 1]);
                Vs_slice_anom = (Vs_slice - Vs_slice_ref) ./ Vs_slice_ref * 100;
                im = pcolor(x_profile,z_grid_eff/1000,Vs_slice_anom');
                set(im, 'facealpha','flat','alphadata', mask')
                shading('interp');
                colormap(cmap_pm);
                hb=colorbar;
                ylabel(hb,'Vs anomaly (%)','fontsize',12)
                max_anom = (nanmax(abs(Vs_slice_anom(:))))
                if isfinite(max_anom) && max_anom > 0
                    set(gca, 'CLim', [-max_anom, max_anom]);
                end
                hold on;contour(x_profile,z_grid_eff/1000,Vs_slice_anom','color',[1 1 1].*0.5,'LineWidth',0.5,'LineStyle',':');
            case 'subtract_constant' % subtract the mean Vs, plot as high-low anomalies.               
                Vs_slice_ref = ones(size(Vs_slice)) .* mean_Vs;
                Vs_slice_anom = (Vs_slice - Vs_slice_ref) ./ Vs_slice_ref * 100;
                im = pcolor(x_profile,z_grid_eff/1000,Vs_slice_anom');
                set(im, 'facealpha','flat','alphadata', mask')
                shading('interp');
                colormap(cmap_pm);
                hb=colorbar;
                ylabel(hb,'Vs anomaly (%)','fontsize',12)
%                 max_anom = 0.7*(nanmax(abs(Vs_slice_anom(:))))
                max_anom = min(abs([nanmin(Vs_slice_anom(:)),nanmax(Vs_slice_anom(:))]))
                if isfinite(max_anom) && max_anom > 0
                    set(gca, 'CLim', [-max_anom, max_anom]);
                end
                hold on;contour(x_profile,z_grid_eff/1000,Vs_slice_anom','color',[1 1 1].*0.5,'LineWidth',0.5,'LineStyle',':');
            case 'absolute' % Plot absolute Vs
                im = pcolor(x_profile,z_grid_eff/1000,Vs_slice');
                set(im, 'facealpha','flat','alphadata', mask')
                shading('interp');
                colormap(cmap_abs);            
                hb=colorbar;
                ylabel(hb,'Vs (m/s)','fontsize',12)
        end     
        if ~isnan(colorlims)
            set(gca, 'CLim', colorlims)
        end        
        axis equal
        axis tight        

        %%% Add stations
        ind = find(abs(y_stat - yslice) < dist_tol);
        if ~isempty(ind)
            hsta = plot(x_stat(ind), zeros(size(ind)), 'vk','linewidth',1.5,'markersize',8,'markerfacecolor',[1 1 1]*0.5,'MarkerEdgeColor','k', 'DisplayName', 'Stations');
        end

        %%% Decoration    
        axis equal
        axis tight               
        set(gca,'ydir','reverse')
        xlabel('distance along Easting (km)'); ylabel('Depth (km)');
        title(['Depth slice at y = ', sprintf('%4.2f', y_profile(yy)), ' km'])
        xlim([min(x_profile), max(x_profile)])
        ylim([0, zmax])
        xlim([min(x_stat), max(x_stat)])        
         
        switch amplitude_type
            case {'subtract_1D', 'subtract_constant'}
%             print([output_fold, '/xzslice_y', sprintf('%4.2f', y_profile(yy)), '_anom.png'], '-dpng')
            export_fig([output_fold, '/xzslice_y', sprintf('%4.2f', y_profile(yy)), '_anom.png']);%,'-transparent')
            case 'absolute'
%             print([output_fold, '/xzslice_y', sprintf('%4.2f', y_profile(yy)), '_abs.png'], '-dpng')
            export_fig([output_fold, '/xzslice_y', sprintf('%4.2f', y_profile(yy)), '_abs.png']);
        end

%         pause(0.5)
%         close(gcf)
    end

end

%% plot one yz slices

if plotyz

    for xx=1:1:length(x_profile)

        xslice = x_profile(xx);
        Vs_slice=squeeze(Vs_smooth(xx,:,:));

        if isempty(find(~isnan(Vs_slice(:))))
            continue
        end

        mask = zeros(size(Vs_slice));
        mask(Vs_slice > 0) = 1;
        
        figure(1); clf; set(gcf,'position',[1945         423        1893         422],'color','w')
        set(gca,'linewidth',1.5,'fontsize',16,'layer','top')

        % Map view
        subplot(1,4,1); cla; hold on
        imagesc(x_map,y_map,map,'alphadata',0.6);          
        h = vline(xslice, "r"); set(h, 'Linewidth',2)
        xlabel('distance along Easting (km)'); 
        ylabel('distance along Northing (km)');

        % Faults
        flist = dir(fullfile(swisstopodir_faults,'*'));
        count_tf = 0; count_uf = 0;
        for k=1:length(flist)
            file = flist(k).name;
            if file == '.'; continue;end
            load(fullfile(swisstopodir_faults,file))
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

        % Stations
        hsta = plot(x_stat,y_stat,'vk','linewidth',1.5,'markersize',2,'markerfacecolor','k', 'DisplayName', 'Stations');    
        
        axis equal
        axis tight
        set(gca,'xlim',[x_map(1) x_map(end)],'ylim',[y_map(end) y_map(1)]);     
        
        % Depth cross-section
        subplot(1,4,[2 3 4]); cla
        hold on
        box on
        
        %%% Plot model
        switch amplitude_type
            case 'subtract_1D'        
%                 mean_profile_slice = nanmean(Vs_slice,1);                
                Vs_slice_ref = repmat(mean_profile',[size(Vs_slice,1), 1]);
            
                Vs_slice_anom = (Vs_slice - Vs_slice_ref) ./ Vs_slice_ref * 100;
                im = pcolor(y_profile,z_grid_eff/1000,Vs_slice_anom');
                set(im, 'facealpha','flat','alphadata', mask')
                colormap(cmap_pm);
                hb=colorbar;
                ylabel(hb,'Vs anomaly (%)','fontsize',12)
                max_anom = nanmax(abs(Vs_slice_anom(:)));
                if isfinite(max_anom)
                    set(gca, 'CLim', [-max_anom, max_anom]);
                end
                hold on;contour(y_profile,z_grid_eff/1000,Vs_slice_anom','color',[1 1 1].*0.5,'LineWidth',0.5,'LineStyle',':');
            case 'subtract_constant'                
                Vs_slice_ref = ones(size(Vs_slice)) .* mean_Vs;
                Vs_slice_anom = (Vs_slice - Vs_slice_ref) ./ Vs_slice_ref * 100;
                im = pcolor(y_profile,z_grid_eff/1000,Vs_slice_anom');
                set(im, 'facealpha','flat','alphadata', mask')
                shading('interp');
                colormap(cmap_pm);
                hb=colorbar;
                ylabel(hb,'Vs anomaly (%)','fontsize',12)
%                 max_anom = nanmax(abs(Vs_slice_anom(:)));
                max_anom = min(abs([nanmin(Vs_slice_anom(:)),nanmax(Vs_slice_anom(:))]))
                if isfinite(max_anom) && max_anom > 0
                    set(gca, 'CLim', [-max_anom, max_anom]);
                end
                hold on;contour(y_profile,z_grid_eff/1000,Vs_slice_anom','color',[1 1 1].*0.5,'LineWidth',0.5,'LineStyle',':');
            case 'absolute'
                im = pcolor(y_profile,z_grid_eff/1000,Vs_slice');
                set(im, 'facealpha','flat','alphadata', mask')
                colormap(cmap_abs);
                hb=colorbar;
                ylabel(hb,'Vs (m/s)','fontsize',12)
        end     
        if ~isnan(colorlims)
            set(gca, 'CLim', colorlims)
        end
        shading('interp');
        axis equal
        axis tight      
        
        %%% Add stations
        ind = find(abs(x_stat - xslice) < dist_tol);
        if ~isempty(ind)
            hsta = plot(y_stat(ind), zeros(size(ind)), 'vk','linewidth',1.5,'markersize',8,'markerfacecolor',[1 1 1]*0.5,'MarkerEdgeColor','k', 'DisplayName', 'Stations');
        end

        %%% Decoration    
        axis equal
        axis tight        
        set(gca,'ydir','reverse')
        xlabel('distance along Northing (km)'); ylabel('Depth (km)');
        title(['Depth slice at x = ', sprintf('%4.2f', x_profile(xx)), ' km'])
        xlim([min(y_profile), max(y_profile)])
        ylim([0, zmax])
        xlim([min(y_stat), max(y_stat)])
        
        pause(0.5)

        switch amplitude_type
            case {'subtract_1D', 'subtract_constant'}
    %             print([output_fold, '/yzslice_x', sprintf('%4.2f', x_profile(xx)), '_anom.png'], '-dpng')
                export_fig([output_fold, '/yzslice_x', sprintf('%4.2f', x_profile(xx)), '_anom.png'])
            case 'absolute'
%             print([output_fold, '/yzslice_x', sprintf('%4.2f', x_profile(xx)), '_abs.png'], '-dpng')
            export_fig([output_fold, '/yzslice_x', sprintf('%4.2f', x_profile(xx)), '_abs.png'])
        end
%         pause(0.5)
%         close(gcf)
        
    end

end
%% Plot xy slices

if plotxy
%     figure('position',get(0,'screensize'));
    figure(1);clf;
    set(gcf, 'Position', [2479 58 1190 903])
    set(gcf, 'color','w')
%     set(gca,'linewidth',1.5,'fontsize',16,'layer','top')
    
    for zdepth=0:100:max(z_grid_eff(:))
        
        zdepth
        [~, zz] = min(abs(z_grid_eff - zdepth));
        Vs_slice = squeeze(Vs_smooth(:,:,zz));
        Vs_slice_ref = repmat(mean_profile(zz), [size(Vs_slice,1) size(Vs_slice,2)]);
        Vs_slice_anom = (Vs_slice - Vs_slice_ref) ./ Vs_slice_ref * 100;        
        mask = nan(size(Vs_slice));
        mask(Vs_slice > 0) = 1;
%         dum = Vs_slice(mask); dum=dum(:); 
%         clims = [nanmean(dum)-1*nanstd(dum), nanmean(dum)+1*nanstd(dum)];
        
        clf    
        hold on
        box on       
        switch amplitude_type
            case {'subtract_1D', 'subtract_constant'}
                im = pcolor(x_profile,y_profile,Vs_slice_anom');
                set(im,'facealpha','flat','alphadata',mask')
                shading('interp');
                imagesc(x_map,y_map,map,'alphadata',1);              
                chi=get(gca,'Children');set(gca,'Children',flipud(chi))
                colormap(cmap_pm);
                hb=colorbar;
                ylabel(hb,'Vs anomaly (%)','fontsize',12)
                hold on; contour(x_profile,y_profile,Vs_slice_anom','color',[1 1 1].*0.5,'LineWidth',0.5,'LineStyle',':');
                max_anom = min(abs([nanmin(Vs_slice_anom(:)),nanmax(Vs_slice_anom(:))]));
                if isfinite(max_anom) && max_anom > 0
                    set(gca, 'CLim', [-max_anom, max_anom]);
                end
            case 'absolute'
                im = pcolor(x_profile,y_profile,Vs_slice');
                set(im,'facealpha','flat','alphadata',mask')
                imagesc(x_map,y_map,map,'alphadata',1.0); % background    
                chi=get(gca,'Children');set(gca,'Children',flipud(chi))
                colormap(cmap_abs);
                hb=colorbar;
                ylabel(hb,'Vs (m/s)','fontsize',12)
        end
        
        shading('interp');
        axis equal
        axis tight
        
        xlabel('distance along Easting (km)'); 
        ylabel('distance along Northing (km)');
        title(['Map slice at z = ', sprintf('%4.2f', zdepth/1000), ' km'])
    
        % Add stations
        hsta = plot(x_stat,y_stat,'vk','linewidth',1.5,'markersize',6,'markerfacecolor','k', 'DisplayName', 'Stations');    

        set(gca,'xlim',[x_map(1) x_map(end)],'ylim',[y_map(end) y_map(1)]); 
        % Save
        pause(0.5)
        switch amplitude_type
            case {'subtract_1D', 'subtract_constant'}
%             print([output_fold, '/xyslice_z', sprintf('%4.2f', zdepth/1000), '_anom.png'], '-dpng')
            export_fig([output_fold, '/xyslice_z', sprintf('%4.2f', zdepth/1000), '_anom.png'])
            case 'absolute'
%             print([output_fold, '/xyslice_z', sprintf('%4.2f', zdepth/1000), '_abs.png'], '-dpng')
            export_fig([output_fold, '/xyslice_z', sprintf('%4.2f', zdepth/1000), '_abs.png'])
        end

    end
end
