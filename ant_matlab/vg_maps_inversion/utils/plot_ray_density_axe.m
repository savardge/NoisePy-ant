function [] = plot_ray_density_axe(G_mat, x_grid, y_grid, dx_grid, dy_grid, x_stat, y_stat, x_map, y_map, map)
%PLOT_RAY_DENSITY Summary of this function goes here
%   Detailed explanation goes here

fontsize = 12;
contour_lev = 3;
thres_dist = 0.01; % km

% in kernel, grid nodes defined at bottom left of cell (should dblcheck); 
% in imagesc, node is at center of cell; these new effective axes compensate for this
x_grid_eff = x_grid + dx_grid/2; 
y_grid_eff = y_grid + dy_grid/2; 

G3D = reshape(G_mat',[length(x_grid) length(y_grid) size(G_mat',2)]);
    
% count ray if dist travelled in cell above threshold of ~100m?
G_count = zeros(size(G3D));
ind_G_ray = G3D(:) > thres_dist; % count ray if >100m in cell
G_count(ind_G_ray) = 1;
G_sum = sum(G_count,3);
mask = zeros(size(G_sum));
min_density = 1;
mask(G_sum > min_density) = 0.4;

%% Plot
cla;
colormap(flipud(hot));
set(gca,'linewidth',1.5,'fontsize', fontsize,'layer','top')
hold on
box on

im = pcolor(x_grid_eff,y_grid_eff,G_sum');
set(im,'facealpha','flat','alphadata',mask')
% shading('interp');
shading('flat');
contour(x_grid_eff,y_grid_eff,G_sum',contour_lev*[1 1],'linecolor','k','linewidth',1.5)  % to draw a single contour element, need to put it twice...

% Plot background
imagesc(x_map,y_map,map,'alphadata',0.6);

% Plot stations
plot(x_stat,y_stat,'vk','linewidth',1.5,'markersize',2,'markerfacecolor','k')
% patch([5 5+dx_grid 5+dx_grid 5],[5 5 5+dy_grid 5+dy_grid],'k')

axis equal
axis tight
%set(gca,'xlim',[x_map(1) x_map(end)])
% title(['Period T=' num2str(T) ' s']);
xlabel('Easting (km)'); ylabel('Northing (km)');
cb = colorbar;
%cb.Ticks = tick_mark ; %Create 8 ticks from zero to 1
%cb.TickLabels = tick_label;
cb.LineWidth = 1.5;
ylabel(cb,'# of rays per cell)','fontsize', fontsize)

%     export_fig(['./density_T' num2str(T) '_' comp '.png'],'-transparent')
%export_fig(['./paths_T' num2str(Tc) '.pdf'],'-transparent')


end

