function []=plot_paths_vg(V_dat, coord_s1_lin, coord_s2_lin, x_stat, y_stat, fignum, x_map, y_map, map)

figure(fignum);clf;
colormap_T=flipud(jet(100)); 
colormap(colormap_T); 
vmin = min(V_dat*1e3); 
vmax = max(V_dat*1e3); 
v_vec = linspace(vmin,vmax,size(colormap_T,1));
tick_mark = 0:0.1:1; % 11 ticks
tick_label = linspace(vmin,vmax,length(tick_mark));

subplot(5,1,[1,2,3]); hold on
set(gca,'linewidth',1.5,'fontsize',16,'layer','top')
hold on
box on
plot(x_stat,y_stat,'vk','linewidth',2,'markersize',10)
% Plot background
imagesc(x_map,y_map,map) %,'alphadata',0.6);
%imagesc(x_map,y_map,map);
azimuths = [];
distances = [];
for cpl=1:length(V_dat)

    vel_plot = V_dat(cpl)*1e3;
    [~, ind_color] = min(abs(vel_plot-v_vec));
    color_plot = colormap_T(ind_color,:);    
    plot([coord_s1_lin(cpl,1) coord_s2_lin(cpl,1)],[coord_s1_lin(cpl,2) coord_s2_lin(cpl,2)],'-','color',color_plot,'linewidth',3);

    % Impose azimuth or distance constraints to plotting
    dist = sqrt( (coord_s2_lin(cpl,2)-coord_s1_lin(cpl,2))^2 + (coord_s2_lin(cpl,1) - coord_s1_lin(cpl,1))^2 );
    distances = [distances, dist];
    azimuth = mod(atan2d((coord_s2_lin(cpl,2) - coord_s1_lin(cpl,2)),(coord_s2_lin(cpl,1) - coord_s1_lin(cpl,1)))+360,360);
    azimuths = [azimuths, azimuth];
%     if azimuth > 45 && azimuth < 90
%         plot([coord_s1_lin(cpl,1) coord_s2_lin(cpl,1)],[coord_s1_lin(cpl,2) coord_s2_lin(cpl,2)],'-','color',color_plot,'linewidth',3);
% %         pause
%     end

%     if mod(cpl,1000) ==0
%         pause
%     end
end

axis equal
axis tight
% title(['T=' num2str(Tc) ' s']);
xlabel('Easting (km)'); ylabel('Northing (km)');
cb=colorbar;
cb.Ticks = tick_mark ; %Create 8 ticks from zero to 1
cb.TickLabels = tick_label;
cb.LineWidth=1.5;
ylabel(cb,'Group velocity (m/s)','fontsize',16)
     
subplot(5,1,4);
histogram(azimuths, 50)
xlabel("azimuth of rays")
subplot(5,1,5);
histogram(distances, 50)
xlabel("Lengths of rays")
end