% Run spike resolution tests for a range of periods, based on picked data
% 
% Must define synthetic model parameters and input paths
%
% By Genevieve Savard @ UniGe (2023)
% 

clear all; clc;

%% Define period ranges to do test on
% Tc_list = [0.5, 1,1.5,2,3.2,3.6,4,4.8]
% Tc_list = [0.5,1,1.5,2,2.5,3,3.5,4,4.5]
Tc_list = [3,3.5,4,4.5]

%% Define data inputs 

comp  =  'ZZ'
datadir = '../../../data-riehen/run3_dcV2_mul2_g200m'; 

load([datadir '/grid/stat_grid.mat'], 'X_GRID', 'x_grid', 'y_grid', 'dx_grid', 'dy_grid', 'x_stat', 'y_stat')
kernel_file = [datadir '/grid/kernel.mat']; % full data kernel (G matrix)
data_kern_path = [datadir '/vg-maps/data_kern_' comp]; % directory with picked data at each period (effective G)
load([datadir, '/grid/map_matrix_terrain_wFaults.mat'], 'map','x_map','y_map') % Map background, for plotting
output_path = [datadir, '/vg-maps/checkerboard'];

if ~exist(output_path,'dir')
    mkdir(output_path);
end

%% Parameters

% Forward problem
nb_el = 1; %number of linear grid cell in each checkerboard case
v_perturb = 0.15;
v_moy = 2.0;
v_up = (1+v_perturb)*v_moy; 
v_down = (1-v_perturb)*v_moy;
%v_up = 2.7; v_down = 2.3;

% Inverse problem
sigma = 4 %8;
LC = 0.8 %.3;
L0 = sqrt(dx_grid^2 + dy_grid^2);
rel_err = 5/100;

% Output path
output_path = [datadir, '/vg-maps/checkerboard_' sprintf('%d', v_perturb*100)]
if ~exist(output_path,'dir')
    mkdir(output_path);
end

% Load grid data
load(kernel_file, 'dx_grid', 'dy_grid', 'X_GRID', 'Y_GRID', 'x_grid', 'y_grid', 'x_stat', 'y_stat') %G_mat

%% Process each period
for indTc=1:length(Tc_list)

    Tc = Tc_list(indTc);

    % Load G
    load([data_kern_path, '/data_and_kern_T' sprintf('%3.1f',Tc) '_' comp '.mat'], 'G_mat') % this uses real G at T=Tc

    % Output filename for figure and .mat
    figfile =      [output_path, '/checkerboard_T' sprintf('%3.1f', Tc) '_LCsigma' num2str(LC) '-' num2str(sigma) '_' num2str(nb_el) 'x' num2str(nb_el) '_' comp '.png']
    output_fname = [output_path, '/checkerboard_T' sprintf('%3.1f', Tc) '_LCsigma' num2str(LC) '-' num2str(sigma) '_' num2str(nb_el) 'x' num2str(nb_el) '_' comp '.mat']

    %% forward problem
    disp('Creating forward model...')

    V_2D = zeros(size(X_GRID));
    ind_x = 1:length(x_grid); ind_y = 1:length(y_grid);
    bool_x = mod(floor((ind_x-1)/nb_el),2)*2-1;
    bool_y = mod(floor((ind_y-1)/nb_el),2)*(-2)+1;
    bool_check = logical(reshape((bool_x'*bool_y+1)/2,[1 numel(X_GRID)]));
    V_2D(bool_check) = v_up; V_2D(~bool_check) = v_down;
    S_2D = 1./V_2D;
    S_lin = reshape(S_2D,[numel(X_GRID),1]);
    TAU = G_mat * S_lin;

    % Add noise
%     var_noise = sqrt(0.01*rel_err*mean(TAU)).*randn(size(TAU));
%     TAU = TAU + var_noise;
    
    %save forward.mat TAU V_2D G_mat;

    %% Inverse problem
    
    disp('Preparing inversion')

    d = TAU;  % d  =  travel time data
    N_d = length(d); % number of data points    
    G = G_mat; % G  =  kernel
    N_m = size(G,2); % number of model cells
    
    % Data prior covariance matrix
    Cd_vec = (rel_err*d).^2;
    CD = diag(Cd_vec);
    CD_inv = diag(1./Cd_vec);    

    % Model prior covariance matrix
    s_prior = 1/v_moy; % prior homogeneous slowness    
    m_prior = s_prior*ones(N_m,1);
    x_cell = reshape(X_GRID,[N_m, 1]); 
    y_cell = reshape(Y_GRID,[N_m, 1]);
    X_CELL = repmat(x_cell,[1 N_m]); 
    Y_CELL = repmat(y_cell,[1 N_m]);
    DIST_CELL = sqrt((X_CELL-X_CELL').^2+(Y_CELL-Y_CELL').^2);
    CM = (sigma*L0/LC)^2*exp(-1/LC*DIST_CELL);
%     CM = (sigma)^2 * exp(- DIST_CELL / LC );
    CM_inv = inv(CM);

    disp('Launching TV inversion...')
    % Inversion Tarantola-Valette
    m_est = m_prior + (G' * CD_inv * G + CM_inv ) \ G' * CD_inv * ( d - G * m_prior);
    S_map = reshape(m_est,[length(x_grid), length(y_grid)]);
    V_map = 1./S_map;

    %% Plot
    % in kernel, grid nodes defined at bottom left of cell (should dblcheck); 
    % in imagesc, node is at center of cell; these new effective axes compensate for this
    x_grid_eff = x_grid + dx_grid/2; 
    y_grid_eff = y_grid + dy_grid/2; 

    % Get density mask    
    min_density = 3;
    thres_dist = 0.01; % km
    G3D = reshape(G_mat',[length(x_grid) length(y_grid) size(G_mat',2)]);
    % count ray if dist travelled in cell above threshold of ~100m?
    G_count = zeros(size(G3D));
    ind_G_ray = G3D(:) > thres_dist; % count ray if >100m in cell
    G_count(ind_G_ray) = 1;
    G_sum = sum(G_count,3);
    mask = nan(size(G_sum));
    mask(G_sum > min_density) = 1;

    disp('Plotting...')
    figure;clf;
    set(gcf, 'position', [44,269,1707 ,693], 'color', 'w')
    
    % Plot synthetic model
    V_2D_perturb = (V_2D-v_moy)./v_moy * 100;
    subtightplot(1,2,1); cla; hold on  
    set(gca,'linewidth',1.5,"FontSize",14)
    % Plot background
    imagesc(x_map,y_map,map,'alphadata',1);
    axis([min(x_map(:)), max(x_map(:)), min(y_map(:)), max(y_map(:))])
    
    %Plot model
    im = pcolor(x_grid_eff,y_grid_eff,V_2D_perturb');    
    set(im,'facealpha','flat','alphadata',mask')
    shading('flat');
    colormap(flipud(jet));
%     hb=colorbar;
%     ylabel(hb,'Velocity perturbation (%)','fontsize',12)
    caxis([-v_perturb, v_perturb]*100)

    % Plot stations
    plot(x_stat,y_stat,'vk','linewidth',1.5,'markersize',4,'markerfacecolor','k')
    % axis equal
    daspect([1 1 1])
    title(['Synthetic model'])
    axis xy
    xlabel('Easting [km]')
    ylabel('Northing [km]')

    % Plot solution
    V_map_perturb = (V_map-v_moy)./v_moy * 100;
    subtightplot(1,2,2);cla;hold on
    set(gca,'linewidth',1.5,"FontSize",14)    
    % Plot background
    imagesc(x_map,y_map,map,'alphadata',1);
    axis([min(x_map(:)), max(x_map(:)), min(y_map(:)), max(y_map(:))])
    
    % Plot solution
    im = pcolor(x_grid_eff,y_grid_eff,V_map_perturb');    
    set(im,'facealpha','flat','alphadata',mask')
    shading('flat');
    colormap(flipud(jet));
    hb=colorbar;
    ylabel(hb,'Velocity perturbation (%)','fontsize',12)
    caxis([-v_perturb, v_perturb]*100)

    
    % Plot stations
    plot(x_stat,y_stat,'vk','linewidth',1.5,'markersize',4,'markerfacecolor','k')
    % axis equal
    daspect([1 1 1])
    title(['Inversion result at T = ' num2str(Tc), ' s'])
    axis xy
    xlabel('Easting [km]')
    ylabel('Northing [km]')

    % subtightplot(1,3,3)
    % set(gca,'linewidth',1.5,"FontSize",14)
    % hold on
    % imagesc(x_grid,y_grid,(V_map'-V_2D')./V_2D'.*100)
    % %pcolor(x_grid,y_grid,V_2D')
    % %shading('interp');
    % colormap(flipud(jet))
    % cb = colorbar;
    % ylabel(cb, "% difference","FontSize",14)
    % caxis([0, 5])
    % axis([min(x_grid(:)), max(x_grid(:)), min(y_grid(:)), max(y_grid(:))])
    % plot(x_stat,y_stat,'sk', 'linewidth',0.5)
    % % axis equal
    % daspect([1 1 1])

    % title(['Synthetic model'])
    %suptitle(["Checkerboard test at T = ", num2str(Tc)])
    % saveas(gcf,figfile)
    
    %% Save
    export_fig(figfile, '-r300', '-png', '-transparent')
    disp(['Figure saved to: ', figfile])
        
    save(output_fname, 'TAU', 'V_2D', 'G_mat','nb_el','v_perturb', 'v_moy','sigma','LC','rel_err','m_est','V_map','V_map_perturb','x_grid','y_grid')
    pause(0.5)

end