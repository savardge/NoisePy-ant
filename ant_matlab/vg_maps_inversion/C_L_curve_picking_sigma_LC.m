% This script plots model and data misfits for each inversion from script B
% to determine the optimal sigma and LC from an L-curve analysis. 
% 
% In practice this does not work so well though... Must check inversion
% results themselves as some parameters lead to unrealistic group
% velocities.
%
% Written by Thomas Planes (2019)
% Modified by Genevieve Savard (2023)

clear all; %close all;
clc

%% USER INPUTS
Tc = 2.0; % Period
comp  =  'ZZ'
datadir = '../../data-riehen/run3_dcV2_mul2_g200m'
kernel_file = [datadir, '/grid/kernel.mat']
kernel_dir = [datadir '/vg-maps/data_kern_' comp ]
data_kernel_file = [kernel_dir '/data_and_kern_T' sprintf('%3.1f',Tc) '_' comp '.mat']
inversion_results_file = [datadir '/vg-maps/full_params_set/inv_T' sprintf('%3.1f',Tc) '_full_params_set_' comp '.mat']

%% Get ray path density and make mask for plotting
min_density = 3; % minimum number of rays crossing a cell
thres_dist = 0.01; % km minimum distance to travel in cell to count
load(kernel_file, 'x_grid','y_grid','x_stat','y_stat','dx_grid','dy_grid')
load(data_kernel_file, '-mat', 'G_mat');
G3D = reshape(G_mat',[length(x_grid) length(y_grid) size(G_mat',2)]);
G_count = zeros(size(G3D));
ind_G_ray = G3D(:) > thres_dist;
G_count(ind_G_ray) = 1;
G_sum = sum(G_count,3);
mask = zeros(size(G_sum));
mask(G_sum > min_density) = 1.0;
maskv = mask(:);
ind_keep = find(maskv == 1);

%% Get inversion results
load(inversion_results_file,'m_est_struc','m_prior','d','d_post_struc','nb_sigma','sigma_vec','nb_LC','LC_vec','rel_err');
% m_prior = m_prior(ind_keep);

%% Plot data misfit vs model misfit L-curve

figure('position',get(0,'screensize'));
set(gca,'fontsize',16,'linewidth',1.5);
box on
grid on
hold on

misfit_data = zeros(nb_sigma,nb_LC);
misfit_model = zeros(nb_sigma,nb_LC);

for ind_sigma = 1:nb_sigma
    for ind_LC = 1:nb_LC
        
        m_est = m_est_struc{ind_sigma,ind_LC};
%         m_est = m_est(ind_keep);
        d_post = d_post_struc{ind_sigma,ind_LC};
        
        misfit_data(ind_sigma,ind_LC) = sqrt(mean(((d_post-d)./d).^2)); % This formulation leads to choosing overfitting sigma
%         misfit_data(ind_sigma,ind_LC) = sqrt(mean(((d_post-d)./(rel_err*d)).^2)); 
        % Here compares to assumed error; 
        % should find a better way to assess error in measurements (maybe causal/anticausal picking? or summer/winter?)
        % actually just prefactor, doesnt change anythin ?!
        
        misfit_model(ind_sigma,ind_LC) = sqrt(mean(((m_est(:)-m_prior(:))./m_prior(:)).^2)); % could calculate this only in correct path-covered area
        
        plot(misfit_model(ind_sigma,ind_LC), misfit_data(ind_sigma,ind_LC), '+', 'linewidth',2)
        hold on
        text(misfit_model(ind_sigma,ind_LC), misfit_data(ind_sigma,ind_LC), ['S' num2str(sigma_vec(ind_sigma)) 'L' num2str(LC_vec(ind_LC))])
        
    end
end

%plot(misfit_model,misfit_data,'+','linewidth',2)
%loglog(misfit_model,misfit_data,'+','linewidth',2)
xlabel('misfit model'); ylabel('misfit data');
hold on
title(['Period T = ' num2str(Tc) ' s'])

%% Interactive picking of bend in L-curve
[x, y] = ginput(1);
dist_pick = sqrt((x-misfit_model).^2+(y-misfit_data).^2);
[~, ind] = min(dist_pick(:));
[ind_sigma_pick, ind_LC_pick] = ind2sub(size(misfit_data),ind);

% Plot "optimal params"
plot(misfit_model(ind_sigma_pick,ind_LC_pick),misfit_data(ind_sigma_pick,ind_LC_pick),'or')
legend({['sigma = ' num2str(sigma_vec(ind_sigma_pick))];['LC = ' num2str(LC_vec(ind_LC_pick))]})

