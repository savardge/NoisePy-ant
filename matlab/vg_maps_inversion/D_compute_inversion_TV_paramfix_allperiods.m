function [] = D_compute_inversion_TV_paramfix_allperiods()
% This script does the group velocity map inversion for a range of periods
% and a chosen regulaization parameters sigma, LC.
% G.S. added consecutive inversions (3 total), downweighting outliers at
% each step.
% 
% Written by Thomas Planes (2019) and Genevieve Savard (2023)

%% set up parpool
p = gcp('nocreate');
if isempty(p)
    if isempty(getenv('SLURM_CPUS_PER_TASK'))
        sz = 4;
    else
        sz = str2num(getenv('SLURM_CPUS_PER_TASK'));
    end
    parpool("local",sz);
end

%% Parameter choice
sigma = 4; % sigma (trade-off factor) 
LC = 0.8;  % LC (correlation distance) to choose
rel_err = 5/100; % relative error on data
Tc_vec = 0.2:0.1:6.5; % range of periods to invert for

%% Paths
datadir = '../../data-aargau/run3_dcV2_g500m_mul2'
comp = 'ZZ'
kernel_dir = [datadir '/vg-maps/data_kern_' comp ]
kernel_file = [datadir '/grid/kernel.mat']
output_dir = [datadir '/vg-maps/all_inv_TV_sigma' num2str(sigma) '_LC' num2str(LC)]
if ~exist(output_dir,'dir')
    mkdir(output_dir);
end

%% load grid and station info
load(kernel_file, 'dx_grid', 'dy_grid', 'X_GRID', 'Y_GRID', 'x_grid', 'y_grid', 'x_stat', 'y_stat')
L0 = sqrt(dx_grid^2 + dy_grid^2); % size of model cells
N_m = length(x_grid) * length(y_grid); % number of model cells
x_cell = reshape(X_GRID,[N_m, 1]); y_cell = reshape(Y_GRID,[N_m, 1]);
X_CELL = repmat(x_cell,[1 N_m]); Y_CELL = repmat(y_cell,[1 N_m]);
DIST_CELL = sqrt((X_CELL-X_CELL').^2+(Y_CELL-Y_CELL').^2);

%% Load data and kernels

nb_Tc = length(Tc_vec);
d_struc = cell(nb_Tc);  
m_prior1_struc = cell(nb_Tc);   
d_post1_struc = cell(nb_Tc);
m_est1_struc = cell(nb_Tc);
d_post2_struc = cell(nb_Tc);
m_est2_struc = cell(nb_Tc);
d_post3_struc = cell(nb_Tc);
m_est3_struc = cell(nb_Tc);

v_moy_list = cell(nb_Tc);
d_list = cell(nb_Tc);
G_list = cell(nb_Tc);
for ind_Tc=1:nb_Tc
    Tc = Tc_vec(ind_Tc);
    load([kernel_dir '/data_and_kern_T' sprintf('%3.1f',Tc) '_' comp '.mat'], '-mat', 'v_moy', 'TAU', 'G_mat');
    v_moy_list{ind_Tc} = v_moy;
    d_list{ind_Tc} = TAU;
    G_list{ind_Tc} = G_mat;
    
end

%% Start TV inversions
parfor ind_Tc=1:nb_Tc
    
    Tc = Tc_vec(ind_Tc);    
    v_moy = v_moy_list{ind_Tc};
    d = d_list{ind_Tc};
    G = G_list{ind_Tc};
%     N_d = length(d); % number of data points
    
    %% Define model prior covariance (same for all inversions)
    CM = (sigma * L0 / LC)^2 * exp(-1/LC*DIST_CELL);
    % CM = (sigma)^2 * exp(-1/LC*DIST_CELL);
    CM_inv = inv(CM);

    %% Inversion 1

    % define prior model and data covariance matrix
    s_prior = 1/v_moy; % prior homogeneous slowness
    m_prior1 = s_prior * ones(N_m,1);
    Cd_vec1 = (rel_err*d).^2;
%     CD1 = diag(Cd_vec1);
    CD_inv1 = diag(1./Cd_vec1);    
    
    % Invert    
    d_prior1 = G * m_prior1;
    m_est1 = m_prior1 + (G' * CD_inv1 * G + CM_inv ) \ G' * CD_inv1 * (d - d_prior1);
    d_post1 = G * m_est1;

    % Misfit
    misfit1 =  d - d_post1;
    misfit_mean1 = mean(misfit1);
    misfit_std1 = std(misfit1);

    %% Inversion 2
    % Change CD
    Cd_vec2 = Cd_vec1;
    ioutliers = abs(misfit1-misfit_mean1) > 2*misfit_std1;
    Cd_vec2(ioutliers) = Cd_vec2(ioutliers).*exp((abs(misfit1(ioutliers))./(2*misfit_std1)-1)); % c.f. Liu and Yao 2016
%     CD2 = diag(Cd_vec2);
    CD_inv2 = diag(1./Cd_vec2);
    
    % Update m_prior and d_prior
    m_prior2 = m_est1;
    d_prior2 = G * m_prior2;
    m_est2 = m_prior2 + (G' * CD_inv2 * G + CM_inv ) \ G' * CD_inv2 * (d - d_prior2);
    d_post2 = G * m_est2;
    
    % New misfit
    misfit2 =  d - d_post2;
    misfit_mean2 = mean(misfit2);
    misfit_std2 = std(misfit2);

    %% Inversion 3
    % Change CD
    Cd_vec3 = Cd_vec2;
    ioutliers = abs(misfit2-misfit_mean2) > 2*misfit_std2;
    Cd_vec3(ioutliers) = Cd_vec3(ioutliers).*exp((abs(misfit2(ioutliers))./(2*misfit_std2)-1)); % c.f. Liu and Yao 2016
%     CD3 = diag(Cd_vec3);
    CD_inv3 = diag(1./Cd_vec3);
    
    % Update m_prior and d_prior
    m_prior3 = m_est2;
    d_prior3 = G * m_prior3;
    m_est3 = m_prior3 + (G' * CD_inv3 * G + CM_inv ) \ G' * CD_inv3 * (d - d_prior3);
    d_post3 = G * m_est3;
    
    % New misfit
%     misfit3 =  d - d_post3;
%     misfit_mean3 = mean(misfit3);
%     misfit_std3 = std(misfit3);

    %% save
    
    d_struc{ind_Tc} = d;   
    m_prior1_struc{ind_Tc} = m_prior1;
    d_prior1_struc{ind_Tc} = d_prior1;    
    d_post1_struc{ind_Tc} = d_post1;
    m_est1_struc{ind_Tc} = m_est1;
    d_post2_struc{ind_Tc} = d_post2;
    m_est2_struc{ind_Tc} = m_est2;
    d_post3_struc{ind_Tc} = d_post3;
    m_est3_struc{ind_Tc} = m_est3;
        
    disp([num2str(ind_Tc) ' out of ' num2str(nb_Tc) '. Tc = ' num2str(Tc)])
    
end

save([output_dir '/all_inv_TV_sigma' num2str(sigma) '_LC' num2str(LC) '_' comp '.mat'], '-mat', ...
    'm_est1_struc','m_est2_struc','m_est3_struc', ...
    'd_struc','d_post1_struc','d_post2_struc','d_post3_struc', ...
    'm_prior1_struc','d_prior1_struc', ...
    'Tc_vec','x_grid','y_grid','x_stat','y_stat','dx_grid','dy_grid'); 

end
