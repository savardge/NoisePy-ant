% Ouput the group velocity map inversions in the format needed for Vs depth inversion. 
%
% Genevieve Savard (2023)

clear all; close all;
clc

%% USER INPUTS
% datadir = '../../data-riehen/F_inv_2D_vg_slices/grid200mX200m'
datadir = '../../data-riehen/run3_dcV2_mul2_g200m/vg-maps'

comp = 'ZZ'

sigma=8; LC=0.3; % put chosen regularization parameters
% sigma=4; LC=0.8;

% Load inversion results
result_dir = [datadir '/all_inv_TV_sigma' num2str(sigma) '_LC' num2str(LC)]
load([result_dir '/all_inv_TV_sigma' num2str(sigma) '_LC' num2str(LC) '_' comp '.mat'])

% Output path
output_fname = [datadir '/all_data_LC' num2str(LC) '_sigma' num2str(sigma) '_' comp '.mat']

%% Extract results and write to output file

DATA_V_all = zeros(length(x_grid),length(y_grid),length(Tc_vec));
for ind_Tc=1:length(Tc_vec) 
    
    m_est = m_est3_struc{ind_Tc}; % inverted model

    S_map = reshape(m_est,[length(x_grid), length(y_grid)]); % slowness map
    V_map = 1./S_map*1000; % velocity map
    
    DATA_V_all(:,:,ind_Tc)=V_map;
        
end

save(output_fname, 'DATA_V_all', 'Tc_vec', 'x_grid', 'y_grid', 'dx_grid', 'dy_grid');

