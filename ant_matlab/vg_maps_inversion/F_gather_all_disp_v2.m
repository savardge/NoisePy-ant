function [] = F_gather_all_disp_v2(datadir,comp, sigma, LC, Tc_vec)
% Ouput the group velocity map inversions in the format needed for Vs depth inversion. 
% Arguments:
% - datadir: parent directory of project 
% - comp: cross-correlation component data used (e.g. ZZ, TT, etc)
% - sigma, LC: regularization parameters
% - Tc_vec: vector of periods for which to export
%
% Genevieve Savard (2023)

% Example inputs:
% datadir = '../../data-aargau/run0_dc2022_g500m'
% datadir = '../../data-aargau/run4_dcV2_mul3_g500m'
% datadir = '../../data-riehen/run4_dcV2_mul3_g200m'
% comp = 'TT'
% sigma=8; LC=0.3; % put chosen parameters
% sigma=4; LC=0.8;
% sigma=5; LC=0.7;
% Tc_vec = 0.2:0.1:5.0%.5

result_dir = [datadir '/vg-maps/all_inv_TV_sigma' num2str(sigma) '_LC' num2str(LC)];
output_fname = [datadir '/vg-maps/all_data_LC' num2str(LC) '_sigma' num2str(sigma) '_' comp '.mat']

load([result_dir '/all_inv_TV_sigma' num2str(sigma) '_LC' num2str(LC) '_' comp '_T' sprintf('%03.1f', Tc_vec(1)) '.mat'], 'x_grid','y_grid')        

DATA_V_all = zeros(length(x_grid),length(y_grid),length(Tc_vec));

for ind_Tc=1:length(Tc_vec) 
    
    Tc = Tc_vec(ind_Tc);
    fname = [result_dir '/all_inv_TV_sigma' num2str(sigma) '_LC' num2str(LC) '_' comp '_T' sprintf('%03.1f', Tc) '.mat'];
    if ~exist(fname,'file')
        continue
    end
%     disp(Tc)
    load(fname)
%     m_est = m_est1;
    
    %% Inv 1
    S_map1 = reshape(m_est1,[length(x_grid), length(y_grid)]);
    V_map1 = 1./S_map1 * 1000; % GS fix!    
    DATA_V_all1(:,:,ind_Tc) = V_map1;

    %% Inv 2
    S_map2 = reshape(m_est2,[length(x_grid), length(y_grid)]);
    V_map2 = 1./S_map2 * 1000; % GS fix!    
    DATA_V_all2(:,:,ind_Tc) = V_map2;

    %% Inv 1
    S_map3 = reshape(m_est3,[length(x_grid), length(y_grid)]);
    V_map3 = 1./S_map3 * 1000; % GS fix!    
    DATA_V_all3(:,:,ind_Tc) = V_map3;        

    DATA_V_all = DATA_V_all3;
end

save(output_fname, 'DATA_V_all', 'DATA_V_all1','DATA_V_all2','DATA_V_all3', 'Tc_vec', 'x_grid', 'y_grid', 'dx_grid', 'dy_grid');

