% Script to find with 2D grid indices have data and which one don't, to
% avoid running depth inversions on indices with insufficient data
% Genevieve Savard (2023)

clear all

%% USER INPUT
% sigma = 8; 
% LC = 0.3; % put chosen parameters
% datadir = '../../data-riehen/run3_dcV2_mul2_g200m'
% sigma = 4; LC = 0.8 %3; % put chosen parameters

sigma = 8; LC = 0.3; % put chosen parameters
datadir = '../../data-riehen/run5_dcV2_mul2_g200m'

% Run for Vg maps corresponding to picks from different CC components:
comps = {'ZZ', 'ZR' 'ZZ-ZR', 'TT'}

for icomp=1:length(comps)
    comp = comps{icomp}
    
    %data_file = [datadir, '/vg-maps/all_data_LC' num2str(LC) '_sigma' num2str(sigma) '_' comp '.mat']   
    %load(data_file, 'DATA_V_all', 'Tc_vec', 'x_grid', 'y_grid','dx_grid','dy_grid');
    kernel_dir = [datadir '/vg-maps/data_kern_' comp ]
    flist = dir([datadir '/vg-maps/all_inv_TV_sigma' num2str(sigma) '_LC' num2str(LC) '/all_inv_TV_sigma' num2str(sigma) '_LC' num2str(LC) '_' comp '*_T*']);
    load(fullfile(flist(1).folder,flist(1).name), 'x_grid', 'y_grid','dx_grid','dy_grid');
    ind_x_range = 1:length(x_grid); 
    ind_y_range = 1:length(y_grid);
    Tc_vec = zeros(length(flist),1);
    for ind_file=1:length(flist)
        dum = strsplit(flist(ind_file).name,'_'); dum=strsplit(dum{7},'.mat'); dum=strsplit(dum{1},'T'); 
        Tc_vec(ind_file) = str2double(dum{2});
    end

    raycount_all = zeros([length(Tc_vec) length(x_grid) length(y_grid)]);
    for ind_Tc=1:length(Tc_vec)
        Tc = Tc_vec(ind_Tc)
        % Get density mask
        load([kernel_dir '/data_and_kern_T' sprintf('%3.1f',Tc) '_' comp '.mat'], '-mat', 'G_mat');
        thres_dist = 0.01; % km
        G3D = reshape(G_mat',[length(x_grid) length(y_grid) size(G_mat',2)]);
        % count ray if dist travelled in cell above threshold of ~100m?
        G_count = zeros(size(G3D));
        ind_G_ray = G3D(:) > thres_dist; % count ray if >10m in cell
        G_count(ind_G_ray) = 1;
        G_sum = sum(G_count,3);
        raycount_all(ind_Tc,:) = reshape(G_sum, [1 length(ind_x_range)*length(ind_y_range)]);
    end
    min_density = 3;
    raycount_total = squeeze(sum(raycount_all,1));
    mask = zeros(size(raycount_total));
    mask(raycount_total > min_density) = 1;
    % pcolor(x_grid,y_grid,mask')
    ind_with_data = reshape(mask, [1 length(ind_x_range)*length(ind_y_range)]);
    
    % x,y position 
    [Xg,Yg] = ndgrid(x_grid,y_grid);
    Xv = Xg(:)'; Yv = Yg(:)'; 

    % Get index with data
    ind = find(ind_with_data);
    raycount = raycount_total(ind);
    Xpos = Xv(ind); Ypos = Yv(ind);

    % Sort
%     [~, isort] = sort(raycount, 'descend');
%     ind = ind(isort);
%     raycount = raycount(isort);
%     Xpos = Xpos(isort); Ypos = Ypos(isort);
    
    % Output only indices with data: 
    % columns: linear index, x coordinate, y coordinate of cell
    fname = [datadir '/vs-model/indwithdata_raycount_' comp '.list']            
    fid = fopen(fname,'w');
    if fid > 0
        fprintf(fid,'%d %d %5.2f %5.2f\n',[ind', raycount', Xpos', Ypos']');
        fclose(fid);
    end

    % Output all indices, with number of ray paths in each. 
    % columns: linear index, # rays, x coordinate, y coordinate of cell 
    fname = [datadir '/vs-model/ind_lin_raycount_' comp '.list']            
    fid = fopen(fname,'w');
    fprintf(fid,'index,ray_count,x,y\n');
    allind = 1:length(raycount_total(:));
    if fid > 0
        fprintf(fid,'%d %d %5.2f %5.2f\n',[allind', raycount_total(:), Xv', Yv']');
        fclose(fid);
    end
    
end