function disp_mat = read_disp_data(mod_name,N_disp,N_samp)
% Script to read the predicted dispersion curves outputed by gpdc
%
% Thomas Planes (2019)

disp_mat = zeros(N_disp,N_samp);  % no need to keep freq vec if always the same...should double check

%% read data

fid = fopen([mod_name '.disp'],'rt');
count_fail = 0;

for ind_N = 1:N_disp

    header = textscan(fid,'%s',3,'delimiter','\n');  %read commented header
    data_cell = textscan(fid,'%f %f\n',N_samp); % does not move cursor during failure (no data found)
    
    %freq_vec=data_cell{1};
    slow_vec = data_cell{2};
    
    if ~isempty(slow_vec) % detects when failure because of LVZ and maintains disp curve at zero; then goes to next one        
        vg_vec = 1./slow_vec;
        disp_mat(ind_N,:) = vg_vec;
    else
        %disp('Warning: dispersion-curve computation failed with current model')
        count_fail = count_fail+1;
    end
        
    clear data_cell slow_vec vg_vec %freq_vec 
  
end

if count_fail == N_disp
    error([num2str(count_fail) ' dispersion-curve computations failed out of ' num2str(N_disp) ' in current set'])
else if count_fail > 0
    warning([num2str(count_fail) ' dispersion-curve computations failed out of ' num2str(N_disp) ' in current set'])
end

fclose(fid);

end



