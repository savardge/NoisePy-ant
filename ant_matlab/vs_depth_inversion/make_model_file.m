function bool_fail = make_model_file(mod_name,Vs_set,Vp_set,rho_set,Z_set)
% For a given proposed velocity model, write the text model file used as
% input by gpdc
%
% Thomas Planes (2019)

N_mod=size(Vs_set,1);  % number of model to write in file
nb_layer=size(Vs_set,2); % number of layers
    
fname = [mod_name '.model']
fid = fopen(fname,'w+t');

for ind_N = 1:N_mod
    
    Vs = Vs_set(ind_N,:);
    Vp = Vp_set(ind_N,:);
    rho = rho_set(ind_N,:);
    z_layer_bottom = Z_set(ind_N,:);
    dz_thickness = [diff([0 z_layer_bottom(1:end-1)]) 0]; % here is the patch, april 2019

    layer_cell = cell(nb_layer,1);
    
    for ii = 1:nb_layer
        layer_cell{ii} = [num2str(dz_thickness(ii)) ' ' num2str(Vp(ii)) ' ' num2str(Vs(ii)) ' ' num2str(rho(ii))];
    end
    
    fprintf(fid,'%s\n',num2str(nb_layer));
    
    for ii = 1:nb_layer
        fprintf(fid,'%s\n',layer_cell{ii});
    end
    
    clear Vs Vp rho z_layer_bottom layer_cell dz_thickness
    
end

bool_fail = fclose(fid); % returns 0 if OK, -1 if fails

end


% from geopsy wiki:

% # My first model: two layers over a half-space
% # First line: number of layers
% 3
% # One line per layer:
% # Thickness(m), Vp (m/s), Vs (m/s) and density (kg/m3)
% 7.5  500  200 1700
% 25  1350  210 1900
% # Last line is the half-space, its thickness is ignored but the first column is still mandatory
% 0   2000 1000 2500 
