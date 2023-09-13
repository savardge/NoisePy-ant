function P_resamp = resamp_voronoi_constrained(P_set,ind_best,N_rand,max_dv,min_dz,Z_range_mat,V_range_mat,bool_v_increase)
% last updated: 07/2018 Thomas Planes

% P_set is the nb_cell by nb_dim matrix of cell coordinates
% ind_best is the integer index of current best cell within P_set
% N_rand is the number of random samples to generate within the given voronoi cell + constraints
% max_dv maximum relative velocity change
% min_dz minimum layer thickness
% Z_range_mat prior depth range
% V_range_mat prior velocity range
% bool_v_increase boolean set to true to force velocity model to increase with depth; set to false otherwise

% compute constraints to "change cell boundaries" during random walk

nb_dim=size(P_set,2);
n_layer=(nb_dim+1)/2;
p_curr=P_set(ind_best,:);
P_set_eff=P_set; P_set_eff(ind_best,:)=[];
nb_cell=size(P_set_eff,1);

P_resamp=zeros(N_rand,nb_dim);

zmin=Z_range_mat(:,1)'; zmax=Z_range_mat(:,2)'; % to compute z and Vs from P
vmin=V_range_mat(:,1)'; vmax=V_range_mat(:,2)';

%% plot if 2D

% if nb_dim==2
%     
%     figure('position',get(0,'screensize'));
%     set(gca,'fontsize',14,'linewidth',1.5);
%     hold on; box on;
%     voronoi(P_set(:,1),P_set(:,2),'+')
%     plot([0 1],[p_curr(2) p_curr(2)],'b-')
%     plot([p_curr(1) p_curr(1)],[0 1],'b-')
%     plot(p_curr(1),p_curr(2),'r+','markersize',10)
%     axis equal
%     
% end

%% initialization of random walk

p_walker=p_curr; % random walk starts at current cell center
d_curr2=0; % squared distance from walker to current best cell thus zero;
d_cell2_vec=sum((repmat(p_walker,[nb_cell, 1])-P_set_eff).^2,2); % squared distance from walker to all cells intialization

for ind_walk=1:N_rand
    
    for ind_dim=1:nb_dim
        
        x_walker=p_walker(ind_dim);
        x_curr=p_curr(ind_dim);
        d_curr_orth2=d_curr2-(x_walker-x_curr)^2; % squared orthogonal distance from current cell to current axis passing through walker
        
        x_cell_vec=P_set_eff(:,ind_dim);
        
        d_cell_orth2_vec=d_cell2_vec-(x_walker-x_cell_vec).^2; % squared orthogonal distance from cell to current axis passing through walker
        
        x_inter_vec=0.5*(x_curr+x_cell_vec+(d_curr_orth2-d_cell_orth2_vec)./(x_curr-x_cell_vec)); %intersection coordinate position on current axis of intersection between current cell and cell
                        
        %% find inferior intersection
        
        bool_inf=x_inter_vec<x_walker;
        
        if sum(bool_inf)>0
            
            x_inter_inf=x_inter_vec(bool_inf);
            [~, ind]=min(x_walker-x_inter_inf);
            x_inf_temp=max([x_inter_inf(ind) 0]); % in case closest intersection outside of the box
            clear ind x_inter_inf
            
        else
            
            disp('warning: current cell is at domain boundary')
            x_inf_temp=0;  % dbl check this
            
        end
                
        %% find superior intersection
        
        bool_sup=x_inter_vec>x_walker;
        
        if sum(bool_sup)>0
            
            x_inter_sup=x_inter_vec(bool_sup);
            [~, ind]=min(x_inter_sup-x_walker);
            x_sup_temp=min(x_inter_sup(ind),1); % % in case closest intersection outside of the box
            clear ind x_inter_sup
            
        else
            
            disp('warning: current cell is at domain boundary')
            x_sup_temp=1;
                   
        end
        
        %% add constraints (forward and backward!)
        
        [x_inf, x_sup]=check_constraint_voronoi(ind_dim,p_walker,x_inf_temp,x_sup_temp,zmin,zmax,vmin,vmax,n_layer,min_dz,max_dv,bool_v_increase);
        clear x_inf_temp x_sup_temp
        
        %% random displacement along current axis
        
        if x_sup>x_inf
            x_walker_new=x_inf+(x_sup-x_inf)*rand; % random on current axis within voronoi cell boundaries + constraints
        else
            disp('error range')
            pause;
        end
         
        p_walker(ind_dim)=x_walker_new;
        
        d_curr2=d_curr_orth2+(x_walker_new-x_curr)^2; % update d_curr2
        d_cell2_vec=d_cell_orth2_vec+(x_walker_new-x_cell_vec).^2; % update d_cell2_vec
        
    end
    
    P_resamp(ind_walk,:)=p_walker;
    
%     if nb_dim==2
%         plot(p_walker(1),p_walker(2),'*r')
%     end
    
end
   
end


