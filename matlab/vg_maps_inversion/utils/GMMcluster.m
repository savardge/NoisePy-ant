function [cluster_id, P, gm, hfig] = GMMcluster(V_dat)
%GMMCLUSTER GMM clustering of velocity distribution into 2 cluster. 
%   Returns membership ID

doplot = true;

% Input data
X = V_dat*1000;
ncluster = 2;
threshold = 0.9;
  
% Fit a two-component Gaussian mixture model (GMM). 
gm = fitgmdist(X,ncluster, ...
    'SharedCovariance',false, ...
    'Options',statset('Display','off', 'MaxIter',5000), ...
    'RegularizationValue',0.0001);
disp(['# components: ' num2str(ncluster) ', BIC = ' num2str(gm.BIC)])
if ~ gm.Converged
    error('No convergence')
end
sprintf('1st Gaussian: mu = %3.1f, sigma = %3.1f\n2nd Gaussian: mu = %3.1f, sigma = %3.1f\n', gm.mu(2), gm.Sigma(2), gm.mu(1), gm.Sigma(1))

% Estimate component-member posterior probabilities for 
% all data points using the fitted GMM gm. 
% These represent cluster membership scores.
P = posterior(gm,X);

thresholding = 'soft';
threshold = 0.9;
switch thresholding

    case 'hard'
        % Cluster w/ hard thresholding
        cluster_id0 = cluster(gm,X);
    case 'soft'
        % Cluster with soft thresholding
        cluster_id0 = zeros(size(X));
        cluster_id0(P(:,1) > threshold) = 1;
        cluster_id0(P(:,2) > threshold) = 2;
        
    case "2sigma" 
        % Cluster with 2*sigma tolerance 
        cluster_id0 = zeros(size(X));
        idx1 = (X > gm.mu(1) - 2*gm.Sigma(1)) & (X < gm.mu(1) + 2*gm.Sigma(1));
        [gm.mu(1) - 2*gm.Sigma(1), gm.mu(1) + 2*gm.Sigma(1) ]
        length(idx1)
        cluster_id0(idx1) = 1;
        if gm.mu(2) > gm.mu(1) + 2*gm.Sigma(1)
            idx2 = (X > gm.mu(2) - 2*gm.Sigma(2)) & (X < gm.mu(2) + 2*gm.Sigma(2));
            [gm.mu(2) - 2*gm.Sigma(2), gm.mu(2) + 2*gm.Sigma(2) ]
            length(idx2)
            cluster_id0(idx2) = 2;
        end
end


% Make sure cluster 1 is slow Vg one
if gm.mu(1) > gm.mu(2) % flip clusters
    cluster_id = ones(size(cluster_id0));
    cluster_id(cluster_id0 == 1) = 2;    
else
    cluster_id = cluster_id0;
%     sprintf('1st Gaussian: mu = %3.1f, sigma = %3.1f\n2nd Gaussian: mu = %3.1f, sigma = %3.1f\n', gm.mu(1), gm.Sigma(1), gm.mu(2), gm.Sigma(2))
end

hfig = [];

if doplot

%     figure(4);clf;
%     subplot(3,1,1);
%     h = histogram(X,100);
%     edges = h.BinEdges;
%     xlim([0.5 4]); 
%     ylimits = get(gca, 'YLim');
%     xlabel("Group velocity (km/s)"); ylabel("PDF")
%     title("Density histogram of data and GMM model fit")
%     
%     subplot(3,1,2)
%     histogram(X(cluster_id==1),edges);
%     xlim([0.5 4]); ylim(ylimits)
%     xlabel("Group velocity (km/s)"); ylabel("PDF")
%     title("Data points in cluster 1 (P > 0.6)")
%     
%     subplot(3,1,3)
%     histogram(X(cluster_id==2),edges);
%     xlim([0.5 4]); ylim(ylimits)
%     xlabel("Group velocity (km/s)"); ylabel("PDF")
%     title("Data points in cluster 2 (P > 0.6)")
%     
    
    cla;hold on
    h = histogram(X,100, 'Normalization', 'pdf');
    edges = h.BinEdges;
    %delete(h)
    v = (min(edges):0.01:max(edges))';
    if gm.mu(1) > gm.mu(2) % flip
%         c1 = histogram(X(cluster_id==1),edges, 'Normalization', 'pdf');
%         set(c1, 'FaceColor', 'r')
%         c2 = histogram(X(cluster_id==2),edges, 'Normalization', 'pdf');
%         set(c2, 'FaceColor', 'b')
        plot(v, pdf('Normal',v,gm.mu(1),sqrt(gm.Sigma(1)))*gm.ComponentProportion(1), "r", 'DisplayName','Cluster 2','LineWidth',2) % cluster 2
        plot(v, pdf('Normal',v,gm.mu(2),sqrt(gm.Sigma(2)))*gm.ComponentProportion(2), "b", 'DisplayName','Cluster 1','LineWidth',2) % cluster 1
    else
%         c1 = histogram(X(cluster_id==1),edges, 'Normalization', 'pdf');
%         set(c1, 'FaceColor', 'r')
%         c2 = histogram(X(cluster_id==2),edges, 'Normalization', 'pdf');
%         set(c2, 'FaceColor', 'b')
        plot(v, pdf('Normal',v,gm.mu(1),sqrt(gm.Sigma(1)))*gm.ComponentProportion(1), "b", 'DisplayName','Cluster 2','LineWidth',2) % cluster 2
        plot(v, pdf('Normal',v,gm.mu(2),sqrt(gm.Sigma(2)))*gm.ComponentProportion(2), "r", 'DisplayName','Cluster 1','LineWidth',2) % cluster 1
    end
    xlabel("Group velocity (km/s)"); ylabel("PDF")
    title("Density histogram of data and GMM model fit")
    
    % for k=1:ncluster
%     plot(v, pdf('Normal',v,gm.mu(k),sqrt(gm.Sigma(k)))*gm.ComponentProportion(k))
% end
    

end

end

