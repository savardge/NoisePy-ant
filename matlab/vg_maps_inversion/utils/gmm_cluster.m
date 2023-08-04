function gm = gmm_cluster(V_dat)

doplot = true;

% Input data
X = V_dat*1000;

% Fit a two-component Gaussian mixture model (GMM). 
% Because there are two components, suppose that any data point with 
% cluster membership posterior probabilities in the interval [0.4,0.6] 
% can be a member of both clusters.
if doplot
    cmap = colormap(lines);
    figure(1);clf;hold on;
    figure(10); clf;
end

bicprevious = 9e9;
for ncluster = 1:2
    
    gm = fitgmdist(X,ncluster, ...
        'SharedCovariance',false, ...
        'Options',statset('Display','off', 'MaxIter',5000), ...
        'RegularizationValue',0.0001);
    disp(['# components: ' num2str(ncluster) ', BIC = ' num2str(gm.BIC)])
    if gm.Converged
        if gm.BIC > bicprevious
            break
        end
        if ncluster == 2
            mincp = min(gm.ComponentProportion);
            maxcp = max(gm.ComponentProportion);
            disp(['Ratio of Component proportion: ' num2str(mincp/maxcp)])
        end

        if doplot
            figure(10); hold on
            subplot(1,2,1);hold on; 
            plot(ncluster, gm.AIC, 'ko');ylabel("AIC"); xlabel("# components")
            subplot(1,2,2);hold on; 
            plot(ncluster, gm.BIC, 'ko'); ylabel("BIC"); xlabel("# components")
    
            figure(1); clf;hold on;
            v = (1:0.01:4)';
            subplot(2,1,1); hold on
            histogram(X,100, 'Normalization', 'pdf')
            plot(v,pdf(gm,v), 'LineWidth',2, 'Color','k')
            xlim([1,4.5])
            subplot(2,1,2);hold on
            plot(v,pdf(gm,v), 'LineWidth',2, 'Color','k')   
            
            for k=1:ncluster
                plot(v, pdf('Normal',v,gm.mu(k),sqrt(gm.Sigma(k)))*gm.ComponentProportion(k), 'Color',cmap(k,:))
            end
            ylabel("PDF")
            xlabel("Group velocity (km/s)")
    %         pause
        end

        bicprevious = gm.BIC;

    else
        disp('Did not converge.')
    end
end

% ncluster = 2;
% gm = fitgmdist(X,ncluster, ...
%         'SharedCovariance',false, ...
%         'Options',statset('Display','off', 'MaxIter',5000), ...
%         'RegularizationValue',0.0001);
% 
% 
% figure(1); clf;hold on;
% v = (1:0.01:4)';
% subplot(2,1,1); hold on
% histogram(X,100, 'Normalization', 'pdf')
% plot(v,pdf(gm,v), 'LineWidth',2, 'Color','k')
% xlim([1,4.5])
% subplot(2,1,2);hold on
% plot(v,pdf(gm,v), 'LineWidth',2, 'Color','k')
% for k=1:ncluster
%     plot(v, pdf('Normal',v,gm.mu(k),sqrt(gm.Sigma(k)))*gm.ComponentProportion(k))
% end
% % plot(v, pdf('Normal',v,gm.mu(1),gm.Sigma(1)), 'r')
% % plot(v, pdf('Normal',v,gm.mu(2),gm.Sigma(2)), 'g')
% xlim([1,4.5])
% hold off




threshold = [0.4 0.6];

% Estimate component-member posterior probabilities for 
% all data points using the fitted GMM gm. 
% These represent cluster membership scores.
P = posterior(gm,X);

% For each cluster, rank the membership scores for all data points. 
n = size(X,1);
[~,order] = sort(P(:,1));

if doplot
    % For each cluster, plot each data points membership score with respect
    % to its ranking relative to all other data points.
    figure(3);clf;
    plot(1:n,P(order,1),'-', 'Color',cmap(1,:)); 
    hold on
    plot(1:n,P(order,2),'-', 'Color',cmap(2,:));
    legend({'Cluster 1', 'Cluster 2'}, 'Location', 'best')
    ylabel('Cluster Membership Score')
    xlabel('Point Ranking')
    title('GMM with Full Unshared Covariances')

    % Plot the data and assign clusters by maximum posterior probability. 
    % Identify points that could be in either cluster.
    idx = cluster(gm,X);
    idxBoth = find(P(:,1)>=threshold(1) & P(:,1)<=threshold(2)); 
    numInBoth = numel(idxBoth);
    
    Xsort = sort(X);
    idxsort = cluster(gm,Xsort);
    if sum(abs(diff(idxsort))) > 1
        disp(['THERE ARE JUMPS'])
    end
    
    figure(4);clf;
    subplot(4,1,1);
    h = histogram(X,100);
    edges = h.BinEdges;
    xlim([0.5 4]); 
    ylimits = get(gca, 'YLim');
    xlabel("Group velocity (km/s)"); ylabel("PDF")
    title("Density histogram of data and GMM model fit")
    
    subplot(4,1,2)
    idx1 = find(P(:,1)>0.6);
    histogram(X(idx1),edges);
    xlim([0.5 4]); ylim(ylimits)
    xlabel("Group velocity (km/s)"); ylabel("PDF")
    title("Data points in cluster 1 (P > 0.6)")
    
    subplot(4,1,3)
    idx2 = find(P(:,2)>0.6);
    histogram(X(idx2),edges);
    xlim([0.5 4]); ylim(ylimits)
    xlabel("Group velocity (km/s)"); ylabel("PDF")
    title("Data points in cluster 2 (P > 0.6)")
    
    subplot(4,1,4)
    histogram(X(idxBoth),edges);
    xlim([0.5 4]); ylim(ylimits)
    xlabel("Group velocity (km/s)"); ylabel("PDF")
    title("Data points shared by both clusters (probability 0.4-0.6)")
    % gscatter(X(:,1),X(:,2),idx,'rb','+o',5)
    % hold on
    % plot(X(idxBoth,1),X(idxBoth,2),'ko','MarkerSize',10)
    % legend({'Cluster 1','Cluster 2','Both Clusters'},'Location','SouthEast')
    % title('Scatter Plot - GMM with Full Unshared Covariances')
    % hold off
end

