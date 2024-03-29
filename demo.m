%This demo shows how to call the weighted fuzzy C-means algorithm described in the paper:
%"Towards Robust Clustering: Integrating Multiple Weighting Factors with Neighborhood Information in Semi-Supervised Intuitionistic Fuzzy C-Means",
%For the demonstration, the iris dataset of the above paper is used.

clc
clear all
close all

%Load the dataset. The last column of dataset is true labels.
X=load('iris.mat');
X=X.iris;

%delete last column (true labels) in clustering process
class=X(:,end);
X(:,end)=[];

%Normalize data between 0 and 1 (optinal)
[N,d]=size(X);
X=(X(:,:)-min(X(:)))./(max(X(:)-min(X(:))));

%---------------------
%Algorithm parameters.
%---------------------
k=size(unique(class),1);  %number of clusters.
q=2;                      %the value for the feature weight updates.
p_init=0;                 %initial p.
p_max=0.5;                %maximum p.
p_step=0.01;              %p step.
t_max=100;                %maximum number of iterations.
beta_memory=0.3;          %amount of memory for the cluster weight updates.
beta = 1;
Restarts=10;              %number of algorithm restarts.
fuzzy_degree=2;           %fuzzy membership degree
I=1;                      %The value of this parameter is in the range of (0 and 1]

landa=I./var(X);
landa(landa==inf)=1;

f = double(class == 1:max(class));% convert class to onehot encoding
f(:,sum(f)==0)=[];
labeled_rate = 20;               % rate of labeled data (0-100)


alpha2 = 0.5;
alpha1 = 1;

NR = 5;  % size of window for finding neighbors
Neig = Find_Neighbors(NR, X, landa);

%---------------------
%Cluster the instances using the propsed procedure.
%---------------------------------------------------------
for repeat=1:Restarts
    fprintf('========================================================\n')
    fprintf('proposed clustering algorithm: Restart %d\n',repeat);
    
    % label indicator vector
    rand('state',repeat)
    b = zeros(N,1);
    tmp1=randperm(N);
    b(tmp1(1:N*labeled_rate/100))=1;
    
    %initialize with labeled data.
    if labeled_rate==0
        %Randomly initialize the cluster centers.
        tmp2=randperm(N);
        M=X(tmp2(1:k),:);
    else
        M = ((b.*f)'*(X))./repmat(sum(b.*f)',1,d);
        if sum(isnan(M))>=1
            tmp2=randperm(N);
            tem3= X(tmp2(1:k),:);
            M(isnan(M))=tem3(isnan(M));
        end
    end
    
    %Sample Weighting
    for j=1:k
        distance(j,:,:) = (exp((-1.*repmat(landa,N,1)).*((X-repmat(M(j,:),N,1)).^2)));
        dNK(:,j) = reshape(distance(j,:,:),[N,d]) * transpose(ones(1,d)/d)   ;
    end
    
    tmp = f.*b;
    tmp(tmp==1) = -1;
    tmp(tmp==0) = 1;
    tmp(tmp==-1) = 0;
    
    sample_weight = sum(dNK.*tmp,2) / k;
    
    %Execute proposed clustering algorithm.
    %Get the cluster assignments, the cluster centers and the cluster weight and feature weight.
    [Cluster_elem,M,W,Z]=IMWFNI_SSIFCM(X,M,k,p_init,p_max,p_step,t_max,beta_memory,N,fuzzy_degree,d,q,landa,f,b, alpha2, alpha1,Neig, NR, beta, sample_weight);
    
    [~,semisupervised_Cluster]=max(Cluster_elem,[],1); %Hard clusters. Select the largest value for each sample among the clusters, and assign that sample to that cluster.
    
    semisupervised_Cluster(tmp1(1:N*labeled_rate/100))=class(tmp1(1:N*labeled_rate/100));
    % Evaluation metrics
    % Accurcy
    EVAL = Evaluate(class,semisupervised_Cluster');
    Accurcy_semisupervised(repeat)=EVAL(1);
    
    fprintf('End of Restart %d\n',repeat);
    fprintf('========================================================\n\n')
end

fprintf('Average semisupervised accurcy over %d restarts: %f.\n',Restarts,mean(Accurcy_semisupervised));