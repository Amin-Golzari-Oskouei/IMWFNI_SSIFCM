function j_fun = object_fun(N,d,k,Cluster_elem,landa,M,fuzzy_degree,W,z,beta_z,p,X,f,b,alpha2, alpha1, Neig, NR, Cluster_elem_star, pi, sample_weight)
j_fun3 = 0;
mf = (Cluster_elem.^fuzzy_degree) + (Cluster_elem_star.^fuzzy_degree);

for j=1:k
    distance(j,:,:) = (1-exp((-1.*repmat(landa,N,1)).*((X-repmat(M(j,:),N,1)).^2)));
    WBETA = transpose(z(j,:).^beta_z);
    WBETA(WBETA==inf)=0;
    dNK(:,j) = (reshape(distance(j,:,:),[N,d]) * WBETA * W(1,j)^p) .* sample_weight;
    
    %term 2
    cc = (1-mf(j,:)).^fuzzy_degree;
    j_fun3=j_fun3+sum((mf(j,:).^fuzzy_degree)'.* sum(cc(Neig),2));
end
j_fun1 = sum(sum(dNK .* transpose(mf)));
j_fun2 = sum(sum(dNK .* transpose((Cluster_elem-(b.*f)').^fuzzy_degree)));

value = pi' .* repmat((exp(1 - ((1/N) * sum(pi', 1)))), N, 1);
j_fun4 = (1/N) * sum(sum(value));


j_fun = j_fun1 + (alpha2 * j_fun2) + ((alpha1/NR)*j_fun3) + j_fun4 ;
end

