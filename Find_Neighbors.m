function result = Find_Neighbors(NR, X, landa)

    function D2 = naneucdist(XI,XJ)
        %NANEUCDIST Euclidean distance ignoring coordinates with NaNs
        sqdx = (XI-XJ).^2;
        D2 = sum(1- exp(-1.*sqdx.*landa),2);
    end

D = pdist(X,@naneucdist);
D = squareform(D);
[~,result] = sort(D, 2);
result = result(:,2:NR+1);
end