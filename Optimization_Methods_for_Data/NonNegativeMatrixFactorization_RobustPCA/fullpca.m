function [L,S,info] = fullpca(L,X,lambda1,beta,maxits,expand, tol)
S = X-L;
ne = numel(X);
info = zeros(maxits,4);
for i=1:maxits
    objold = Inf;
    obj = @(L,S) get_robpcaobj(L,S,X,lambda1,beta);
    objnew = obj(L,S);
    while objold - objnew > tol
        objold = objnew;
        Ltemp = 0.5*(L-S+X);
        S = 0.5*(S-L+X);
        L = proxnuc(Ltemp,1/(2*beta));
        S = prox(S,lambda1/(2*beta));
        objnew = obj(L,S);
    end
    info(i,:) = [get_robpcaobj(L,S,X,lambda1,beta), rank(L), nnz(S)/ne, ...
        norm(L+S-X,'fro')];
    beta = beta*expand;
end


end

