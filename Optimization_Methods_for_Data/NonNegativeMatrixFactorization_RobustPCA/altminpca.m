% function [L,S] = altminpca(L,X,lambda1,beta,maxits,expand)
% S = X-L;
% % [m,n] = size(X);
% for i=1:maxits
%     % minimize w.r.t. L using forward-backward
%     L = proxnuc(X-S,1/beta);
%     % minimize w.r.t. S using forward-backward
%     S = prox(X-L,lambda1/beta);
%     beta = beta*expand;
%     get_robpcaobj(L,S,X,lambda1,0)
% end
% 
% end


function [L,S, info] = altminpca(L,X,lambda1,beta,maxits,expand,tol)
S = X-L;
% [m,n] = size(X);
ne = numel(X);
info = zeros(maxits,4);
for i=1:maxits
    objold = Inf;
    obj = @(L,S) get_robpcaobj(L,S,X,lambda1,beta);
    objnew = obj(L,S);
    while objold - objnew > tol
        objold = objnew;
        % minimize w.r.t. L using forward-backward proximal splitting
        L = proxnuc(X-S,1/beta);
        % minimize w.r.t. S using forward-backward proximal splitting
        S = prox(X-L,lambda1/beta);
        objnew = obj(L,S);
    end
    info(i,:) = [get_robpcaobj(L,S,X,lambda1,beta), rank(L), nnz(S)/ne, ...
        norm(L+S-X,'fro')];
    beta = beta*expand;
end

end

%     while abs(objnew - objold) > tol
%         objL = objnew
%         objold = objnew;
%         L = proxnuc(X-S,1/beta);
%         objnew = obj(L);
%         beta = 1.5*beta;
%     end
%     % minimize w.r.t. S using forward-backward
%     obj = @(S) get_robpcaobj(L,S,X,lambda1,beta);
%     objold = Inf;
%     objnew = obj(S);
%     beta = beta0;
%     while abs(objnew - objold) > tol
%         objS = obj
%         objold = objnew;
%         S = prox(X-L,lambda1/beta);
%         objnew = obj(S);
%         beta = 1.5*beta;
%     end
