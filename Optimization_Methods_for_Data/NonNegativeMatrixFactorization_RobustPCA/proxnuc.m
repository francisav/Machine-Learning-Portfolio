function pr = proxnuc(L,tau)

[U,S,V] = svd(L,'econ');
s = diag(S);
s = prox(s,tau);
pr = U*diag(s)*V';

end
