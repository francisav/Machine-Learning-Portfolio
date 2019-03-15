function obj = get_robpcaobj(L,S,X,lambda1, beta)

obj = trace(sqrt(L'*L)) + lambda1*sum(sum(abs(S))) + norm(L+S-X,'fro')^2;

end
