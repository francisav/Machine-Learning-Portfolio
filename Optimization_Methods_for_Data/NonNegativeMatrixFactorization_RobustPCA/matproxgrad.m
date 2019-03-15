function [W,Xbar,objs] = matproxgrad(X, W0, H0, alpha, maxits)
gradfun = @(W,H) get_grad3(X, W, H);
W = W0;
H = H0;
objs = zeros(maxits,1);
for i=1:maxits
    [gradW, gradH] = gradfun(W,H);
    W = W - alpha*gradW;
    H = H - alpha*gradH;
    W(W<0) = 0;
    H(H<0) = 0;
    W = proj2unitball(W);
    objs(i) = get_nnmatfacobj(W,H,X);
end
Xbar = W*H';
end