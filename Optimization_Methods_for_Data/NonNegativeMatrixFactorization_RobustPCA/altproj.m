function [W,Xbar,objs] = altproj(X, W, H, maxits)
objs = zeros(maxits,1);
gradfunW = @(W,H) get_grad3_W(X, W, H);
gradfunH = @(W,H) get_grad3_H(X, W, H);
for i=1:maxits
    % proj grad for W
    alphaW = 1/norm(H'*H,2);
    gradW = gradfunW(W, H);
    W = W - alphaW*gradW;
    W(W<0) = 0;
    W = proj2unitball(W);
    % proj grad for H
    alphaH = 1/norm(W'*W,2);
    gradH = gradfunH(W, H);
    H = H - alphaH*gradH;
    H(H<0) = 0;
    Xbar = W*H';
    objs(i) = get_nnmatfacobj(W,H,X);
end
end
