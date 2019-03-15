function [W,Xbar,objs] = altmin(X, W, H, maxits, gradtol, tol)
objs = zeros(maxits,1);
for i=1:maxits
    gradfunW = @(W) get_grad3_W(X, W, H);
    gradW = gradfunW(W);
    gradWprev = Inf;
    alphaW = 1/norm(H'*H,2);
    while norm(gradW-gradWprev,'fro')>gradtol && norm(X-W*H','fro')>tol && norm(gradW,'fro')>gradtol
        gradWprev = gradW;
        W = W - alphaW*gradW;
        W(W<0) = 0;
        W = proj2unitball(W);
        gradW = gradfunW(W);
    end
    gradfunH = @(H) get_grad3_H(X, W, H);
    gradH = gradfunH(H);
    gradHprev = Inf;
    alphaH = 1/norm(W'*W,2);
    while norm(gradH-gradHprev,'fro')>gradtol && norm(X-W*H','fro')>tol && norm(gradH,'fro')>gradtol
        gradHprev= gradH;
        H = H - alphaH*gradH;
        H(H<0) = 0;
        gradH = gradfunH(H);
    end
    objs(i) = get_nnmatfacobj(W,H,X);
    Xbar = W*H';
end

end