function [gradW,gradH] = get_grad3(X, W, H)
 gradW = (W*H'-X)*H; 
 gradH = (W*H'-X)'*W;
 % first element is gradient with respect to W, second is wrt H
end