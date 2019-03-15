function M = proj2unitball(M)
cnorms = colnorms(M);
infeasiblenorms = cnorms > 1;
M(:,infeasiblenorms) = vecnorm(M(:,infeasiblenorms)); % vecnorms normalizes columns of a matrix
end

