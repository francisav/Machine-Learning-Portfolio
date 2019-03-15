function z = prox(z,tau)
% if z > tau
%     z = z-tau;
% elseif z < -tau
%     z = z + tau;
% else z = 0;
% end

cond1 = z > tau;
cond2 = z < -tau;
cond3 = z>=-tau & z<=tau;
z(cond1) = z(cond1) - tau;
z(cond2) = z(cond2) + tau;
z(cond3) = 0;

end
