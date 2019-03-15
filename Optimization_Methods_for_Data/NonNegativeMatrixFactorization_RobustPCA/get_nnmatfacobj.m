function obj = get_nnmatfacobj(W,H,X)

obj = 0.5*norm(W*H'-X,'fro')^2;  % assuming projection already happened

end