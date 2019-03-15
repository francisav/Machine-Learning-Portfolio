% Be sure to run the initial guess section before running the sections
% containing the minimization algorithms, as the algorithms all use the
% same random initialization.  The hyperparameters of each algorithm come pre-set 
% to the values displayed in tables 1 and 2 in the paper. I have indicated, 
% as comments, what the errors (and other relevant algorithm output) should be when using the pre-set hyperparameters.
% Changes necessary to reproduce plots from the paper are indicated wherever necessary.
%% load data
load('Swimmer.mat')
dims = size(Swimmer);
X = reshape(Swimmer, dims(1)*dims(2), dims(3));
X(X==1) = 0; 
X(X==39) = 1;
r = 17;
[m,n] = size(X);
%% Initial Guesses
rng(10)
W0 = rand(m,r);
H0 = rand(n,r);

%% full projected gradient
alpha = 1e-3;
maxits = 10000;
% The error should be 1.0887.  This section takes some time to run...about 5-10 minutes 
% on my computer, which is a 2013 macbook pro with 8GB RAM.
[W,Xbar,objs] = matproxgrad(X, W0, H0, alpha, maxits);
Wreshape = reshape(W,[32,32,r]);
fprintf('Done with non-neg matrix factorization via full projected gradient descent.  The frobenius error is %.*f .\n', 4, norm(Xbar-X,'fro'));


%% plot
figure(1)
for i=1:r
    subplot(3,6,i)    
    imshow(Wreshape(:,:,i))
    title(i)
end

%% alternating minimization
maxits = 200;
% Error should be 0.0030.  Runs fast, <30 seconds on my machine.
[W_alt,Xbar_alt,objs_alt] = altmin(X, W0, H0, maxits, 0.001, 0.0001);
Wreshape_alt = reshape(W_alt,[32,32,r]);
fprintf('Done with non-neg matrix factorization via alternating minimization.  The frobenius error is %.*f .\n', 4, norm(Xbar_alt-X,'fro'));
%% plot
figure(2)
for i=1:r
    subplot(3,6,i)    
    imshow(Wreshape_alt(:,:,i))
    title(i)
end
%% alternating projected gradient descent
% Error should be 0.0030, runs fast...<30 secs on my machine
maxits = 800;
[W_altproj,Xbar_altproj,objs_altproj] = altproj(X, W0, H0, maxits);
Wreshape_altproj = reshape(W_altproj,[32,32,r]);
fprintf('Done with non-neg matrix factorization via alternating projected gradient descent.  The frobenius error is %.*f .\n', 4, norm(Xbar_altproj-X,'fro'));
%% plot
figure(3)
for i=1:r
    subplot(3,6,i)    
    imshow(Wreshape_altproj(:,:,i))
    title(i)
end
%% compare per-iteration performance
% To reproduce plots in paper, rerun the 3 algorithms above with maxits=1000.
% Then run this section.
num_its = 1000;
figure(11)
clf
hold on
a1 = plot(1:1000,objs(1:1000),'-r'); M1 = 'Full Projected Gradient';
a2 = plot(1:1000,objs_alt(1:1000),'-k'); M2 = 'Alternating Minimization';
a3 = plot(1:1000,objs_altproj(1:1000),'-c'); M3 = 'Alternating Projected Gradient';
legend([a1;a2;a3],M1, M2, M3);
xlabel('Iteration'); ylabel('Objective Value');
%% Robust PCA
% read in data
load('escalator_data.mat')
[m,n] = size(X);
mx = max(max(X));
mn = min(min(X));
X = (X-mn)/(mx-mn);
dubX = double(X);

%% alternating min
[L_alt,S_alt, info_alt] = altminpca(dubX,dubX,10e-4,10e-3,10,3,0.1);
err = norm(L_alt+S_alt - dubX, 'fro');
rnk = rank(L_alt);
nz = nnz(S_alt)/(m*n);
% Should get err=0.0065, rnk=2, nz=0.2765
fprintf('Done with robust pca via alternating minimization:\n');
fprintf('The frobenius error is %.*f .\n', 4, err);
fprintf('The rank of L is %.*f .\n', 0, rnk);
fprintf('S is %.*f%% nonzero.\n', 2, nz*100);
%% full proximal
[L,S, info] = fullpca(dubX,dubX,5e-3,1e-3,15,2,1);
err = norm(L+S - dubX, 'fro');
rnk = rank(L);
nz = nnz(S)/(m*n);
% should get err=0.4756, rnk=61, nz=0.2503
fprintf('Done with robust pca via full proximal gradient descent:\n');
fprintf('The frobenius error is %.*f .\n', 4, err);
fprintf('The rank of L is %.*f .\n', 0, rnk);
fprintf('S is %.*f%% nonzero.\n', 2, nz*100);
%% compare per-iteration performance
% To reproduce plots in paper, rerurn the two above sections using 20
% iterations.   then run this section.
num = size(info,1);
num_alt = size(info_alt,1);

figure(12)
clf
hold on
a1 = plot(1:num,info(:,1),'-r'); M1 = 'Full Proximal Gradient';
a2 = plot(1:num_alt,info_alt(:,1),'-k'); M2 = 'Alternating Proximal Minimization';
legend([a1;a2],M1, M2);
xlabel('Iteration'); ylabel('Objective Value');

figure(13)
clf
hold on
a1 = plot(1:num,info(:,2),'-r'); M1 = 'Full Proximal Gradient';
a2 = plot(1:num_alt,info_alt(:,2),'-k'); M2 = 'Alternating Proximal Minimization';
legend([a1;a2],M1, M2);
xlabel('Iteration'); ylabel('rank(L)');

figure(14)
clf
hold on
a1 = plot(1:num,info(:,3),'-r'); M1 = 'Full Proximal Gradient';
a2 = plot(1:num_alt,info_alt(:,3),'-k'); M2 = 'Alternating Proximal Minimization';
legend([a1;a2],M1, M2);
xlabel('Iteration'); ylabel('%nnz(S)');

figure(15)
clf
hold on
a1 = plot(1:num,info(:,4),'-r'); M1 = 'Full Proximal Gradient';
a2 = plot(1:num_alt,info_alt(:,4),'-k'); M2 = 'Alternating Proximal Minimization';
legend([a1;a2],M1, M2);
xlabel('Iteration'); ylabel('Frobenius Error');
%% reshape
Lreshape_alt = reshape(L_alt,130,160,200);
Sreshape_alt = reshape(S_alt,130,160,200);
Lreshape = reshape(L,130,160,200);
Sreshape = reshape(S,130,160,200);
%% show selected images from L and S
% increase iterating range of imnum to display more pictures
for imnum=28
    figure(imnum)
    subplot(2,2,1)
    imshow(Lreshape(:,:,imnum))
    title('L')
    subplot(2,2,2)
    imshow(Sreshape(:,:,imnum))
    title('S')
    subplot(2,2,3)
    imshow(Lreshape_alt(:,:,imnum))
    title('L')
    subplot(2,2,4)
    imshow(Sreshape_alt(:,:,imnum))
    title('S')
    imnum
end

%% implay
% play image stream from S.  L not played because it is essentially static,
% as it captures the underlying structure.  S captures the noise.  In this
% case, the noise is the moving escalator steps and the people riding them.
implay(Sreshape_alt) 
%% d
implay(Sreshape)
