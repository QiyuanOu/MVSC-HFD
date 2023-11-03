function [UU,A,W,Z,iter,alpha] = algoMVDSC_v10(X,Y,d_vec,numanchor)
% difference from _v4: no use for lambda, also, no use for term 2(the frobenius norm of Z)
% difference from _v8: fix the error in updating Z( parameter alpha^2)
% m      : the number of anchor. the size of Z is m*n.
% lambda : the hyper-parameter of regularization term.
% X      : n*di

%% initialize
maxIter = 50 ; % the number of iterations

deep = length(d_vec);
d = d_vec(deep);
m = numanchor;
numclass = length(unique(Y));
numview = length(X);
numsample = size(Y,1);


%% Wi
W = cell(numview,deep);            % deep=3::::::>d3 * d2, d2 * d1, d1 * d XXXXXXXwrong, d_vec: di*d1,d1*d2,...,d_(deep-1)*d(anchor)
Wpro = cell(numview,1);            % product of all W_1, W_2, ..., W_deep

for i = 1:numview
    di = size(X{i},1);  %% di: dimension of the i-th view
    for j = 1:deep
        if j==1
            W{i,j} = zeros(di,d_vec(j));
        else
            W{i,j} = zeros(d_vec(j-1),d_vec(j));
        end
    end
end

%% A
% A = zeros(d,m);         % d  * m
A = eye(d);
%initialize centroids
% [initiallabel, center] = litekmeans(X{i}', numclass, 'MaxIter', 100,'Replicates',10);
% center = center'; % original center--m*di
% zerocenter = zeros(d,m);
% for i = 1:d
%     zerocenter[i,] = i;
% end
% opt.disp = 0;
% [A, ~] = eigs(center, d, 'la', opt);
% formerA = litekmeans(X, numanchor, 'MaxIter', 100,'Replicates',10);
A_vec = cell(deep,1);
for i = 1:numview
    for j = 1:deep
        count = deep - j + 1;
        if j==1
            A_vec{count} = A;
        else
            A_vec{count} = W{i,count+1} * A_vec{count+1};
        end
    end
end

%% Z
Z = zeros(m,numsample); % m  * n
Z(:,1:m) = eye(m);


alpha = ones(1,numview)/numview;
opt.disp = 0;

flag = 1;
iter = 0;
%%
while flag
    iter = iter + 1;
    
    %% optimize W_i
%     parfor iv=1:numview
    phi = cell(numview,deep);
    for iv=1:numview
        for im=1:deep
            if im == deep   % A_vec
                A_vec{im} = A;
            else
                A_vec{im} = W{iv,im+1}*A_vec{im+1};
            end
            if im == 1
                phi{iv,im} = eye(size(X{iv},1));
                C = X{iv}*(A_vec{im}*Z)';      
                [U,~,V] = svd(C,'econ');
                W{iv,im} = U*V';  %% W{iv,1}
            else
%                 for ip = 2:im  % may not use iteration
%                     phi{iv,im} = phi{iv,im-1} * W{iv,ip-1};
%                 end
                phi{iv,im} = phi{iv,im-1} * W{iv,im-1};
                C = phi{iv,im}'*X{iv}*(A_vec{im}*Z)';      
                [U,~,V] = svd(C,'econ');
                W{iv,im} = U*V';  %% W{iv,im}
            end
        end
        Wpro{iv}= phi{iv,deep} * W{iv,deep};  %% product of all W_1, W_2, ..., W_deep
    end
    
    %% optimize A
    sumAlpha = 0;
    part1 = 0;
    for ia = 1:numview
        al2 = alpha(ia)^2;
        sumAlpha = sumAlpha + al2;
%         part1 = part1 + al2 * W{ia}' * X{ia} * Z';
        % new code
        part1 = part1 + al2 * Wpro{ia}' * X{ia} * Z';
    end
    [Unew,~,Vnew] = svd(part1,'econ');
%     A = (part1/sumAlpha) * inv(Z*Z');
    A = Unew*Vnew';
    % Replace inv(A)*b with A\b
    % Replace b*inv(A) with b/A
    
    %% optimize Z
    % % QP
    % Sbar=[];
    % H = 2*sumAlpha*A'*A+2*lambda*eye(m);
    H = 2*sumAlpha*eye(m);
    H = (H+H')/2;
    % [r,q] = chol(H);

    options = optimset( 'Algorithm','interior-point-convex','Display','off'); % Algorithm 默认为 interior-point-convex
    parfor ji=1:numsample
        ff=0;
        for j=1:numview
%           C = W{j} * A;
        % new code
            C = Wpro{j} * A;
            ff = ff - 2*(alpha(j)^2)*X{j}(:,ji)'*C;
%         ff = ff - 2*X{j}(:,ji)'*C;
        end
        Z(:,ji) = quadprog(H,ff',[],[],ones(1,m),1,zeros(m,1),ones(m,1),[],options);
    end

    %% optimize alpha
    M = zeros(numview,1);
    for iv = 1:numview
%         M(iv) = norm( X{iv} - W{iv} * A * Z,'fro');
        % new code
        M(iv) = norm( X{iv} - Wpro{iv} * A * Z,'fro');
    end
    Mfra = M.^-1;
    Q = 1/sum(Mfra);
    alpha = Q*Mfra;

    %% calculate obj
%     term1 = 0;
%     for iv = 1:numview
% %         term1 = term1 + alpha(iv)^2 * norm(X{iv} - W{iv} * A * Z,'fro')^2;
%         % new code
%         term1 = term1 + alpha(iv)^2 * norm(X{iv} - Wpro{iv} * A * Z,'fro')^2;
%     end
% %     term2 = lambda * norm(Z,'fro')^2;
%     obj(iter) = term1;
    
    %% Stop criteria
%     if (iter>9) && (abs((obj(iter-1)-obj(iter))/(obj(iter-1)))<1e-3 || iter>maxIter || obj(iter) < 1e-10)
%         [UU,~,~]=svd(Z','econ');
%         flag = 0;
%     end
    if iter>9
        [UU,~,~]=svd(Z','econ');
        flag = 0;
    end
end
         
         
    
