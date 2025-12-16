function [X_res, y_res] = meb_plot(X, y)
rng(2);
minority = X(y==1, :)'; majority = X(y==0, :)';
d = size(X,2); numToGen = sum(y==0)-sum(y==1);

[~, ~, ~, ~, ~, ~, S_pairs] = LS_Test(majority, minority);
A_idx = S_pairs(1,:); B_idx = S_pairs(2,:);
A_sup = majority(:,A_idx); B_sup = minority(:,B_idx);
Nodes = [A_sup, B_sup];
n_nodes = size(Nodes,2);

coords = [A_sup; B_sup];
D = pdist2(coords', coords');
G = graph(D); T = minspantree(G);
Adj = adjacency(T);
[row, col] = find(triu(Adj)); 


internal_idx = setdiff(1:size(minority,2), unique(B_idx));
B_internal = minority(:, internal_idx);
[idx_sup, dists] = knnsearch(A_sup', B_internal');
sigma = mean(dists);
rbf_w = exp(- (dists.^2)/(2*sigma^2));
rbf_w = rbf_w / sum(rbf_w);

avg_min_dist = mean(pdist(minority'));
density = 1 ./ (mean(pdist2(B_sup', minority'),2) + eps);
k_base = 0.5 + 0.5*tanh(mean(density));

Nodes_new = Nodes;
velocity = zeros(d,size(Nodes_new,2));
rest_len = D(sub2ind(size(D), row, col));

lr = 0.1; momentum = 0.9;

max_iter = 100;
for iter=1:max_iter
    F = zeros(size(Nodes_new));
    node_pairs = find(triu(Adj));
    [rows, cols] = find(triu(Adj));
    for k=1:numel(rows)
        i=rows(k); j=cols(k);
        diff = Nodes_new(:,j)-Nodes_new(:,i);
        dist0 = rest_len(k);
        dist_now=norm(diff);
        if dist_now>0
            Fspr = k_base*(dist_now-dist0)*diff/dist_now;
            F(:,i)=F(:,i)+Fspr;
            F(:,j)=F(:,j)-Fspr;
        end
    end
    for i=1:size(B_internal,2)
        q = B_internal(:,i);
        s = idx_sup(i);
        diff = q - Nodes_new(:,s);
        F(:,s)= F(:,s)+ rbf_w(i)*k_base*diff;
    end
    vel_prev = velocity;
    velocity = momentum*velocity + lr*(F - 0.1*velocity);
    N_nodes = size(Nodes_new,2);
    for j=1:N_nodes
        if min(vecnorm(majority - Nodes_new(:,j))) < min(vecnorm(minority - Nodes_new(:,j)))
            proj = pca_project(Nodes_new(:,j), minority);
            Nodes_new(:,j)=proj;
            velocity(:,j)=0;
        end
    end
    Nodes_new = Nodes_new + velocity;
    if max(vecnorm(velocity))<1e-4, break; end
end

A_deform = Nodes_new(:,1:numel(A_idx));
B_deform = Nodes_new(:,numel(A_idx)+1:end);
newSamps = zeros(d, numToGen);

edge_lengths = vecnorm(A_deform(:,rows) - B_deform(:,rows));
bpi = std(edge_lengths) / (mean(edge_lengths) + eps);
line_ratio = 1 / (1 + exp(-5 * (bpi - 0.3))); 
n_line = round(numToGen * line_ratio);
n_tri = numToGen - n_line;


for i = 1:n_line
    k = randi(numel(A_idx));
    a = A_deform(:, k);
    b = B_deform(:, k);
    
    % 插值位置，0~1区间均匀覆盖整线段
    t = rand();
    base = b + t * (a - b);
    
    % 局部密度影响扰动尺度（密度越大扰动越小）
    density_a = density(k);
    density_b = density(k);
    density_edge = (density_a + density_b) / 2;
    perturb_scale = 0.1 / (1 + 5 * density_edge);
    
    % 计算垂直扰动方向并归一化
    perp_dir = randn(d, 1);
    perp_dir = perp_dir - (perp_dir' * (a - b)) / (norm(a - b)^2) * (a - b);
    perp_dir = perp_dir / norm(perp_dir);
    
    % 垂直扰动幅度（正态分布），增强多样性
    perturb = norm(a - b) * perturb_scale * randn();
    
    newSamps(:, i) = base + perturb * perp_dir;
end


for i=1:n_tri
    k = randi(numel(A_idx));
    a = A_deform(:,k); b = B_deform(:,k);
    % 按距离采权选择 c 点（离 a,b 较近的内部点概率更高）
    dists_to_ab = vecnorm(B_internal - (a+b)/2, 2, 1);
    weights_c = exp(-dists_to_ab / (mean(dists_to_ab) + eps));
    weights_c = weights_c / sum(weights_c);
    c_idx = randsample(1:size(B_internal,2),1,true,weights_c);
    c = B_internal(:,c_idx);

    alpha_dir = [1,1,1];
    bary = gamrnd(alpha_dir,1);
    bary = bary / sum(bary);

    newSamps(:,n_line+i) = bary(1)*a + bary(2)*b + bary(3)*c;
end


X_res = [X; newSamps'];
y_res = [y; ones(numToGen,1)];
end

function proj = pca_project(x, pts)
C=cov(pts');
[U,~,~]=svd(C);
proj = mean(pts,2) + U(:,1)*(U(:,1)'*(x-mean(pts,2)));
end

