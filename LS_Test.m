function [LS, LS_Degree, time, final_x, final_p,z,SupportPairs] = LS_Test(A, B)

tic

% Get size information
n = size(B, 1);         % Dimension
sizeA = size(A, 2);     % Number of points in A
sizeB = size(B, 2);     % Number of points in B
m = sizeA + sizeB;

global epsTol epsTol2 epsTol3 count
epsTol = 1e-4;  
epsTol2 = 1e-4; 
epsTol3 = 1e-6;
count = 0;

% Initial setup
Diff = A(:, 1) - B(:, 1);
p1 = Diff / norm(Diff);

Diff = A(:, 1) - B(:, 2);
p2 = Diff / norm(Diff);

% Initialization
[Q, R] = qr([p1, p2]);      
x = (p1 + p2) / 2;          
z = 0.5 * norm(p1 - p2);    

iter = 0;
isOpt = 0;

normsOfPsq = ones(1, sizeA * sizeB);
p = p1;
radii = [];

% Store support pair indices
SupportA_idx = [];
SupportB_idx = [];

while isOpt == 0   
    iter = iter + 1;
    distances = zeros(sizeA, sizeB); 
    x_reshaped = reshape(x, [length(x), 1, 1]);

    for blockA = 1:ceil(sizeA / sizeA)
        for blockB = 1:ceil(sizeB / sizeB)
            rangeA = (blockA-1)*sizeA + 1:min(blockA*sizeA, sizeA);
            rangeB = (blockB-1)*sizeB + 1:min(blockB*sizeB, sizeB);

            A_block = A(:, rangeA);
            B_block = B(:, rangeB);

            Diff = A_block - permute(B_block, [1, 3, 2]);
            norm_Diff = sqrt(sum(Diff.^2, 1));
            p = bsxfun(@rdivide, Diff, norm_Diff);
            distances(rangeA, rangeB) = -2 * squeeze(sum(x_reshaped .* p, 1));
        end
    end

    distances = reshape(distances', 1, []);
    distances = distances + normsOfPsq;

    [maxdist, ip] = max(distances);
    if sqrt(maxdist + x' * x) < z + epsTol
        break;
    end

    iA = ceil(ip / sizeB);
    iB = ip - (iA - 1) * sizeB;

    Diff = A(:, iA) - B(:, iB);
    norm_Diff = norm(Diff);
    p = Diff / norm_Diff;

    % Save support pair indices
    SupportA_idx = [SupportA_idx, iA];
    SupportB_idx = [SupportB_idx, iB];

    % Update support set S
    [Q, R] = LARGEupdateS(Q, R, p, x);

    % Solve for new x
    flag = 1;
    while flag ~= 0
        [x, flag] = LARGElineSearch(x, Q, R, p); 
        if flag == 0
            z = norm(x - p);
            break;
        else
            [Q, R] = qrdelete(Q, R, flag, 'col'); 
        end
    end
    [Q, R] = qrinsert(Q, R, 1, p, 'col');  

    radii = [radii, z];
    if iter >= 6
        recent_increases = diff(radii(end-4:end));
        if all(abs(recent_increases) < epsTol2)
            break;
        end
    end
end

if z > 1 - epsTol3
    LS = 0;
else
    LS = 1;
end

LS_Degree = 1 - z;
time = toc;

% Final return values
final_x = x;
final_p = p;
SupportPairs = [SupportA_idx; SupportB_idx];

end
