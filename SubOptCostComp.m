function [RMat, A3, B3, tanhArg ] = SubOptCostComp(N1, N2, NTrials, R1Mat, R2Mat, R3Mat, a3, b3, type, RNNParams, V)
% Function that finds the parameters for the sub-optimal models

%rng(12);
% load VR; load VB;

M = NTrials;

if strcmp(type, 'Q')
    % Quadratic case 
    N       = N1*N2;
    % Generating RMat for the quadratic case
    RMat    = zeros(N1*N2,NTrials);
    
    for k = 1:NTrials
        r1 = transpose(R1Mat(k,:));
        r2 = transpose(R2Mat(k,:));
        r12 = r1*transpose(r2);
        RMat(:,k) = r12(:);
    end
    tanhArg = [];
    
elseif strcmp(type, 'L')
    % Linear Case
    N   = N1 + N2;
    RMat = [R1Mat'; R2Mat'];
    tanhArg = [];
    
else
    % RNN case 
    gN      = RNNParams.gN;
    gB      = RNNParams.gB;
    gC      = RNNParams.gC;
    WMat    = RNNParams.WMat;
    Nratio  = RNNParams.Nratio;
    N       = round(N1*N2*Nratio);
    W       = gN*WMat(1:N,1:N1+N2);
    WBias   = RNNParams.WBias;
    Wb      = gB*WBias(1:N); % If bias is not needed, put gB = 0
    Wb      = repmat(Wb,1,NTrials);
    tanhArg = W*transpose([R1Mat, R2Mat]) + Wb + gC;
    RMat    = tanh(tanhArg);

    ThisReq = 0;
    if (ThisReq)
        meanRMat = mean(RMat,2);
        idx      = find(meanRMat < -0.95 | meanRMat > 0.5);
        RMat     = RMat(idx,:);
    end
    
end

% Least squares solution: Ax = y

% First construct the A matrix
A = [transpose(RMat), zeros(M,size(RMat,1)); zeros(M,size(RMat,1)), transpose(RMat)];
% Then the y vector
y = [R3Mat*a3; R3Mat*b3];

% xhat = pinv(A)*y;
xhat = A\y;
A3 = xhat(1:size(RMat,1));
B3 = xhat(1+size(RMat,1):end);
