function [RMat, A3, B3 ] = SubOptCostComp(N1, N2, NTrials, R1Mat, R2Mat, AVec, BVec, type, RNNParams)
% Function that finds the parameters for the sub-optimal models

M = NTrials;

if strcmp(type, 'Q')
    % Quadratic case 
    N   = N1*N2;
    % Generating RMat for the quadratic case
    RMat = zeros(N1*N2,NTrials);

    for k = 1:NTrials
        r1 = transpose(R1Mat(k,:));
        r2 = transpose(R2Mat(k,:));
        r12 = r1*transpose(r2);
        RMat(:,k) = r12(:);
    end
    
elseif strcmp(type, 'L')
    % Linear Case
    N   = N1 + N2;
    RMat = [R1Mat'; R2Mat'];
    
else
    % RNN case 
    gN      = RNNParams.gN;
    gB      = RNNParams.gB;
    WMat    = RNNParams.WMat;
    Nratio  = RNNParams.Nratio;
    N       = round(N1*N2*Nratio);
    W       = gN*WMat(1:N,1:N1+N2);
    Wbias   = gB*WMat(end-N+1:end,end); % If bias is not needed, put gB = 0
    Wbias   = repmat(Wbias,1,NTrials);
    RMat    = tanh(W*transpose([R1Mat, R2Mat]) + Wbias);
    
end

% Least squares solution: Ax = y

% First construct the A matrix
A = [transpose(RMat), zeros(M,N); zeros(M,N), transpose(RMat)];
% Then the y vector
y = [AVec; BVec];

% xhat = pinv(A)*y;
xhat = A\y;
A3 = xhat(1:N);
B3 = xhat(N+1:end);
