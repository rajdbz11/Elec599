% Comparing a chain of QDN modules with a chain of RNN modules
% Trying to understand how the error (Information Loss) behaves 
% as the number of stages in the chain increases. Would it increase? 
%
% At each stage j we compute the marginal that corresponds to the sum of 
% s_prevstage + s_(j+1), 
% s_prevstage = s_1 + s_2 + ... + s_j,
% j = 1,2 ..., NVars - 1
% 
% Example: At stage 1: we want to get p(s1 + s2|r)
%          At stage 2: we want to get p(s1 + s2 + s3|r)

clear;
LoadModelParams2;
load WMat; load WBias;
load BMat; load BBias;

NVars   = 2; % number of input stimulus variables
NTrials = 5000;
M       = NTrials;
gainMat = 14*rand(NTrials, NVars) + 1; 
% No. of neurons in each population. Assuming the same for each stage. 
NN      = 20; 

% Range of stimulus: 
% Only for visualization purposes. For now, using a Gaussian prior of unit
% variance for each stimuls variable. 
minval  = -NVars;
maxval  =  NVars;
ds      = 0.05;
s       = (minval:ds:maxval)'; % vector of s values to evaluate the posterior

% Prior Param
% These alphas are the inverse of the variance. But, here we are using unit
% variance, therefore the alphas are also unity.
PriorAlphaVec = ones(NVars,1);

% Generate the stimulus values for each trial
sMat    = randn(NTrials, NVars);

% Initialize cells for storing the Network
% Outputs at each intermediate stage

RQDNOutCell     = cell(NVars-1,1); % This is for the QDN model
RRNNOutCell     = cell(NVars-1,1); % This is for the RNN model

% Generate the activities of populations encoding each stim var
% Each cell entry is a matrix of activities for population r_i that encodes
% stimulus variable s_i
RInCell = cell(NVars,1);

for kk = 1:NVars
    RInMat = zeros(NTrials,NN);
    for jj = 1:NTrials
        sVal    = sMat(jj,kk);
        gainVal = gainMat(jj,kk);
        % f(s): mean firing rate
        f_s     = gainVal*exp(-((sVal - S0).^2)/(2*var_w)); 
        RInMat(jj,:) = poissrnd(f_s);
    end
    RInCell{kk} = RInMat;
end
clear RInMat;

% Compute the true Posteriors for each stage
AAMat = zeros(NTrials, NVars-1);
BBMat = zeros(NTrials, NVars-1);
MIVec = zeros(NVars-1,1);

for stage = 1:(NVars-1) % For each stage
    for jj = 1:NTrials  % For each trial
        
        % R_n is new population input for the new stim var s_(j+1)
        R_n = RInCell{stage+1}(jj,:);
        R_n = reshape(R_n,NN,1);
        
        if stage == 1
            AA_p = RInCell{1}(jj,:)*QDNParams.a + PriorAlphaVec(1);
            BB_p = RInCell{1}(jj,:)*QDNParams.b;
        else
            AA_p = AAMat(jj,stage-1);
            BB_p = BBMat(jj,stage-1);
        end
        AAInv = 1/(AA_p) + 1/(QDNParams.a'*R_n + PriorAlphaVec(stage+1));
        AA = 1/AAInv;
        AAMat(jj,stage) = AA;
        
        BB = AA*( BB_p/AA_p + QDNParams.b'*R_n/(QDNParams.a'*R_n + PriorAlphaVec(stage+1)) );
        BBMat(jj,stage) = BB;
        
        RQDNOutCell{stage}(jj,:) = AA*QDNParams.at_d + BB*QDNParams.bt_d + QDNParams.ft*QDNParams.ct_d;
        % PSTrueCell{stage}(jj,:)  = normpdf(s,BB/AA,sqrt(1/AA));
        
    end
    
    % Compute the mutual information at each stage
    % Mean and Standard Deviation for the true posteriors at each stage
    Mu1Vec      = BBMat(:,stage)./AAMat(:,stage);
    Sig1Vec     = sqrt(1./AAMat(:,stage));
    % Mean and Std Dev for the prior at each stage
    Mu2         = 0; 
    Sig2        = sqrt(sum(PriorAlphaVec(1:stage+1)));
    % Using the expression for KL Div between two Gaussians here
    MI_JVec         = ((Mu1Vec - Mu2).^2 + (Sig1Vec.^2 - Sig2^2))/(2*Sig2^2) + log(Sig2./Sig1Vec);
    MIVec(stage)    = sum(MI_JVec)/NTrials;
end


% Now to do the RNN computations
type    = 'RNN';
Train   = 1;

% % Using Gaussian distribution for the weights
% RNNParams.Nratio    = 1;
% RNNParams.gN        = 0.015;
% RNNParams.gB        = 0.85;
% RNNParams.WMat      = WMat;
% RNNParams.WBias     = WBias;
% RNNParams.gC        = 0;

% Using Beta distribution for the weights
RNNParams.Nratio    = 1;
RNNParams.gN        = 0.01;
RNNParams.gB        = 1.5;
RNNParams.WMat      =  BMat; %2*betarnd(0.05, 0.4, N1*N2, N1+N2) - 1;
RNNParams.WBias     =  BBias; %2*betarnd(0.05, 0.4, N1*N2, 1) - 1;
RNNParams.gC        = 0;


NormCostVec        = zeros(NVars-1,1);

for stage = 1:(NVars-1)
    
    % Each RNN module has two inputs: r1 and r2
    % r1 corresponds to the output from the previous stage
    % r2 corresponds to the input corresponding to the new variable being
    % added in this stage, which is s_(j+1)

    if stage == 1
        R1Mat = RInCell{stage};
    else
        R1Mat = RRNNOutCell{stage-1};
    end
    
    R2Mat = RInCell{stage+1}; % input activity corresponding to s_(j+1)
    
    RIMat = RQDNOutCell{stage}; % The ideal QDN model output
    
    % PSTrueMat_I = PSTrueCell{stage}; % No need of the true posteriors
    MI = MIVec(stage);
    
%     [NormCostVec(stage) , RNNOut] = ...
%         RNNModule(NN, NN, NTrials, R1Mat, R2Mat, RIMat, QDNParams.at, QDNParams.bt, type, RNNParams, PSTrueMat_I, s, MI, Train);
    
    [RMat, A3, B3 ]    = SubOptCostComp(NN, NN, NTrials, R1Mat, R2Mat, RIMat, QDNParams.at, QDNParams.bt, type, RNNParams);
    RNNOut = repmat(QDNParams.at_d,1,NTrials).*repmat(A3'*RMat,NN,1) + repmat(QDNParams.bt_d,1,NTrials).*repmat(B3'*RMat,NN,1);  

    RRNNOutCell{stage} = RNNOut';
    
    % Computing the KL Div now
    AMat        = repmat(A3,1,M);
    BMat        = repmat(B3,1,M);
    atermvec    = sum(AMat.*RMat)';
    btermvec    = sum(BMat.*RMat)';
    
    Mu1Vec      = BBMat(:,stage)./AAMat(:,stage);
    Sig1Vec     = sqrt(1./AAMat(:,stage));
    
    Mu2Vec      = btermvec./atermvec;
    Sig2Vec     = sqrt(1./atermvec);
    
    KL_JVec     = ((Mu1Vec - Mu2Vec).^2 + (Sig1Vec.^2 - Sig2Vec.^2))./(2*Sig2Vec.^2) + log(Sig2Vec./Sig1Vec);
    KLDiv       = sum(KL_JVec)/NTrials;
    NormCostVec(stage) = 100*KLDiv/MI;
    
    
end