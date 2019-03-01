function [Results, RMatData, PosteriorData, s ] = GenerateTruth(CorrC, gainMat, NTrials)

LoadModelParams;

% Generate the stimulus values.
% Generate the activites r1 and r2
% Generate the true posterior

mu = [0; 0]; % Mean vector
sigma1 = sqrt(1/alpha_p1); 
sigma2 = sqrt(1/alpha_p2);

% Covariance matrix
SigmaMat = [sigma1^2 CorrC*sigma1*sigma2; CorrC*sigma1*sigma2 sigma2^2];

sMat = mvnrnd(mu, SigmaMat, NTrials);

s1 = sMat(:,1);
s2 = sMat(:,2);

% These parameters define the values of s3 over which to compute the
% posteriors.
minval = -6;
maxval =  6;

ds = 0.05;
s = (minval:ds:maxval)'; % vector of s values to evaluate the posterior

Prior1 = exp(-(s - mu(1)).^2/(2*sigma1^2))/sqrt(2*pi*sigma1^2);

% Recording the true posteriors and the input activity values for each
% trial. 

PSTrueMat = zeros(NTrials,length(s));

R1Mat     = zeros(NTrials,N1);
R2Mat     = zeros(NTrials,N2);

AVec      = zeros(NTrials,1);
BVec      = zeros(NTrials,1);

A1Vec     = zeros(NTrials,1);
B1Vec     = zeros(NTrials,1);

TrueMean  = zeros(NTrials,1);
TrueVar   = zeros(NTrials,1);

TrueMean1 = zeros(NTrials,1);
TrueVar1  = zeros(NTrials,1);



% Simplest implementation: using a for loop over trials
% If this is too slow, will try to vectorize later

for trial = 1:NTrials
    
    if size(gainMat,2) == 1
        gain = gainMat;
    else
        gain = gainMat(:,trial);
    end
    
    [PSTrueVec, r1, r2, A, B, A1, B1]     = GenTruePosVec(s1(trial), s2(trial), S0, gain, var_w, sigma1, CorrC, s);
    
    AVec(trial) = A;
    BVec(trial) = B;
    
    A1Vec(trial) = A1;
    B1Vec(trial) = B1;
    
    TrueMean(trial) = B/A;
    TrueVar(trial)  = 1/A;
    
    TrueMean1(trial) = B1/A1;
    TrueVar1(trial)  = 1/A1;
    
    % Update entries: this will be used to find suboptimal model params
    PSTrueMat(trial,:) = PSTrueVec;
    R1Mat(trial,:)     = r1;
    R2Mat(trial,:)     = r2;
    
        
end

PSTrueMat1  = transpose(PSTrueMat) + 1e-30;

PriorMat1   = repmat(Prior1,1,NTrials);
IJVec1      = sum( PSTrueMat1.*(log(PSTrueMat1) - log(PriorMat1)) );
I           = sum(IJVec1)/NTrials;

Results.I       = I;
Results.s1      = s1;
Results.s2      = s2;
Results.TrueMean    = TrueMean;
Results.TrueVar     = TrueVar;
Results.TrueMean1   = TrueMean1;
Results.TrueVar1    = TrueVar1;

RMatData.R1Mat  = R1Mat;
RMatData.R2Mat  = R2Mat;
RMatData.AVec   = AVec;
RMatData.BVec   = BVec;
RMatData.A1Vec  = A1Vec;
RMatData.B1Vec  = B1Vec;

PosteriorData.PSTrueMat = PSTrueMat;