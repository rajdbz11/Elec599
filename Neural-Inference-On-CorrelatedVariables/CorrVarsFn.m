function SubOptCost = CorrVarsFn(CorrC)



LoadModelParams;
load WMat;

NTrials = 5000;
% M       = NTrials;
% gainMat = [2; 2];
gainMat = 5*rand(2,NTrials) + 1;
% CorrC = 0.8;

[Results, RMatData, PosteriorData, s ] = GenerateTruth(CorrC, gainMat, NTrials);

I       = Results.I;

R1Mat = RMatData.R1Mat;
R2Mat = RMatData.R2Mat;
AVec  = RMatData.AVec;
BVec  = RMatData.BVec;

PSTrueMat = PosteriorData.PSTrueMat;


%-------------------------- Suboptimal networks----------------------------

RNNParams.gN        = 0.0005;
RNNParams.gB        = 0.85;
RNNParams.WMat      = WMat;
RNNParams.Nratio    = 1;

% type = 'Q';
type = 'RNN';
[RMat, A3, B3 ] = SubOptCostComp(N1, N2, NTrials, R1Mat, R2Mat, AVec, BVec, type, RNNParams);


[Jfinal, PSSubOptMat, DJVec, SubOptMean, SubOptVar]       = ComputeCostVec(PSTrueMat, RMat, A3, B3, s);

SubOptCost = 100*Jfinal/I;
PSSubOptMat = PSSubOptMat';

TrueMean = Results.TrueMean;
TrueVar = Results.TrueVar;

TrueMean1 = Results.TrueMean1;
TrueVar1 = Results.TrueVar1;

