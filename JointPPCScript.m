% First obtain a joint PPC representation for two correlate stimulus
% variables. Once the joint PPC is obtained, perform marginalization on the
% posterior distribution to obtain a marginal posterior distribution of one
% of the stimulus variables.

clear;
Visuals = 1;
% -------------------------------------------------------------------------
% 1. Define the prior distribution of the stimulus variables s1 and s2
Mu1     = 0; Mu2    = 0;
Sig1    = 1; Sig2   = 1;
Rho     = 0.5;
CovMat  = [Sig1^2 Rho*Sig1*Sig2; Rho*Sig1*Sig2 Sig2^2];
alpha_11 = 1/(1-Rho^2)/Sig1^2;
alpha_22 = 1/(1-Rho^2)/Sig2^2;
alpha_12 = Rho/ (1-Rho^2)/Sig1/Sig2;
alpha_1  = (Mu1/Sig1^2 - Rho*Mu2/Sig1/Sig2)/(1-Rho^2);
alpha_2  = (Mu2/Sig2^2 - Rho*Mu1/Sig1/Sig2)/(1-Rho^2);


% -------------------------------------------------------------------------
% 2. Code for the joint tuning curves

% First get the set of preferred orientations
N = 20;             % Spacing parameter
PO                  = linspace(-5,5,N)';
[POMatA,POMatB]     = meshgrid(PO,PO); %2D grid of preferred orientations
Mu1Vec              = POMatA(:);
Mu2Vec              = POMatB(:);
% These parameters don't have to be the same for all neurons
Sig1Vec             = sqrt(1)*ones(N^2,1);
Sig2Vec             = sqrt(1)*ones(N^2,1);
RhoVec              = 0.5*ones(N^2,1);
% Sig1Vec             = abs(2 + 0.25*randn(N^2,1));
% Sig2Vec             = abs(2 + 0.25*randn(N^2,1));
% RhoVec              = 0.25*randn(N^2,1);

% Generate the stimulus values using the prior distribution
NTrials = 5000;
sMat = mvnrnd([Mu1, Mu2], CovMat, NTrials);
s1Vec = sMat(:,1); 
s2Vec = sMat(:,2);

% Now create copies of the Mean and Variance vectors to generate mean spike
% counts for all the trials
Mu1Mat  = repmat(Mu1Vec',NTrials,1);
Mu2Mat  = repmat(Mu2Vec',NTrials,1);
Sig1Mat = repmat(Sig1Vec',NTrials,1);
Sig2Mat = repmat(Sig2Vec',NTrials,1);
RhoMat  = repmat(RhoVec',NTrials,1);

% Create copies of the stimulus variables for all the trials
s1Mat   = repmat(s1Vec,1,N^2);
s2Mat   = repmat(s2Vec,1,N^2);

% Matrix of mean spike counts for all the trials
fMat    = exp(-( (s1Mat - Mu1Mat).^2./(2*Sig1Mat.^2) - ...
        RhoMat.*(s1Mat - Mu1Mat).*(s2Mat - Mu2Mat)./(Sig1Mat.*Sig2Mat) + ...
        (s2Mat - Mu2Mat).^2./(2*Sig2Mat.^2) )./(1-RhoMat.^2));

% fs = exp(-( (s1 - Mu1Vec).^2./(2*Sig1Vec.^2) - ...
%     RhoVec.*(s1 - Mu1Vec).*(s2 - Mu2Vec)./(Sig1Vec.*Sig2Vec) + ...
%     (s2 - Mu2Vec).^2./(2*Sig2Vec.^2) )./(1-RhoVec.^2));


% Generate the spike counts with these mean spike counts as the mean of a
% Poisson distribution; use a gain paramter to modulate the spike counts
gainVal = 1;
rMat    = poissrnd(gainVal*fMat);

% -------------------------------------------------------------------------
% 3. Verify that the sum of the tuning curves is constant in the area of
% interest

if (Visuals)
    % Verification Done
    x1      = -5:0.05:5;
    x2      = -5:0.05:5;
    [X1Mat, X2Mat] = meshgrid(x1,x2);
    X1Vec   = X1Mat(:);
    X2Vec   = X2Mat(:);

    ZVec = zeros(length(X1Vec),1);

    for kk = 1:N^2
        ZTemp   = exp(-( (X1Vec - Mu1Vec(kk)).^2./(2*Sig1Vec(kk).^2) - ...
                RhoVec(kk).*(X1Vec - Mu1Vec(kk)).*(X2Vec - Mu2Vec(kk))./(Sig1Vec(kk).*Sig2Vec(kk)) + ...
                (X2Vec - Mu2Vec(kk)).^2./(2*Sig2Vec(kk).^2) )./(1-RhoVec(kk).^2));
        ZVec = ZVec + ZTemp;    
    end
    ZMat = reshape(ZVec,length(x1),length(x1));
    % figure; mesh(X1Mat, X2Mat, ZMat);
    figure; contour(X1Mat, X2Mat, ZMat);
    title('Sum of all the tuning curves');
end

% -------------------------------------------------------------------------
% 4. Construct the joint likelhood and posterior distributions

% First get the parameters 'u' which are used to represent the likelihood
% function. 
u_11 = 1./((1 - RhoVec.^2).*(Sig1Vec.^2));
u_22 = 1./((1 - RhoVec.^2).*(Sig2Vec.^2));
u_12 = RhoVec./((1 - RhoVec.^2).*(Sig1Vec.*Sig2Vec));
u_1  = Mu1Vec./(Sig1Vec.^2) - (RhoVec.*Mu2Vec)./(Sig1Vec.*Sig2Vec);
u_2  = Mu2Vec./(Sig2Vec.^2) - (RhoVec.*Mu1Vec)./(Sig1Vec.*Sig2Vec);

% Now take the dot product with the spike counts
UVec_11 = rMat*u_11;
UVec_22 = rMat*u_22;
UVec_12 = rMat*u_12;
UVec_1  = rMat*u_1;
UVec_2  = rMat*u_2;

% Now we need to generate a whole matrix of the likelihood distributions
L       = 100;
x1      = linspace(-5,5,L);
x2      = linspace(-5,5,L);
[X1Mat, X2Mat] = meshgrid(x1,x2);
X1Vec   = X1Mat(:);
X2Vec   = X2Mat(:);
dx      = x1(2) - x1(1);


% Doing the same to obtain the joint posterior distribution
AVec_11 = UVec_11 + alpha_11;
AVec_22 = UVec_22 + alpha_22;
AVec_12 = UVec_12 + alpha_12;
AVec_1  = UVec_1  + alpha_1;
AVec_2  = UVec_2  + alpha_2;

  
if (0)
    % Create copies to generate the likelihood functions in one matrix
    % operation
    X1Rep   = repmat(X1Vec,1,NTrials);
    X2Rep   = repmat(X2Vec,1,NTrials);

    UMat_11 = repmat(UVec_11',L^2,1);
    UMat_22 = repmat(UVec_22',L^2,1);
    UMat_12 = repmat(UVec_12',L^2,1);
    UMat_1  = repmat(UVec_1',L^2,1);
    UMat_2  = repmat(UVec_2',L^2,1);

    LikelihoodMat = exp(-0.5*X1Rep.^2.*UMat_11 -0.5*X2Rep.^2.*UMat_22 + ...
                X1Rep.*X2Rep.*UMat_12 + X1Rep.*UMat_1 + X2Rep.*UMat_2);
            
    PosteriorMat = exp(-0.5*X1Rep.^2.*(UMat_11 + alpha_11) -0.5*X2Rep.^2.*(UMat_22 + alpha_22) + ...
                X1Rep.*X2Rep.*(UMat_12 + alpha_12) + X1Rep.*(UMat_1 + alpha_1) + X2Rep.*(UMat_2 + alpha_2));
            
    % Visualizing a likelihood function
    trialNum = 1;
    LLMat    = reshape(LikelihoodMat(:,trialNum),L,L);
    LLMat    = LLMat/sum(sum(LLMat))/dx^2;
    figure; contour(X1Mat,X2Mat,(LLMat));
    hold on
    plot(s1Vec(trialNum),s2Vec(trialNum),'rx','MarkerSize',10,'LineWidth',3);
    title('Contour plot of the likelihood function')

    % Visualizing a posterior distribution
    PSMat    = reshape(PosteriorMat(:,trialNum),L,L);
    PSMat    = PSMat/sum(sum(PSMat))/dx^2;
    figure; contour(X1Mat,X2Mat,(PSMat));
    hold on
    plot(s1Vec(trialNum),s2Vec(trialNum),'rx','MarkerSize',10,'LineWidth',3);
    title('Contour plot of the posterior distribution')
end

% 5. Marginalization step
AMarVec = AVec_11 - AVec_12.^2./(AVec_22);
BMarVec = AVec_1 + AVec_2.*AVec_12./AVec_22;

% AMarVec = UVec_11 - UVec_12.^2./(UVec_22);
% BMarVec = UVec_1 + UVec_2.*UVec_12./UVec_22;

MeanMargVec = BMarVec./AMarVec;

% Visualize a marginal posterior distribution
L = 200;
x1 = linspace(-5,5,L);
dx = x1(2) - x1(1);
trialNum = 1;
PSVec = exp(-0.5*x1.^2*AMarVec(trialNum) + x1*BMarVec(trialNum));
PSVec = PSVec/sum(PSVec)/dx;
figure; plot(x1,PSVec,'b.-'); 
hold on
stem(s1Vec(trialNum),1,'r')



% -------------------------------------------------------------------------
% % Now for the RNN part
% % Now to do the RNN computations
% type    = 'RNN';
% Train   = 1;
% 
% % % Using Gaussian distribution for the weights
% % RNNParams.Nratio    = 1;
% % RNNParams.gN        = 0.015;
% % RNNParams.gB        = 0.85;
% % RNNParams.WMat      = WMat;
% % RNNParams.WBias     = WBias;
% % RNNParams.gC        = 0;
% 
% % Using Beta distribution for the weights
% RNNParams.Nratio    = 1;
% RNNParams.gN        = 0.01;
% RNNParams.gB        = 1.5;
% RNNParams.WMat      =  BMat; %2*betarnd(0.05, 0.4, N1*N2, N1+N2) - 1;
% RNNParams.WBias     =  BBias; %2*betarnd(0.05, 0.4, N1*N2, 1) - 1;
% RNNParams.gC        = 0;
