clear;
% Define the parameters of the bivariate Gaussian distribution
Var1    = 0.2;
Mu1     = 0;  
Var3    = 0.2;
Mu3     = 0;
Ro      = -0.5;

% A11 = 1/((1-Ro^2)*Var1);
% A33 = 1/((1-Ro^2)*Var3);
% A13 = Ro/((1-Ro^2)*sqrt(Var1*Var3));
% A1  = (Mu1/Var1 - Ro*Mu3/sqrt(Var1*Var3))/(1-Ro^2);
% A3  = (Mu3/Var3 - Ro*Mu1/sqrt(Var1*Var3))/(1-Ro^2);
% C   = Mu1^2/Var1 + Mu3^2/Var3 - 2*Ro*Mu1*Mu3/sqrt(Var1*Var3);
% 
% 
% s1Vec = -3:0.1:3;
% s3Vec = -3:0.1:3;
% [S1Mat, S3Mat] = meshgrid(s1Vec,s3Vec);
% S1Vec = S1Mat(:);
% S3Vec = S3Mat(:);
% 
% % Phi13Vec = exp(-0.5*A11*S1Vec.^2 + A1*S1Vec + A13*S1Vec.*S3Vec -0.5*A33*S3Vec.^2 + A3*S3Vec);
% Phi13Vec = exp(-0.5*A11*S1Vec.^2 + A1*S1Vec + A13*S1Vec.*S3Vec);
% Phi13Mat = reshape(Phi13Vec,length(s1Vec),length(s1Vec));
% figure; surf(S1Mat,S3Mat, Phi13Mat)

% Now generate a PPC and look at the posterior

% Preferred orientations
N = 20;
PO                  = linspace(-5,5,N)';
[PO1Mat,PO3Mat]     = meshgrid(PO,PO); % We have a 2D grid of preferred orientations now
PO1Vec              = PO1Mat(:);
PO3Vec              = PO3Mat(:);

% Present some stimulus value now
s1Val   = 2;
s3Val   = 0;

CovMat  = [Var1 Ro*sqrt(Var1*Var3); Ro*sqrt(Var1*Var3) Var3];
gainval = 5;
K       = 2*pi*sqrt(Var1*Var3*(1-Ro^2))*gainval;

% Generate the mean firing rates
F       = K*mvnpdf(repmat([s1Val, s3Val],length(PO1Vec),1), [PO1Vec, PO3Vec], CovMat);

% Generate one instance of neural activity
R       = poissrnd(F);

% Now to get the posterior distribution
% First define all the "kernel" vectors
h11 = ones(N^2,1)/(1-Ro^2)/Var1;
h33 = ones(N^2,1)/(1-Ro^2)/Var3;
h13 = Ro*ones(N^2,1)/(1-Ro^2)/sqrt(Var1*Var3);
h1  = PO1Vec/(1-Ro^2)/Var1 - Ro*PO3Vec/(1-Ro^2)/sqrt(Var1*Var3);
h3  = PO3Vec/(1-Ro^2)/Var3 - Ro*PO1Vec/(1-Ro^2)/sqrt(Var1*Var3);

s1Vec = -5:0.1:5;
s3Vec = -5:0.1:5;
[S1Mat, S3Mat] = meshgrid(s1Vec,s3Vec);
S1Vec = S1Mat(:);
S3Vec = S3Mat(:);


A11 = h11'*R;
A33 = h33'*R;
A13 = h13'*R;
A1  = h1'*R;
A3  = h3'*R;

Phi13Vec = exp(-0.5*A11*S1Vec.^2 + A1*S1Vec + A13*S1Vec.*S3Vec -0.5*A33*S3Vec.^2 + A3*S3Vec);
Phi13Mat = reshape(Phi13Vec,length(s1Vec),length(s1Vec));
figure; contour(S1Mat,S3Mat, log(Phi13Mat)); hold on
plot(s1Val, s3Val, 'rx', 'MarkerSize',10,'LineWidth',2);
grid on






