function [J, PSAppMat, DJVec, SubOptMean, SubOptVar] = ComputeCostVec(PSTrueMat, RMat, A3, B3, s3)
% Vectorized form of the ComputeCost function

[M, L]  = size(PSTrueMat);
ds3     = s3(2) - s3(1);

AMat        = repmat(A3,1,M);
BMat        = repmat(B3,1,M);

atermvec    = sum(AMat.*RMat);
btermvec    = sum(BMat.*RMat);

atermMat    = repmat(atermvec,L,1);
btermMat    = repmat(btermvec,L,1);
s3Mat       = repmat(s3,1,M);

PSAppMat    = exp(-(atermMat/2).*(s3Mat.^2) + btermMat.*s3Mat);
PSAppMat    = PSAppMat./repmat(sum(PSAppMat),L,1)/ds3;

PSTrueMat   = transpose(PSTrueMat) + 1e-30;
PSAppMat    = PSAppMat + 1e-30;

DJVec = sum( PSTrueMat.*(log(PSTrueMat) - log(PSAppMat)) );
J = sum(DJVec)/M;

SubOptMean = (btermvec./atermvec)';
SubOptVar  = (1./atermvec)';


