clear;
CorrVec = -1:0.1:1;
SubOptCostVec = zeros(1,21);
for k = 1:21
    SubOptCostVec(k) = CorrVarsFn(CorrVec(k));
end

% RNN to just get the indirect evidence p(s1/r2)
% Understand why the low nonlinearity gain case does well but not linear? Check the 'L'
% code
% Handle cases where r2 provides evidence to r1 
