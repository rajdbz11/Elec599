function [PSTrueVec, r1, r2, A, B, A1, B1] = GenTruePosVec(s1, s2, S0, gain, var_w, sigma1, CorrC, s)


% Generating the mean firing rates
f1 = gain(1)*exp(-((s1 - S0).^2)/(2*var_w));
f2 = gain(2)*exp(-((s2 - S0).^2)/(2*var_w));

% Now to generate one instance of input populations

r1 = poissrnd(f1); % Input population 1
r2 = poissrnd(f2); % Input population 2

% This is the true posterior

% if sum(r1) == 0
%     keyboard;
% end

A = (1/sigma1^2) + sum(r1)/var_w + CorrC^2*sum(r2)/(var_w + sigma1^2*(1-CorrC^2)*sum(r2));
B = sum(r1.*S0)/var_w + CorrC*sum(r2.*S0)/(var_w + sigma1^2*(1-CorrC^2)*sum(r2)); % Already assuming mu = 0

A1 = (1/sigma1^2) + sum(r1)/var_w;
B1 = sum(r1.*S0)/var_w;

Z = sqrt(2*pi/A)*exp(B^2/(2*A));
PSTrueVec = exp(-(A/2)*s.^2 + B*s);

PSTrueVec = PSTrueVec/Z;