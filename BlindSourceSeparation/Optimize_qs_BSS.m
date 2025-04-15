function qs_cur = Optimize_qs_BSS(o_cur,qlnA_prev,qcA_prev)
%__________________________________________________________________________
% This code was created for the following work.
% Synaptic pruning facilitates online Bayesian model selection
% Ukyo T. Tazawa, Takuya Isomura
%
% Copyright (C) 2024 Ukyo Towbish Tazawa
%
% 2024-06-06
%__________________________________________________________________________
% Subroutine to optimize(update) posteriors of the states.
%
% Input
% o_cur:		Observations at the current time-point
% qlnA_prev:	Posterior of the ln(A) at the previous session
% qcA_prev:		Posterior of the C_A-matrix at the previous session
% 
% Output
% qs_cur:		Posterior of the state at the current time-point
%__________________________________________________________________________
Ns = size(qlnA_prev,4);
if nargin<3 % follow the conventional model
	termA = squeeze(pagemtimes(reshape(permute(qlnA_prev,[2,1,3,4]),2,[],Ns) , repmat(reshape(o_cur,[],1),1,1,Ns)));
else % follow BSyMP
	termA = squeeze(pagemtimes(reshape(permute(qlnA_prev.*permute(repmat(qcA_prev,1,1,2,2),[3,4,1,2]),[2,1,3,4]),2,[],Ns) , repmat(reshape(o_cur,[],1),1,1,Ns)));
end
qs_cur = softmax(termA);
end