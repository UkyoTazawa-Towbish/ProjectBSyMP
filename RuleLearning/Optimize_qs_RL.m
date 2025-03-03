function [qs_cur,qsPred] = Optimize_qs_RL(qs_prev,o_cur,qlnA_prev,qlnB_prev,qcA_prev,qcB_prev)
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
% qs_prev:		Posterior of the state at the previous session
% o_cur:		Observations at the current session
% qlnA_prev:	Posterior of the ln(A) at the previous session
% qlnB_prev:	Posterior of the ln(B) at the previous session
% qcA_prev:		Posterior of the C_A-matrix at the previous session
% qcB_prev:		Posterior of the C_B-matrix at the previous session
% 
% Output
% qs_cur:		Posterior of the state at the current session
% qsPred:		Prediction about the state at the current session before
%				obtaining the observations 
%				(prediction based only on the previous state)
%__________________________________________________________________________
Ns = size(qs_prev,2);
if nargin<5 % follow the conventional model
	termA = squeeze(pagemtimes(reshape(permute(qlnA_prev,[2,1,3,4]),2,[],Ns) , repmat(reshape(o_cur,[],1),1,1,Ns)));
	termB = squeeze(pagemtimes(reshape(permute(qlnB_prev,[1,2,4,3]),2,[],Ns) , repmat(reshape(qs_prev,[],1),1,1,Ns)));
else % follow BSyMP
	termA = squeeze(pagemtimes(reshape(permute(qlnA_prev.*permute(repmat(qcA_prev,1,1,2,2),[3,4,1,2]),[2,1,3,4]),2,[],Ns) , repmat(reshape(o_cur,[],1),1,1,Ns)));
	termB = squeeze(pagemtimes(reshape(permute(qlnB_prev.*permute(repmat(qcB_prev,1,1,2,2),[3,4,1,2]),[1,2,4,3]),2,[],Ns) , repmat(reshape(qs_prev,[],1),1,1,Ns)));
end
qs_cur = softmax(termA+termB);
qsPred = softmax(termB);
end