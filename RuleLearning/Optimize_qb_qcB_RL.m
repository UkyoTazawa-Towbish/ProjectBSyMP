function [qb_cur,qcB_cur] = Optimize_qb_qcB_RL(qb_prev,qs,qcB_prev,qlnB_prev)
%__________________________________________________________________________
% This code was created for the following work.
% Synaptic pruning facilitates online Bayesian model selection
% Ukyo T. Tazawa, Takuya Isomura
%
% Copyright (C) 2024 Ukyo Towbish Tazawa
%
% 2024-06-06
%__________________________________________________________________________
% Subroutine to optimize(update) posteriors of B-matrix and C_B-matrix.
%
% Input
% qb_prev:		Posterior of the B-matrix at the previous session
% qs:			Posterior of the states at the current session
% qcB_prev:		Posterior of the C_B-matrix at the previous session
% qlnB_prev:	Posterior of the ln(B) at the previous session
% 
% Output
% qb_cur:		Posterior of the B-matrix at the current session
% qcB_cur:		Posterior of the C_B-matrix at the current session
%__________________________________________________________________________
Ns = size(qs,2); T = size(qs,3);
s_cross_s = sum( pagemtimes(repmat(reshape(qs(:,:,2:T),2,1,Ns,1,T-1),1,1,1,Ns,1) , repmat(reshape(qs(:,:,1:T-1),1,2,1,Ns,T-1),1,1,Ns,1,1)), 5);
if nargin<3 % follow the conventional model
	qb_cur = qb_prev + s_cross_s;
else % follow BSyMP
	qb_cur = qb_prev + s_cross_s.*permute(repmat(qcB_prev,1,1,2,2),[3,4,1,2]);
	qcB_cur = sig(logit(qcB_prev) + squeeze(sum(qlnB_prev.*s_cross_s,[1,2])) + (T-1)*log(2));
end
end