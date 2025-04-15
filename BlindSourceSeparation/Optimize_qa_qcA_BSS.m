function [qa_cur,qcA_cur] = Optimize_qa_qcA_BSS(qa_prev,qs,o,qcA_prev,qlnA_prev)
%__________________________________________________________________________
% This code was created for the following work.
% Synaptic pruning facilitates online Bayesian model selection
% Ukyo T. Tazawa, Takuya Isomura
%
% Copyright (C) 2024 Ukyo Towbish Tazawa
%
% 2024-06-06
%__________________________________________________________________________
% Subroutine to optimize(update) posteriors of A-matrix and C_A-matrix.
%
% Input
% qa_prev:		Posterior of the A-matrix at the previous session
% qs:			Posterior of the states at the current session
% o:			Observations at the current session
% qcA_prev:		Posterior of the C_A-matrix at the previous session
% qlnA_prev:	Posterior of the ln(A) at the previous session
% 
% Output
% qa_cur:		Posterior of the A-matrix at the current session
% qcA_cur:		Posterior of the C_A-matrix at the current session
%__________________________________________________________________________
Ns = size(qs,2); No = size(o,2);
if size(qs,3)==size(o,3); T = size(qs,3);
else; error('size(qs,3) and size(o,3) mismatched.'); end
o_cross_s = sum( pagemtimes(repmat(reshape(o,2,1,No,1,T),1,1,1,Ns,1) , repmat(reshape(qs,1,2,1,Ns,T),1,1,No,1,1)), 5);
if nargin<4 % follow the conventional model
	qa_cur = qa_prev + o_cross_s;
else % follow BSyMP
	qa_cur = qa_prev + o_cross_s.*permute(repmat(qcA_prev,1,1,2,2),[3,4,1,2]);
	qcA_cur = sig(logit(qcA_prev) + squeeze(sum(qlnA_prev.*o_cross_s,[1,2])) + T*log(2));
end
end