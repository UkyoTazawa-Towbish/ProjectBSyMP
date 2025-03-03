function Q = RunOptimization_RL(P,qa_init,qb_init,qd_init,qcA_init,qcB_init,ShowProgress)
%__________________________________________________________________________
% This code was created for the following work.
% Synaptic pruning facilitates online Bayesian model selection
% Ukyo T. Tazawa, Takuya Isomura
%
% Copyright (C) 2024 Ukyo Towbish Tazawa
%
% 2024-06-06
%__________________________________________________________________________
% Subroutine to run optimization(update) of posteriors.
%
% Input
% P:			Info of outer world (truth)
% qa_init:		Prior of the A-matrix (Initial posterior)
% qb_init:		Prior of the B-matrix (Initial posterior)
% qd_init:		Prior of the D-vector (Initial posterior)
% qcA_init:		Prior of the C_A-matrix (Initial posterior)
% qcB_init:		Prior of the C_B-matrix (Initial posterior)
% ShowProgress:	Flag to switch whether to show progress or not
% 
% Output
% Q:			Optimized beliefs of the agent (full model or BSyMP)
%__________________________________________________________________________
Q = struct;
if isnumeric(qcA_init) % follow BSyMP
	Q = OptCMR(P,Q,qa_init,qb_init,qd_init,qcA_init,qcB_init,ShowProgress);
	Q.Label = 'BSyMP';
else % follow the conventional model
	Q = OptFullModel(P,Q,qa_init,qb_init,qd_init,ShowProgress);
	Q.Label = 'Full Model';
end
end


%% nested functions
function Q = OptFullModel(P,Q,qa_init,qb_init,qd_init,ShowProgress)
Q.a_init = qa_init; qlnA_init = psi(qa_init)-repmat(psi(sum(qa_init,1)),[2,1,1,1]);
Q.b_init = qb_init; qlnB_init = psi(qb_init)-repmat(psi(sum(qb_init,1)),[2,1,1,1]);
Q.d_init = qd_init; qlnD_init = psi(qd_init)-repmat(psi(sum(qd_init,1)),[2,1,1,1]);

Q.a = NaN(2,2,P.No,P.Ns,P.T); Q.lnA = NaN(2,2,P.No,P.Ns,P.T);
Q.b = NaN(2,2,P.Ns,P.Ns,P.T); Q.lnB = NaN(2,2,P.Ns,P.Ns,P.T);
if ShowProgress
	wb = waitbar(0,'Optimizing qs,qa,qb...');
	wb.NumberTitle = 'off';
	wb.Name = 'Progress Status';
end

% Update Q.s(posterior of s) and Q.d(posterior of D-vector)
%--------------------------------------------------------------------------
t=1;
termA = squeeze(pagemtimes(reshape(permute(qlnA_init,[2,1,3,4]),2,[],P.Ns) , repmat(reshape(P.o(:,:,t),[],1),1,1,P.Ns)));
Q.s(:,:,t) = softmax(termA+qlnD_init);
Q.sPred(:,:,t) = softmax(qlnD_init);
Q.d = qd_init + Q.s(:,:,t);
Q.lnD = psi(Q.d)-repmat(psi(sum(Q.d,1)),[2,1,1,1]);

for t=2:P.T
	if ShowProgress; waitbar(t/P.T,wb); end
	[Q.s(:,:,t),Q.sPred(:,:,t)] = Optimize_qs_RL(Q.s(:,:,t-1),P.o(:,:,t),qlnA_init,qlnB_init);
end

% Update Q.a(posterior of A-matrix) and Q.b(posterior of B-matrix)
%--------------------------------------------------------------------------
Q.a = Optimize_qa_qcA_RL(Q.a_init,Q.s,P.o);
Q.b = Optimize_qb_qcB_RL(Q.b_init,Q.s);
Q.lnA = psi(Q.a)-repmat(psi(sum(Q.a,1)),[2,1,1,1]);
Q.lnB = psi(Q.b)-repmat(psi(sum(Q.b,1)),[2,1,1,1]);
if ShowProgress; close(wb); end
end

function Q = OptCMR(P,Q,qa_init,qb_init,qd_init,qcA_init,qcB_init,ShowProgress)
Q.a_init = qa_init; qlnA_init = psi(qa_init)-repmat(psi(sum(qa_init,1)),[2,1,1,1]);
Q.b_init = qb_init; qlnB_init = psi(qb_init)-repmat(psi(sum(qb_init,1)),[2,1,1,1]);
Q.d_init = qd_init; qlnD_init = psi(qd_init)-repmat(psi(sum(qd_init,1)),[2,1,1,1]);
Q.cA_init = qcA_init; Q.cB_init = qcB_init;

Q.a = NaN(2,2,P.No,P.Ns,P.T); Q.lnA = NaN(2,2,P.No,P.Ns,P.T);
Q.b = NaN(2,2,P.Ns,P.Ns,P.T); Q.lnB = NaN(2,2,P.Ns,P.Ns,P.T);
Q.cA = NaN(P.No,P.Ns,P.T); Q.cB = NaN(P.Ns,P.Ns,P.T);
if ShowProgress
	wb = waitbar(0,'Optimizing qs,qa,qb,qcA,qcB...');
	wb.NumberTitle = 'off';
	wb.Name = 'Progress Status';
end

% Update Q.s(posterior of s) and Q.d(posterior of D-vector)
%--------------------------------------------------------------------------
t=1;
termA = squeeze(pagemtimes(reshape(permute(qlnA_init.*permute(repmat(qcA_init,1,1,2,2),[3,4,1,2]),[2,1,3,4]),2,[],P.Ns) , repmat(reshape(P.o(:,:,t),[],1),1,1,P.Ns)));
Q.s(:,:,t) = softmax(termA+qlnD_init);
Q.sPred(:,:,t) = softmax(qlnD_init);
Q.d = qd_init + Q.s(:,:,t);
Q.lnD = psi(Q.d)-repmat(psi(sum(Q.d,1)),[2,1,1,1]);

for t=2:P.T
	if ShowProgress; waitbar(t/P.T,wb); end
	[Q.s(:,:,t),Q.sPred(:,:,t)] = Optimize_qs_RL(Q.s(:,:,t-1),P.o(:,:,t),qlnA_init,qlnB_init,qcA_init,qcB_init);
end

% Update Q.a(posterior of A-matrix) and Q.b(posterior of B-matrix)
% and Q.cA(posterior of C_A) and Q.cB(posterior of C_B)
%--------------------------------------------------------------------------
[Q.a,Q.cA] = Optimize_qa_qcA_RL(Q.a_init,Q.s,P.o,qcA_init,qlnA_init);
[Q.b,Q.cB] = Optimize_qb_qcB_RL(Q.b_init,Q.s,qcB_init,qlnB_init);
Q.lnA = psi(Q.a)-repmat(psi(sum(Q.a,1)),[2,1,1,1]);
Q.lnB = psi(Q.b)-repmat(psi(sum(Q.b,1)),[2,1,1,1]);
if ShowProgress; close(wb); end
end