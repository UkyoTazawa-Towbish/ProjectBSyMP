function Q = RunOptimization_BSS(P,qa_init,qcA_init,ShowProgress)
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
% qcA_init:		Prior of the C_A-matrix (Initial posterior)
% ShowProgress:	Flag to switch whether to show progress or not
% 
% Output
% Q:			Optimized beliefs of the agent (full model or BSyMP)
%__________________________________________________________________________
Q = struct;
if isnumeric(qcA_init) % follow BSyMP
	Q = OptBSyMP(P,Q,qa_init,qcA_init,ShowProgress);
	Q.Label = 'BSyMP';
else % follow the conventional model
	Q = OptFullModel(P,Q,qa_init,ShowProgress);
	Q.Label = 'Full Model';
end
end


%% nested functions
function Q = OptFullModel(P,Q,qa_init,ShowProgress)
Q.a_init = qa_init; qlnA_init = psi(qa_init)-repmat(psi(sum(qa_init,1)),[2,1,1,1]);

Q.a = NaN(2,2,P.No,P.Ns,P.T); Q.lnA = NaN(2,2,P.No,P.Ns,P.T);
if ShowProgress
	wb = waitbar(0,'Optimizing qs,qa...');
	wb.NumberTitle = 'off';
	wb.Name = 'Progress Status';
end

% Update Q.s(posterior of s)
%--------------------------------------------------------------------------
t=1;
termA = squeeze(pagemtimes(reshape(permute(qlnA_init,[2,1,3,4]),2,[],P.Ns) , repmat(reshape(P.o(:,:,t),[],1),1,1,P.Ns)));
Q.s = NaN([size(termA),P.T]); % preallocate Q.s
Q.s(:,:,t) = softmax(termA);
for t=2:P.T
	if ShowProgress; waitbar(t/P.T,wb); end
	Q.s(:,:,t) = Optimize_qs_BSS(P.o(:,:,t),qlnA_init);
end

% Update Q.a(posterior of A-matrix)
%--------------------------------------------------------------------------
Q.a = Optimize_qa_qcA_BSS(Q.a_init,Q.s,P.o);
Q.lnA = psi(Q.a)-repmat(psi(sum(Q.a,1)),[2,1,1,1]);
if ShowProgress; close(wb); end
end

function Q = OptBSyMP(P,Q,qa_init,qcA_init,ShowProgress)
Q.a_init = qa_init; qlnA_init = psi(qa_init)-repmat(psi(sum(qa_init,1)),[2,1,1,1]);
Q.cA_init = qcA_init;

Q.a = NaN(2,2,P.No,P.Ns,P.T); Q.lnA = NaN(2,2,P.No,P.Ns,P.T);
Q.cA = NaN(P.No,P.Ns,P.T);
if ShowProgress
	f = waitbar(0,'Optimizing qs,qa,qcA...');
	f.NumberTitle = 'off';
	f.Name = 'Progress Status';
end

% Update Q.s(posterior of s)
%--------------------------------------------------------------------------
t=1;
termA = squeeze(pagemtimes(reshape(permute(qlnA_init.*permute(repmat(qcA_init,1,1,2,2),[3,4,1,2]),[2,1,3,4]),2,[],P.Ns) , repmat(reshape(P.o(:,:,t),[],1),1,1,P.Ns)));
Q.s = NaN([size(termA),P.T]); % preallocate Q.s
Q.s(:,:,t) = softmax(termA);
for t=2:P.T
	if ShowProgress; waitbar(t/P.T,f); end
	Q.s(:,:,t) = Optimize_qs_BSS(P.o(:,:,t),qlnA_init,qcA_init);
end

% Update Q.a(posterior of A-matrix) and Q.cA(posterior of C_A)
%--------------------------------------------------------------------------
[Q.a,Q.cA] = Optimize_qa_qcA_BSS(qa_init,Q.s,P.o,qcA_init,qlnA_init);
Q.lnA = psi(Q.a)-repmat(psi(sum(Q.a,1)),[2,1,1,1]);
if ShowProgress; close(f); end
end