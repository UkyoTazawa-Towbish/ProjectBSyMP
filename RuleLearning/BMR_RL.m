function Q_rdc = BMR_RL(P,Q_raw,MagnA,MagnB,MagnD,threshA,threshB,threshD,ShowProgress)
%__________________________________________________________________________
% This code was created for the following work.
% Synaptic pruning facilitates online Bayesian model selection
% Ukyo T. Tazawa, Takuya Isomura
%
% Copyright (C) 2024 Ukyo Towbish Tazawa
%
% 2024-06-06
%__________________________________________________________________________
% Perform BMR in the blind source separation task.
%
% Input
% P:			Info of outer world (truth) 
% Q_raw:		Beliefs of full model agent
% MagnA:		Magnification for A-matrix to be reduced
%				The reduced elements are replaced with qa_init*MagnA
%				It should usually be a large number
% MagnB:		Magnification for B-matrix to be reduced
% MagnD:		Magnification for D-vector to be reduced
% threshA:		Threshold for A-matrix
%				The element is reduced if Delta F is smaller than threshA
% threshB:		Threshold for B-matrix
% threshD:		Threshold for D-vector
% ShowProgress:	Flag to switch whether to show progress or not
% 
% Output
% Q_rdc:		Beliefs of BMR agent; Q_ReDuCed-->Q_rdc
%__________________________________________________________________________
if ShowProgress
	wb = waitbar(0,'Applying BMR...');
	wb.NumberTitle = 'off';
	wb.Name = 'Progress Status';
end

Q_rdc = Q_raw; Q_rdc(1).MagnA = MagnA; Q_rdc(1).MagnB = MagnB; Q_rdc(1).MagnD = MagnD;
Q_rdc(1).threshA = threshA; Q_rdc(1).threshB = threshB; Q_rdc(1).threshD = threshD; Q_rdc(1).Label = 'BMR';
qa_init_raw = Q_raw(1).a_init; qb_init_raw = Q_raw(1).b_init; qd_init_raw = Q_raw(1).d_init;
qa_init_rdc = ones(size(qa_init_raw))*(sum(qa_init_raw(:,:,1,1),'all')/4)*MagnA;
qb_init_rdc = ones(size(qb_init_raw))*(sum(qb_init_raw(:,:,1,1),'all')/4)*MagnB;
qd_init_rdc = ones(size(qd_init_raw))*(sum(qd_init_raw(:,1))/2)*MagnD;
Q_rdc(1).a_init_rdc = qa_init_rdc(1,1,1,1);
Q_rdc(1).b_init_rdc = qb_init_rdc(1,1,1,1);
Q_rdc(1).d_init_rdc = qd_init_rdc(1,1,1,1);
for h=1:length(Q_rdc)
	if ShowProgress; waitbar(h/length(Q_rdc),wb); end

	% Reduce qa, qb, qd
	%----------------------------------------------------------------------
	[Q_rdc(h).a,DFmatA,MatRetainedA,DF_A] = reduceMat(Q_raw(h).a,qa_init_raw,qa_init_rdc,threshA);
	[Q_rdc(h).b,DFmatB,MatRetainedB,DF_B] = reduceMat(Q_raw(h).b,qb_init_raw,qb_init_rdc,threshB);
	[Q_rdc(h).d,DFmatD,MatRetainedD,DF_D] = reduceMat(Q_raw(h).d,qd_init_raw,qd_init_rdc,threshD);
	Q_rdc(h).lnA = psi(Q_rdc(h).a)-repmat(psi(sum(Q_rdc(h).a,1)),[2,1,1,1]);
	Q_rdc(h).lnB = psi(Q_rdc(h).b)-repmat(psi(sum(Q_rdc(h).b,1)),[2,1,1,1]);
	Q_rdc(h).lnD = psi(Q_rdc(h).d)-repmat(psi(sum(Q_rdc(h).d,1)),[2,1,1,1]);
	Q_rdc(h).DFmatA = DFmatA; Q_rdc(h).MatRetainedA = MatRetainedA;
	Q_rdc(h).DFmatB = DFmatB; Q_rdc(h).MatRetainedB = MatRetainedB;
	Q_rdc(h).DFmatD = DFmatD; Q_rdc(h).MatRetainedD = MatRetainedD;
	Q_rdc(h).DF = DF_A + DF_B + DF_D;

	% Update qs with reduced parameters
	%----------------------------------------------------------------------
	if h==length(Q_rdc); break; end
	t=1;
	termA = squeeze(pagemtimes(reshape(permute(Q_rdc(h).lnA,[2,1,3,4]),2,[],P(h).Ns) , repmat(reshape(P(h+1).o(:,:,t),[],1),1,1,P(h).Ns)));
	Q_rdc(h+1).s(:,:,t) = softmax(termA+Q_rdc(h).lnD);
	Q_rdc(h+1).sPred(:,:,t) = softmax(Q_rdc(h).lnD);

	for t=2:P(h).T
		if ShowProgress; waitbar(((h-1)*P(h).T+t)/(length(Q_rdc)*P(h).T),wb); end
		[Q_rdc(h+1).s(:,:,t),Q_rdc(h+1).sPred(:,:,t)] = Optimize_qs_RL(Q_rdc(h+1).s(:,:,t-1),P(h+1).o(:,:,t),Q_rdc(h).lnA,Q_rdc(h).lnB);
	end
end
if ShowProgress; close(wb); end
end


%% a nested function
function [qmat_rdc,DF_mat,MatRetained,DF] = reduceMat(qmat_raw,init_raw,init_rdc,thresh)
if ~ismatrix(qmat_raw) % if qmat_raw is A-matrix or B-matrix
	% Prepare variables
	% Please refer to the paper of BMR for the mathematical backgrouds
	% of this part.
	%----------------------------------------------------------------------
	qmat_rdc = qmat_raw + init_rdc - init_raw;
	dPrior = squeeze(sum(betaln(init_rdc(1,:,:,:),init_rdc(2,:,:,:)),2)...
			- sum(betaln(init_raw(1,:,:,:),init_raw(2,:,:,:)),2));
	dPost = squeeze(sum(betaln(qmat_raw(1,:,:,:),qmat_raw(2,:,:,:)),2)...
			- sum(betaln(qmat_rdc(1,:,:,:),qmat_rdc(2,:,:,:)),2));
	DF_mat = dPrior + dPost;

	% Execute the reduction
	% Replace the elements with large (almost) flat 2-by-2 matrix
	% where Delta F is smaller than threshA (or threshB).
	%----------------------------------------------------------------------
	MatRetained = DF_mat>=thresh;
	idx_retained = reshape(repmat(reshape(MatRetained,[],1)',4,1),[],1);
	qmat_rdc(idx_retained) = qmat_raw(idx_retained);
	DF = sum(DF_mat(~MatRetained),'all');
else % if qmat_raw is D-matrix
	% Prepare variables
	%----------------------------------------------------------------------
	qmat_rdc = qmat_raw + init_rdc - init_raw;
	dPrior = betaln(init_rdc(1,:,:,:),init_rdc(2,:,:,:))...
			- betaln(init_raw(1,:,:,:),init_raw(2,:,:,:));
	dPost = betaln(qmat_raw(1,:,:,:),qmat_raw(2,:,:,:))...
			- betaln(qmat_rdc(1,:,:,:),qmat_rdc(2,:,:,:));
	DF_mat = dPrior + dPost;

	% Execute the reduction
	%----------------------------------------------------------------------
	MatRetained = DF_mat>=thresh;
	idx_retained = repmat(MatRetained,2,1);
	qmat_rdc(idx_retained) = qmat_raw(idx_retained);
	DF = sum(DF_mat(~MatRetained),'all');
end
end