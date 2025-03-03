function Q_rdc = BMR_BSS(P,Q_raw,MagnA,threshA,FlagOnline,ShowProgress)
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
% threshA:		Threshold for A-matrix
%				The element is reduced if Delta F is smaller than threshA
% FlagOnline:	Flag to switch between BMR and online-BMR
% ShowProgress:	Flag to switch whether to show progress or not
% 
% Output
% Q_rdc:		Beliefs of BMR (or online-BMR) agent; Q_ReDuCed-->Q_rdc
%__________________________________________________________________________
if ShowProgress
	if FlagOnline
		wb = waitbar(0,'Applying online-BMR...');
	else
		wb = waitbar(0,'Applying BMR...');
	end
	wb.NumberTitle = 'off';
	wb.Name = 'Progress Status';
end

Q_rdc = Q_raw; Q_rdc(1).MagnA = MagnA; Q_rdc(1).threshA = threshA; Q_rdc(1).Label = AttachLabel(FlagOnline);
qa_init_raw = Q_raw(1).a_init;
qa_init_rdc = ones(size(qa_init_raw))*(sum(qa_init_raw(:,:,1,1),'all')/4)*MagnA;
Q_rdc(1).a_init_rdc = qa_init_rdc(1,1,1,1);
for h=1:length(Q_rdc)
	if ShowProgress; waitbar(h/length(Q_rdc),wb); end

	% Reduce qa
	%----------------------------------------------------------------------
	if h==1
		[Q_rdc(h).a,DFmatA,MatRetainedA,DF] = reduceMat(Q_raw(h).a,qa_init_raw,qa_init_rdc,threshA,FlagOnline,true);
	else
		[Q_rdc(h).a,DFmatA,MatRetainedA,DF] = reduceMat(Q_raw(h).a,qa_init_raw,qa_init_rdc,threshA,FlagOnline,Q_rdc(h-1).MatRetainedA);
	end
	Q_rdc(h).lnA = psi(Q_rdc(h).a)-repmat(psi(sum(Q_rdc(h).a,1)),[2,1,1,1]);
	Q_rdc(h).DFmatA = DFmatA; Q_rdc(h).MatRetainedA = MatRetainedA; Q_rdc(h).DF = DF;

	if h==length(Q_rdc); break; end
	if FlagOnline
		% Update qs and qa with reduced parameters
		%------------------------------------------------------------------
		Q_raw(h+1) = RunOptimization_BSS(P(h+1),Q_rdc(h).a,false,false);
	else
		% Update qs with reduced parameters
		%------------------------------------------------------------------
		t=1;
		termA = squeeze(pagemtimes(reshape(permute(Q_rdc(h).lnA,[2,1,3,4]),2,[],P(h).Ns) , repmat(reshape(P(h+1).o(:,:,t),[],1),1,1,P(h).Ns)));
		Q_rdc(h+1).s(:,:,t) = softmax(termA);

		for t=2:P(h).T
			if ShowProgress; waitbar(((h-1)*P(h).T+t)/(length(Q_rdc)*P(h).T),wb); end
			% re-optimize qs(t+1) with reduced parameters
			Q_rdc(h+1).s(:,:,t) = Optimize_qs_BSS(P(h+1).o(:,:,t),Q_rdc(h).lnA);
		end
	end
end
if ShowProgress; close(wb); end
end


%% nested functions
function [qmat_rdc,DF_mat,MatRetained,DF] = reduceMat(qmat_raw,init_raw,init_rdc,thresh,FlagOnline,MatRetained_prev)
% Prepare variables
% Please refer to the paper of BMR for the mathematical backgrouds
% of this part.
%--------------------------------------------------------------------------
qmat_rdc = qmat_raw + init_rdc - init_raw; 
dPrior = squeeze(sum(betaln(init_rdc(1,:,:,:),init_rdc(2,:,:,:)),2)...
		- sum(betaln(init_raw(1,:,:,:),init_raw(2,:,:,:)),2));
dPost = squeeze(sum(betaln(qmat_raw(1,:,:,:),qmat_raw(2,:,:,:)),2)...
		- sum(betaln(qmat_rdc(1,:,:,:),qmat_rdc(2,:,:,:)),2));
DF_mat = dPrior + dPost;

% Execute the reduction
% Replace the elements with large (almost) flat 2-by-2 matrix
% where Delta F is smaller than threshA.
%--------------------------------------------------------------------------
if FlagOnline
	MatRetained = (DF_mat>=thresh)&MatRetained_prev;
	idx_retained = ~logical(reshape(repmat(reshape(MatRetained_prev-MatRetained,[],1)',4,1),[],1));
else
	MatRetained = DF_mat>=thresh;
	idx_retained = reshape(repmat(reshape(MatRetained,[],1)',4,1),[],1);
end
DF = sum(DF_mat(~MatRetained),'all');
qmat_rdc(idx_retained) = qmat_raw(idx_retained);
end

function Label = AttachLabel(FlagOnline)
if FlagOnline
	Label = 'Online BMR';
else
	Label = 'BMR';
end
end