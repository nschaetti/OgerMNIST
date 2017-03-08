% simulation des données d'entrée de la dynamique à retard passe-bas simple
% délai, et de la réponse dynamique à cette entrée
% Les digits sont traités avec un ordre aléatoire qui utilise la
% permutation effectuée de toute façon pour le choix aléatoire des
% partitions de test et d'apprentissage.
% duration: 140s for 150 nodes and all the 500 digits
tic

clear all
close all

% original sampling period of the AWG
%Tawg=1/628319.622; % =T/Nnode=tau/5
% sampling period for the recorded time trace
%Ts=100e-9;
%
% number of time traces recorded to process all the digits
Nttmax=20;
% Nb of full "zero-filled" single digit to be extracted in ttrace
Ndptt=25;
% Nb of nodes
Nnode=306; %306
% Sparsity
spsty=0.1;
% Sample length of a "zero-filled" single digit
Nspd=150;

%-------------------------------------------------------------------------

input=9
% load the permutation rule of the original ordered 500 digits
%prm = load('permutation.txt');
%
% Load the table of the individual digit lenghts
digtlen = load(['./inputs' num2str(input) '/load_lengthfile1.dat']);
% Number of actual spoken digits (500)
Nsd=length(digtlen);
%
% Load labels
labels = load(['./inputs' num2str(input) '/load_labels1.dat']);

% Number of BIG calculations (loop round)
Cmax=100;
Vpmin=-2.5e-4;
Vpmax=2.5e-4;

% Varied parameter, and Results matrix where the NTC performance will be
% stored while scanning the parameter Vp (see line 104-130)
%Vp=Vpmin:10:Vpmax;
%Cmax=length(Vp);
Vp=linspace(Vpmin,Vpmax,Cmax);
ResVp=zeros(Cmax,3);

%-------------------------------------------------------------------------
% Load the mask for input connectivity
load(['./inputs' num2str(input) '/demo3/mask_' num2str(Nnode) '.mat']);
%load ./input/mask_test.mat
%Wi=load(['./inputs/maskb' num2str(Nnode) '_1.dat'],'-ascii');
% normalization factor for the input signal; if set to zero, the next
% condition will calculate it according to the loaded Wi
% before performing this calculation, tests were done with Delta=0.5, with
% which single error performance were achieved. With normalization, nWi was
% evaluated to 2.6142, thus resulting in an equivalent Delta of 1.307 if
% one wants to recover the same single error result.
nWi=1.0;
%%If loaded mask already normalized
%nWi=1;
%2.6142;
if (nWi==0)
    display('Calculation of the normalization factor')
    nMax=0;
    nmin=0;
    for m=1:Nsd
        % load a cochleagram
        Mc=load(['./inputs ' num2str(input) '/load_inputs1_' num2str(m) '.dat']);
        % construct the input with the mask
        Mu=Wi*Mc;
        nMax=max(max(max(Mu)),nMax);
        nmin=min(min(min(Mu)),nmin);
    end
    nWi=nMax-nmin;
end
display(['Normalization factor for the input: ' num2str(nWi)])
Wi=Wi/nWi;
%-------------------------------------------------------------------------
% './inputs/maska150.dat': mask for 150 Nodes
% '../../../Palma/LaurentReservoir/ExperimentData/EOSetup/InputsTargetsSpee
% ch/load_mask1.dat': mask for 400 Nodes

% Construction of a sparse matrix with normal distribution of the non-zero
% elements
% Wi=sprand(Nnode,86,spsty);
%transform matrix Wi into a ternary elements sparse matrix (-1,0,+1)
% Wi(Wi~=0)=Wi(Wi~=0)-0.5;
% Wi(Wi<0)=-1;
% Wi(Wi>0)=1;
%MWi=zeros(Nnode,86,Cmax);
%-------------------------------------------------------------------------
% Define a random permutation of all the digits
prm=randperm(Nsd);
%-------------------------------------------------------------------------
% Loop for the scanning of the varied parameter Vp(C)
for C=1:Cmax
    %
    % Temporal parameters
    %--------------------
    % ratio tau/delta\tau between high cut-off response time and node
    % separation
    rho=2.5;
    % integration time step (in units of tau)
    h=0.01;
    % Nb of integration time steps between two nodes
    Nhpn=floor(1/h/rho);
    % Integer nb of integration time step per delay
    nbe=Nnode*Nhpn;
    % time shift of the origine, in Nb of samples, for ttrace
    %N0=11;
    % time shift to choose the node position between 0 and floor(Tawg/Ts)
    %dT0=13;
    % time delay (in units of tau)
    %T=nbe*h;
    %    
    % physical response time of the high cut-off (in integ. time step units)
    tau=1/h;
    %
    % Amplitude parameters
    %---------------------
    % Operating conditions for the delay dynamics
    nl=0;
    beta=1.1;
    % around a linear operating point: phi=+-pi/4-beta/2
    % around a parabolic operating point: phi close to 0 or +-pi/2
    phi=0.01;
    alpha=1;
    % relative amplitude of the input information
    % input information is injected as an arugument of the NL-function
    Delta=2.5;
    % open (0) or closed (1) loop operation, feedback off or on
    fb_on=1;
    if (fb_on==1)
        display('Closed loop operation (with delayed feedback)')
    else
        display('Open loop operation')
    end    
    %
    %---------------------------------------------------------------------
    % 2D pattern built from the calculated transient, for each digit of the
    % computed transient sequence of Ndptt digits
    %
    % Time origine for cutting each digit in the sequence
    N0=0;
    % Time shift for choosing the node position between the several possible
    % ones, with a constant spacing between the nodes, 0<=dT0<Nhpn
    dT0=36;
    % Small (ca. 1e-3) change in the pitch of the Read Out compared to the
    % input node spacing when addressing each sample of the input signal (can
    % be >0 or <0), this results in a down or up tilt of the 2D pattern
    Npitch=Vp(C);
    % Concatenation matrix of the transients of the successive digits
    A=[];
    B=[];
    %    
    %ttrace=ttrace/(max(ttrace)-min(ttrace));
    tps=[tau ; nbe ; 0 ; Nspd*Ndptt];
    param=[nl ; beta ; phi ; alpha ; fb_on];
    % Determine the stable steady state, to avoid initial "unrealistic"
    % transient for the first digit of the sequence
    cdi=fzero(@(x)(x-beta*(sin(x+phi))^2),0.5);
    %
    % Loop for the Nttmax sequences of Ndptt digits
    %p=1;
    %
    % Counters for the digit answer vectors (train and learn)
   % Dgv=zeros(1,Nsd);
    Dgv2=zeros(Nttmax,Ndptt);
    %
    %---------------------------------------------------------------------
    % Numerical integration calculating the nonlinear transient response
    %
    % loop for the Nttmax sequences of each Ndptt spoken digits
    for p=1:Nttmax
        % Vector for an input sequence of Ndptt digits to be processed
        
        ttrace=zeros(Nspd*nbe*Ndptt,1);
        
        Mc=0;
        Mu=0;
        
        %
        
        % Loop for Ndptt digits in a single sequence
        
        for m=1:Ndptt
            % load a cochleagram
            Mc=load(['./inputs' num2str(input) '/load_inputs1_' num2str(prm(Ndptt*(p-1)+m)) '.dat']);
            % construct the input with the mask
            Mu=Wi*Mc;
            %
            % Calculate the input signal
            for k=1:size(Mu,2)
                %ind=Nspd*nbe*(m-1)+(k-1)*nbe;
                for l=1:Nnode
                    
                    %ttrace=[ttrace; Mu(l,k)*ones(Nhpn,1)];
                    ttrace(Nspd*nbe*(m-1)+(k-1)*nbe+ ( ( (Nhpn*(l-1)) + 1 ) : (Nhpn*l)) )=Mu(l,k)*ones(Nhpn,1);
                end
            end
            %plot(ttrace)
            %drawnow
        end
        %
        
        % Calculate the transient nonlinear delay dynamics for one sequence
        yc=rk4_ntc(cdi*ones(nbe+1,1),tps,param,Delta*ttrace);
        
        %lyc=length(yc)-1;
        %
        %plot((1:length(ttrace))*h/T,ttrace,'r',(1:lyc)*h/T,yc(1:lyc),'g')
        %drawnow
        %
        % Loop for extracting the 2D pattern from the ReadOut transient,
        % for each of the Ndptt digits in a sequence
        %m=1;
        DD=zeros(1,Ndptt);
        for m=1:Ndptt
            % Define the position of the digit in the ordered data base
            no=prm(Ndptt*(p-1)+m);
            % Store the digit value of currently processed response
            %dgo=no-floor((no-1)/10)*10;
            
            %DD(m)=dgo;
            
            dgo=labels(no);
            DD(m)=dgo;
            
            %Dgv(Ndptt*(p-1)+m)=dgo;
            %Dgv2(p,m)=dgo;
            % Recover the digit length of the currently processed response
            Kpd=digtlen(no);
            % starting date of the current digit in the sequence
            start=(m-1)*nbe*Nspd+1+N0+dT0;
            % sample dates for the currently processed digit + 1 delay
            %display(num2str(no))
            %display(num2str(Kpd))
            tnop=start+(1:((Kpd+1)*Nnode*Nhpn));
            yci=interp1(tnop,yc(tnop),start+Nhpn*(1+Npitch)*(1:(Kpd*Nnode)));
            
            % 2D pattern
            P2D=zeros(Nnode,Kpd);
            M=zeros(10,Kpd);
            % 2D pattern construction for one digit of the sequence
            for k=1:Kpd
                P2D(:,k)=yci((k-1)*Nnode+(1:Nnode));
                %yc(start+floor(Nhpn*(1+Npitch)*((k-1)*Nnode+(1:Nnode))));
            end
            A=horzcat(A, P2D);
            % Construct B matrix, and concatenate the ones for the learning
            M(dgo,:)=ones(1,Kpd);
            B=horzcat(B, M);
            % Count occurrences of each digit in the learning set
            %Nfl(dgo)=Nfl(dgo)+1;
            
            % Each 2D pattern of each processed digit is saved individually
            %save(['./outputs/output2D' num2str(Ndptt*(p-1)+m) '.dat'],'P2D','-ASCII');
        end
        
        Dgv2(p,:)=DD;
    end
    
    
    Dgv2=Dgv2';
    Dgv2=Dgv2(:)';
    Dgv=Dgv2;
   % Dgv2-Dgv
   % Dgv=Dgv2;
    
    display(' ')
    display(['Nb of nodes: ' num2str(Nnode) ', offset phase: '...
        num2str(phi) ', beta: ' num2str(beta) ', input gain: ' num2str(Delta)])
    %
    %---------------------------------------------------------------------
    % Training and testing with cross-validation
    %
    % cross-validation loop
    % each round in the loop corresponds to a different partition for training
    % and testing sub-sets, and thus to a different calculation of W
    %
    % Error counter
    Err=0;
    % Define the size of the test set
    Stest=25;
    % Counter for the digit statistics in the learning set (for Fisher
    % Labelling issues)
    Nfl=zeros(10,1);
    % vector for the identification of the best and worth recognized digit
    sdid=zeros(floor(Nsd/Stest),2);
    % table of margin performance over the cross-validation partitions
    mP=zeros(floor(Nsd/Stest),4);
    % histogram for the wrongly identified digits
    wsdH=zeros(Nsd,1);
    
    % regression parameter
    lambda=1e-4;
    
    % table memorizing on the first line the wrong digits during testing (in
    % the ordered sequence), and on the second line the wrongly interpreted
    % digit value from 1 to 10:
    ErrlisT=[];
    % table memorizing on the first line the wrong digits during learning (in
    % the ordered sequence), and on the second line the wrongly interpreted
    % digit value from 1 to 10:
    ErrList=[];
    % count for the total number of errors (learning & testing):
    Errtot=0;
    %
    %---------------------------------------------------------------------
    % 
    % Loop for the cross-validation steps
    for p=1:floor(Nsd/Stest) 
        %Al=[];
        %Bl=[];
        %
        % Construction of the matrices A and B according to the partition
        % of the current cross-validation, for the learning set only
        Al=A(:,sum(digtlen(prm(1:p*Stest)))+1:end);
        Bl=B(:,sum(digtlen(prm(1:p*Stest)))+1:end);
        if p>1
            Al=horzcat(A(:,1:sum(digtlen(prm(1:(p-1)*Stest)))), Al);
            Bl=horzcat(B(:,1:sum(digtlen(prm(1:(p-1)*Stest)))), Bl);
        end
        %
        % Calculation of the Read-Out Matrix W via a More-Penrose inversion
        W=(Al*Al'+lambda*eye(Nnode))\(Al*Bl');
        %
        % Evaluate the Read Out with the calculated W for all the processed
        % digits, in the random order defined by the permutation prm
        E=W'*A;
        %
        % Summed evaluation
        sE=zeros(10,Nsd);
        % Vector containing the margin of each digit answer
        margin=zeros(1,Nsd);
        % Vector memorizing the values of the wrong answers of a processed
        % Digit; the memorized value is a number from 1 to 10
        wA=zeros(1,Nsd);
        % Counter indexing the beginning of each digit in the concatenated
        % sequence 
        cmpt=0;
        
        % Calculate the answer from the evaluation of the Read Out for all
        % the digits
        for n=1:Nsd
            % Determine the end index of the processed digit in the full
            % Read Out evaluation
            end_c=cmpt+digtlen(prm(n));
            % Sum the Read Out evaluation over its duration to "score" each
            % possible digit value
            sE(:,n)=sum(E(:,cmpt+1:end_c),2);
            cmpt=end_c;
            % Calculate the distance of each score from the right answer
            sE(:,n)=sE(:,n)-sE(Dgv(n),n);
            % Calculate the maximum of the distance (all distances should
            % be negative if all the answers are correct)
            margin(n)=max(sE(setdiff(1:10,Dgv(n)),n));
            % Detect a wrong evaluation via a positive distance
            if any(sE(:,n)>0)
                wA(n)=find(sE(:,n)==max(sE(:,n)));
                wsdH(prm(n))=wsdH(prm(n))+1;
            end
        end
        %
        % Results within a partitioning of one cross-validation
        % Indices (in the permuted sequence) for the training set:
        Tind=((p-1)*Stest+1):p*Stest;
        Lind=setdiff(1:Nsd,Tind);
        % Indices in the full permutated sequence for the wrong answers:
        ind=find(wA>0);
        Errtot=Errtot+length(ind);
        %
        % Indices for the wrong answers in the Learning set only
        El=intersect(ind,Lind);
        if ~isempty(El)
            ErrList=horzcat(ErrList,[prm(El);wA(El)]);
            %display(' ')
            %display(['Partition ' num2str(p) '/' num2str(floor(Nsd/Stest))...
            %    ', errors in the learning set: ' num2str(prm(El)) ' (found as ' num2str(wA(El)) ')']);
            % plot the Reservoir Response, the Read Out Evaluation, and the 10
            % digits scores for the erroneous digits
            %RCdproc(prm(El),prm,digtlen,A,E);
            % forces a pause (type "return" in the Command Line to continue)
            %keyboard
            display(['Learning errors in cross-validation step ' num2str(p) '/' num2str(floor(Nsd/Stest))...
                ': ' num2str(prm(El)) ' (interpreted as ' num2str(wA(El)) ')']);
        end
        %
        % Indices for the wrong answers in the Training set only
        Et=intersect(ind,Tind);
        if ~isempty(Et)
            %display(' ')
            %display(['Partition ' num2str(p) '/' num2str(floor(Nsd/Stest))...
            %', errors in the testing set: ' num2str(prm(Et)) ' (found as ' num2str(wA(Et)) ')']);
            Err=Err+length(wA(Et));
            ErrlisT=horzcat(ErrlisT,[prm(Et);wA(Et)]);
            % Plot the Reservoir Response, the Read Out Evaluation, and the
            % 10 digits scores for the erroneous digits
            %RCdproc(prm(Et),prm,digtlen,A,E);
            % Forces pause (type "return" in the Command Line to continue)
            %keyboard
            display(['Testing errors in cross-validation step ' num2str(p) '/' num2str(floor(Nsd/Stest))...
                ': ' num2str(prm(Et)) ' (interpreted as ' num2str(wA(Et)) ')']);
        end
        mP(p,:)=[mean(margin(Tind)) max(margin(Tind)) min(margin(Tind)) std(margin(Tind))];
        % For each partition of the cross-validation, record the best and
        % worth digit separation in the testing set
        sdid(p,:)=[prm(margin==max(margin(Tind))) prm(margin==min(margin(Tind)))];
    end % end of a cross-validation step
    % end of the full cross-validation
    %
    
    Npart=1:floor(Nsd/Stest);
    %figure(1)
    %plot(Npart,mP(:,1),'r',Npart,mP(:,2),'g',Npart,mP(:,3),'b',Npart,mP(:,4),'k')
    %title('Margin distance: mean (red), min (green), max (blue), std (black)')
    %xlabel('Partition No.')
    display(['Word Error Rate in %: ' num2str(Err/Nsd*100) ,...
        ', relative mean smallest margin: ' num2str(mean(mP(:,2))/mean(mP(:,1))*100)])
    %display('List of erroneus digits for the testing set (and below their wrongly interpreted value):')
    %ErrlisT %#ok<NOPTS>
    display(['Total number of errors (learning + testing): ' num2str(Errtot)])
    %figure(4)
    %plot(wsdH,'o')
    %title('Histogram of the wrong digits')
    %    
    toc
    %
    ResVp(C,:)=[Err mean(mP(:,2)) mean(mP(:,1))];
    display(['loop processed: ' num2str(C) '/' num2str(Cmax) '. Parameter value: ' num2str(Vp(C))])
    display(['Results. Nb of errors: ' num2str(ResVp(C,1)) ', mean smallest margin: ' num2str(ResVp(C,2)) ' (in %)'])
    display(' ');
    
end
%
figure(1)
set(1,'pos',[57 374 560 420])
plot(Vp,ResVp(:,1))
title('Nb of errors (500 tests)')
figure(2)
set(2,'pos',[1040 374 560 420])
plot(Vp,ResVp(:,2))
title('Mean, over all X-valid, of the worst Margin (for test digits)')
figure(3)
set(3,'pos',[575 2 560 420])
plot(Vp,ResVp(:,3))
title('Mean, over all X-valid, of the mean Margin (for test digits)')

delete(gcp);
