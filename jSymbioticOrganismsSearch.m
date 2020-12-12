%[2014]-"Symbiotic organisms search: A new metaheuristic optimization 
%algorithm" 

% (9/12/2020)

function SOS = jSymbioticOrganismsSearch(feat,label,opts)
% Parameters
lb    = 0;
ub    = 1; 
thres = 0.5; 

if isfield(opts,'N'), N = opts.N; end
if isfield(opts,'T'), max_Iter = opts.T; end
if isfield(opts,'thres'), thres = opts.thres; end 

% Objective function
fun = @jFitnessFunction; 
% Number of dimensions
dim = size(feat,2); 
% Initial 
X   = zeros(N,dim); 
for i = 1:N
  for d = 1:dim
    X(i,d) = lb + (ub - lb) * rand();
  end
end  
% Fitness
fit  = zeros(1,N);
fitG = inf;
for i = 1:N 
  fit(i) = fun(feat,label,(X(i,:) > thres),opts); 
  % Global best
  if fit(i) < fitG
    fitG = fit(i); 
    Xgb  = X(i,:);
  end
end
% Pre
Xi = zeros(1,dim); 
Xj = zeros(1,dim);

curve = zeros(1,max_Iter);
curve(1) = fitG;
t = 2;  
% Iteration
while t <= max_Iter
  for i = 1:N
    % {1} Mutualism phase
    R = randperm(N); R(R == i) = []; 
    J = R(1); 
    % Benefit factor [1 or 2]
    BF1 = randi([1,2]); 
    BF2 = randi([1,2]);
    for d = 1:dim
      % Mutual vector (3)
      MV    = (X(i,d) + X(J,d)) / 2;
      % Update solution (1-2)
      Xi(d) = X(i,d) + rand() * (Xgb(d) - MV * BF1); 
      Xj(d) = X(J,d) + rand() * (Xgb(d) - MV * BF2);
    end
    % Boundary
    Xi(Xi > ub) = ub; Xi(Xi < lb) = lb;
    Xj(Xj > ub) = ub; Xj(Xj < lb) = lb; 
    % Fitness
    fitI = fun(feat,label,(Xi > thres),opts); 
    fitJ = fun(feat,label,(Xj > thres),opts);
    % Update if better solution
    if fitI < fit(i) 
      fit(i) = fitI; 
      X(i,:) = Xi;
    end
    if fitJ < fit(J)
      fit(J) = fitJ;
      X(J,:) = Xj;
    end
    
    % {2} Commensalism phase
    R = randperm(N); R(R == i) = []; 
    J = R(1); 
    for d = 1:dim
      % Random number in [-1,1]
      r1    = -1 + 2 * rand();
      % Update solution (4)
      Xi(d) = X(i,d) + r1 * (Xgb(d) - X(J,d));
    end
    % Boundary
    Xi(Xi > ub) = ub; Xi(Xi < lb) = lb; 
    % Fitness
    fitI = fun(feat,label,(Xi > thres),opts); 
    % Update if better solution
    if fitI < fit(i) 
      fit(i) = fitI;
      X(i,:) = Xi;
    end
    
    % {3} Parasitism phase
    R  = randperm(N); R(R == i) = [];
    J  = R(1);   
    % Parasite vector
    PV = X(i,:);  
    % Randomly select random variables 
    r_dim  = randperm(dim); 
    dim_no = randi([1,dim]);
    for d = 1:dim_no
      % Update solution
      PV(r_dim(d)) = lb + (ub - lb) * rand();
    end
    % Boundary
    PV(PV > ub) = ub; PV(PV < lb) = lb; 
    % Fitness
    fitPV = fun(feat,label,(PV > thres),opts); 
    % Replace parasite if it is better than j
    if fitPV < fit(J) 
      fit(J) = fitPV; 
      X(J,:) = PV;
    end
  end
  
  % Update global best
  for i = 1:N
    if fit(i) < fitG
      fitG = fit(i); 
      Xgb  = X(i,:);
    end
  end
  curve(t) = fitG; 
  fprintf('\nIteration %d GBest (SOS)= %f',t,curve(t))
  t = t + 1;
end
% Select features based on selected index
Pos   = 1:dim;
Sf    = Pos((Xgb > thres) == 1);
sFeat = feat(:,Sf); 
% Store results
SOS.sf = Sf; 
SOS.ff = sFeat; 
SOS.nf = length(Sf); 
SOS.c  = curve; 
SOS.f  = feat;
SOS.l  = label;
end



