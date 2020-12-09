%[2016]-"A novel metaheuristic method for solving constrained 
%engineering optimization problems: Crow search algorithm"

% (9/12/2020)

function CSA = jCrowSearchAlgorithm(feat,label,opts)
% Parameters
lb    = 0;
ub    = 1; 
thres = 0.5; 
AP    = 0.1;   % awareness probability
fl    = 1.5;   % flight length

if isfield(opts,'T'), max_Iter = opts.T; end
if isfield(opts,'N'), N = opts.N; end
if isfield(opts,'AP'), AP = opts.AP; end 
if isfield(opts,'fl'), fl = opts.fl; end 
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
  % Global update
  if fit(i) < fitG
    fitG = fit(i);
    Xgb  = X(i,:);
  end
end
% Save memory
fitM = fit;
Xm   = X;
% Pre
Xnew = zeros(N,dim);

curve = zeros(1,max_Iter);
curve(1) = fitG;
t = 2;
% Iteration
while t <= max_Iter
	for i = 1:N
    % Random select 1 memory crow to follow
    k = randi([1,N]);
    % Awareness of crow m (2)
    if rand() >= AP    
      r = rand();
      for d = 1:dim
      	% Crow m does not know it has been followed (1)
        Xnew(i,d) = X(i,d) + r * fl * (Xm(k,d) - X(i,d));
      end
    else
      for d = 1:dim
        % Crow m fools crow i by flying randomly
        Xnew(i,d) = lb + (ub - lb) * rand();
      end
    end
  end
  % Fitness
  for i = 1:N
    % Fitness
    Fnew = fun(feat,label,(Xnew(i,:) > thres),opts); 
    % Check feasibility
    if all(Xnew(i,:) >= lb) && all(Xnew(i,:) <= ub)
      % Update crow
      X(i,:) = Xnew(i,:);
      fit(i) = Fnew;
      % Memory update (5)
      if fit(i) < fitM(i)
        Xm(i,:) = X(i,:);
        fitM(i) = fit(i);
      end
      % Global update
      if fitM(i) < fitG
        fitG = fitM(i);
        Xgb  = Xm(i,:);
      end
    end
  end
  curve(t) = fitG; 
  fprintf('\nIteration %d Best (CSA)= %f',t,curve(t))
  t = t + 1;
end
% Select features
Pos   = 1:dim; 
Sf    = Pos((Xgb > thres) == 1);
sFeat = feat(:,Sf);
% Store results
CSA.sf = Sf; 
CSA.ff = sFeat;
CSA.nf = length(Sf); 
CSA.c  = curve; 
CSA.f  = feat;
CSA.l  = label;
end





