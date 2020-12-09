%[2010]-"A new metaheuristic bat-inspired algorithm"

% (9/12/2020)

function BA = jBatAlgorithm(feat,label,opts)
% Parameters
lb     = 0;
ub     = 1; 
thres  = 0.5; 
fmax   = 2;     % maximum frequency
fmin   = 0;     % minimum frequency
alpha  = 0.9;   % constant
gamma  = 0.9;   % constant
A_max  = 2;     % maximum loudness
r0_max = 1;     % maximum pulse rate

if isfield(opts,'N'), N = opts.N; end
if isfield(opts,'T'), max_Iter = opts.T; end
if isfield(opts,'fmax'), fmax = opts.fmax; end 
if isfield(opts,'fmin'), fmin = opts.fmin; end 
if isfield(opts,'alpha'), alpha = opts.alpha; end 
if isfield(opts,'gamma'), gamma = opts.gamma; end
if isfield(opts,'A'), A_max = opts.A; end 
if isfield(opts,'r'), r0_max = opts.r; end 
if isfield(opts,'thres'), thres = opts.thres; end

% Objective function
fun = @jFitnessFunction; 
% Number of dimensions
dim = size(feat,2);  
% Initial 
X   = zeros(N,dim);
V   = zeros(N,dim); 
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
% Loudness of each bat, [1 ~ 2]
A  = unifrnd(1, A_max, [N 1]);
% Pulse rate of each bat, [0 ~ 1]
r0 = unifrnd(0, r0_max, [N 1]);
r  = r0;
% Pre
Xnew = zeros(N,dim);

curve = zeros(1,max_Iter);
curve(1) = fitG;
t = 2;
% Iterations
while t <= max_Iter 
	for i = 1:N
    % Beta [0~1]
    beta = rand();
    % Frequency (2)
    freq = fmin + (fmax - fmin) * beta; 
    for d = 1:dim
      % Velocity update (3)
      V(i,d)    = V(i,d) + (X(i,d) - Xgb(d)) * freq; 
      % Position update (4)
      Xnew(i,d) = X(i,d) + V(i,d);
    end
    % Generate local solution around best solution
    if rand() > r(i)
      for d = 1:dim
        % Epsilon in [-1,1]
        eps       = -1 + 2 * rand(); 
        % Random walk (5)
        Xnew(i,d) = Xgb(d) + eps * mean(A);
      end
    end
    % Boundary
    XB = Xnew(i,:); XB(XB > ub) = ub; XB(XB < lb) = lb;
    Xnew(i,:) = XB;
  end
  % Fitness
  for i = 1:N
    % Fitness
    Fnew = fun(feat,label,(Xnew(i,:) > thres),opts);
    % Greedy selection
    if rand() < A(i) && Fnew <= fit(i)
      X(i,:) = Xnew(i,:); 
      fit(i) = Fnew;
      % Loudness update (6)
      A(i)   = alpha * A(i); 
      % Pulse rate update (6)
      r(i)   = r0(i) * (1 - exp(-gamma * t));
    end
    % Global best
    if fit(i) < fitG
      fitG = fit(i);
      Xgb  = X(i,:);
    end
  end
  curve(t) = fitG;
  fprintf('\nIteration %d Best (BA)= %f',t,curve(t))
  t = t + 1;
end
% Select features
Pos   = 1:dim;
Sf    = Pos((Xgb > thres) == 1);
sFeat = feat(:,Sf);
% Store results
BA.sf = Sf; 
BA.ff = sFeat; 
BA.nf = length(Sf);
BA.c  = curve; 
BA.f  = feat; 
BA.l  = label;
end





