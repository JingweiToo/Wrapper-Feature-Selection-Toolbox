%[2020]-"Slime mould algorithm: A new method for stochastic 
%optimization"

% (8/12/2020)

function SMA = jSlimeMouldAlgorithm(feat,label,opts)
% Parameters
lb    = 0;
ub    = 1; 
thres = 0.5; 
z     = 0.03;  % control local & global 

if isfield(opts,'N'), N = opts.N; end
if isfield(opts,'T'), max_Iter = opts.T; end
if isfield(opts,'z'), z = opts.z; end 
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
% Pre
fit  = zeros(1,N); 
fitG = inf; 
W    = zeros(N,dim);

curve = inf; 
t = 1; 
% Iteration
while t <= max_Iter
  % Fitness
  for i = 1:N
    fit(i) = fun(feat,label,(X(i,:) > thres),opts);
    % Best
    if fit(i) < fitG
      fitG = fit(i);
      Xb   = X(i,:);
    end
  end
  % Sort smell index (2.6)
  [fitS, idxS] = sort(fit,'ascend'); 
  % Best fitness & worst fitness
  bF = min(fit);
  wF = max(fit);
  % Compute W (2.5)
  for i = 1:N
    for d = 1:dim
      % Condition 
      r = rand();
      if i <= N / 2
        W(idxS(i),d) = 1 + r * log10(((bF - fitS(i)) / ....
          (bF - wF + eps)) + 1);
      else
        W(idxS(i),d) = 1 - r * log10(((bF - fitS(i)) / ...
          (bF - wF + eps)) + 1);
      end
    end
  end
  % Compute a (2.4)
  a = atanh(-(t / max_Iter) + 1);
  % Compute b 
  b = 1 - (t / max_Iter);
  % Update (2.7)
  for i = 1:N
    if rand() < z
      for d = 1:dim
        X(i,d) = rand() * (ub - lb) + lb;
      end
    else
      % Update p (2.2)
      p  = tanh(abs(fit(i) - fitG));
      % Update vb (2.3)
      vb = unifrnd(-a,a,[1,dim]);
      % Update vc 
      vc = unifrnd(-b,b,[1,dim]);
      for d = 1:dim
        % Random in [0,1]
        r = rand();
        % Two random individuals
        A = randi([1,N]);
        B = randi([1,N]);
        if r < p 
          X(i,d) = Xb(d) + vb(d) * (W(i,d) * X(A,d) - X(B,d));
        else
          X(i,d) = vc(d) * X(i,d);
        end
      end
    end
    % Boundary
    XB = X(i,:); XB(XB > ub) = ub; XB(XB < lb) = lb;
    X(i,:) = XB;
  end 
  % Save
  curve(t) = fitG;
  fprintf('\nIteration %d Best (SMA)= %f',t,curve(t))
  t = t + 1;
end
% Select features based on selected index
Pos   = 1:dim;
Sf    = Pos((Xb > thres) == 1); 
sFeat = feat(:,Sf); 
% Store results
SMA.sf = Sf;
SMA.ff = sFeat; 
SMA.nf = length(Sf); 
SMA.c  = curve; 
SMA.f  = feat; 
SMA.l  = label;
end



