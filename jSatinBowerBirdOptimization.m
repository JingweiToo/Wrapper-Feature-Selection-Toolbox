%[2017]-"Satin bowerbird optimizer: A new optimization algorithm to 
%optimize ANFIS for software development effort estimation"

% (8/12/2020)

function SBO = jSatinBowerBirdOptimization(feat,label,opts)
% Parameters
lb    = 0;
ub    = 1; 
thres = 0.5;
alpha = 0.94;   % constant
z     = 0.02;   % constant
MR    = 0.05;   % mutation rate

if isfield(opts,'T'), max_Iter = opts.T; end
if isfield(opts,'N'), N = opts.N; end
if isfield(opts,'alpha'), alpha = opts.alpha; end 
if isfield(opts,'z'), z = opts.z; end
if isfield(opts,'MR'), MR = opts.MR; end 
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
fitE = inf;
for i = 1:N
  fit(i) = fun(feat,label,(X(i,:) > thres),opts);
  % Elite update
  if fit(i) < fitE
    fitE = fit(i); 
    Xe   = X(i,:);
  end
end
% Sigma (7)
sigma = z * (ub - lb);
% Pre
Xnew = zeros(N,dim); 
Fnew = zeros(1,N);

curve = zeros(1,max_Iter);
curve(1) = fitE;
t = 2; 
% Iterations
while t <= max_Iter
  % Calculate probability (1-2)
  Ifit = 1 ./ (1 + fit); 
  prob = Ifit / sum(Ifit); 
  for i = 1:N
    for d = 1:dim
      % Select a bower using roulette wheel 
      rw = jRouletteWheelSelection(prob);
      % Compute lambda (4)
      lambda = alpha / (1 + prob(rw));
      % Update position (3)
      Xnew(i,d) = X(i,d) + lambda * (((X(rw,d) + Xe(d)) / 2) - ...
        X(i,d));
      % Mutation
      if rand() <= MR
        % Normal distribution & Position update (5-6) 
        r_normal  = randn();
        Xnew(i,d) = X(i,d) + (sigma * r_normal);
      end
    end
    % Boundary
    XB = Xnew(i,:); XB(XB > ub) = ub; XB(XB < lb) = lb;
    Xnew(i,:) = XB;
  end
  % Fitness
  for i = 1:N
    Fnew(i) = fun(feat,label,(Xnew(i,:) > thres),opts);
  end
  % Merge & Select best N solutions
  XX        = [X; Xnew]; 
  FF        = [fit, Fnew];
  [FF, idx] = sort(FF,'ascend');
  X         = XX(idx(1:N),:);
  fit       = FF(1:N);
  % Elite update
  if fit(1) < fitE
    fitE = fit(1); 
    Xe   = X(1,:);
  end
  % Save
  curve(t) = fitE; 
  fprintf('\nIteration %d Best (SBO)= %f',t,curve(t))
  t = t + 1;
end
% Select features
Pos   = 1:dim; 
Sf    = Pos((Xe > thres) == 1); 
sFeat = feat(:,Sf);
% Store results
SBO.sf = Sf; 
SBO.ff = sFeat;
SBO.nf = length(Sf);
SBO.c  = curve; 
SBO.f  = feat; 
SBO.l  = label;
end


%// Roulette Wheel Selection //
function Index = jRouletteWheelSelection(prob)
% Cummulative summation 
C = cumsum(prob);
% Random one value, most probability value [0~1]
P = rand();
% Route wheel
for i = 1:length(C)
	if C(i) > P
    Index = i;
    break;
  end
end
end



