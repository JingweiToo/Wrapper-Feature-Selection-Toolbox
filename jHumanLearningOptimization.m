%[2015]-"A human learning optimization algorithm and its application 
%to multi-dimensional knapsack problems"

% (9/12/2020)

function HLO = jHumanLearningOptimization(feat,label,opts)
% Parameters
pi = 0.85;   % probability of individual learning
pr = 0.1;    % probability of exploration learning

if isfield(opts,'N'), N = opts.N; end
if isfield(opts,'T'), max_Iter = opts.T; end
if isfield(opts,'pi'), pi = opts.pi; end 
if isfield(opts,'pr'), pr = opts.pr; end

% Objective function
fun = @jFitnessFunction; 
% Number of dimensions
dim = size(feat,2);
% Initial 
X   = jInitialPopulation(N,dim); 
% Fitness 
fit    = zeros(1,N);
fitSKD = inf;
for i = 1:N
  fit(i) = fun(feat,label,X(i,:),opts);
  % Update SKD/gbest
  if fit(i) < fitSKD
    fitSKD = fit(i); 
    SKD    = X(i,:);
  end
end
% Get IKD/pbest
fitIKD = fit; 
IKD    = X;
% Pre
curve = zeros(1,max_Iter);
curve(1) = fitSKD;
t = 2; 
% Generations
while t <= max_Iter
  for i = 1:N
    % Update solution (8)
    for d = 1:dim
      % Radom probability in [0,1]
      r = rand();
      if r >= 0 && r < pr
        % Random exploration learning operator (7)
        if rand() < 0.5
          X(i,d) = 0;
        else
          X(i,d) = 1;
        end
      elseif r >= pr && r < pi
        X(i,d) = IKD(i,d);
      else
        X(i,d) = SKD(d);
      end
    end
  end
  % Fitness
  for i = 1:N
    % Fitness
    fit(i) = fun(feat,label,X(i,:),opts);
    % Update IKD/pbest
    if fit(i) < fitIKD(i)
      fitIKD(i) = fit(i);
      IKD(i,:)  = X(i,:);
    end
    % Update SKD/gbest
    if fitIKD(i) < fitSKD
      fitSKD = fitIKD(i);
      SKD    = IKD(i,:);
    end
  end
  curve(t) = fitSKD;
  fprintf('\nGeneration %d Best (HLO)= %f',t,curve(t))
  t = t + 1;
end
% Select features based on selected index
Pos   = 1:dim;
Sf    = Pos(SKD == 1); 
sFeat = feat(:,Sf); 
% Store results
HLO.sf = Sf; 
HLO.ff = sFeat;
HLO.nf = length(Sf);
HLO.c  = curve; 
HLO.f  = feat; 
HLO.l  = label;
end


% Binary initialization strategy
function X = jInitialPopulation(N,dim)
X = zeros(N,dim);
for i = 1:N
  for d = 1:dim
    if rand() > 0.5
      X(i,d) = 1;
    end
  end
end
end


    
