% Fitness Function KNN (9/12/2020)

function cost = jFitnessFunction(feat,label,X,opts)
% Default of [alpha; beta]
ws = [0.99; 0.01];

if isfield(opts,'ws'), ws = opts.ws; end

% Check if any feature exist
if sum(X == 1) == 0
  cost = 1;
else
  % Error rate
  error    = jwrapper_KNN(feat(:,X == 1),label,opts);
  % Number of selected features
  num_feat = sum(X == 1);
  % Total number of features
  max_feat = length(X); 
  % Set alpha & beta
  alpha    = ws(1); 
  beta     = ws(2);
  % Cost function 
  cost     = alpha * error + beta * (num_feat / max_feat); 
end
end


%---Call Functions-----------------------------------------------------
function error = jwrapper_KNN(sFeat,label,opts)
if isfield(opts,'k'), k = opts.k; end
if isfield(opts,'Model'), Model = opts.Model; end

% Define training & validation sets
trainIdx = Model.training;    testIdx = Model.test;
xtrain   = sFeat(trainIdx,:); ytrain  = label(trainIdx);
xvalid   = sFeat(testIdx,:);  yvalid  = label(testIdx);
% Training model
My_Model = fitcknn(xtrain,ytrain,'NumNeighbors',k); 
% Prediction
pred     = predict(My_Model,xvalid);
% Accuracy
Acc      = sum(pred == yvalid) / length(yvalid);
% Error rate
error    = 1 - Acc; 
end












