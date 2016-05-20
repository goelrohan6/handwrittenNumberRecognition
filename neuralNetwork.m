%load training and testing data using functions loadMNISTImages and loadMNISTLabels

trainingImages = loadMNISTImages('train-images.idx3-ubyte')';
trainingLabels = loadMNISTLabels('train-labels.idx1-ubyte');
testingImages = loadMNISTImages('t10k-images.idx3-ubyte')';
testingLabels = loadMNISTLabels('t10k-labels.idx1-ubyte');

alpha = 0.1; %learning rate
trainigDataDimenisons = size(trainingImages);
inputSize = trainigDataDimenisons(1); %number of images i.e 60,000
numberInputLayers = trainigDataDimenisons(2); %784 as the image size is 28*28 px
numberHiddenLayers = 300;
numberOutputLayers = 10;

theta1 = 2*rand(numberInputLayers+1,numberHiddenLayers)-1; % randomise weight matrix for layer 1 to layer 2
theta2 = 2*rand(numberHiddenLayers+1,numberOutputLayers)-1; % randomise weight matrix for layer 2 to layer 3
identityMatrix = eye(numberOutputLayers);

maxIterations = 100 ; %number of iterations

for i=1:maxIterations
	for j=1:inputSize
		disp(' for iteration'),disp(i);
		% set respective input
		inputX = [1;trainingImages(j,:)'];

		% compute hidden layer output 
		hiddenOutput = [1;sigmf(theta1'*inputX, [0.1 0])];
		% compute final layer output 
		Predictedoutput = sigmf(theta2'*hiddenOutput, [0.1 0]);
		%compute errors in respective layers using back propogaion
		delta2 = (Predictedoutput-identityMatrix(:,trainingLabels(j)+1)).*Predictedoutput.*(Predictedoutput-1);
		delta1 = (theta2*delta2).*hiddenOutput.*(1-hiddenOutput);
		delta1 = delta1(2:end);

		%update weight matrix
		theta1 = theta1 - alpha*(inputX*delta1');
		theta2 = theta2 - alpha*(hiddenOutput*delta2');
	end
end