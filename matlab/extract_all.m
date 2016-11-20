methodName = 'BSIF'; % BSIF,LPQ,WLD
collectorName = 'Bio'; % Bio,Dig,Ita,Sag

switch(methodName)
    case 'BSIF'
        cd BSIF;
        % load filter
        filename=['./texturefilters/ICAtextureFilters_9x9_12bit'];
        load(filename, 'ICAtextureFilters');
        
        Value_Real_Training = zeros(1000,4096);
        Value_Real_Testing = zeros(1000,4096);
        Value_Spoof_Training = zeros(1000,4096);
        Value_Spoof_Testing = zeros(1000,4096);
    case 'LPQ'
        cd LPQ;
        Value_Real_Training = zeros(1000,256);
        Value_Real_Testing = zeros(1000,256);
        Value_Spoof_Training = zeros(1000,256);
        Value_Spoof_Testing = zeros(1000,256);
    case 'WLD'
        cd WLD
        Value_Real_Training = zeros(1000,960);
        Value_Real_Testing = zeros(1000,960);
        Value_Spoof_Training = zeros(1000,960);
        Value_Spoof_Testing = zeros(1000,960);
end

imgNum = 1;
for i=1:200
    for j = 1:5
        % load image
        if strcmp(collectorName,'Bio') == 1 ||strcmp(collectorName,'Ita') == 1
            imgRealTrain = rgb2gray(imread(strcat('../TestData/BiometrikaTrain/','Live/',num2str(i),'_',num2str(j),'.png')));
            imgRealTest = rgb2gray(imread(strcat('../TestData/BiometrikaTest/','Live/',num2str(i),'_',num2str(j),'.png')));
        else
            imgRealTrain = imread(strcat('../TestData/BiometrikaTrain/','Live/',num2str(i),'_',num2str(j),'.bmp'));
            imgRealTest = imread(strcat('../TestData/BiometrikaTest/','Live/',num2str(i),'_',num2str(j),'.bmp'));
        end
        
        % extract features
        switch(methodName)
            case 'BSIF'
                histTrainReal = bsif(imgRealTrain ,ICAtextureFilters,'h');
                histTestReal = bsif(imgRealTest,ICAtextureFilters,'h');
            case 'LPQ'
                histTrainReal = lpq(imgRealTrain,3,1,1,'h');
                histTestReal = lpq(imgRealTest,3,1,1,'h');
            case 'WLD'
                histTrainReal = wld(imgRealTrain,3,8);
                histTestReal = wld(imgRealTest,3,8);
        end
        % store the feature
        Value_Real_Training(imgNum,:)=histTrainReal;
        Value_Real_Testing(imgNum,:)=histTestReal;
        imgNum=imgNum+1;
    end
end

if strcmp(collectorName,'Bio') == 1 ||strcmp(collectorName,'Ita') == 1
    Spoof_Method = {'EcoFlex','Gelatin','Latex','Silgum','WoodGlue'};
end

imgNum = 1;
for k=1:5
    for i=1:20
        for j = 1:5
            % load image
            if strcmp(collectorName,'Bio') == 1 ||strcmp(collectorName,'Ita') == 1
                imgSpoofTrain=rgb2gray(imread(strcat('../TestData/BiometrikaTrain/','Spoof/',Spoof_Method{k},'/',num2str(i),' (',num2str(j),').png')));
                imgSpoofTest=rgb2gray(imread(strcat('../TestData/BiometrikaTest/','Spoof/',Spoof_Method{k},'/',num2str(i),' (',num2str(j),').png')));
            else
                imgSpoofTrain=imread(strcat('../TestData/BiometrikaTrain/','Spoof/',Spoof_Method{k},'/',num2str(i),' (',num2str(j),').bmp'));
                imgSpoofTest=imread(strcat('../TestData/BiometrikaTest/','Spoof/',Spoof_Method{k},'/',num2str(i),' (',num2str(j),').bmp'));
            end
            
            % extract features
            switch(methodName)
                case 'BSIF'
                    histTrainReal = bsif(imgSpoofTrain ,ICAtextureFilters,'h');
                    histTestReal = bsif(imgSpoofTest,ICAtextureFilters,'h');
                case 'LPQ'
                    histTrainReal = lpq(imgSpoofTrain,3,1,1,'h');
                    histTestReal = lpq(imgSpoofTest,3,1,1,'h');
                case 'WLD'
                    histTrainReal=WLD_new(imgSpoofTrain,3,8);
                    histTestReal=WLD_new(imgSpoofTest,3,8);
            end
            % store the feature
            Value_Spoof_Training(imgNum,:)=histTrainReal;
            Value_Spoof_Testing(imgNum,:)=histTestReal;
            imgNum=imgNum+1;
            
        end
    end
end

cd ..
nameTrainReal = strcat(methodName,'_7_12_motion_','Train_Real_',collectorName);
eval([nameTrainReal,'=Value_Real_Training;']);
save(strcat('./',methodName,'/',methodName,'_7_12_motion_','Train_Real_',collectorName),nameTrainReal);

nameTestReal = strcat(methodName,'_7_12_motion_','Test_Real_',collectorName);
eval([nameTestReal,'=Value_Real_Testing;']);
save(strcat('./',methodName,'/',methodName,'_7_12_motion_','Test_Real_',collectorName),nameTrainReal);

nameTrainSpoof = strcat(methodName,'_7_12_motion_','Train_Spoof_',collectorName);
eval([nameTrainSpoof,'=Value_Spoof_Training;']);
save(strcat('./',methodName,'/',methodName,'_7_12_motion_','Train_Spoof_',collectorName),nameTrainReal);

nameTestSpoof = strcat(methodName,'_7_12_motion_','Test_Spoof_',collectorName);
eval([nameTestSpoof,'=Value_Spoof_Testing;']);
save(strcat('./',methodName,'/',methodName,'_7_12_motion_','Test_Spoof_',collectorName),nameTrainReal);

training_data = [Value_Real_Training;Value_Spoof_Training];
testing_data = [Value_Real_Testing;Value_Spoof_Testing];
real_label = ones(1000,1);
fake_label = zeros(1000,1);
training_label = [real_label;fake_label];
testing_label = training_label;
SVM_model = svmtrain(training_data,training_label);
Predict_Real = svmclassify(SVM_model,Value_Real_Testing);
correct_Real = sum(Predict_Real);
Predict_Spoof = svmclassify(SVM_model,Value_Spoof_Testing);
correct_Spoof = 1000-sum(Predict_Spoof);
acc = (correct_Real + correct_Spoof) / 2000
