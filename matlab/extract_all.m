methodName = 'BSIF'; % BSIF LPQ WLD
collectorName = 'DigPerson'; % Bio DigPerson Ita Sag
train_dir = '..\data-livdet-2015\Training_augmented';
test_dir = '..\data-livdet-2015\Testing_augmented';

switch (methodName)
    case 'BSIF'
        load('./ICAtextureFilters_9x9_12bit', 'ICAtextureFilters');
        n_features = 4096;
    case 'LPQ'
        n_features = 256;
    case 'WLD'
        n_features = 960;
end
Value_Real_Training = zeros(1000, n_features);
Value_Real_Testing = zeros(1000, n_features);
Value_Spoof_Training = zeros(1000, n_features);
Value_Spoof_Testing = zeros(1500, n_features);


dir = strcat(train_dir, '\Digital_Persona\Live\');
files = ls(strcat(dir, '*.png'));
for i = 1:length(files)
    if mod(i, 1000) == 0
        disp(strcat('Train\Live:', num2str(i)))
    end
    f = files(i, :);
    img = imread(strcat(dir, strtrim(f)));
    img = rgb2gray(img);

    res = [];
    switch (methodName)
        case 'BSIF'
            res = bsif(img, ICAtextureFilters, 'h');
        case 'LPQ'
            res = lpq(img, 3, 1, 1, 'h');
        case 'WLD'
            res = wld(img, 3, 8).';
    end
    Value_Real_Training(i, :) = res;
end
nameTrainReal = strcat(methodName, '_7_12_motion_', 'Train_Real_', collectorName);
eval([nameTrainReal, '=Value_Real_Training;']);
save(strcat('./', methodName, '_7_12_motion_', 'Train_Real_', collectorName), nameTrainReal);
dlmwrite(strcat('./', nameTrainReal, '.txt'), Value_Real_Training, 'delimiter', '\t', 'newline', 'pc');


dir = strcat(test_dir, '\Digital_Persona\Live\');
files = ls(strcat(dir, '*.png'));
for i = 1:length(files)
    if mod(i, 1000) == 0
        disp(strcat('Test\Live:', num2str(i)))
    end
    f = files(i, :);
    img = imread(strcat(dir, strtrim(f)));
    img = rgb2gray(img);

    res = [];
    switch (methodName)
        case 'BSIF'
            res = bsif(img, ICAtextureFilters, 'h');
        case 'LPQ'
            res = lpq(img, 3, 1, 1, 'h');
        case 'WLD'
            res = wld(img, 3, 8);
    end
    Value_Real_Testing(i, :) = res;
end
nameTestReal = strcat(methodName, '_7_12_motion_', 'Test_Real_', collectorName);
eval([nameTestReal, '=Value_Real_Testing;']);
save(strcat('./', methodName, '_7_12_motion_', 'Test_Real_', collectorName), nameTestReal);
dlmwrite(strcat('./', nameTestReal, '.txt'), Value_Real_Testing, 'delimiter', '\t', 'newline', 'pc');


dir = strcat(train_dir, '\Digital_Persona\Fake\');
files = ls(strcat(dir, '*.png'));
for i = 1:length(files)
    if mod(i, 1000) == 0
        disp(strcat('Train\Fake:', num2str(i)))
    end
    f = files(i, :);
    img = imread(strcat(dir, strtrim(f)));
    img = rgb2gray(img);

    res = [];
    switch (methodName)
        case 'BSIF'
            res = bsif(img, ICAtextureFilters, 'h');
        case 'LPQ'
            res = lpq(img, 3, 1, 1, 'h');
        case 'WLD'
            res = wld(img, 3, 8);
    end
    Value_Spoof_Training(i, :) = res;
end
nameTrainSpoof = strcat(methodName, '_7_12_motion_', 'Train_Spoof_', collectorName);
eval([nameTrainSpoof, '=Value_Spoof_Training;']);
save(strcat('./', methodName, '_7_12_motion_', 'Train_Spoof_', collectorName), nameTrainSpoof);
dlmwrite(strcat('./', nameTrainSpoof, '.txt'), Value_Spoof_Training, 'delimiter', '\t', 'newline', 'pc');


dir = strcat(test_dir, '\Digital_Persona\Fake\');
files = ls(strcat(dir, '*.png'));
for i = 1:length(files)
    if mod(i, 1000) == 0
        disp(strcat('Test\Fake:', num2str(i)))
    end
    f = files(i, :);
    img = imread(strcat(dir, strtrim(f)));
    img = rgb2gray(img);

    res = [];
    switch (methodName)
        case 'BSIF'
            res = bsif(img, ICAtextureFilters, 'h');
        case 'LPQ'
            res = lpq(img, 3, 1, 1, 'h');
        case 'WLD'
            res = wld(img, 3, 8);
    end
    Value_Spoof_Testing(i, :) = res;
end
nameTestSpoof = strcat(methodName, '_7_12_motion_', 'Test_Spoof_', collectorName);
eval([nameTestSpoof, '=Value_Spoof_Testing;']);
save(strcat('./', methodName, '_7_12_motion_', 'Test_Spoof_', collectorName), nameTestSpoof);
dlmwrite(strcat('./', nameTestSpoof, '.txt'), Value_Spoof_Testing, 'delimiter', '\t', 'newline', 'pc');
