clear;clc;close all; 
workPath = './1-degree-data';
addpath(workPath);

outPath = fullfile(workPath,'day_1d_production');
if ~exist(outPath,'dir')
    mkdir(outPath);
end

outputVarName = {'lat_center','lon_center',...
    'SenZ_mean','Ice_Fraction','High_Fraction','Cloud_Fraction','clear_sky','LWP_ave','Re_ave','COT_ave',...
    'emis_29_CB','emis_31_CB','emis_32_CB','lat_128','lon_128'};

inputVarName = {'lon', 'lat',...
    'emis_29','emis_31','emis_32',...
    'SenZ','CTT','CTH',...
    'COT','Re','Cloud_mask','LWP','COTave'};


interp = {'lon_grid', 'lat_grid',...
    'emis_29_interp','emis_31_interp','emis_32_interp',...
    'SenZ_interp','CTT_interp','CTH_interp',...
    'COT_interp','Re_interp','Cloud_mask_interp','LWP_interp','COT_jisuan'};


%% MODIS data path      
    year = '2021';

    sourcePathMYD021    = fullfile('./MYD021',year);
    fileListMYD021      = dir(fullfile(sourcePathMYD021,'**/*.hdf'));


    sourcePathMYD06     = fullfile('./MYD06',year);
    fileListMYD06       = dir(fullfile(sourcePathMYD06,'**/*.hdf'));

 
    sourcePathCOT     = fullfile('./Retrieved_COT_CER',year);
    fileListCOT       = dir(fullfile(sourcePathCOT,'**/*.h5'));



    COT_substr_list = {};
    for idx = 1:length(fileListCOT)
        COT_substr = fileListCOT(idx).name(7:18);
        disp(COT_substr);
        COT_substr_list{end+1} = COT_substr;
    end



    if length(fileListMYD021)~=length(fileListMYD06)
        error('Error: Check number of files of MODIS L1/L2.')
    end

    fprintf('Total number of files: %d\n', length(fileListMYD021));


    %% Process
    for kk = 1:length(fileListMYD021)
        tic;
        fprintf('start with file %d \n', kk); 
        MYD02_substr = fileListMYD021(kk).name(11:22);
        matching_index = find(strcmp(COT_substr_list, MYD02_substr));

        if isempty(matching_index)
            continue; 
        end

        COT_file = fileListCOT(matching_index);
        fileNameCOT = fullfile(COT_file.folder, COT_file.name);
        disp(fileNameCOT);

        %% check the name of MYD021 and MYD06 is the same
        if ~strcmp(fileListMYD021(kk).name(11:22),COT_file.name(7:18))
           error('Error: The order of MODIS L1 and cot is different, please check!');
        end 

        %% read MODIS data
        fileNameMYD021 = fullfile(fileListMYD021(kk).folder, fileListMYD021(kk).name);
        fileNameMYD06 = fullfile(fileListMYD06(kk).folder, fileListMYD06(kk).name);
                
        [~,~,~,emis_29,emis_31,emis_32,~,~,~,SenZ,~,~,~,lat,lon]...
        = func_read_MYD021(fileNameMYD021);

        if any(isnan(emis_29(:)))
           continue;
        end       
        
        if any(lat(:) == -999) || any(lon(:) == -639)
            continue;
        end


        if abs(mean(lat(:))) > 60
            disp(mean(lat(:)));
            continue;
        end


        [CTT,CTH,~,~,~,Cloud_mask,~,~]...
        = func_read_MYD06(fileNameMYD06);


        COT = h5read(fileNameCOT, '/CNN_COT');
        COT = COT';

        sizeCOT = size(COT);
        sizeLat = size(lat);
        
        if ~isequal(sizeCOT, sizeLat)
            disp(sizeCOT);
            error('The sizes of COT and lat are different!');
        end

        Re = h5read(fileNameCOT, '/CNN_CER');
        Re = Re';
        Re(Re < 1) = NaN;

        sizeRe = size(Re);
        if ~isequal(sizeRe, sizeLat)
            disp(sizeRe);
            error('The sizes of Re and lat are different!');
        end

        %% Interpolate the whole granuel into grided window 
        tempPath = fullfile(outPath,fileListMYD021(kk).name(11:14),fileListMYD021(kk).name(15:17));
        if ~exist(tempPath,'dir')
            mkdir(tempPath);
        end


        [lat_values, lon_values] = func_lat_lon_values(lat, lon);

        [lon_grid, lat_grid] = meshgrid(lon_values, lat_values);

        lon_grid = double(lon_grid);
        lat_grid = double(lat_grid);

        for k = 1:length(inputVarName)-2
            evalStr = [inputVarName{k},'= double(', inputVarName{k},');'];
            eval(evalStr);
        end
        
        [emis_29_interp, emis_31_interp, emis_32_interp, CTT_interp, CTH_interp, SenZ_interp, COT_interp, Re_interp, Cloud_mask_interp]...
         = func_night_interpolate(lon, lat, lon_grid, lat_grid, emis_29, emis_31, emis_32, CTT, CTH, SenZ, COT, Re, Cloud_mask);

        % LWP_interp
        COT_jisuan = COT_interp;
        COT_jisuan(COT_jisuan <= 0.1) = NaN;

        LWP_interp = (2/3) * COT_jisuan .* Re_interp;

        %% Process every 1-degree grid
        windowSize = 128;
        ibin = 1:128:length(lat_values);
        jbin = 1:128:length(lon_values);


        for k = 1:length(outputVarName)-5
            evalStr = [outputVarName{k},' = nan(length(ibin)*length(jbin),1);'];
            eval(evalStr) 
        end

        lat_128 = nan(128,128,length(ibin)*length(jbin));
        lon_128 = nan(128,128,length(ibin)*length(jbin));
        emis_29_CB = nan(128,128,length(ibin)*length(jbin));
        emis_31_CB = nan(128,128,length(ibin)*length(jbin));
        emis_32_CB = nan(128,128,length(ibin)*length(jbin));
        COT_CB = nan(128,128,length(ibin)*length(jbin));


        l=0;
        for i = 1:length(ibin)
            for j = 1:length(jbin)
                l = l+1;

                for k = 1:length(inputVarName)
                    evalStr = [inputVarName{k},'_wz = ',interp{k},...
                            '(ibin(i):ibin(i)+127,jbin(j):jbin(j)+127);'];
                    eval(evalStr) 
                end


                % Sensor zenith angle 
                SenZ_mean(l) =  mean(SenZ_wz(:),'omitnan');


                % cloud fraction: CF
                Cloud_Fraction(l) = sum(Cloud_mask_wz(:),'omitnan')/length(Cloud_mask_wz(:));


                ind_CF = find(Cloud_mask_wz(:)==1);
                if isempty(ind_CF)
                    clear_sky(l) = 1;
                    Ice_Fraction(l) = 0;
                    High_Fraction(l) = 0;

                else
                    clear_sky(l) = 0;

                    % ice/mixed phase clouds
                    ind_ice = find(CTT_wz<=273.15);
                    Ice_Fraction(l) = length(ind_ice)/length(ind_CF);


                    % high cloud
                    ind_high = find(CTH_wz>6000);
                    High_Fraction(l) = length(ind_high)/length(ind_CF);
                end


                lat_128(:,:,l) = lat_wz;
                lon_128(:,:,l) = lon_wz;
                lat_center(l) = mean(lat_wz(:));
                lon_center(l) = mean(lon_wz(:));
                emis_29_CB(:,:,l) = emis_29_wz;
                emis_31_CB(:,:,l) = emis_31_wz;
                emis_32_CB(:,:,l) = emis_32_wz;
                COT_CB(:,:,l) = COT_wz;
                LWP_ave(l) = mean(LWP_wz(:),'omitnan');
                Re_ave(l) = mean(Re_wz(:),'omitnan');
                COT_ave(l) = mean(COTave_wz(:),'omitnan');
            end
        end



            SenZ_mean = single(SenZ_mean);
            lat_center = single(lat_center);
            lon_center = single(lon_center);
            emis_29_CB = single(emis_29_CB);
            emis_31_CB = single(emis_31_CB);
            emis_32_CB = single(emis_32_CB);
            COT_CB = single(COT_CB);
            lat_128 = single(lat_128);
            lon_128 = single(lon_128);
            Ice_Fraction = single(Ice_Fraction);
            High_Fraction = single(High_Fraction);
            Cloud_Fraction = single(Cloud_Fraction);
            clear_sky = single(clear_sky);
            LWP = single(LWP_ave);
            Re = single(Re_ave);
            COT = single(COT_ave);


            tempName = [fileListMYD021(kk).name(1:end-4),'.mat'];
            outName = fullfile(tempPath,tempName);
            save(outName,'SenZ_mean','lat_center','lon_center',...
                'COT_CB','Cloud_Fraction',...
                'LWP','Re','COT',...
                'emis_29_CB','clear_sky', ...
                'emis_31_CB','emis_32_CB','lat_128','lon_128','Ice_Fraction','High_Fraction','-v7.3');

        time_elapsed_outer = toc;
        fprintf(kk, time_elapsed_outer); 
    end
