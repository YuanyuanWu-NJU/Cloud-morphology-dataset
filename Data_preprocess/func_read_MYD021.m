function [refl_01,refl_03,refl_04,emis_29,emis_31,emis_32,emis_33,emis_34,emis_35,...
    SenZ,SenA,SolZ,SolA,lat1,lon1] ...
    = func_read_MYD021(fileNameMYD021)

    % calibrated radiance data from MYD021KM
    EV_250_Aggr1km_RefSB = hdfread(fileNameMYD021, 'EV_250_Aggr1km_RefSB');
    EV_500_Aggr1km_RefSB = hdfread(fileNameMYD021, 'EV_500_Aggr1km_RefSB');
    EV_1KM_Emissive = hdfread(fileNameMYD021, 'EV_1KM_Emissive');
    
    %从info文件中提取radiance_scales和radiance_offsets
    info = hdfinfo(fileNameMYD021);%读取hdf文件信息
    % info250m = {info.Vgroup.Vgroup(2).SDS(5).Attributes.Name};
    info250mv = {info.Vgroup.Vgroup(2).SDS(5).Attributes.Value};
    fillValue_250m = double(cell2mat(info250mv(4)));
    reflectance_scales_250m = double(cell2mat(info250mv(9)));
    reflectance_offsets_250m = double(cell2mat(info250mv(10)));
    % info500m = {info.Vgroup.Vgroup(2).SDS(8).Attributes.Name};
    info500mv = {info.Vgroup.Vgroup(2).SDS(8).Attributes.Value};
    fillValue_500m = double(cell2mat(info500mv(4)));
    radiance_scales_500m = double(cell2mat(info500mv(9)));
    radiance_offsets_500m = double(cell2mat(info500mv(10)));
    %EV_1KM_Emissive,band 20 to band 36
    % info1000m = {info.Vgroup.Vgroup(2).SDS(3).Attributes.Name};
    info1000mv = {info.Vgroup.Vgroup(2).SDS(3).Attributes.Value};
    fillValue_1000m = double(cell2mat(info1000mv(4)));
    radiance_scales_1000m = double(cell2mat(info1000mv(6)));
    radiance_offsets_1000m = double(cell2mat(info1000mv(7)));
    
    refl_01 = double(EV_250_Aggr1km_RefSB(1,:,:)); %band1: 0.65um, for cloud boundaries
    refl_01(refl_01==fillValue_250m) = nan;
    refl_01 = reflectance_scales_250m(1).*(squeeze(refl_01) - reflectance_offsets_250m(1));
    refl_03 = double(EV_500_Aggr1km_RefSB(1,:,:)); %band3: 0.47um, for cloud properties
    refl_03(refl_03==fillValue_500m) = nan;
    refl_03 = radiance_scales_500m(1).*(squeeze(refl_03) - radiance_offsets_500m(1));
    refl_04 = double(EV_500_Aggr1km_RefSB(2,:,:)); %band4: 0.55um, for cloud properties
    refl_04(refl_04==fillValue_500m) = nan;
    refl_04 = radiance_scales_500m(2).*(squeeze(refl_04) - radiance_offsets_500m(2));
    % clearvars EV_250_Aggr1km_RefSB EV_500_Aggr1km_RefSB 
    
    emis_29 = double(EV_1KM_Emissive(9,:,:)); %band29: 8.55um, for cloud properties
    emis_29(emis_29==fillValue_1000m) = nan;
    emis_29 = radiance_scales_1000m(9).*(squeeze(emis_29) - radiance_offsets_1000m(9));
    emis_31 = double(EV_1KM_Emissive(11,:,:)); %band31: 11.0um, for cloud temperature
    emis_31(emis_31==fillValue_1000m) = nan;
    emis_31 = radiance_scales_1000m(11).*(squeeze(emis_31) - radiance_offsets_1000m(11));
    emis_32 = double(EV_1KM_Emissive(12,:,:)); 
    emis_32(emis_32==fillValue_1000m) = nan;
    emis_32 = radiance_scales_1000m(12).*(squeeze(emis_32) - radiance_offsets_1000m(12));

    emis_33 = double(EV_1KM_Emissive(13,:,:)); 
    emis_33(emis_33==fillValue_1000m) = nan;
    emis_33 = radiance_scales_1000m(13).*(squeeze(emis_33) - radiance_offsets_1000m(13));
    emis_34 = double(EV_1KM_Emissive(14,:,:)); 
    emis_34(emis_34==fillValue_1000m) = nan;
    emis_34 = radiance_scales_1000m(14).*(squeeze(emis_34) - radiance_offsets_1000m(14));
    emis_35 = double(EV_1KM_Emissive(15,:,:)); 
    emis_35(emis_35==fillValue_1000m) = nan;
    emis_35 = radiance_scales_1000m(15).*(squeeze(emis_35) - radiance_offsets_1000m(15));
    % clearvars EV_1KM_Emissive
    
    [row,col] = size(refl_01);
    
    % SensorZenith, SensorAzimuth, SolarZenith, SolarAzimuth,
    SensorZenith = double(hdfread(fileNameMYD021,'SensorZenith'));
    infoSenZ = {info.Vgroup.Vgroup(2).SDS(12).Attributes.Value};
    SenZ_fillValue = double(cell2mat(infoSenZ(3)));
    SensorZenith(SensorZenith==SenZ_fillValue) = nan;
    SenZ_scales = double(cell2mat(infoSenZ(4)));
    SensorZenith = SenZ_scales*SensorZenith;
    SenZ = imresize(SensorZenith,[row col],'bilinear');
    
    SensorAzimuth = double(hdfread(fileNameMYD021,'SensorAzimuth'));
    infoSenA = {info.Vgroup.Vgroup(2).SDS(13).Attributes.Value};
    SenA_fillValue = double(cell2mat(infoSenA(3)));
    SensorAzimuth(SensorAzimuth==SenA_fillValue) = nan;
    SenA_scales = double(cell2mat(infoSenA(4)));
    SensorAzimuth = SenA_scales*SensorAzimuth;
    SenA = imresize(SensorAzimuth,[row col],'bilinear');
    
    SolarZenith = double(hdfread(fileNameMYD021,'SolarZenith'));
    infoSolZ = {info.Vgroup.Vgroup(2).SDS(15).Attributes.Value};
    SolZ_fillValue = double(cell2mat(infoSolZ(3)));
    SolarZenith(SolarZenith==SolZ_fillValue) = nan;
    SolZ_scales = double(cell2mat(infoSolZ(4)));
    SolarZenith = SolZ_scales*SolarZenith;
    SolZ = imresize(SolarZenith,[row col],'bilinear');
    
    SolarAzimuth = double(hdfread(fileNameMYD021,'SolarAzimuth'));
    infoSolA = {info.Vgroup.Vgroup(2).SDS(16).Attributes.Value};
    SolA_fillValue = double(cell2mat(infoSolA(3)));
    SolarAzimuth(SolarAzimuth==SolA_fillValue) = nan;
    SolA_scales = double(cell2mat(infoSolA(4)));
    SolarAzimuth = SolA_scales*SolarAzimuth;
    SolA = imresize(SolarAzimuth,[row col],'bilinear');
    
    %----consider that this scene may contain 180 longitude----
    lat_MOD021 = hdfread(fileNameMYD021,'Latitude');
    lon_MOD021 = hdfread(fileNameMYD021,'Longitude');
    
    templon = lon_MOD021;
    templat = lat_MOD021;
    
    lon_max = max(templon,[],'all');
    lon_min = min(templon,[],'all');
    if lon_max-lon_min>300
        templon(templon<0) = templon(templon<0) + 360;
    end
    lat1 = imresize(templat,[row col],'bilinear');
    lon1 = imresize(templon,[row col],'bilinear');
    lon1(lon1>180) = lon1(lon1>180)-360;
    % clearvars templat templon
    
    return
end