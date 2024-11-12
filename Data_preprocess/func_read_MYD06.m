function [CTT,CTH,CLWP,Re,COT,Cloud_mask,...
    Cloud_Phase_Optical_Properties,Cloud_Multi_Layer_Flag] ...
    = func_read_MYD06(fileNameMYD06)

    % CTH, CLWP, Re21, Re37, COT from MYD06
    fileinfo_L2 = hdfinfo(fileNameMYD06);
    
    temp = {fileinfo_L2.Vgroup.Vgroup(2).SDS(59).Attributes.Value};
    CTT = double(hdfread(fileNameMYD06,'cloud_top_temperature_1km'));
    CTT = func_cal_L2( CTT,temp);
    
    temp = {fileinfo_L2.Vgroup.Vgroup(2).SDS(58).Attributes.Value};
    CTH = double(hdfread(fileNameMYD06,'cloud_top_height_1km'));
    CTH = func_cal_L2( CTH,temp);
    
    temp = {fileinfo_L2.Vgroup.Vgroup(2).SDS(83).Attributes.Value};
    CLWP = double(hdfread(fileNameMYD06,'Cloud_Water_Path'));
    CLWP = func_cal_L2( CLWP,temp);
    
    temp = {fileinfo_L2.Vgroup.Vgroup(2).SDS(67).Attributes.Value};
    Re = double(hdfread(fileNameMYD06,'Cloud_Effective_Radius'));
    Re = func_cal_L2( Re,temp);
    
    temp = {fileinfo_L2.Vgroup.Vgroup(2).SDS(73).Attributes.Value};
    COT = double(hdfread(fileNameMYD06,'Cloud_Optical_Thickness'));
    COT = func_cal_L2( COT,temp);
    
    % Cloud_Phase_Optical_Properties and Cloud_Multi_Layer_Flag from MYD06
    temp = {fileinfo_L2.Vgroup.Vgroup(2).SDS(105).Attributes.Value};
    Cloud_Phase_Optical_Properties = double(hdfread(fileNameMYD06,'Cloud_Phase_Optical_Properties'));
    Cloud_Phase_Optical_Properties = func_cal_L2( Cloud_Phase_Optical_Properties,temp);
   
    temp = {fileinfo_L2.Vgroup.Vgroup(2).SDS(106).Attributes.Value};
    Cloud_Multi_Layer_Flag = double(hdfread(fileNameMYD06,'Cloud_Multi_Layer_Flag'));
    Cloud_Multi_Layer_Flag = func_cal_L2( Cloud_Multi_Layer_Flag,temp);
    
    % Cloud mask from MYD06
    Cloud_Mask_1km = hdfread(fileNameMYD06,'Cloud_Mask_1km');
    Cloud_Mask_1km1 = squeeze(Cloud_Mask_1km(:,:,1));
    cm=int16(Cloud_Mask_1km1) ;
    flag_1=bitand(int16(1),cm);
    cm2=bitshift(cm,-1);

    flag_2=bitand(int16(3),cm2);
    
    Cloud_mask = zeros(size(Cloud_Mask_1km,1),size(Cloud_Mask_1km,2));
    % Cloud_mask_p = zeros(size(Cloud_Mask_1km,1),size(Cloud_Mask_1km,2));
    % 1-Determined,0-cloudy, flag_1
    % index1= flag_1 ==1 & (flag_2 == 0 | flag_2 == 1); % CF_L2, 5km
    % 0-Confident Cloudy, flag_2
    
    index1= flag_1 ==1 & flag_2 == 0 ;
    Cloud_mask(index1) = 1;
    
    % indexp= flag_1 ==1 & (flag_2 == 0 | flag_2 == 1);
    % Cloud_mask_p(indexp) = 1;
    
    return
end