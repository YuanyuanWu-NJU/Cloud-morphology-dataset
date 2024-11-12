function [ float_value ] = func_cal_L2( rars,temp)

    fillvalue = double(cell2mat(temp(2)));

    rars(rars==fillvalue) = nan;

    scale_factor = double(cell2mat(temp(5)));
    add_offset =  double(cell2mat(temp(6)));

    float_value = scale_factor.*(rars - add_offset);

end