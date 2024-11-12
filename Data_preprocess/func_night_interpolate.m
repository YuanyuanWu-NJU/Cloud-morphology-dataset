function [emis_29_interp, emis_31_interp, emis_32_interp, CTT_interp, CTH_interp, SenZ_interp, COT_interp, Re_interp, Cloud_mask_interp]...
         = func_night_interpolate(lon, lat, lon_grid, lat_grid, emis_29, emis_31, emis_32, CTT, CTH, SenZ, COT, Re, Cloud_mask)


    P = [lon(:), lat(:)];
    [P_unique, iP, ~] = unique(P, 'rows');
    
    v = emis_29(:);
    v = v(iP);
    F = scatteredInterpolant(P_unique, v);
    F2 = scatteredInterpolant(P_unique, v, "nearest");
    emis_29_interp = F(lon_grid, lat_grid);

    F.Values = emis_31(:);
    F.Values = F.Values(iP);
    emis_31_interp = F(lon_grid, lat_grid);

    F.Values = emis_32(:);
    F.Values = F.Values(iP);
    emis_32_interp = F(lon_grid, lat_grid);

    F.Values = CTT(:);
    F.Values = F.Values(iP);
    CTT_interp = F(lon_grid, lat_grid);

    F.Values = CTH(:);
    F.Values = F.Values(iP);
    CTH_interp = F(lon_grid, lat_grid);

    F.Values = SenZ(:);
    F.Values = F.Values(iP);
    SenZ_interp = F(lon_grid, lat_grid);


    F.Values = Re(:);
    F.Values = F.Values(iP);
    Re_interp = F(lon_grid, lat_grid);

    % nanIndices = isnan(COT);
    % COT(nanIndices) = 0;
    F.Values = COT(:);
    F.Values = F.Values(iP);
    COT_interp = F(lon_grid, lat_grid);


    F2.Values = Cloud_mask(:);
    F2.Values = F2.Values(iP);
    Cloud_mask_interp = F2(lon_grid, lat_grid);
end
