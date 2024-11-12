function [lat_values, lon_values] = func_lat_lon_values(lat, lon)
    lon_min = floor(min(lon(:)));
    lon_max = ceil(max(lon(:)));

    if lon_max - lon_min > 300
        % If lon contains the 180th meridian
        lon(lon < 0) = lon(lon < 0) + 360;

        % Extract the first and last values from the first column and last column
        lon_left_1 = lon(1, 1);
        lon_left_end = lon(end, 1);
        lon_right_1 = lon(1, end);
        lon_right_end = lon(end, end);

        % Sort and get the middle two values
        sorted_values = sort([lon_left_1, lon_left_end, lon_right_1, lon_right_end]);
        lon_min = floor(sorted_values(2)) + 1;  % Smaller value
        lon_max = ceil(sorted_values(3)) - 1;   % Larger value

        % Extract the first and last values from the first row and last row of lat
        lat_up_1 = lat(1, 1);
        lat_up_end = lat(1, end);
        lat_bottom_1 = lat(end, 1);
        lat_bottom_end = lat(end, end);

        % Sort and get the middle two values
        sorted_values = sort([lat_up_1, lat_up_end, lat_bottom_1, lat_bottom_end]);
        lat_min = floor(sorted_values(2)) + 1;  % Smaller value
        lat_max = ceil(sorted_values(3)) - 1;   % Larger value

        nlat = lat_max - lat_min;
        nlon = lon_max - lon_min;

        % Generate 128*nlat by 128*nlon pixel points within the latitude and longitude range
        lat_values = linspace(lat_min, lat_max, 128 * nlat);
        lon_values = linspace(lon_min, lon_max, 128 * nlon);

        % Adjust values for wrap-around at 180 degrees
        lon(lon > 180) = lon(lon > 180) - 360;
        lon_values(lon_values > 180) = lon_values(lon_values > 180) - 360;

    else
        % Similar logic for the case when lon does not contain the 180th meridian
        lon_left_1 = lon(1, 1);
        lon_left_end = lon(end, 1);
        lon_right_1 = lon(1, end);
        lon_right_end = lon(end, end);

        sorted_values = sort([lon_left_1, lon_left_end, lon_right_1, lon_right_end]);
        lon_min = floor(sorted_values(2)) + 1;
        lon_max = ceil(sorted_values(3)) - 1;

        lat_up_1 = lat(1, 1);
        lat_up_end = lat(1, end);
        lat_bottom_1 = lat(end, 1);
        lat_bottom_end = lat(end, end);

        sorted_values = sort([lat_up_1, lat_up_end, lat_bottom_1, lat_bottom_end]);
        lat_min = floor(sorted_values(2)) + 1;
        lat_max = ceil(sorted_values(3)) - 1;

        nlat = lat_max - lat_min;
        nlon = lon_max - lon_min;

        lat_values = linspace(lat_min, lat_max, 128 * nlat);
        lon_values = linspace(lon_min, lon_max, 128 * nlon);
    end
end
