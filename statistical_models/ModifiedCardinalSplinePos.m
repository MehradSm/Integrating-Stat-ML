function [spline] = ModifiedCardinalSplinePos(data,cpt,s)

spline = zeros(length(data),length(cpt));

for i=1:length(data)  
    if data(i)==cpt(1)
        nearest_c_pt_index = find(cpt==data(i));
    else   
        nearest_c_pt_index = max(find(cpt<data(i)));     
    end
    nearest_c_pt_time = cpt(nearest_c_pt_index);
    next_c_pt_time = cpt(nearest_c_pt_index+1);
    u = (data(i)-nearest_c_pt_time)./(next_c_pt_time-nearest_c_pt_time);  
    lb = (cpt(3) - cpt(1))/(cpt(2)-cpt(1));
    le = (cpt(end) - cpt(end-2))/(cpt(end) - cpt(end-1));
    % Beginning knot
    if nearest_c_pt_time==cpt(1) 
           p=[u^3 u^2 u 1]*[2-(s/lb) -2 s/lb;(s/lb)-3 3 -s/lb;0 0 0;1 0 0];
           spline(i,nearest_c_pt_index:nearest_c_pt_index+2) = p; 
     % End knot
    elseif nearest_c_pt_time==cpt(end-1) 
           p=[u^3 u^2 u 1]*[-s/le 2 -2+(s/le);2*s/le -3 3-(2*s/le);-s/le 0 s/le;0 1 0];
           spline(i,nearest_c_pt_index-1:nearest_c_pt_index+1) = p;
     % Interior knots      
    else
           privious_c_pt = cpt(nearest_c_pt_index-1);
           next2 = cpt(nearest_c_pt_index+2);
           l1 = next_c_pt_time - privious_c_pt;
           l2 = next2 - nearest_c_pt_time;
           p=[u^3 u^2 u 1]*[-s/l1 2-(s/l2) (s/l1)-2 s/l2;2*s/l1 (s/l2)-3 3-2*(s/l1) -s/l2;-s/l1 0 s/l1 0;0 1 0 0];
           spline(i,nearest_c_pt_index-1:nearest_c_pt_index+2) = p; 
    end
end

end

