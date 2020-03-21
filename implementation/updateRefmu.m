function [ref_mu,occ]= updateRefmu(response_diff,init_mu,p,frame)
   
        [ref_mu,occ]=varphiFunction(response_diff,init_mu,p,frame);
end
function [y,occ]=varphiFunction(response_diff,init_mu,p,frame)
%         global eta_list;
        phi=0.3; %0.3
        m=init_mu;
        eta=norm(response_diff,2)/1e4;
%         eta_list(frame)=eta;
        if eta<phi
            y=m/(1+log(p*eta+1));
            occ=false;
        else
            y=50;
            occ=true;
        end
end