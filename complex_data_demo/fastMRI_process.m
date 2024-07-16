clc
clear
close all;

%%%% Please download the data in dicom format from the FastMRI public dataset.
path = '/Users/yymacpro13/Desktop/MRI/DuSR_data/select_knee_2/';
d1 = dir(path);
shape_1=[];
shape_2=[];

for i= [2,6,13,15,24,27,28,31,33,34,40,45,47,61,64,73,79,81,84,95]
    data_name = d1(i+3).name;
    d2 = dir(strcat(path,data_name,'/'));
    mc_name_1 = d2(3).name;
    mc_name_2 = d2(4).name;
    dir_mc_name_1 = dir(strcat(path,data_name,'/',mc_name_1,'/'));
    dir_mc_name_2 = dir(strcat(path,data_name,'/',mc_name_2,'/'));
    
    for j=length(dir_mc_name_1)-2
        
        pattern_g = '.dcm';
        str_1 = regexp(dir_mc_name_1(j+2).name, pattern_g, 'split');
        str_2 = regexp(dir_mc_name_2(j+2).name, pattern_g, 'split');
        
        dir_mc_1 = dicominfo(strcat(path,data_name,'/',mc_name_1,'/',str_1{1,1},''));
        dir_mc_2 = dicominfo(strcat(path,data_name,'/',mc_name_2,'/',str_2{1,1},''));
        
        shape_1(j)=dir_mc_1.Width;
        shape_2(j)=dir_mc_2.Width;
        disp(dir_mc_1.ScanOptions);
        disp(dir_mc_2.ScanOptions);
        
        if (strcmp(dir_mc_1.ScanOptions,'FS'))
            T2 = dicomread(dir_mc_1);
            T1 = dicomread(dir_mc_2);
        else
            T2 = dicomread(dir_mc_2);
            T1 = dicomread(dir_mc_1);
        end
        
        if(dir_mc_1.Width==320)
            T1 = T1/max(T1(:));
            T1_ks=fft2c(T1);
            %---------T1 256
            T1_256_ks = T1_ks(32:287,32:287);
            T1_256_im = ifft2c(T1_256_ks);
            %---------T1 128
            k_T1_128_lr = T1_256_ks(64:191,64:191,:);
            im_T1_128_lr = ifft2c(k_T1_128_lr);
            %---------T1 64
            k_T1_64_lr = k_T1_128_lr(32:95,32:95,:);
            im_T1_64_lr = ifft2c(k_T1_64_lr);
            %%%%%%%%%%%%%%%%%%%%%%%
            T2 = T2/max(T2(:));
            T2_ks = fft2c(T2);
            %---------T2 256
            T2_256_ks = T2_ks(32:287,32:287);
            T2_256_im = ifft2c(T2_256_ks);
            %---------T2 128
            k_T2_128_lr = T2_256_ks(64:191,64:191,:);
            im_T2_128_lr = ifft2c(k_T2_128_lr);
            %---------T2 64
            k_T2_64_lr = k_T2_128_lr(32:95,32:95,:);
            im_T2_64_lr = ifft2c(k_T2_64_lr);
            
            T1 = T1_256_im;
            T1_128 = im_T1_128_lr;
            T1_64 = im_T1_64_lr;
            
            T2 = T2_256_im;
            T2_128 = im_T2_128_lr;
            T2_64 = im_T2_64_lr;
            
            mkdir(strcat('mc_knee/valid/'));
            k = j-10;
            if (j-10<10)
                save(strcat('mc_knee/valid/',data_name,'_0',num2str(k),'.mat'),"T1","T1_128","T1_64","T2","T2_128","T2_64");   
            else
                save(strcat('mc_knee/valid/',data_name,'_',num2str(k),'.mat'),"T1","T1_128","T1_64","T2","T2_128","T2_64");   
            end
            
        end
        
        
        
    end
%     as(T1);
%     as(T2);

    
    
    
end
