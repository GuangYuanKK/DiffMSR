clc;
clear;

mask_2 = zeros([256,256]);
mask_2(64:191,64:191)=1;
mask_4 = zeros([256,256]);
mask_4(96:159,96:159)=1;

mas_2 = logical(mask_2);
a_2 = sum(mas_2(:))/256/256;

mas_4 = logical(mask_4);
a_4 = sum(mas_4(:))/256/256;

lr_mask = mask_2;
% lr_mask = mask_4;

save(['lr_2x.mat'],'lr_mask');

