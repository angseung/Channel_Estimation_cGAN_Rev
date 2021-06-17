clear; clc; close all;

D1 = load("Gan_Comb\Comb_3_12_25_rev3.mat");
D2 = load("Gan_Comb\Gan_10_dB_25_path_Indoor2p5_64ant_32users_8pilot_r7.mat");

input_da_1 = D1.input_da;
input_da_2 = D2.input_da;

input_da_test_1 = D1.input_da_test;
input_da_test_2 = D2.input_da_test;

output_da_1 = D1.output_da;
output_da_2 = D2.output_da;

output_da_1_test = D1.output_da_test;
output_da_2_test = D2.output_da_test;

input_da = cat(1, input_da_1, input_da_2);
input_da_test = cat(1, input_da_test_1, input_da_test_2);

output_da = cat(1, output_da_1, output_da_2);
output_da_test = cat(1, output_da_1_test, output_da_2_test);

save("Gan_Comb\Comb_3_12_25_rev3",'input_da','output_da','input_da_test','output_da_test','-v7.3');