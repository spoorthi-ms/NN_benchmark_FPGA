////////////////////////////////////////////////////////////////////////////////////////
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void __attribute__ ((reqd_work_group_size(32,32,1))) fL(__global int *Layer1_Weights_CPU, __global int *Data_Layer_CPU_R, __global int *Data_Layer_CPU_G, __global int *Data_Layer_CPU_B, __global double *Layer1_Features)
		{
	int x = get_global_id(0);
	int y = get_global_id(1);

	for(int f=0; f<32; f++)
	{
		double result = 0;

		for(int i = x-2; i<=x+2; i++)
		{
			for(int j=y-2; j<=y+2; j++)
			{
				int x_index = i-x+2;
				int y_index = j-y+2;
				int m = (y_index)+(x_index)*5;
				if(i<0 || j<0)
				{
					result+= 0;
				}
				else if(j>31 || i>31)
				{
					result+= 0;
				}
				else
				{
					result += Data_Layer_CPU_R[(y_index-2) + x*32 + y + (x_index-2)*32]*Layer1_Weights_CPU[m+f*75] + Data_Layer_CPU_G[(y_index-2) + x*32 + y + (x_index-2)*32]*Layer1_Weights_CPU[m+25+f*75] + Data_Layer_CPU_B[(y_index-2) + x*32 + y + (x_index-2)*32]*Layer1_Weights_CPU[m+50+f*75];
				}
			}
		}
		Layer1_Features[f*32*32+x*32+y] = result;
	}
		}
