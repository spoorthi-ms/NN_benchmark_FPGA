__kernel void __attribute__ ((reqd_work_group_size(13,1,1))) fire9sq1x1(__global float *fire9squeeze1x1_Weights_hw, __global float *fire9expand1x1_Weights_hw, __global float *fire9expand3x3_Weights_hw, __global float *fire9squeeze1x1_Features,  __global float *Pool_Layer8_Features,  __global float *fire9_Features)
		{
	int x = get_local_id(0);
	int y = get_group_id(0);

	float Features_squeeze_9;
	float Features_expand1x1_9;
	float Features_expand3x3_9;


	for(int f=0; f<64; f++)
	{
		Features_squeeze_9 = 0;
		for(int n=0; n<512; n++)
		{

			Features_squeeze_9+= Pool_Layer8_Features[n*13*13 + x*13 + y]*fire9squeeze1x1_Weights_hw[f*512+n];
		}
		//ReLU activation function computation
		if(Features_squeeze_9<0)
			Features_squeeze_9 = 0;
		fire9squeeze1x1_Features[f*13*13 + x*13 + y] = Features_squeeze_9;// + fire8squeeze1x1_Weights_GPU[24576 + f];
	}

	for(int f=0; f<256; f++)
	{
		Features_expand1x1_9 = 0;
		for(int n=0; n<64; n++)
		{
			float result = 0;
			result = fire9squeeze1x1_Features[n*13*13 + x*13 + y]*fire9expand1x1_Weights_hw[f*64+n];
			//if(x==0 && y==0 && f==0)
			//printf("(%.8f,%.8f) ", result, fire2expand1x1_Weights_GPU[f*16+n]);
			Features_expand1x1_9+= result;
		}
		//ReLU activation function computation
		if(Features_expand1x1_9<0)
			Features_expand1x1_9 = 0;
		fire9_Features[f*13*13 + x*13+ y] = Features_expand1x1_9;
	}

	fire9_Features = fire9_Features+(13*13*256);
	for(int f=0; f<256; f++)
	{
		Features_expand3x3_9 = 0;
		for(int n=0; n<64; n++)
		{	float result = 0;
		for(int i = x-1; i<=x+1; i++)
		{
			for(int j=y-1; j<=y+1; j++)
			{
				int x_index = i-x+1;
				int y_index = j-y+1;
				int m = (y_index)+(x_index)*3;
				if(i<0 || j<0)
				{
					result+=0;
				}
				else if(j>12 || i>12)
				{
					result+=0;
				}
				else
				{
					result+= fire9squeeze1x1_Features[n*13*13 + i*13 + j]*fire9expand3x3_Weights_hw[m+f*9*64+n*9];
				}
			}
		}
		Features_expand3x3_9 += result;
		}
		//ReLU activation function computation
		if(Features_expand3x3_9<0)
			Features_expand3x3_9 = 0;
		fire9_Features[f*13*13 + x*13 + y] = Features_expand3x3_9;
	}



		}

