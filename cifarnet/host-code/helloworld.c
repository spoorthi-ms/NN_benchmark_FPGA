#include <stdlib.h>
#include "platform.h"
#include "xil_mmu.h"
#include "xil_cache.h"
#include "xil_cache_l.h"
#include <string.h>
#include <stdio.h>

void print(char *str);

// Base address of registers for FPGA
volatile char *control = (volatile char*)0x43C00000;
volatile int *wg_x = (volatile int*)0x43C00010;
volatile int *wg_y = (volatile int*)0x43C00018;
volatile int *wg_z = (volatile int*)0x43C00020;
volatile int *o_x = (volatile int*)0x43C00028;
volatile int *o_y = (volatile int*)0x43C00030;
volatile int *o_z = (volatile int*)0x43C00038;
volatile int *Layer1_Weights_hw = (volatile int*)0x43C00040;
volatile int *Data_R_hw = (volatile int*)0x43C00048;
volatile int *Data_G_hw = (volatile int*)0x43C00050;
volatile int *Data_B_hw = (volatile int*)0x43C00058;
volatile double *Layer1_Features_hw = (volatile double*)0x43C00060;


#define WG_SIZE_X 32
#define WG_SIZE_Y 32
#define WG_SIZE_Z 1


void InitHostMem(int* Layer1_Weights_host)
{
	// initial layer 1 weight
	FILE * pFile1 = fopen ("data/conv1.txt","rb");
	if (pFile1 != NULL)
	{
		print("File Opened\n");
		char s[300000] = "";
		fread(s,sizeof(s),1,pFile1);
		print("Done2\n");
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{
			double temp_num = atof(temp_string);
			Layer1_Weights_host[i] = temp_num;
			i++;
			index++;
			if(i==2400)
			{
				printf("Breaking\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		fclose (pFile1);
	}
}



int main()
{
	init_platform();
	/* more initialization */
	Xil_SetTlbAttributes(0x43c00000,0x10c06); /* non cacheable */
	int *Layer1_Weights_host;
	int *Data_R_host;
	int *Data_G_host;
	int *Data_B_host;
	double* Layer1_Features_host;
	//int ok = 1;

	/* Allocating host memory */
	Layer1_Weights_host = (int*) malloc (3*32*32 * sizeof(int));
	Data_R_host = (int*) malloc (32*32*sizeof(int));
	Data_G_host = (int*) malloc (32*32*sizeof(int));
	Data_B_host = (int*) malloc (32*32*sizeof(int));
	Layer1_Features_host = (double*) malloc (32*32*32 * sizeof(double));

	//InitHostMem(Layer1_Weights_host);

	Xil_DCacheFlush();

	/* Program registers of layer1 unit */
	*Layer1_Weights_hw = (int)Layer1_Weights_host;
	*Data_R_hw = (int)Data_R_host;
	*Data_G_hw = (int)Data_G_host;
	*Data_B_hw = (int)Data_B_host;
	*Layer1_Features_hw = (int)Layer1_Features_host;

	InitHostMem(Layer1_Weights_host);

	/* set the workgroup identity */
	*wg_y = 0;
	*wg_z = 0;
	*wg_x = 0;
	*o_x = 0;
	*o_y = 0;
	*o_z = 0;




	print("Status of control register: \n\r");
	unsigned int con = *control;
	for (int i = 0; i < 8; i ++) {
		if (con & (1 << i) ) {
			print("1");
		} else {
			print("0");
		}
	}
	print("\n\r");
	print("Starting OpenCL kernel execution\n\r");
	*control = *control | 1; /* start */
	/* waiting for hardware to report "done" */
	while (! ((*control) & 2));
	print("DONE!\n\r");
	Xil_DCacheInvalidate();

#if 1
	for (int i = 0; i < 32; i ++)
	{
		printf("\n %f",Layer1_Features_host[i]);
	}
	print("Result print complete!\n\r");
#endif

	cleanup_platform();
	return 0;
}
