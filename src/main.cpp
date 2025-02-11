
#include "GPU_TP.hpp"
#include "CPU_TP.hpp"

int main(int, char*[])
{
	//runOnCPU_GREY();
	runOnCPU_RGB();
	//runOnGPU_GREY();
	runOnGPU_RGB();
	return 0;
}