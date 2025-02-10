
#include "GPU_TP.hpp"
#include "CPU_TP.hpp"

int main(int, char*[])
{
	//nx2_plus_ny_GPU();
	//check_prime_glob_GPU();
	check_prime_shrd_GPU();
	return 0;
}