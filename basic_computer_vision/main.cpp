#include "test/test_k_means.h"
#include "test/test_gvf.h"
#include "test/test_snake.h"
#include "test/test_optical_flow_tracker.h"

int main(int argc, char** argv)
{
	//Test_Kmeans::Run();
	//TestGVF::Run();
	TestOpticalFlow::Run();
	system("pause");
}