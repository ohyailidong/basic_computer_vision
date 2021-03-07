#include "test/test_k_means.h"
#include "test/test_gvf.h"
#include "test/test_snake.h"
#include "test/test_optical_flow_tracker.h"
#include "test/test_ms_segmentation.h"
#include "test/test_ms_tracking.h"
int main(int argc, char** argv)
{
	//Test_Kmeans::Run();
	//TestGVF::Run();
	//TestOpticalFlow::Run();
	TestMeanShift_Seg::Run();
	//TestMeanShift_Tracking::Run();
	system("pause");
}