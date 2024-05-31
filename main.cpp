#include"config.h"
#include"Neuron.h"
#include"Sample.h"
#include"utils.h"
#include"NetWork.h"

using namespace std;

int main()
{
	const string trainDataFile("D:\\��ҵ\\����ѧϰ\\BPNN\\traindata.txt"),
		testDataFile("D:\\��ҵ\\����ѧϰ\\BPNN\\testdata.txt");
	vector<Sample>trainDataSet, testDataSet;
	
	Utils::getTrainData(trainDataFile, trainDataSet);
	Utils::getTestData(testDataFile, testDataSet);

	//Utils::display(trainDataSet);
	//Utils::display(testDataSet);


	NetWork nw;
	nw.train(trainDataSet);
	nw.predict(testDataSet);

	Utils::display(testDataSet);

	return 0;
}