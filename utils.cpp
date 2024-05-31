#include"config.h"
#include"utils.h"

#include <fstream>
#include<iostream>



using std::vector;
using std::string;
using std::ifstream;
using std::ofstream;

double Utils::sigmoid(double x)
{
	return 1.0/(1.0+std::exp(-x));
}

void Utils::getFileData(const std::string&filename, std::vector<double>&buffer)
{
	ifstream infile(filename);
	
	double t;
	if (infile.is_open())
	{
		while (!infile.eof())
		{
			infile >> t;
			buffer.emplace_back(t);
		}
	}
	else
	{
		// Data file not found
		printf("[ERROR] '%s' not found.\n", filename.c_str());
		
		exit(1);
	}
}

void Utils::getTrainData(const std::string& filename, std::vector<Sample>&trainSample)
{
	vector<double>buffer;
	getFileData(filename, buffer);

	vector<double>feature(Config::INNODE), label(Config::OUTNODE);

	const int bufferSize = buffer.size();

	int i, j;
	for (i = 0; i < bufferSize; i+=Config::INNODE+Config::OUTNODE)
	{

		for (j = 0; j < Config::INNODE; ++j)
			feature[j]=buffer[i + j];
		for (j = 0; j < Config::OUTNODE; ++j)
			label[j]=buffer[i +Config::INNODE + j];
		
		trainSample.emplace_back(feature, label);
	}


}

void Utils::getTestData(const std::string& filename, std::vector<Sample>& testSample)
{
	vector<double>buffer;
	getFileData(filename, buffer);

	const int bufferSize = buffer.size();

	vector<double>feature(Config::INNODE), label(Config::OUTNODE);

	int i, j;
	for (i = 0; i < bufferSize; i += Config::INNODE + Config::OUTNODE)
	{
		
		for (j = 0; j < Config::INNODE; ++j)
			feature[j]=buffer[i + j];
		/*for (j = 0; j < Config::OUTNODE; ++j)
			tempSample.label.emplace_back(buffer[i + Config::INNODE + j]);*/

		testSample.emplace_back(feature,label);
	}
}





void Utils::display(const std::vector<Sample>&data)
{
	for (const auto& sample : data)
		std::cout << sample;
	
}