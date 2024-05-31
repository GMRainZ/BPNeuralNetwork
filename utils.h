#pragma once

#include<vector>
#include<string>
#include<fstream>
#include<cmath>
#include"Sample.h"


namespace Utils
{
	double sigmoid(double x);

	void getFileData(const std::string&, std::vector<double>&);
	void getTrainData(const std::string&, std::vector<Sample>&);
	void getTestData(const std::string&, std::vector<Sample>&);


	void display(const std::vector<Sample>&);
}