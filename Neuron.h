#pragma once

#include<vector>


class Neuron
{
public:
	double value, bias, bias_delta;
	std::vector<double>weight, weight_delta;

	Neuron(const int n = 0) 
		:weight(n,0.f),weight_delta(n,0.f){}
};

