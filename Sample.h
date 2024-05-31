#pragma once
#include<vector>
#include<iostream>

class Sample
{
	friend std::ostream& operator<<(std::ostream&, const Sample&);
public:
	std::vector<double>feature, label;

	Sample(){}
	Sample(std::vector<double>&_feature, std::vector<double>& _label)
		:feature(_feature),label(_label){}
};

