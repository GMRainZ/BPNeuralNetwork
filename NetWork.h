#pragma once

#include"Neuron.h"
#include"Sample.h"
#include"config.h"
#include"utils.h"

class NetWork
{
private:
	std::vector<Neuron*>inputLayer;
	std::vector<Neuron*>hiddenLayer;
	std::vector<Neuron*>outputLayer;
public:	
	NetWork();

private:
	//梯度置零
	void gradZero();

	//前向传播
	void forward();

	//计算损失函数
	double calculateLoss(const std::vector<double>&);

	//反向传播
	void backward(const std::vector<double>&);

	//修正函数
	void revise(const int batch_size);

	void predict(const std::vector<double>&,std::vector<double>&);

public:
	//训练函数
	bool train(const std::vector<Sample>& trainDataSet);
	
	//预测函数
	void predict(std::vector<Sample>& testDataSet);
};

