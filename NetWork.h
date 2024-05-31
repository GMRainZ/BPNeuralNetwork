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
	//�ݶ�����
	void gradZero();

	//ǰ�򴫲�
	void forward();

	//������ʧ����
	double calculateLoss(const std::vector<double>&);

	//���򴫲�
	void backward(const std::vector<double>&);

	//��������
	void revise(const int batch_size);

	void predict(const std::vector<double>&,std::vector<double>&);

public:
	//ѵ������
	bool train(const std::vector<Sample>& trainDataSet);
	
	//Ԥ�⺯��
	void predict(std::vector<Sample>& testDataSet);
};

