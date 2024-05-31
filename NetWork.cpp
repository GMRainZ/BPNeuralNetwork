#include "NetWork.h"
#include <random>
#include <algorithm>

NetWork::NetWork() 
	:inputLayer(Config::INNODE),
	hiddenLayer(Config::HIDENODE),
	outputLayer(Config::OUTNODE)
{
	std::mt19937 rd;
	rd.seed(std::random_device()());

	std::uniform_real_distribution<double> distribution(-1, 1);

	int i,j;

	//�����
	for (i = 0; i < Config::INNODE; ++i)
	{
		inputLayer[i] = new Neuron(Config::HIDENODE);

		for (j = 0; j < Config::HIDENODE; ++j)
		{
			inputLayer[i]->weight[j] = distribution(rd);
			inputLayer[i]->weight_delta[j] = 0.f;
		}
	}

	//���ز�
	for (i = 0; i < Config::HIDENODE; ++i)
	{
		hiddenLayer[i] = new Neuron(Config::OUTNODE);

		hiddenLayer[i]->bias = distribution(rd);
		hiddenLayer[i]->bias_delta = 0.f;

		for (j = 0; j < Config::OUTNODE; ++j)
		{
			hiddenLayer[i]->weight[j] = distribution(rd);
			hiddenLayer[i]->weight_delta[j] = 0.f;
		}

	}


	//�����
	for (i = 0; i < Config::OUTNODE; ++i)
	{
		outputLayer[i] = new Neuron;

		outputLayer[i]->bias = distribution(rd);
		outputLayer[i]->bias_delta = 0.f;

		//for (j = 0; j < Config::OUTNODE; ++j)
		//{
		//	hiddenLayer[i]->weight[j] = distribution(rd);
		//	hiddenLayer[i]->weight_delta[j] = 0.f;
		//}

	}
}

void NetWork::gradZero()
{
	for (auto& nodeOfInputLayer : inputLayer)
		nodeOfInputLayer->weight_delta.assign(nodeOfInputLayer->weight_delta.size(), 0.f);

	for (auto& nodeOfHiddenLayer : hiddenLayer)
		nodeOfHiddenLayer->weight_delta.assign(nodeOfHiddenLayer->weight_delta.size(), 0.f);

	for (auto& nodeOfOutputLayer : outputLayer)
		nodeOfOutputLayer->weight_delta.assign(nodeOfOutputLayer->weight_delta.size(), 0.f);
}

void NetWork::forward()
{
	int i,j;
	double sum = 0;

	//�������ز����ֵ
	for (j = 0; j < Config::HIDENODE; ++j)
	{
		sum = 0.f;
		for (i = 0; i < Config::INNODE; ++i)
		{
			sum += inputLayer[i]->value *
				inputLayer[i]->weight[j];
		}
		sum -= hiddenLayer[j]->bias;

		hiddenLayer[j]->value = Utils::sigmoid(sum);
	}


	//������������ֵ
	for (j = 0; j < Config::OUTNODE; ++j)
	{
		sum = 0.f;
		for (i = 0; i < Config::HIDENODE; ++i)
		{
			sum += hiddenLayer[i]->value *
				hiddenLayer[i]->weight[j];
		}
		sum -= outputLayer[j]->bias;

		outputLayer[j]->value = Utils::sigmoid(sum);
	}

}

double NetWork::calculateLoss(const std::vector<double>& label)
{
	double loss = 0.f,t;

	int i;
	for(i=0;i<Config::OUTNODE;++i)
	{
		t = std::fabs(outputLayer[i]->value - label[i]);
		
		//printf("outputLayer[i]->value=%lf\n", outputLayer[i]->value);

		loss += t * t / 2.0;
	}

	return loss;
}


void NetWork::backward(const std::vector<double>& label)
{
	int i, j;
	double temp_delta = 0.f;
	for (i = 0; i < Config::OUTNODE; ++i)
	{
		//���򴫲��ڼ������sigmoid���������֮��
		// ��������ƫ��ֵ������ֵΪ
		//delta_bias=-learning_rate*(y-yhat)*yhat*(1-yhat)
		temp_delta = /*learning_rate**/
			-(label[i] - outputLayer[i]->value) *
			outputLayer[i]->value *
			(1.f - outputLayer[i]->value);

		outputLayer[i]->bias_delta += temp_delta;
	}

	//���ز��㵽������� Ȩ�� ������ֵ
	//����µĹ�ʽΪdelta_ji=learning_rate*(yi-yihat)*yihat*(1-yihat)*hj
	for (j = 0; j < Config::HIDENODE; ++j)
	{
		//temp_delta = 0.f;
		for (i = 0; i < Config::OUTNODE; ++i)
		{
			temp_delta = /*learning_rate**/
				(label[i] - outputLayer[i]->value) *
				outputLayer[i]->value *
				(1.f - outputLayer[i]->value) *
				hiddenLayer[j]->value;

			hiddenLayer[j]->weight_delta[i] += temp_delta;
		}
	}



	//���ز��� ƫ��ֵ ������ֵ
	//delta_ji=-learning_rate*sigma(yi-yihat)*yihat*(1-yihat)*vji*hj*(1-hj)

	for (j = 0; j < Config::HIDENODE; ++j)
	{
		temp_delta = 0.f;
		for (i = 0; i < Config::OUTNODE; ++i)
		{
			temp_delta += -(label[i] - outputLayer[i]->value) *
				outputLayer[i]->value *
				(1 - outputLayer[i]->value) *
				hiddenLayer[j]->weight[i];
		}
		temp_delta *= /*learning_rate**/
			hiddenLayer[j]->value *(1.f - hiddenLayer[j]->value);

		hiddenLayer[j]->bias_delta = temp_delta;
	}

	//������㵽���ز��� Ȩ�� ������ֵΪ
	//delta_ij=learing_rate*sigma((yk-ykhat))
	int k;
	for (i = 0; i < Config::INNODE; ++i)
	{
		temp_delta = 0.f;
		for (j = 0; j < Config::HIDENODE; ++j)
		{
			for (k = 0; k < Config::OUTNODE; ++k)
			{
				temp_delta+= (label[k] - outputLayer[k]->value) *
					outputLayer[k]->value *
					(1 - outputLayer[k]->value) *
					hiddenLayer[j]->weight[k];
			}
			temp_delta *= hiddenLayer[j]->value *
				(1.0 - hiddenLayer[j]->value) *
				inputLayer[i]->value;

			inputLayer[i]->weight_delta[j] += temp_delta;
		}
	}
}

void NetWork::revise(const int batch_size)
{
	double batch_size_double = (double)batch_size;

	int i, j;
	//����������Ȩ��
	for (i = 0; i < Config::INNODE; ++i)
	{
		for (j = 0; j < Config::HIDENODE; ++j)
			inputLayer[i]->weight[j] += Config::learning_rate *
				inputLayer[i]->weight_delta[j] / batch_size_double;
	}

	//�������ز��Ȩ�غ�ƫ��
	for (i = 0; i < Config::HIDENODE; ++i)
	{
		hiddenLayer[i]->bias += Config::learning_rate *
			hiddenLayer[i]->bias_delta / batch_size_double;

		for (j = 0; j < Config::OUTNODE; ++j)
			hiddenLayer[i]->weight[j] += Config::learning_rate *
			hiddenLayer[i]->weight_delta[j] / batch_size_double;
	}

	//���������� ƫ��
	for (i = 0; i < Config::OUTNODE; ++i)
		outputLayer[i]->bias += Config::learning_rate *
			outputLayer[i]->bias_delta / batch_size_double;
}

void NetWork::predict(const std::vector<double>&feature, std::vector<double>&label)
{
	int i;
	for (i = 0; i < Config::INNODE; ++i)
		inputLayer[i]->value = feature[i];

	forward();

	for (i = 0; i < Config::OUTNODE; ++i)
		label[i] = outputLayer[i]->value;
}

bool NetWork::train(const std::vector<Sample>& trainDataSet)
{
	int iteration, i;
	double loss,max_loss;

	for (iteration = 0; iteration < Config::max_iteration; ++iteration)
	{
		//����ݶȣ���һ�ε������ݶȲ��ܶ���һ�ε��ݶ���Ӱ��
		gradZero();

		max_loss = 0.f;

		for (const auto& trainSample : trainDataSet)
		{
			//׼��������Ԫ
			for (i = 0; i < Config::INNODE; ++i)
				inputLayer[i]->value = trainSample.feature[i];
			
			//ǰ�򴫲�
			forward();

			//������ʧ
			loss = calculateLoss(trainSample.label);
			
			//printf("��ʱ��loss=%lf\n", loss);

			max_loss = std::max(loss, max_loss);


			backward(trainSample.label);
			
		}
		// Deciding whether to stop training

		if (max_loss < Config::threshold)
		{
			printf("Training SUCCESS in %d epochs.\n", iteration);
			printf("Final maximum error(loss): %lf\n", max_loss);
			return true;
		}
		else if (iteration % Config::printEpoch)
		{
			printf("#epoch %-7lu - max_loss: %lf\n", iteration, max_loss);
		}

		revise(trainDataSet.size());
	}

	printf("Failed within %lu epoch.", Config::max_iteration);

	return false;
}

void NetWork::predict(std::vector<Sample>& testDataSet)
{

	for (auto& testSample : testDataSet)
		predict(testSample.feature, testSample.label);
}


