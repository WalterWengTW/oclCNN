kernel void Conv2D(constant int* info, global float* X, constant float* F, constant float* B, global float* Y)
{
	// info[0] : input map height, info[1] : input map width, info[2] : input map channel
	// info[3] : output map height, info[4] : output map width, info[5] : output map channel
	// info[6] : stride, info[7] : kernel size, info[8] : pad
	
	int width  = get_global_id(0);
	int height = get_global_id(1); 
	int batch  = get_global_id(2); 
	
	int fSubSize = info[7]*info[7]*info[2];
	int xOneBatchSize = info[0]*info[1]*info[2];
	int xMapSize = info[0]*info[1];
	int yOneBatchSize = info[3]*info[4]*info[5];
	int yMapSize = info[3]*info[4];
	
	for(int Co = 0; Co < info[5]; ++Co)
	{
		float sum = 0;
		int fInd = 0;
		for(int Ci = 0; Ci < info[2]; ++Ci)
		{
			for(int kh = 0; kh < info[7]; ++kh)
			{
				for(int kw = 0; kw < info[7]; ++kw, ++fInd)
				{
					int hp = height*info[6]+kh-info[8];
					int wp = width *info[6]+kw-info[8];
					if(hp >= 0 && wp >=0 && hp < info[0] && wp < info[1])
						sum += F[Co*fSubSize+fInd] * X[batch*xOneBatchSize+Ci*xMapSize+hp*info[1]+wp];
				}
			}
		}
		if(B != NULL)
			Y[batch*yOneBatchSize+Co*yMapSize+height*info[4]+width] = sum + B[Co];
		else
			Y[batch*yOneBatchSize+Co*yMapSize+height*info[4]+width] = sum ;
	}
}


kernel void PWConv2D(constant int* info, global float* X, constant float* F, constant float* B, global float* Y)
{
	// info[0] : input map height, info[1] : input map width, info[2] : input map channel
	// info[3] : output map height, info[4] : output map width, info[5] : output map channel
	
	int width  = get_global_id(0);
	int height = get_global_id(1); 
	int batch  = get_global_id(2); 
	
	int xOneBatchSize = info[0]*info[1]*info[2];
	int xMapSize = info[0]*info[1];
	int yOneBatchSize = info[3]*info[4]*info[5];
	int yMapSize = info[3]*info[4];
	
	for(int Co = 0; Co < info[5]; ++Co)
	{
		float sum = 0;
		for(int Ci = 0; Ci < info[2]; ++Ci)
			sum += F[Co*info[2]+Ci] * X[batch*xOneBatchSize+Ci*xMapSize+height*info[1]+width];
		if(B != NULL)
			Y[batch*yOneBatchSize+Co*yMapSize+height*info[4]+width] = sum + B[Co];
		else
			Y[batch*yOneBatchSize+Co*yMapSize+height*info[4]+width] = sum;
	}
}


kernel void DWConv2D(constant int* info, global float* X, constant float* F, constant float* B, global float* Y)
{
	// info[0] : input map height, info[1] : input map width, info[2] : input map channel
	// info[3] : output map height, info[4] : output map width, info[5] : output map channel
	// info[6] : stride, info[7] : kernel size, info[8] : pad
	
	int width  = get_global_id(0);
	int height = get_global_id(1); 
	int batch  = get_global_id(2); 
	
	int fSubSize = info[7]*info[7];
	int xOneBatchSize = info[0]*info[1]*info[2];
	int xMapSize = info[0]*info[1];
	int yOneBatchSize = info[3]*info[4]*info[5];
	int yMapSize = info[3]*info[4];

	
	for(int Co = 0; Co < info[2]; ++Co)
	{
		float sum = 0;
		int fInd = 0;
		for(int kh = 0; kh < info[7]; ++kh)
		{
			for(int kw = 0; kw < info[7]; ++kw, ++fInd)
			{
				int hp = height*info[6]+kh-info[8];
				int wp = width *info[6]+kw-info[8];
				if(hp >= 0 && wp >=0 && hp < info[0] && wp < info[1])
					sum += F[Co*fSubSize+fInd] * X[batch*xOneBatchSize+Co*xMapSize+hp*info[1]+wp];
			}
		}
		if(B != NULL)
			Y[batch*yOneBatchSize+Co*yMapSize+height*info[4]+width] = sum + B[Co];
		else
			Y[batch*yOneBatchSize+Co*yMapSize+height*info[4]+width] = sum;
	}

}


kernel void Dense(constant int* info, global const float* X, constant float* W, constant float* B, global float* Y)
{
	// info[0] : input channels(neurons)
	
	int batch = get_global_id(1);
	int neuron = get_global_id(0);
	int Nneuron = get_global_size(0);
	float sum = 0;
	for(int c = 0; c < info[0]; ++c)
		sum += X[batch * info[0] + c] * W[neuron * info[0] + c];
	if(B != NULL)
		Y[batch * Nneuron + neuron] = sum + B[neuron];
	else
		Y[batch * Nneuron + neuron] = sum;
}


kernel void ReLU(global const float* X, global float* Y)
{
	int i = get_global_id(0);
	Y[i] = X[i] < 0? 0:X[i];
}

kernel void ReLU6(global const float* X, global float* Y)
{
	int i = get_global_id(0);
	Y[i] = X[i] < 0? 0:(X[i]>6? 6:X[i]);
}


kernel void Softmax(global const float* X, global float* Y)
{
	int neuron = get_global_id(0);
	int Nneuron = get_global_size(0);
	int batch = get_global_id(1);
	float sum = 0;
	for(int n=0; n<Nneuron; ++n)
		sum += exp(X[batch*Nneuron+n]);
	Y[batch*Nneuron+neuron] = exp(X[batch*Nneuron+neuron]) / sum;
}

kernel void BatchNorm(constant int* info, global const float* X, constant float* P, global float* Y)
{
	//info[0] : Channel,  info[1] : Epsilon
	
	int width  = get_global_id(0);
	int height = get_global_id(1); 
	int batch  = get_global_id(2);
	int oneBatchSize = info[0]*get_global_size(0)*get_global_size(1);
	int oneMapSize = get_global_size(0)*get_global_size(1);
	int colSize = get_global_size(0);
	float epsilon = (float)info[1]/(float)1e7;
	for(int c = 0; c < info[0]; ++c)
	{
		float gamma = P[0 * info[0] + c];
		float beta  = P[1 * info[0] + c];
		float mean  = P[2 * info[0] + c];
		float var   = P[3 * info[0] + c];
		Y[batch*oneBatchSize+c*oneMapSize+height*colSize+width] = gamma * ((X[batch*oneBatchSize+c*oneMapSize+height*colSize+width] - mean)/sqrt(var+epsilon)) + beta;
	}
}

kernel void Add(global const float* X, global const float* SC, global float* Y)
{
	int i = get_global_id(0);
	Y[i] = X[i] + SC[i];
}

kernel void GlobalAvg(constant int* info, global const float* X, global float* Y)
{
	// info[0] : input map height, info[1] : input map width, info[2] : input map channel
	
	int channel = get_global_id(0);
	int batch = get_global_id(1);
	int xOneBatchSize = info[0]*info[1]*info[2];
	int xMapSize = info[0]*info[1];
	float sum = 0;
	for(int i = 0; i < xMapSize; ++i)
		sum += X[batch*xOneBatchSize+channel*xMapSize+i];
	Y[batch*info[2]+channel] = sum / xMapSize;
}