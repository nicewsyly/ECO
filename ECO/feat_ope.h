#ifndef	FEAT_OPERATION
#define FEAT_OPERATION

#include <opencv2/features2d/features2d.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <algorithm>
#include "fftTool.h"

//using namespace std;
template<typename T>
extern std::vector<T> operator+(const std::vector<T>& a, const std::vector<T>& b)
{
	assert(a.size() == b.size());
	std::vector<T> result;
	for (int i = 0; i < a.size(); ++i)
	{
		result.push_back(a[i]+b[i]);
	}
	return result;
}
template<typename T>
extern std::vector<T> operator-(const std::vector<T>& a, const std::vector<T>& b)
{
	assert(a.size() == b.size());
	std::vector<T> result;
	for (int i = 0; i < a.size(); ++i)
	{
		result.push_back(a[i]-b[i]);
	}
	return result;
}

typedef   std::vector<std::vector<cv::Mat> >  ECO_FEATS;
typedef   cv::Vec<float, 2>                   COMPLEX;


extern  ECO_FEATS  featDotMul(const ECO_FEATS& a, const ECO_FEATS& b);   // two features dot multiplication
extern  ECO_FEATS  FeatDotDivide(ECO_FEATS data1, ECO_FEATS data2);

extern  std::vector<cv::Mat>   computeFeatSores(const ECO_FEATS& x, const ECO_FEATS& f); // compute socres  Sum(x * f)
extern  ECO_FEATS              computerFeatScores2(const ECO_FEATS& x, const ECO_FEATS& f);

extern  ECO_FEATS  FeatScale(ECO_FEATS data, float scale);
//extern  ECO_FEATS  FeatAdd(ECO_FEATS data1, ECO_FEATS data2);
//extern  ECO_FEATS  FeatMinus(ECO_FEATS data1, ECO_FEATS data2);

extern  void       symmetrize_filter(ECO_FEATS& hf);
extern  float      FeatEnergy(ECO_FEATS& feat);             
extern  std::vector<cv::Mat>      FeatVec(const ECO_FEATS& x);   // vectorize features
extern  ECO_FEATS  FeatProjMultScale(const ECO_FEATS& x, const std::vector<cv::Mat>& projection_matrix);

extern  std::vector<cv::Mat>  ProjScale(std::vector<cv::Mat> data, float scale);
//extern  std::vector<cv::Mat>  ProjAdd(std::vector<cv::Mat> data1, std::vector<cv::Mat> data2);
//extern  std::vector<cv::Mat>  ProjMinus(std::vector<cv::Mat> data1, std::vector<cv::Mat> data2);



#endif