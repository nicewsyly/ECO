#include "feature_operator.h"

ECO_FEATS   featDotMul(const ECO_FEATS& a, const ECO_FEATS& b)
{
	ECO_FEATS res;

	if (a.size() != b.size())
		assert("Unamtched feature size");

	for (size_t i = 0; i < a.size(); i++)
	{
		std::vector<cv::Mat> temp;
		for (size_t j = 0; j < a[i].size(); j++)
		{
			temp.push_back(FFTTools::complexMultiplication( a[i][j],b[i][j]));
		}
		res.push_back(temp);
	}
	return res;
}

ECO_FEATS  FeatDotDivide(ECO_FEATS a, ECO_FEATS b)
{
	ECO_FEATS res;

	if (a.size() != b.size())
		assert("Unamtched feature size");

	for (size_t i = 0; i < a.size(); i++)
	{
		std::vector<cv::Mat> temp;
		for (size_t j = 0; j < a[i].size(); j++)
		{
			temp.push_back(FFTTools::complexDivision(a[i][j], b[i][j]));
		}
		res.push_back(temp);
	}
	return res;

}

std::vector<cv::Mat>    computeFeatSores(const ECO_FEATS& x, const ECO_FEATS& f)
{
	std::vector<cv::Mat> res;

	ECO_FEATS res_temp = featDotMul(x, f);
	for (size_t i = 0; i < res_temp.size(); i++)
	{
		cv::Mat temp(cv::Mat::zeros(res_temp[i][0].size(), res_temp[i][0].type()));
		for (size_t j = 0; j < res_temp[i].size(); j++)
		{
			temp = temp + res_temp[i][j];
		}
		res.push_back(temp);
	}

	return res;
}
 

float      FeatEnergy(ECO_FEATS& feat)
{
	float res = 0;
	if (feat.empty())
		return res;

	for (size_t i = 0; i < feat.size(); i++)
	{
		for (size_t j = 0; j < feat[i].size(); j++)
		{
			res += FFTTools::mat_sum(FFTTools::real(FFTTools::complexMultiplication(FFTTools::mat_conj(feat[i][j]), feat[i][j])));
		}
	}

	return res;
}


///**** features operation **************

ECO_FEATS  FeatScale(ECO_FEATS data,float scale)
{
	ECO_FEATS res;

	for (size_t i = 0; i < data.size(); i++)
	{
		std::vector<cv::Mat> tmp;
		for (size_t j = 0; j < data[i].size(); j++)
		{
			tmp.push_back(data[i][j] * scale);
		}
		res.push_back(tmp);
	}
	return res;
}

 
ECO_FEATS  FeatAdd(ECO_FEATS data1, ECO_FEATS data2)
{
	ECO_FEATS res;

	for (size_t i = 0; i < data1.size(); i++)
	{
		std::vector<cv::Mat> tmp;
		for (size_t j = 0; j < data1[i].size(); j++)
		{
			tmp.push_back(data1[i][j] + data2[i][j]);
		}
		res.push_back(tmp);
	}

	return res;

}

ECO_FEATS  FeatMinus(ECO_FEATS data1, ECO_FEATS data2)
{
	ECO_FEATS res;

	for (size_t i = 0; i < data1.size(); i++)
	{
		std::vector<cv::Mat> tmp;
		for (size_t j = 0; j < data1[i].size(); j++)
		{
			tmp.push_back(data1[i][j] - data2[i][j]);
		}
		res.push_back(tmp);
	}

	return res;

}

void       symmetrize_filter(ECO_FEATS& hf)
{

	for (size_t i = 0; i < hf.size(); i++)
	{
		int dc_ind = (hf[i][0].rows + 1) / 2;
		for (size_t j = 0; j < hf[i].size(); j++)
		{
			int c = hf[i][j].cols - 1;
			for (size_t r = dc_ind; r < hf[i][j].rows; r++)
			{
				//cout << hf[i][j].at<COMPLEX>(r, c);
				hf[i][j].at<COMPLEX>(r, c) = hf[i][j].at<COMPLEX>(2 * dc_ind - r - 2, c).conj();
			}
		}

	}
}

ECO_FEATS  FeatProjMultScale(const ECO_FEATS& x, const std::vector<cv::Mat>& projection_matrix)
{
	ECO_FEATS result;
	//vector<cv::Mat> featsVec = FeatVec(x);
	for (size_t i = 0; i < x.size(); i++)
	{
		int org_dim = projection_matrix[i].rows;
		int numScales = x[i].size() / org_dim;
		std::vector<cv::Mat> temp;

		for (size_t s = 0; s < numScales; s++)   // for every scale 
		{
			cv::Mat x_mat;
			for (size_t j = s * org_dim; j < (s + 1) * org_dim; j++)
			{
				cv::Mat t = x[i][j].t();
				x_mat.push_back(cv::Mat(1, x[i][j].size().area(), CV_32FC1, t.data));
			}

			x_mat = x_mat.t();

			cv::Mat res_temp = x_mat * FFTTools::real( projection_matrix[i]);

			//**** reconver to standard formation ****
			for (size_t j = 0; j < res_temp.cols; j++)
			{
				cv::Mat temp2 = res_temp.col(j);
				cv::Mat tt; temp2.copyTo(tt);                                 // the memory should be continous!!!!!!!!!! 
				cv::Mat temp3(x[i][0].cols, x[i][0].rows, CV_32FC1, tt.data); //(x[i][0].cols, x[i][0].rows, CV_32FC2, temp2.data) int size[2] = { x[i][0].cols, x[i][0].rows };cv::Mat temp3 = temp2.reshape(2, 2, size)
				temp.push_back(temp3.t());
			}
		}
		result.push_back(temp);
	}
	return result;
}

std::vector<cv::Mat>      FeatVec(const ECO_FEATS& x)
{
	if (x.empty())
		return std::vector<cv::Mat>();

	std::vector<cv::Mat> res;
	for (size_t i = 0; i < x.size(); i++)
	{
		cv::Mat temp;
		for (size_t j = 0; j < x[i].size(); j++)
		{
			cv::Mat temp2 = x[i][j].t();
			temp.push_back(cv::Mat(1, x[i][j].size().area(), CV_32FC2, temp2.data));
		}
		res.push_back(temp);
	}
	return res;
}

///**** projection operation **************
std::vector<cv::Mat>  ProjScale(std::vector<cv::Mat> data, float scale)
{
	std::vector<cv::Mat> res;
	for (size_t i = 0; i < data.size(); i++)
	{
		res.push_back(data[i] * scale);
	}
	return res;
}

std::vector<cv::Mat>  ProjAdd(std::vector<cv::Mat> data1, std::vector<cv::Mat> data2)
{
	std::vector<cv::Mat> res;
	for (size_t i = 0; i < data1.size(); i++)
	{
		res.push_back(data1[i] + data2[i]);
	}
	return res;
}

std::vector<cv::Mat>  ProjMinus(std::vector<cv::Mat> data1, std::vector<cv::Mat> data2)
{
	std::vector<cv::Mat> res;
	for (size_t i = 0; i < data1.size(); i++)
	{
		res.push_back(data1[i] - data2[i]);
	}
	return res;
}
