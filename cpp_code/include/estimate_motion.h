#ifndef _INCLUDE_ESTIMATE_MOTION_H_
#define _INCLUDE_ESTIMATE_MOTION_H_

#include "utility.h"

namespace p3dv
{
class MotionEstimator
{
public:
  bool estimateE8Points(std::vector<cv::KeyPoint> &keypoints1,
                        std::vector<cv::KeyPoint> &keypoints2,
                        std::vector<cv::DMatch> &matches,
                        Eigen::Matrix3f &K,
                        Eigen::Matrix4f &T);

  bool estimate2D2D_E5P_RANSAC(frame_t &cur_frame_1, frame_t &cur_frame_2,
                               std::vector<cv::DMatch> &matches, std::vector<cv::DMatch> &inlier_matches,
                               Eigen::Matrix4f &T,
                               double ransac_prob = 0.99, double ransac_thre = 1.0, bool show = true);

  bool doTriangulation(frame_t &cur_frame_1, frame_t &cur_frame_2,
                       const std::vector<cv::DMatch> &matches,
                       pointcloud_sparse_t &sparse_pointcloud, bool show = true);

  bool estimate2D3D_P3P_RANSAC(frame_t &cur_frame, pointcloud_sparse_t &cur_map_3d,
                               int iterationsCount = 5000, double ransac_prob = 0.99, double ransac_thre = 1.0, bool show = true);

  bool transformCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_in, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_out,
                      Eigen::Matrix4f &trans);

private:
  //transform pixel to camera space coordiante system
  cv::Point2f pixel2cam(const cv::Point2f &p, const cv::Mat &K)
  {
    cv::Point2f cam_pt((p.x - K.at<float>(0, 2)) / K.at<float>(0, 0), (p.y - K.at<float>(1, 2)) / K.at<float>(1, 1));
    //std::cout<<cam_pt<<std::endl;
    return cam_pt;
  }
};
} // namespace p3dv

#endif