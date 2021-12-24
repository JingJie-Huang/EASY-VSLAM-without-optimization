#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <chrono>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <vector>

//#include "common/rotation.h"

using namespace std;
using namespace cv;


bool readYaml( Mat& cameraMatrix, Mat& distCoeffs );

Eigen::Quaterniond Rotation2quaternion( const Mat& R );

void find_feature_matches(
    const Mat& img_1, const Mat& img_2,
    vector<KeyPoint>& keypoints_1,
    vector<KeyPoint>& keypoints_2,
    vector< DMatch >& matches,
    Mat& descriptors_1,
    Mat& descriptors_2   );

// 像素坐标转相机归一化坐标
Point2d pixel2cam( const Point2d& p, const Mat& K );

void initialization_Rt( vector<KeyPoint> keypoints_1,
                        vector<KeyPoint> keypoints_2,
                        const Mat& depth_1,
                        vector<Eigen::Quaterniond>& qua_global,
                        vector<Eigen::Vector3d>& tran_global,
                        vector<DMatch>& matches,
                        vector< vector<Point3d> >& pointcloud_all,
                        Mat& cameraMatrix  );

void bundle_adjustment( Mat& r, Mat& t, std::vector<Point2d>& pts_2d, std::vector<Point3d>& pts_3d );


struct cost_function_define
{
    cost_function_define(Point2d observation_): observation(observation_) {}
    template<typename T>
    bool operator()(const T* const cere_r, const T* const cere_t, const T* pose3d, T* residual) const
    {
        T proj[3]; // after rotation
        //cout<<"point_3d: "<<p_1[0]<<" "<<p_1[1]<<"  "<<p_1[2]<<endl;
        ceres::AngleAxisRotatePoint(cere_r, pose3d, proj);
        // x2 = R*x1 + t
        proj[0] = proj[0]+cere_t[0];
        proj[1] = proj[1]+cere_t[1];
        proj[2] = proj[2]+cere_t[2];

        const T x = proj[0]/proj[2];
        const T y = proj[1]/proj[2];
        //三维点重投影计算的像素坐标
        const T u = x*520.9 + 325.1; // u = fx*x + cx
        const T v = y*521.0 + 249.7; // v = fy*y + cy
        
        //观测的在图像坐标下的值
        T u1 = T(observation.x);
        T v1 = T(observation.y);
        // reprojection error
        residual[0] = u-u1;
        residual[1] = v-v1;
        return true;
    }
    private:
    Point2d observation;
};


int main ( int argc, char** argv )
{
    Mat cameraMatrix, distCoeffs;
    bool state;
    
    if ( argc != 2 )
    {
        cout << "usage: pose_3d2d_depth_ceres ../" << endl;
        return 1;
    }

    state = readYaml(cameraMatrix, distCoeffs);
    if ( !state ){
        cout << "Error reading camera .yaml file" << endl;
        return 1;
    }

    string path_to_dataset = argv[1];
    string associate_file = path_to_dataset + "/associate_rgb_depth.txt";

    // 0.4183 -0.4920 1.6849 -0.8156 0.0346 -0.0049 0.5775 (tx,ty,tz,qx,qy,qz,qw)
    // initialization of translation and rotation from groundtruth.txt
    Eigen::Quaterniond q_eigen( 0.5775, -0.8156, 0.0346, -0.0049 );
    Eigen::Matrix3d R_eigen = q_eigen.toRotationMatrix();
    // store global translation and rotation array
    vector<Eigen::Quaterniond> quaternion_global;
    vector<Eigen::Vector3d> translation_global;

    quaternion_global.push_back(q_eigen);
    Mat r;
    Mat t = ( Mat_<double> ( 3,1 ) << 0.4183, -0.4920, 1.6849 );
    Mat R = ( Mat_<double> ( 3,3 ) << R_eigen(0,0), R_eigen(0,1), R_eigen(0,2),
                                      R_eigen(1,0), R_eigen(1,1), R_eigen(1,2),
                                      R_eigen(2,0), R_eigen(2,1), R_eigen(2,2) );

    Eigen::Vector3d t_eigen;
    t_eigen << t.at<double>(0,0), t.at<double>(1,0), t.at<double>(2,0);
    translation_global.push_back(t_eigen);
    cout << "Initial pose of the first image:\n";
    cout << "R = \n" << R << endl;
    cout << "t = \n" << t << endl;
    q_eigen = Rotation2quaternion( R );
    cout << "q = \n" << q_eigen.vec().transpose() << " " << q_eigen.w() << endl; 

    ifstream fin ( associate_file );

    string rgb_file, depth_file, time_rgb, time_depth;
    Mat color_prev, color_curr, color_undist, depth_prev, depth_curr;

    vector<KeyPoint> keypoints_1, keypoints_2;
    Mat descriptors_1, descriptors_2;
    vector<DMatch> matches;
    vector< vector<DMatch> > matches_all;
    vector< Mat > descriptors_all;
    vector< vector<Point3d> > pointcloud_all;
    vector< vector<KeyPoint> > keypoints_all;

    // read first image
    fin >> time_rgb >> rgb_file >> time_depth >> depth_file;
    color_prev = cv::imread ( path_to_dataset+"/"+rgb_file, CV_LOAD_IMAGE_COLOR);
    undistort( color_prev, color_undist, cameraMatrix, distCoeffs, getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, color_prev.size(), 0, color_prev.size() ) );
    color_prev = color_undist.clone();
    depth_prev = cv::imread ( path_to_dataset+"/"+depth_file, -1 );
    // read second image
    fin >> time_rgb >> rgb_file >> time_depth >> depth_file;
    color_curr = cv::imread ( path_to_dataset+"/"+rgb_file, CV_LOAD_IMAGE_COLOR);
    undistort( color_curr, color_undist, cameraMatrix, distCoeffs, getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, color_curr.size(), 0, color_curr.size() ) );
    color_curr = color_undist.clone();
    depth_curr = cv::imread ( path_to_dataset+"/"+depth_file, -1 );

    find_feature_matches ( color_prev, color_curr, keypoints_1, keypoints_2, matches, descriptors_1, descriptors_2 );
    matches_all.push_back(matches);
    descriptors_all.push_back(descriptors_2);
    keypoints_all.push_back(keypoints_1);
    keypoints_all.push_back(keypoints_2);
    initialization_Rt( keypoints_1, keypoints_2, depth_prev, quaternion_global, translation_global, matches, pointcloud_all, cameraMatrix );
    // swap state
    color_prev = color_curr.clone();
    depth_prev = depth_curr.clone();


    int k = 0; // image match index

    while( !fin.eof() ){

        // for debug
        if(k == 0)
            break;
        cout << "*********** loop " << k << " ************" << endl << endl;
        fin >> time_rgb >> rgb_file >> time_depth >> depth_file;
   

        if( !fin.fail() ){
            color_curr = cv::imread ( path_to_dataset+"/"+rgb_file, CV_LOAD_IMAGE_COLOR);
            undistort( color_curr, color_undist, cameraMatrix, distCoeffs, getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, color_curr.size(), 0, color_curr.size() ) );
            color_curr = color_undist.clone();
            depth_curr = cv::imread ( path_to_dataset+"/"+depth_file, -1 );
            find_feature_matches ( color_prev, color_curr, keypoints_1, keypoints_2, matches, descriptors_1, descriptors_2 );
            cout << "There are total: " << matches.size() << "sets of pairs" << endl;

            
            quaternion_global.push_back(q_eigen);
            t_eigen << t.at<double>(0,0), t.at<double>(1,0), t.at<double>(2,0);
            translation_global.push_back(t_eigen);
            // swap state
            color_prev = color_curr.clone();
            depth_prev = depth_curr.clone();
            // reset
            keypoints_1.clear(); 
            keypoints_2.clear();
            matches.clear();

            k++; 


        }

    }
    
    return 0;
}


Eigen::Quaterniond Rotation2quaternion( const Mat& R )
{
    Eigen::Matrix3d R_eigen;
    R_eigen << R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2),
               R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2),
               R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2);

    Eigen::Quaterniond q = Eigen::Quaterniond( R_eigen );

    return q;
}


void initialization_Rt( vector<KeyPoint> keypoints_1,
                        vector<KeyPoint> keypoints_2,
                        const Mat& depth_1,
                        vector<Eigen::Quaterniond>& qua_global,
                        vector<Eigen::Vector3d>& tran_global,
                        vector<DMatch>& matches,
                        vector< vector<Point3d> >& pointcloud_all,
                        Mat& cameraMatrix  )
{
    vector<Point3d> pts_3d;
    vector<Point2d> pts_2d;
    Mat r ,t, R;
    Eigen::Quaterniond q_eigen;
    Eigen::Vector3d t_eigen;

    cout << "Initialization of the rotation, translation and pointcloud for first match" << endl << endl;

    for( DMatch m:matches )
    {
        ushort d = depth_1.ptr<unsigned short> (int ( keypoints_1[m.queryIdx].pt.y )) [ int ( keypoints_1[m.queryIdx].pt.x ) ];
        if ( d == 0 )   // bad depth
            continue;
        double dd = d/5000.0;
        Point2d p1 = pixel2cam ( keypoints_1[m.queryIdx].pt, cameraMatrix );
        // transform point cloud from current frame to global frame
        // from x2 = R*x1+t
        // x1 = R(Transpose)*(x2-t), x1 are the pointcloud in frame 1; x2 are pointcloud in frame 2
        Eigen::Vector3d pts_3d_proj;
        pts_3d_proj << p1.x*dd-tran_global[0](0,0), 
                        p1.y*dd-tran_global[0](1,0),
                        dd-tran_global[0](2,0) ;
        //cout << "pts_3d_tran = " << pts_3d_tran << endl;
        pts_3d_proj = qua_global[0].toRotationMatrix().transpose() * pts_3d_proj;
        //pts_3d.push_back ( Point3d ( p1.x*dd, p1.y*dd, dd ) );
        pts_3d.push_back ( Point3d( pts_3d_proj(0,0), pts_3d_proj(1,0), pts_3d_proj(2,0) ) );
        pts_2d.push_back ( keypoints_2[m.trainIdx].pt );
    }

    cout << "pts_3d: " << pts_3d.size() << endl;
    cout << "pts_2d: " << pts_2d.size() << endl;

    solvePnPRansac( pts_3d, pts_2d, cameraMatrix, Mat(), r, t, false ); // 调用OpenCV 的 PnP 求解，可选择EPNP，DLS等方法
    cout << "calling bundle adjustment using Ceres Solver" << endl;
    bundle_adjustment( r, t, pts_2d, pts_3d );
  
    cv::Rodrigues( r, R ); // r为旋转向量形式，用Rodrigues公式转换为矩阵
    t_eigen << t.at<double>(0,0), t.at<double>(1,0), t.at<double>(2,0);

    cout << "R=\n" << R <<endl;
    cout << "r =\n" << endl << r << endl;
    cout << "t=\n" << endl << t << endl;
    q_eigen = Rotation2quaternion( R );
    cout << "q = \n" << q_eigen.vec().transpose() << " " << q_eigen.w() << endl;
    
    pointcloud_all.push_back(pts_3d);
    qua_global.push_back(q_eigen);
    tran_global.push_back(t_eigen);
}


void bundle_adjustment( Mat& r, Mat& t, std::vector<Point2d>& pts_2d, std::vector<Point3d>& pts_3d )
{
    //给rot，和tranf初值
    double cere_rot[3], cere_tran[3];
    cere_rot[0] = r.at<double>(0,0);
    cere_rot[1] = r.at<double>(1,0);
    cere_rot[2] = r.at<double>(2,0);

    cere_tran[0] = t.at<double>(0,0);
    cere_tran[1] = t.at<double>(1,0);
    cere_tran[2] = t.at<double>(2,0);

    ceres::Problem problem;
    for(int i=0; i<pts_3d.size(); i++){
        ceres::CostFunction* costfunction = new ceres::AutoDiffCostFunction<cost_function_define,2,3,3,3>(new cost_function_define(pts_2d[i]));
        problem.AddResidualBlock(costfunction, new ceres::HuberLoss(4), cere_rot, cere_tran, &(pts_3d[i].x)); //注意，cere_rot不能为Mat类型      
    }

    ceres::Solver::Options option;
    option.linear_solver_type=ceres::DENSE_SCHUR;
    //输出迭代信息到屏幕
    option.minimizer_progress_to_stdout = false;
    //显示优化信息
    ceres::Solver::Summary summary;
    //开始求解
    ceres::Solve(option,&problem,&summary);
    //显示优化信息
    cout<<summary.BriefReport()<<endl;

    //Mat cam_3d = ( Mat_<double> ( 3,1 ) << cere_rot[0], cere_rot[1], cere_rot[2]);
    //Mat cam_9d;
    //cv::Rodrigues ( cam_3d, cam_9d ); // r为旋转向量形式，用Rodrigues公式转换为矩阵

    //cout << "cam_9d:" << endl << cam_9d << endl;
    //cout << "cam_t:" << cere_tranf[0] << "  " << cere_tranf[1] << "  " << cere_tranf[2] << endl;

    // update rotation and translation
    r.at<double>(0,0) = cere_rot[0];
    r.at<double>(1,0) = cere_rot[1];
    r.at<double>(2,0) = cere_rot[2];

    t.at<double>(0,0) = cere_tran[0];
    t.at<double>(1,0) = cere_tran[1];
    t.at<double>(2,0) = cere_tran[2];

}


void find_feature_matches ( const Mat& img_1, const Mat& img_2,
                            vector<KeyPoint>& keypoints_1,
                            vector<KeyPoint>& keypoints_2,
                            vector< DMatch >& matches,
                            Mat& descriptors_1,
                            Mat& descriptors_2 )
{

    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce-Hamming" );
    //-- 第一步:检测 Oriented FAST 角点位置
    detector->detect ( img_1,keypoints_1 );
    detector->detect ( img_2,keypoints_2 );

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute ( img_1, keypoints_1, descriptors_1 );
    descriptor->compute ( img_2, keypoints_2, descriptors_2 );

    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<DMatch> match;
    // BFMatcher matcher ( NORM_HAMMING );
    matcher->match ( descriptors_1, descriptors_2, match );

    //-- 第四步:匹配点对筛选
    double min_dist=10000, max_dist=0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        double dist = match[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        if ( match[i].distance <= max ( 2*min_dist, 25.0 ) )
        {
            matches.push_back ( match[i] );
            //cout << "queryIdx = " << match[i].queryIdx << endl;
            //cout << "trainIdx = " << match[i].trainIdx << endl << endl;
        }
    }
}


Point2d pixel2cam ( const Point2d& p, const Mat& K )
{
    return Point2d
           (
               ( p.x - K.at<double> ( 0,2 ) ) / K.at<double> ( 0,0 ),
               ( p.y - K.at<double> ( 1,2 ) ) / K.at<double> ( 1,1 )
           );
}


bool readYaml( Mat& cameraMatrix, Mat& distCoeffs )
{
    std::string filename = "../camera_calibration_parameters.yaml";
    FileStorage fs(filename, FileStorage::READ);
    if (!fs.isOpened()) {
        cout << "failed to open file " << filename << endl;
        return false;
    }
    
    fs["cameraMatrix"] >> cameraMatrix;
    fs["distCoeffs"] >> distCoeffs;
    cout << "Reading .yaml file ..." << endl << endl;
    cout << "cameraMatrix = \n" << cameraMatrix << endl;
    cout << "distCoeffs = \n" << distCoeffs << endl;
 
    // read string
    string timeRead;
    fs["calibrationDate"] >> timeRead;
    cout << "calibrationDate = " << timeRead << endl;
    cout << "Finish reading .yaml file" << endl;

    fs.release();
    return true;
}

