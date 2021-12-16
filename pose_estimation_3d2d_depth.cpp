#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <chrono>

using namespace std;
using namespace cv;


bool readYaml( Mat& cameraMatrix, Mat& distCoeffs );

void find_feature_matches (
    const Mat& img_1, const Mat& img_2,
    std::vector<KeyPoint>& keypoints_1,
    std::vector<KeyPoint>& keypoints_2,
    std::vector< DMatch >& matches );

// 像素坐标转相机归一化坐标
Point2d pixel2cam ( const Point2d& p, const Mat& K );

void bundleAdjustment (
    const vector<Point3f> points_3d,
    const vector<Point2f> points_2d,
    const Mat& K,
    Mat& R, Mat& t
);

int main ( int argc, char** argv )
{
    Mat cameraMatrix, distCoeffs;
    bool state;
     
    if ( argc != 2 )
    {
        cout<<"usage: path_to_dataset"<<endl;
        return 1;
    }

    state = readYaml(cameraMatrix, distCoeffs);
    if ( !state ){
        cout << "Error reading camera .yaml file" << endl;
        return 1;
    }

    string path_to_dataset = argv[1];
    string associate_file = path_to_dataset + "/associate_rgb_depth.txt";

    ifstream fin ( associate_file );

    string rgb_file, depth_file, time_rgb, time_depth;
    cv::Mat color_prev, color_curr, color_undist, depth_prev, depth_curr;
    /* Camera Intrinsic 
    float cx = 325.5;
    float cy = 253.5;
    float fx = 518.0;
    float fy = 519.0;
    float depth_scale = 5000.0;
    */
    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    Mat r;
    Mat t = ( Mat_<double> ( 3,1 ) << 0, 0, 0 );
    Mat R = ( Mat_<double> ( 3,3 ) << 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0 );

    // initialization of translation and rotation



    for(int k=0; k<10; k++){
        cout << "*********** loop " << k << " ************" << endl;
        fin >> time_rgb >> rgb_file >> time_depth >> depth_file;
        if(k == 0){
            color_prev = cv::imread ( path_to_dataset+"/"+rgb_file, CV_LOAD_IMAGE_COLOR);
            undistort( color_prev, color_undist, cameraMatrix, distCoeffs, getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, color_prev.size(), 0, color_prev.size() ) );
            color_prev = color_undist.clone();
            depth_prev = cv::imread ( path_to_dataset+"/"+depth_file, -1 );
            continue;
        }    
        color_curr = cv::imread ( path_to_dataset+"/"+rgb_file, CV_LOAD_IMAGE_COLOR);
        undistort( color_curr, color_undist, cameraMatrix, distCoeffs, getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, color_curr.size(), 0, color_curr.size() ) );
        color_curr = color_undist.clone();
        depth_curr = cv::imread ( path_to_dataset+"/"+depth_file, -1 );

        find_feature_matches ( color_prev, color_curr, keypoints_1, keypoints_2, matches );
        cout << "There are total: " << matches.size() << "sets of pairs" << endl;
        cout << keypoints_1.size()  << endl;
        // 建立3D点
        //Mat d1 = imread ( argv[3], CV_LOAD_IMAGE_UNCHANGED );       // 深度图为16位无符号数，单通道图像
        //Mat K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );
        vector<Point3f> pts_3d;
        vector<Point2f> pts_2d;
        

        for ( DMatch m:matches )
        {
            ushort d = depth_prev.ptr<unsigned short> (int ( keypoints_1[m.queryIdx].pt.y )) [ int ( keypoints_1[m.queryIdx].pt.x ) ];	
            if ( d == 0 )   // bad depth
                continue;
            float dd = d/5000.0;
            Point2d p1 = pixel2cam ( keypoints_1[m.queryIdx].pt, cameraMatrix );
            // transform point cloud from current frame to global frame
            Mat pts_3d_tran = ( Mat_<double>(3,1) << p1.x*dd-t.at<double>(0,0), p1.y*dd-t.at<double>(1,0), dd-t.at<double>(2,0) );
            //cout << "pts_3d_tran = " << pts_3d_tran << endl;
            pts_3d_tran = R.t()*pts_3d_tran;
            //cout << "pts_3d_tran = " << pts_3d_tran << endl;
            //pts_3d.push_back ( Point3f ( p1.x*dd, p1.y*dd, dd ) );
            pts_3d.push_back ( Point3f ( pts_3d_tran.at<double>(0,0), pts_3d_tran.at<double>(1,0), pts_3d_tran.at<double>(2,0) ) );
            pts_2d.push_back ( keypoints_2[m.trainIdx].pt );
        }

        cout<<"3d-2d pairs: "<<pts_3d.size() <<endl;

        
        solvePnPRansac( pts_3d, pts_2d, cameraMatrix, Mat(), r, t, false );
        
        cv::Rodrigues ( r, R ); // Rodrigues to rotation matrix

        cout << "R = " << endl << R << endl;
        cout << "R transpose = " << R.t() << endl;
        cout << "t=" << endl << t << endl;

        cout<<"calling bundle adjustment"<<endl;

        bundleAdjustment ( pts_3d, pts_2d, cameraMatrix, R, t );            


        // swap state
        color_prev = color_curr.clone();
        depth_prev = depth_curr.clone();
        // reset
        keypoints_1.clear(); 
        keypoints_2.clear();
        matches.clear(); 
    }


    
}

void find_feature_matches ( const Mat& img_1, const Mat& img_2,
                            std::vector<KeyPoint>& keypoints_1,
                            std::vector<KeyPoint>& keypoints_2,
                            std::vector< DMatch >& matches )
{
    //-- 初始化
    Mat descriptors_1, descriptors_2;
    // used in OpenCV3
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    // use this if you are in OpenCV2
    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
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

    //printf ( "-- Max dist : %f \n", max_dist );
    //printf ( "-- Min dist : %f \n", min_dist );

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        if ( match[i].distance <= max ( 2*min_dist, 25.0 ) )
        {
            matches.push_back ( match[i] );
        }
    }
}


bool readYaml( Mat& cameraMatrix, Mat& distCoeffs ) {
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


Point2d pixel2cam ( const Point2d& p, const Mat& K )
{
    return Point2d
           (
               ( p.x - K.at<double> ( 0,2 ) ) / K.at<double> ( 0,0 ),
               ( p.y - K.at<double> ( 1,2 ) ) / K.at<double> ( 1,1 )
           );
}

void bundleAdjustment (
    const vector< Point3f > points_3d,
    const vector< Point2f > points_2d,
    const Mat& K,
    Mat& R, Mat& t )
{
    // 初始化g2o
    typedef g2o::BlockSolver< g2o::BlockSolverTraits<6,3> > Block;  // pose 维度为 6, landmark 维度为 3
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverCSparse<Block::PoseMatrixType>(); // 线性方程求解器
    Block* solver_ptr = new Block ( linearSolver );     // 矩阵块求解器
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( solver_ptr );
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm ( solver );

    // vertex
    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap(); // camera pose
    Eigen::Matrix3d R_mat;
    R_mat <<
          R.at<double> ( 0,0 ), R.at<double> ( 0,1 ), R.at<double> ( 0,2 ),
               R.at<double> ( 1,0 ), R.at<double> ( 1,1 ), R.at<double> ( 1,2 ),
               R.at<double> ( 2,0 ), R.at<double> ( 2,1 ), R.at<double> ( 2,2 );
    pose->setId ( 0 );
    pose->setEstimate ( g2o::SE3Quat (
                            R_mat,
                            Eigen::Vector3d ( t.at<double> ( 0,0 ), t.at<double> ( 1,0 ), t.at<double> ( 2,0 ) )
                        ) );
    optimizer.addVertex ( pose );

    int index = 1;
    for ( const Point3f p:points_3d )   // landmarks
    {
        g2o::VertexSBAPointXYZ* point = new g2o::VertexSBAPointXYZ();
        point->setId ( index++ );
        point->setEstimate ( Eigen::Vector3d ( p.x, p.y, p.z ) );
        point->setMarginalized ( true ); // g2o 中必须设置 marg 参见第十讲内容
        optimizer.addVertex ( point );
    }

    // parameter: camera intrinsics
    g2o::CameraParameters* camera = new g2o::CameraParameters (
        K.at<double> ( 0,0 ), Eigen::Vector2d ( K.at<double> ( 0,2 ), K.at<double> ( 1,2 ) ), 0
    );
    camera->setId ( 0 );
    optimizer.addParameter ( camera );

    // edges
    index = 1;
    for ( const Point2f p:points_2d )
    {
        g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
        edge->setId ( index );
        edge->setVertex ( 0, dynamic_cast<g2o::VertexSBAPointXYZ*> ( optimizer.vertex ( index ) ) );
        edge->setVertex ( 1, pose );
        edge->setMeasurement ( Eigen::Vector2d ( p.x, p.y ) );
        edge->setParameterId ( 0,0 );
        edge->setInformation ( Eigen::Matrix2d::Identity() );
        optimizer.addEdge ( edge );
        index++;
    }

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.setVerbose ( true );
    optimizer.initializeOptimization();
    optimizer.optimize ( 100 );
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>> ( t2-t1 );
    cout<<"optimization costs time: "<<time_used.count() <<" seconds."<<endl;

    cout<<endl<<"after optimization:"<<endl;
    cout<<"T="<<endl<<Eigen::Isometry3d ( pose->estimate() ).matrix() <<endl;
}
