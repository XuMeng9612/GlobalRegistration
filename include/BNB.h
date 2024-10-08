//
// Created by sakura on 23-7-24.
//

#ifndef GLOBALREGISTRATION_BNB_H
#define GLOBALREGISTRATION_BNB_H

#include "tool.h"
#include "nanoflann.hpp"

#include <queue>
#include <tuple>
#include <algorithm>
#include <utility>
#include <random>
#include <ceres/ceres.h>
#include <pcl/common/common.h>

// all configuration parameters
struct configBNB {

// parameters for BNB blocks
    int maxWidth = 20;
    double minTransWidth = 0.5;
    double downSampleRate = 0.1;
    int bnbType = 0;

// correspondence estimation parameters
    int maxIteration = 30, lmMaxIteration = 30;
    double borderRate = 0.01;
    double correspondenceDistanceThreshold = 2.0;
    double checkRate = 0.9;

    // loss function parameters
    double rho = 10.0;
    double maxCorrespondenceDistance = 1.0;

    // callbacks function parameters
    double functionTolerance = 1e-5;
    int correspondenceStepThreshold = 10;

// debug parameters
    int debugPrintLevel = 0;
};

//cost function for ceres 2d icp
struct CostFunctionCorrespond {
    CostFunctionCorrespond(double ax, double ay, double bx, double by)
            : ax_(ax), ay_(ay), bx_(bx), by_(by) {}

    //\sum \frac{1}{1+e^(\rho(t-(A_i-RB_i-t)^2)}
    template<typename T>
    bool operator()(const T *const q, T *residual) const {
        auto deltaX = T(ax_) - cos(q[2]) * T(bx_) + sin(q[2]) * T(by_) - q[0];
        auto deltaY = T(ay_) - sin(q[2]) * T(bx_) - cos(q[2]) * T(by_) - q[1];
        residual[0] = sqrt(deltaX * deltaX + deltaY * deltaY);
        return true;
    }

    const double ax_, ay_, bx_, by_;
};

//loss function for ceres 2d icp
class sigmoid : public ceres::LossFunction {
public:
    explicit sigmoid(double rho, double t) : rho_(rho), t_(t) {
    }

    void Evaluate(double s, double rho[3]) const override {


        // /rho(x)=-1/(1+e^{\rho_(x-t)})，-1 is used to convert problem into a minimization problem
        rho[0] = -1.0 / (1.0 + exp(rho_ * (s - t_)));
        rho[1] = (rho_ * exp(rho_ * (s - t_))) / pow((exp(rho_ * (s - t_)) + 1), 2);
        rho[2] = -(rho_ * rho_ * exp(rho_ * (s - t_)) * (exp(rho_ * (s - t_)) - 1))
                 / pow((exp(rho_ * (s - t_)) + 1), 3);
        //avoid CHECK_GT(rho[1], 0.0) assert

        if (rho[1] < 1e-100 || isnan(rho[1]))
            rho[1] = 1e-100;
    };

private:
    const double rho_;
    const double t_;
};

struct bnb2dCallBack : public ceres::IterationCallback {
    explicit bnb2dCallBack(double *x, double functionTolerance, double parameterTolerance) : x_(x), functionTolerance_(
            functionTolerance), parameterTolerance_(parameterTolerance) {
        xOld.resize(3, 0);
        minX = FLT_MIN;
        maxX = FLT_MAX;
        minY = FLT_MIN;
        maxY = FLT_MAX;
    }

    ceres::CallbackReturnType operator()(const ceres::IterationSummary &summary) override {


        double costRatio = std::abs(summary.cost_change / summary.cost);
//        LOG(INFO) << "iter number: " << summary.iteration << " step: " << summary.step_norm << " costRatio: "
//                  << costRatio << " cost: " << -2 * summary.cost << " cost change: " << summary.cost_change;
        if (summary.iteration == 0) {
            xOld[0] = x_[0];
            xOld[1] = x_[1];
            xOld[2] = x_[2];
        }

        if (x_[0] - minX < parameterTolerance_ || maxX - x_[0] < parameterTolerance_ ||
            x_[1] - minY < parameterTolerance_ || maxY - x_[1] < parameterTolerance_)
            return ceres::SOLVER_TERMINATE_SUCCESSFULLY;

        if (summary.iteration > 0 && costRatio < functionTolerance_)
            return ceres::SOLVER_TERMINATE_SUCCESSFULLY;
        return ceres::SOLVER_CONTINUE;
    }

    double getFinalUpdate() {
        double XUpdate = std::sqrt((x_[0] - xOld[0]) * (x_[0] - xOld[0]) + (x_[1] - xOld[1]) * (x_[1] - xOld[1]) +
                                   (x_[2] - xOld[2]) * (x_[2] - xOld[2]));
        return XUpdate;
    }

    double *x_, functionTolerance_, parameterTolerance_;
    std::vector<double> xOld;
    double minX, maxX, minY, maxY;
};

//bnb single block
struct BNBBlock {

    BNBBlock() : pose_(3, 0), minPose_(3, 0), blockSize_(2, 0), bestPose_(3, 0) {
        upLimit_ = -1;
        downLimit_ = -1;
        depth_ = 0;
        randomNumber_ = 0;
        valid_ = false;
        terminationType_ = ceres::NO_CONVERGENCE;
    }

    BNBBlock(double minX, double minY, std::vector<double> initialPose, int depth) : depth_(depth),
    pose_(std::move(initialPose)), minPose_{minX, minY}, blockSize_(2, 0), bestPose_(3, 0) {
        upLimit_ = -1;
        downLimit_ = -1;
        randomNumber_ = 0;
        valid_ = false;
        terminationType_ = ceres::NO_CONVERGENCE;
    }

    BNBBlock(std::vector<double> minPose, std::vector<double> initialPose, int depth) : depth_(depth),
    pose_(std::move(initialPose)), minPose_(std::move(minPose)), blockSize_(2, 0), bestPose_(3, 0) {
        upLimit_ = -1;
        downLimit_ = -1;
        randomNumber_ = 0;
        valid_ = false;
        terminationType_ = ceres::NO_CONVERGENCE;
    }

    BNBBlock(std::vector<double> minPose, std::vector<double> initialPose, std::vector<double> blockSize, int depth)
            : depth_(depth), pose_(std::move(initialPose)), minPose_(std::move(minPose)),
            blockSize_(std::move(blockSize)), bestPose_(3, 0) {
        upLimit_ = -1;
        downLimit_ = -1;
        randomNumber_ = std::max(20, (int) (blockSize_[0] * blockSize_[1]));
        valid_ = false;
        terminationType_ = ceres::NO_CONVERGENCE;
    }

    //depth for single block
    bool valid_;
    int depth_, randomNumber_;
    // pose and min pose for single block
    std::vector<double> pose_, minPose_, blockSize_, bestPose_;
    // icp error and  estimated min error for single block
    double upLimit_, downLimit_;
    // termination type for icp
    ceres::TerminationType terminationType_;
};

class BNB {

public:
    explicit BNB(const configBNB &configBnb);

    ~BNB();

    static void loadBNBConfig(const std::string &configPath, configBNB &config) {
        YAML::Node configNode = YAML::LoadFile(configPath);
        config.maxWidth = configNode["maxWidth"].as<int>();
        config.minTransWidth = configNode["minTransWidth"].as<double>();
        config.downSampleRate = configNode["downSampleRate_"].as<double>();
        config.bnbType = configNode["bnbType"].as<int>();

        config.maxIteration = configNode["maxIteration"].as<int>();
        config.lmMaxIteration = configNode["lmMaxIteration"].as<int>();
        config.borderRate = configNode["borderRate"].as<double>();
        config.correspondenceDistanceThreshold = configNode["correspondenceDistanceThreshold"].as<double>();
        config.checkRate = configNode["checkRate"].as<double>();

        config.rho = configNode["rho"].as<double>();
        config.maxCorrespondenceDistance = configNode["maxCorrespondenceDistance"].as<double>();

        config.functionTolerance = configNode["functionTolerance"].as<double>();
        config.correspondenceStepThreshold = configNode["correspondenceStepThreshold"].as<int>();

        config.debugPrintLevel = configNode["debugPrintLevel"].as<int>();
    }

    //set source and target cloud
    void setInput(const pcl::PointCloud<pcl::PointXYZI>::Ptr &sourceCloud,
                  const pcl::PointCloud<pcl::PointXYZI>::Ptr &targetCloud);

    // Registering
    void run();

    //check 2d icp result
    void icp2dCheck(BNBBlock &block);

    // Get the final transformation
    std::vector<double> getFinalTransformation() const;

    // Get the final score
    double getFinalScore() const;

    void getFinalCorrespondence(std::vector<std::pair<int, int>> &correspondences);

    template<typename type>
    static Eigen::Isometry2f fromVector(const std::vector<type> &vec) {
        Eigen::Isometry2f pose;
        pose.matrix() << cos(vec[2]), -sin(vec[2]), vec[0],
                sin(vec[2]), cos(vec[2]), vec[1],
                0, 0, 1;
        return pose;
    }

    template<typename type>
    static std::vector<type> toVector(const Eigen::Isometry2f &matrix) {
        std::vector<type> vec;
        vec.push_back(matrix.translation().x());
        vec.push_back(matrix.translation().y());
        vec.push_back(atan2(-matrix(0, 1), matrix(0, 0)));
        return vec;
    }

    inline void setGtPose(const std::vector<double> &pose) {
        gtPose_ = pose;
    }

    void
    pointTest(const std::vector<double> &pointPose, double &error, double &correspondenceModified, int &correspondence);

    BNBBlock getBlock(const std::vector<double> &pointPose, const int &depth);

    std::vector<BNBBlock> checkList_;

//private:
    //source and target point clouds
    pcl::PointCloud<pcl::PointXYZI>::Ptr sourceCloud_, targetCloud_;

    // kdTree related variables
    PointCloudXY<float> FLANNTargetCloud_;
    nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, PointCloudXY<float>>, PointCloudXY<float>, 2> flannTarget_;

    // precomputed variables
    std::vector<double> sigmaXs_, sigmaYs_;

    // bnb parameter
    configBNB config_;
    double transMinX_, transMinY_, widthX_, widthY_;
    std::vector<std::vector<BNBBlock>> blocks_;
    int splitX_, splitY_, depth_;
//    ceres::Solver::Options options;

    // bnb result and ground truth
    std::vector<double> pose_, gtPose_;
    double maxCorrespondencesFound_;

    // calculate correspondence for given pose
    void updateCorrespondence(std::vector<std::pair<int, int>> &correspondence, const std::vector<double> &vector);

    // initialize and precompute parameters
    void initialize();

    // Build Distance Transform
    void buildDT();

    //2d icp
    void icp(BNBBlock &block);

    void bottomIcp(BNBBlock &block);

    void topIcp(BNBBlock &block);

    ceres::Solver::Summary
    singleRegistration(const Eigen::Matrix3f &transformation, BNBBlock &block, const ceres::Solver::Options &options,
                       const double &searchDistance);

//    double

};

#endif //GLOBALREGISTRATION_BNB_H