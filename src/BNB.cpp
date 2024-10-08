//
// Created by sakura on 23-7-24.
//

#include "BNB.h"

BNB::BNB(const configBNB &configBnb) : flannTarget_(2, FLANNTargetCloud_, {10}) {

    targetCloud_.reset(new pcl::PointCloud<pcl::PointXYZI>);
    sourceCloud_.reset(new pcl::PointCloud<pcl::PointXYZI>);
    config_ = configBnb;
    transMinY_ = 0;
    transMinX_ = 0;
    widthX_ = 0;
    widthY_ = 0;
    maxCorrespondencesFound_ = 0;
    splitX_ = 0;
    splitY_ = 0;
    depth_ = 0;

    //    options.minimizer_type = ceres::TRUST_REGION;
    //    options.linear_solver_type = ceres::DENSE_QR;
    //    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    //    options.use_nonmonotonic_steps = true;
    //    options.max_consecutive_nonmonotonic_steps = 10;
    //    options.max_num_iterations = config_.lmMaxIteration;
    //    options.update_state_every_iteration = true;
    //    options.num_threads = omp_get_max_threads();
}

void BNB::buildDT() {

    FLANNTargetCloud_.pts.resize(targetCloud_->points.size());
    for (size_t i = 0; i < targetCloud_->points.size(); ++i) {
        FLANNTargetCloud_.pts[i].x = targetCloud_->points[i].x;
        FLANNTargetCloud_.pts[i].y = targetCloud_->points[i].y;
    }
    flannTarget_.buildIndex();
}

void BNB::run() {

    if (sourceCloud_->empty() || targetCloud_->empty()) {
        LOG(INFO) << "source cloud or target cloud is empty";
        return;
    }

    // set first topBlock
    std::vector<double> initialPose{transMinY_ + widthX_ / 2, transMinY_ + widthY_ / 2, 0};
    BNBBlock topBlock(transMinX_, transMinY_, initialPose, 0);
    if (config_.bnbType)
        bottomIcp(topBlock);
    else
        icp(topBlock);
    blocks_[0].push_back(topBlock);
    maxCorrespondencesFound_ = topBlock.downLimit_;

    if (config_.debugPrintLevel > 0)
        LOG(INFO) << "BNB start!";

    // BNB main loop
    int emptyBlockList = depth_ - 1;
    int handleBlockId = 0;
    while (emptyBlockList < depth_) {
        if (config_.debugPrintLevel > 0)
            LOG(INFO) << "handleBlockId: " << handleBlockId << ", blocks_[handleBlockId].size(): "
                      << blocks_[handleBlockId].size() << ", maxCorrespondencesFound_: " << maxCorrespondencesFound_;
        if (!blocks_[handleBlockId].empty()) {
            // pop best topBlock of specific depth
            auto blockNow = blocks_[handleBlockId].front();
            std::pop_heap(blocks_[handleBlockId].begin(), blocks_[handleBlockId].end(),
                          [](const BNBBlock &a, const BNBBlock &b) {
                              return a.downLimit_ < b.downLimit_;
                          });
            blocks_[handleBlockId].pop_back();
            if (blocks_[handleBlockId].empty())
                emptyBlockList++;
            handleBlockId = (handleBlockId + 1) % depth_;

            //skip to split leaf node
            if (blockNow.depth_ == depth_ - 1)
                continue;

            // pruning check
            if (blockNow.upLimit_ < config_.checkRate * maxCorrespondencesFound_ || blockNow.depth_ == depth_ - 1) {
                if (config_.debugPrintLevel > 0 && blockNow.upLimit_ < config_.checkRate * maxCorrespondencesFound_)
                    LOG(INFO) << "Skip Block id: " << (blockNow.minPose_[0] - transMinX_) / sigmaXs_[blockNow.depth_]
                              << " " << (blockNow.minPose_[1] - transMinY_) / sigmaYs_[blockNow.depth_] << ", depth: "
                              << blockNow.depth_ << ", upLimit: " << blockNow.upLimit_;
                continue;
            }
            // split blocks
            if (config_.debugPrintLevel > 0)
                LOG(INFO) << "Split Block id: " << (blockNow.minPose_[0] - transMinX_) / sigmaXs_[blockNow.depth_]
                          << " " << (blockNow.minPose_[1] - transMinY_) / sigmaYs_[blockNow.depth_] << ", depth: "
                          << blockNow.depth_ << ", uplimit: " << blockNow.upLimit_;
            std::vector<BNBBlock> splitBlock(splitX_ * splitY_);
            int depth = blockNow.depth_ + 1;
#pragma omp parallel for collapse(2) shared(splitX_, splitY_, blockNow, splitBlock, sourceCloud_, targetCloud_, flannTarget_, config_) num_threads(omp_get_max_threads())
            for (int x = 0; x < splitX_; x++)
                for (int y = 0; y < splitY_; y++) {
                    // initial topBlock parameters
                    auto minPose = blockNow.minPose_;
                    minPose[0] += x * sigmaXs_[depth];
                    minPose[1] += y * sigmaYs_[depth];
                    std::vector<double> initial(3);
                    initial[0] = minPose[0] + 0.5 * sigmaXs_[depth];
                    initial[1] = minPose[1] + 0.5 * sigmaYs_[depth];
                    initial[2] = 0;
                    BNBBlock blockNew(minPose, initial, depth);

                    // topBlock handle
                    if (config_.bnbType)
                        bottomIcp(blockNew);
                    else
                        icp(blockNew);

                    if (config_.debugPrintLevel > 2)
                        LOG(INFO) << "New Block id: "
                                  << (blockNew.minPose_[0] - transMinX_) / sigmaXs_[depth] << " "
                                  << (blockNew.minPose_[1] - transMinY_) / sigmaYs_[depth] << ", depth: "
                                  << depth << ", upLimit: " << blockNew.upLimit_ << ", downLimit: "
                                  << blockNew.downLimit_;
                    splitBlock[x * splitY_ + y] = blockNew;
                }
            int validBlock(0);
            double maxDownLimit(0);
            BNBBlock maxBlock;
            for (auto &block: splitBlock)
                if (block.downLimit_ > 0) {
                    validBlock++;
                    // insert topBlock into heap
                    if (blocks_[depth].empty())
                        emptyBlockList--;
                    blocks_[depth].push_back(block);

                    // non-maximum suppression
                    if (maxDownLimit < block.downLimit_ && depth == depth_ - 1) {
                        maxBlock = block;
                        maxDownLimit = block.downLimit_;
                    }

                    // result analysis
                    if (block.downLimit_ > maxCorrespondencesFound_ ) {
                        if (config_.debugPrintLevel > 0)
                            LOG(INFO) << "MaxCorrespondences update from " << maxCorrespondencesFound_ << " to "
                                      << block.downLimit_ << " at " << block.bestPose_;
                        maxCorrespondencesFound_ = block.downLimit_;
                        pose_ = block.bestPose_;
                    }
                }
            if (depth == depth_ - 1 && maxDownLimit > 0) {
                checkList_.push_back(maxBlock);
            }

            std::make_heap(blocks_[depth].begin(), blocks_[depth].end(),
                           [](const BNBBlock &a, const BNBBlock &b) { return a.downLimit_ < b.downLimit_; });
            if (config_.debugPrintLevel > 0)
                LOG(INFO) << "Split end, get " << validBlock << " valid sub-blocks.";

        } else
            handleBlockId = (handleBlockId + 1) % depth_;
    }

    auto rule = [&](const BNBBlock &i, const BNBBlock &j) -> bool {
        return i.downLimit_ > j.downLimit_;
    };
    sort(checkList_.begin(), checkList_.end(), rule);
}

void BNB::updateCorrespondence(std::vector<std::pair<int, int>> &correspondence, const std::vector<double> &vector) {

    Eigen::Isometry2f transform = fromVector(vector);

    size_t num_results = 1;
    std::vector<uint32_t> ret_index(num_results);
    std::vector<float> out_dist_sqr(num_results);
    for (int i = 0; i < sourceCloud_->size(); i++) {
        Eigen::Vector3f pose = sourceCloud_->at(i).getVector3fMap();
        pose = transform * pose;
        const float query_pt[2] = {pose.x(), pose.y()};
        num_results = flannTarget_.knnSearch(&query_pt[0], num_results, &ret_index[0], &out_dist_sqr[0]);
        if (num_results == 1 && out_dist_sqr[0] < config_.maxCorrespondenceDistance)
            correspondence.emplace_back(i, ret_index[0]);
    }
}

void BNB::icp(BNBBlock &block) {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.05, 0.95);

    // iteration termination border threshold
    double borderWidth(std::max(0.1, config_.borderRate * std::min(sigmaXs_[block.depth_], sigmaYs_[block.depth_])));

    double searchDistance;

    if (block.blockSize_[0] * block.blockSize_[1] < 50)
        searchDistance = config_.correspondenceDistanceThreshold * config_.maxCorrespondenceDistance;
    else
        searchDistance = 5 * config_.maxCorrespondenceDistance;

    double minX, minY, maxX, maxY;
    minX = block.minPose_[0];
    maxX = block.minPose_[0] + sigmaXs_[block.depth_];
    minY = block.minPose_[1];
    maxY = block.minPose_[1] + sigmaXs_[block.depth_];
    block.downLimit_ = 0;
    block.upLimit_ = 0;

    for (int j = 0; j < config_.maxIteration; j++) {
        ceres::Problem problem;
        ceres::LossFunction *loss_function;
        loss_function = new sigmoid(config_.rho, config_.maxCorrespondenceDistance);
        Eigen::Matrix2d rot;
        Eigen::Vector2d trans;
        rot << cos(block.pose_[2]), -sin(block.pose_[2]), sin(block.pose_[2]), cos(block.pose_[2]);
        trans << block.pose_[0], block.pose_[1];
        double maxCorrespondModifiedSplit = 0;
        for (size_t sourceId = 0; sourceId < sourceCloud_->size(); sourceId++) {
            size_t num_results = 1;
            std::vector<uint32_t> ret_index(num_results);
            std::vector<float> out_dist_sqr(num_results);
            Eigen::Vector2f poseEigen(sourceCloud_->at(sourceId).x, sourceCloud_->at(sourceId).y);
            poseEigen = rot.cast<float>() * poseEigen + trans.cast<float>();
            const float query_pt[2] = {poseEigen.x(), poseEigen.y()};
            num_results = flannTarget_.knnSearch(&query_pt[0], num_results, &ret_index[0], &out_dist_sqr[0]);
            if (num_results == 1 && out_dist_sqr[0] < searchDistance) {
                auto targetPoint = targetCloud_->points[ret_index[0]];
                auto sourcePoint = sourceCloud_->points[sourceId];
                ceres::CostFunction *cost_function = new ceres::AutoDiffCostFunction<CostFunctionCorrespond, 1, 3>(
                        new CostFunctionCorrespond(targetPoint.x, targetPoint.y, sourcePoint.x, sourcePoint.y));
                problem.AddResidualBlock(cost_function, loss_function, block.pose_.data());
                double xPartMin, xPartMax, yPartMin, yPartMax, fullMin, fullMax;
                if ((minX - targetPoint.x) * (minX - targetPoint.x) <
                    (maxX - targetPoint.x) * (maxX - targetPoint.x)) {
                    xPartMin = (minX - targetPoint.x) * (minX - targetPoint.x);
                    xPartMax = (maxX - targetPoint.x) * (maxX - targetPoint.x);
                } else {
                    xPartMin = (maxX - targetPoint.x) * (maxX - targetPoint.x);
                    xPartMax = (minX - targetPoint.x) * (minX - targetPoint.x);
                }

                if (targetPoint.x > minX && targetPoint.x < maxX)
                    xPartMin = 0;

                if ((minY - targetPoint.y) * (minY - targetPoint.y) <
                    (maxY - targetPoint.y) * (maxY - targetPoint.y)) {
                    yPartMin = (minY - targetPoint.y) * (minY - targetPoint.y);
                    yPartMax = (maxY - targetPoint.y) * (maxY - targetPoint.y);
                } else {
                    yPartMin = (maxY - targetPoint.y) * (maxY - targetPoint.y);
                    yPartMax = (minY - targetPoint.y) * (minY - targetPoint.y);
                }
                if (targetPoint.y > minY && targetPoint.y < maxY)
                    yPartMin = 0;

                fullMin = xPartMin + yPartMin;
                fullMax = xPartMax + yPartMax;

                double S = sourcePoint.x * sourcePoint.x + sourcePoint.y * sourcePoint.y;
                double minError;
                if (S < fullMin)
                    minError = -2 * sqrt(S * fullMin) + fullMin + S;
                else if (S > fullMax)
                    minError = -2 * sqrt(S * fullMax) + fullMax + S;
                else
                    minError = 0;
                maxCorrespondModifiedSplit +=
                        1.0 / (1.0 + exp(config_.rho * (minError - config_.maxCorrespondenceDistance)));
            }
        }
        ceres::Solver::Options options;
        options.minimizer_type = ceres::TRUST_REGION;
        options.linear_solver_type = ceres::DENSE_QR;
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        options.use_nonmonotonic_steps = true;
        options.max_consecutive_nonmonotonic_steps = 10;
        options.max_num_iterations = config_.lmMaxIteration;
        //        options.minimizer_progress_to_stdout = true;
        bnb2dCallBack callBack(block.pose_.data(), config_.functionTolerance, borderWidth);
        callBack.minX = minX;
        callBack.minY = minY;
        callBack.maxX = maxX;
        callBack.maxY = maxY;
        options.update_state_every_iteration = true;
        options.callbacks.push_back(&callBack);
        options.num_threads = omp_get_max_threads();
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        //        LOG(INFO) << summary.FullReport();

        double correspondenceStep = 2 * (summary.initial_cost - summary.final_cost);
        if (-summary.final_cost * 2 > block.downLimit_) {
            block.downLimit_ = -summary.final_cost * 2;
            block.terminationType_ = summary.termination_type;
            block.bestPose_ = block.pose_;
        }
        if (block.upLimit_ < maxCorrespondModifiedSplit)
            block.upLimit_ = maxCorrespondModifiedSplit;

        double disToBoarderX, disToBoarderY;
        if ((block.pose_[0] - minX) * (block.pose_[0] - minX) <
            (block.pose_[0] - maxX) * (block.pose_[0] - maxX))
            disToBoarderX = block.pose_[0] - minX;
        else
            disToBoarderX = maxX - block.pose_[0];
        if ((block.pose_[1] - minY) * (block.pose_[1] - minY) <
            (block.pose_[1] - maxY) * (block.pose_[1] - maxY))
            disToBoarderY = block.pose_[1] - minY;
        else
            disToBoarderY = maxY - block.pose_[1];

        if (disToBoarderX < borderWidth || disToBoarderY < borderWidth ||
            correspondenceStep < config_.correspondenceStepThreshold) {
            block.pose_[0] = block.minPose_[0] + sigmaXs_[block.depth_] * dis(gen);
            block.pose_[1] = block.minPose_[1] + sigmaYs_[block.depth_] * dis(gen);
            //            block.pose_[2] = 2 * M_PI * dis(gen);
            block.pose_[2] = gtPose_[2];
        }
    }
}

BNB::~BNB() = default;

void BNB::setInput(const pcl::PointCloud<pcl::PointXYZI>::Ptr &sourceCloud,
                   const pcl::PointCloud<pcl::PointXYZI>::Ptr &targetCloud) {
    *sourceCloud_ = *sourceCloud;
    *targetCloud_ = *targetCloud;

    initialize();
}

void BNB::initialize() {

    // t_x,t_y,\theta
    pose_ = std::vector<double>(3, 0);
    // estimate occupy rate and pose boundary
    Eigen::Vector4f min_p, max_p;
    Eigen::Matrix<float, 2, 3> sourceMinMax;
    pcl::getMinMax3D(*sourceCloud_, min_p, max_p);
    sourceMinMax.row(0) = min_p.head<3>();
    sourceMinMax.row(1) = max_p.head<3>();
    pcl::getMinMax3D(*targetCloud_, min_p, max_p);
    double maxDist = sourceMinMax.colwise().lpNorm<2>().maxCoeff();
    transMinX_ = min_p[0] - maxDist;
    transMinY_ = min_p[1] - maxDist;

    // fix spilt theta 7 to for quadratic error calculation
    widthX_ = max_p[0] + maxDist - transMinX_;
    widthY_ = max_p[1] + maxDist - transMinY_;
    double widthX = widthX_ / config_.minTransWidth;
    double widthY = widthY_ / config_.minTransWidth;
    if (config_.debugPrintLevel)
        LOG(INFO) << "WidthX: " << widthX << ", widthY: " << widthY << "\n";
    double widthXY = std::max(widthX, widthY);
    if (pow(config_.maxWidth, 2) < widthXY)
        depth_ = 4;
    else
        depth_ = 3;

    splitX_ = (int) ceil(exp(log(widthX) / (depth_ - 1)));
    splitY_ = (int) ceil(exp(log(widthY) / (depth_ - 1)));

    if (config_.debugPrintLevel)
        LOG(INFO) << "Width of x and y are: " << widthX_ << " " << widthY_ << ". resolution of x and y are : "
                  << splitX_ << " " << splitY_ << ", depth: " << depth_;

    blocks_.resize(depth_);

    for (int i = 0; i < depth_; i++) {
        sigmaXs_.push_back(widthX_ / pow(splitX_, i));
        sigmaYs_.push_back(widthY_ / pow(splitY_, i));
    }
    if (config_.debugPrintLevel)
        LOG(INFO) << "SigmaXs: " << sigmaXs_ << ", sigmaYs: " << sigmaYs_;

    // build DT
    buildDT();

    checkList_.resize(0);
}

std::vector<double> BNB::getFinalTransformation() const {

    return pose_;
}

double BNB::getFinalScore() const {

    return maxCorrespondencesFound_;
}

void BNB::icp2dCheck(BNBBlock &block) {

    Eigen::Matrix3f transformation;
    transformation << (float) cos(block.bestPose_[2]), -(float) sin(block.bestPose_[2]), (float) block.bestPose_[0],
            (float) sin(block.bestPose_[2]), (float) cos(block.bestPose_[2]), (float) block.bestPose_[1],
            0, 0, 1;

    std::vector<int> trueIndex;
    std::vector<int> falseIndex;
    std::vector<int> validIndex;
    PointCloudXY<float> fullCloud_;
    nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, PointCloudXY<float>>, PointCloudXY<float>, 2> fullKdTree(
            2, fullCloud_, {10});
    fullCloud_.pts.resize(targetCloud_->size() + sourceCloud_->size());
    for (size_t i = 0; i < targetCloud_->points.size(); ++i) {
        fullCloud_.pts[i].x = targetCloud_->points[i].x;
        fullCloud_.pts[i].y = targetCloud_->points[i].y;
    }
    for (size_t i = 0; i < sourceCloud_->points.size(); ++i) {
        auto point = sourceCloud_->at(i);
        Eigen::Vector3f transPose = transformation * point.getVector3fMap();
        fullCloud_.pts[i + targetCloud_->size()].x = transPose[0];
        fullCloud_.pts[i + targetCloud_->size()].y = transPose[1];
    }
    fullKdTree.buildIndex();

    for (int i = 0; i < sourceCloud_->size(); i++) {
        auto point = sourceCloud_->at(i);
        Eigen::Vector3f transPose = transformation * point.getVector3fMap();
        std::vector<nanoflann::ResultItem<uint32_t, float>> outPut;
        fullKdTree.radiusSearch(transPose.data(), config_.maxCorrespondenceDistance, outPut);
        double targetNumber(0), sourceNumber(0);
        for (auto &iter: outPut)
            if (iter.first < targetCloud_->points.size())
                targetNumber++;
            else
                sourceNumber++;
        if (targetNumber == 0) {
            validIndex.push_back(i);
            continue;
        }
        double ratio =
                (targetNumber * targetNumber + sourceNumber * sourceNumber) / (2 * targetNumber * sourceNumber);
        if (ratio < 2)
            trueIndex.push_back(i);
        else
            falseIndex.push_back(i);
    }
    //
    //    if (trueIndex.size() > falseIndex.size())
    //        block.valid_ = true;
    //    else
    //        block.valid_ = false;

    block.valid_ = true;
}

void BNB::getFinalCorrespondence(std::vector<std::pair<int, int>> &correspondences) {
    updateCorrespondence(correspondences, pose_);
}

void BNB::bottomIcp(BNBBlock &block) {

    std::random_device rd;
    std::mt19937 gen(rd());
    double minRatio, maxRatio;
    if (block.depth_ < 2) {
        minRatio = 0.05;
        maxRatio = 0.95;
    } else {
        minRatio = 0.1;
        maxRatio = 0.9;
    }
    std::uniform_real_distribution<double> dis(minRatio, maxRatio), rot(0, 1);

    // iteration termination border threshold
    double borderWidth = 0.1;
    double searchDistance;
    if (block.blockSize_[0] * block.blockSize_[1] < 50)
        searchDistance = config_.correspondenceDistanceThreshold * config_.maxCorrespondenceDistance;
    else
        searchDistance = 10 * config_.correspondenceDistanceThreshold * config_.maxCorrespondenceDistance;
    auto minX = block.minPose_[0];
    auto maxX = block.minPose_[0] + sigmaXs_[block.depth_];
    auto minY = block.minPose_[1];
    auto maxY = block.minPose_[1] + sigmaXs_[block.depth_];
    block.downLimit_ = 0;
    block.upLimit_ = 0;

    ceres::Solver::Options options;
    options.minimizer_type = ceres::TRUST_REGION;
    options.linear_solver_type = ceres::DENSE_QR;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.use_nonmonotonic_steps = true;
    options.max_consecutive_nonmonotonic_steps = 10;
    options.max_num_iterations = config_.lmMaxIteration;
    options.update_state_every_iteration = true;
    options.num_threads = omp_get_max_threads();
    //    options.minimizer_progress_to_stdout = true;

    bnb2dCallBack callBack(block.pose_.data(), config_.functionTolerance, borderWidth);
    callBack.minX = minX;
    callBack.minY = minY;
    callBack.maxX = maxX;
    callBack.maxY = maxY;
    options.callbacks.push_back(&callBack);
    int loopIterations;
    if (block.depth_ > 1)
        loopIterations = config_.maxIteration;
    else
        loopIterations = 49;

    for (int j = 0; j < loopIterations; j++) {
        Eigen::Matrix3f transformation;
        transformation << (float) cos(block.pose_[2]), -(float) sin(block.pose_[2]), (float) block.pose_[0],
                (float) sin(block.pose_[2]), (float) cos(block.pose_[2]), (float) block.pose_[1],
                0, 0, 1;
        double correspondModified(0), correspond(0);
        for (size_t sourceId = 0; sourceId < sourceCloud_->size(); sourceId++) {
            std::vector<uint32_t> ret_index(1, 0);
            std::vector<float> out_dist_sqr(1, 0);
            Eigen::Vector3f transPose = transformation * sourceCloud_->points[sourceId].getVector3fMap();
            auto num_results = flannTarget_.knnSearch(transPose.data(), 1,
                                                      &ret_index[0], &out_dist_sqr[0]);
            if (num_results == 1 && out_dist_sqr[0] < searchDistance) {
                auto targetPoint = targetCloud_->points[ret_index[0]];
                auto sourcePoint = sourceCloud_->points[sourceId];
                double xPartMin, xPartMax, yPartMin, yPartMax, fullMin, fullMax;
                if ((minX - targetPoint.x) * (minX - targetPoint.x) <
                    (maxX - targetPoint.x) * (maxX - targetPoint.x)) {
                    xPartMin = (minX - targetPoint.x) * (minX - targetPoint.x);
                    xPartMax = (maxX - targetPoint.x) * (maxX - targetPoint.x);
                } else {
                    xPartMin = (maxX - targetPoint.x) * (maxX - targetPoint.x);
                    xPartMax = (minX - targetPoint.x) * (minX - targetPoint.x);
                }

                if (targetPoint.x > minX && targetPoint.x < maxX)
                    xPartMin = 0;

                if ((minY - targetPoint.y) * (minY - targetPoint.y) <
                    (maxY - targetPoint.y) * (maxY - targetPoint.y)) {
                    yPartMin = (minY - targetPoint.y) * (minY - targetPoint.y);
                    yPartMax = (maxY - targetPoint.y) * (maxY - targetPoint.y);
                } else {
                    yPartMin = (maxY - targetPoint.y) * (maxY - targetPoint.y);
                    yPartMax = (minY - targetPoint.y) * (minY - targetPoint.y);
                }
                if (targetPoint.y > minY && targetPoint.y < maxY)
                    yPartMin = 0;

                fullMin = xPartMin + yPartMin;
                fullMax = xPartMax + yPartMax;

                double S = sourcePoint.x * sourcePoint.x + sourcePoint.y * sourcePoint.y;
                double minError;
                if (S < fullMin)
                    minError = -2 * sqrt(S * fullMin) + fullMin + S;
                else if (S > fullMax)
                    minError = -2 * sqrt(S * fullMax) + fullMax + S;
                else
                    minError = 0;
                if (out_dist_sqr[0] < config_.maxCorrespondenceDistance)
                    correspond++;
                correspondModified +=
                        1.0 / (1.0 + exp(config_.rho * (minError - config_.maxCorrespondenceDistance)));
            }
        }

        if (block.downLimit_ < correspond)
            block.downLimit_ = correspond;

        if (block.depth_ > 1) {
            auto summary = singleRegistration(transformation, block, options, searchDistance);
            if (-summary.final_cost * 2 > block.downLimit_) {
                block.downLimit_ = -summary.final_cost * 2;
                block.terminationType_ = summary.termination_type;
                block.bestPose_ = block.pose_;
                //                LOG(INFO) << block.bestPose_;
            }
            //            LOG(INFO)<<block.pose_<<" "<<-summary.final_cost * 2<<" "<<summary.termination_type;
        }

        if (block.upLimit_ < correspondModified)
            block.upLimit_ = correspondModified;

        int w = sqrt(loopIterations);
        block.pose_[0] = block.minPose_[0] + sigmaXs_[block.depth_] * dis(gen);
        //        block.pose_[0] = block.minPose_[0] + sigmaXs_[block.depth_] / w * (j / w + 0.5);
        block.pose_[1] = block.minPose_[1] + sigmaYs_[block.depth_] * dis(gen);
        //        block.pose_[1] = block.minPose_[1] + sigmaYs_[block.depth_] / w * (j % w + 0.5);
        block.pose_[2] = gtPose_[2];
        // block.pose_[2] = 2 * rot(gen) * M_PI;
        //        block.pose_=gtPose_;
    }

    if (block.depth_ <= 1)
        block.downLimit_ = block.upLimit_ / 10;
}

ceres::Solver::Summary
BNB::singleRegistration(const Eigen::Matrix3f &transformation, BNBBlock &block, const ceres::Solver::Options &options,
                        const double &searchDistance) {

    bool converged = false;
    ceres::Solver::Summary summary;

    while (!converged) {
        ceres::Problem problem;
        ceres::LossFunction *loss_function = new sigmoid(config_.rho, config_.maxCorrespondenceDistance);

        for (size_t sourceId = 0; sourceId < sourceCloud_->size(); sourceId++) {
            size_t num_results = 1;
            std::vector<uint32_t> ret_index(num_results);
            std::vector<float> out_dist_sqr(num_results);
            Eigen::Vector3f transPose = transformation * sourceCloud_->points[sourceId].getVector3fMap();
            num_results = flannTarget_.knnSearch(transPose.data(), num_results, &ret_index[0], &out_dist_sqr[0]);
            if (num_results == 1 && out_dist_sqr[0] < searchDistance) {
                auto targetPoint = targetCloud_->points[ret_index[0]];
                auto sourcePoint = sourceCloud_->points[sourceId];
                ceres::CostFunction *cost_function = new ceres::AutoDiffCostFunction<CostFunctionCorrespond, 1, 3>(
                        new CostFunctionCorrespond(targetPoint.x, targetPoint.y, sourcePoint.x, sourcePoint.y));
                problem.AddResidualBlock(cost_function, loss_function, block.pose_.data());
            }
        }

        ceres::Solve(options, &problem, &summary);
        //        LOG(INFO) << summary.FullReport();
        //        LOG(INFO) << block.pose_;

        double correspondenceStep = 2 * (summary.initial_cost - summary.final_cost);
        if (summary.termination_type == ceres::CONVERGENCE || summary.termination_type == ceres::USER_SUCCESS ||
            correspondenceStep < config_.correspondenceStepThreshold)
            converged = true;
    }

    return summary;
}

void BNB::topIcp(BNBBlock &block) {
}

void BNB::pointTest(const std::vector<double> &pointPose, double &error, double &correspondenceModified,
                    int &correspondence) {

    Eigen::Matrix2d rot;
    Eigen::Vector2d trans;
    rot << cos(pointPose[2]), -sin(pointPose[2]),
            sin(pointPose[2]), cos(pointPose[2]);
    trans << pointPose[0], pointPose[1];
    correspondence = 0;
    correspondenceModified = 0;
    error = 0;
    for (auto point: *sourceCloud_) {
        size_t num_results = 1;
        std::vector<uint32_t> ret_index(num_results);
        std::vector<float> out_dist_sqr(num_results);
        Eigen::Vector2f poseEigen(point.x, point.y);
        poseEigen = rot.cast<float>() * poseEigen + trans.cast<float>();
        const float query_pt[2] = {poseEigen.x(), poseEigen.y()};
        num_results = flannTarget_.knnSearch(&query_pt[0], num_results, &ret_index[0],
                                             &out_dist_sqr[0]);

        if (num_results == 1 && out_dist_sqr[0] < config_.correspondenceDistanceThreshold *
                                                  config_.maxCorrespondenceDistance) {
            error += out_dist_sqr[0] * out_dist_sqr[0];
            correspondenceModified += 1.0 / (1.0 + exp(config_.rho * (out_dist_sqr[0] * out_dist_sqr[0] -
                                                                      config_.maxCorrespondenceDistance)));
            correspondence++;
        }
    }
}

BNBBlock BNB::getBlock(const std::vector<double> &pointPose, const int &depth) {
    if (depth > depth_) {
        LOG(WARNING) << "Too large depth!";
        return {};
    }

    int idx, idy;
    idx = std::floor((pointPose[0] - transMinX_) / sigmaXs_[depth]);
    idy = std::floor((pointPose[1] - transMinY_) / sigmaYs_[depth]);
    std::vector<double> minPose{transMinX_ + idx * sigmaXs_[depth], transMinY_ + idy * sigmaYs_[depth]}, initialPose(3);
    initialPose[0] = minPose[0] + 0.5 * sigmaXs_[depth];
    initialPose[1] = minPose[1] + 0.5 * sigmaYs_[depth];
    initialPose[2] = 0;
    LOG(INFO) << "Block depth: " << depth << ", idx: " << idx << ", idy: " << idy;
    LOG(INFO) << "minPose: " << minPose << ", maxPose: [" << minPose[0] + sigmaXs_[depth] << ", "
              << minPose[1] + sigmaYs_[depth] << ", 0]";
    BNBBlock block(minPose, initialPose, depth);

    return block;
}
