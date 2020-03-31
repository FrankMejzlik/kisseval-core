
#include "BooleanBucketModel.h"

//
// Initialize static member variables
//
float BooleanBucketModel::m_trueTresholdFrom{0.01f};
float BooleanBucketModel::m_trueTresholdTo{0.9f};
float BooleanBucketModel::m_trueTresholdStep{0.1f};
std::vector<float> BooleanBucketModel::m_trueTresholds;
std::vector<uint8_t> BooleanBucketModel::m_inBucketOrders{{0, 1, 2}};
