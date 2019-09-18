
#include "ViretModel.h"

//
// Initialize static member variables
//
float ViretModel::m_trueTresholdFrom{ 0.01f };
float ViretModel::m_trueTresholdTo{ 0.9f };
float ViretModel::m_trueTresholdStep{ 0.1f };
std::vector<float> ViretModel::m_trueTresholds;
std::vector<uint8_t> ViretModel::m_queryOperations{ {0,1} };