
#include "GridTest.h"

using namespace image_ranker;

std::vector<InputDataTransformId> GridTest::m_aggregations{ { InputDataTransformId::cSoftmax } };
std::vector<UserDataSourceId> GridTest::m_queryOrigins{ { UserDataSourceId::cDeveloper } };
std::vector<RankingModelId> GridTest::m_rankingModels{ { RankingModelId::cBooleanBucket, RankingModelId::cViretBase } };
std::vector<TestSettings> GridTest::m_testSettings;

std::atomic<size_t> GridTest::numCompletedTests{ 0ULL };
