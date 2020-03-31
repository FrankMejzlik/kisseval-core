
#include "GridTest.h"

std::vector<InputDataTransformId> GridTest::m_aggregations{{InputDataTransformId::cSoftmax}};
std::vector<DataSourceTypeId> GridTest::m_queryOrigins{{DataSourceTypeId::cDeveloper}};
std::vector<RankingModelId> GridTest::m_rankingModels{{RankingModelId::cBooleanBucket, RankingModelId::cViretBase}};
std::vector<TestSettings> GridTest::m_testSettings;

std::atomic<size_t> GridTest::numCompletedTests{0ULL};