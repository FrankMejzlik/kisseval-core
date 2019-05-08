
#include "GridTest.h"

std::vector<AggregationId> GridTest::m_aggregations{ { AggregationId::cSoftmax } };
std::vector<QueryOriginId> GridTest::m_queryOrigins{ { QueryOriginId::cDeveloper } };
std::vector<RankingModelId> GridTest::m_rankingModels{ {RankingModelId::cBooleanBucket, RankingModelId::cViretBase } };
std::vector<TestSettings> GridTest::m_testSettings;

std::atomic<size_t> GridTest::numCompletedTests{ 0ULL };