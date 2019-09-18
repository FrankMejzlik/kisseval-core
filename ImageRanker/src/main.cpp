

#include <iostream>
#include <thread>
#include <chrono>
#include "ImageRanker.h"

int main()
{
  ImageRanker ranker{
    R"(c:\Users\devwe\source\repos\ImageRankerApp\public\images\)",
    std::vector<KeywordsFileRef>{
        std::tuple(eKeywordsDataType::cViret1, SOLUTION_DIR + R"(data/imageset1/viret_keywords/dataset1/keyword_classes.txt)"s)
    },
    std::vector<ScoringDataFileRef>{
      std::tuple(std::tuple(eKeywordsDataType::cViret1, eImageScoringDataType::cNasNet, SOLUTION_DIR + R"(data/imageset1/viret_keywords/dataset1/Studenti_NasNetLarge.pre-softmax)"))
    },
    std::vector<ScoringDataFileRef>{
      std::tuple(std::tuple(eKeywordsDataType::cViret1, eImageScoringDataType::cNasNet, SOLUTION_DIR + R"(data/imageset1/viret_keywords/dataset1/Studenti_NasNetLarge.softmax)"))
    }, 
    std::vector<ScoringDataFileRef>{},
    SOLUTION_DIR + R"(data/imageset1/viret_keywords/dataset1/files.txt)",
    1ULL,
    ImageRanker::eMode::cFullAnalytical
  };

  ranker.Initialize();

  auto result1{
  ranker.GetRelevantImages(
    std::tuple{eKeywordsDataType::cViret1, eImageScoringDataType::cNasNet},
    std::vector<std::string>({"3438257"}), 100,
    InputDataTransformId::cXToTheP, RankingModelId::cViretBase,
    std::vector<std::string>({ "0"s, "0.0"s, "2"s, "1"s, "2"s }), std::vector<std::string>({ "1"s, "0"s }),
    5_z, false
  )};



  /*auto result15{
  ranker.RunModelTestWrapper(
    NetDataTransformation::cXToTheP, RankingModelId::cViretBase, (QueryOriginId)20000,
    std::vector<std::string>({ "4"s }), std::vector<std::string>({ "0"s, "0.0"s, "2"s, "1"s, "2"s }), std::vector<std::string>({ "1"s, "0"s })
  )
  };

  for (auto&& i : result15)
  {
    std::cout << i.first << " => " << i.second << std::endl;
  }*/

  std::cout << "========================" << std::endl;
  
  
/*
  auto result11{
  ranker.RunModelTestWrapper(
    AggregationId::cXToTheP, RankingModelId::cViretBase, (QueryOriginId)999,
    std::vector<std::string>({ "0"s, "0.0"s, "0"s }), std::vector<std::string>({ "0"s })
  )
  };

  auto result12{
  ranker.RunModelTestWrapper(
    AggregationId::cXToTheP, RankingModelId::cViretBase, (QueryOriginId)999,
    std::vector<std::string>({ "0"s, "0.0"s, "1"s }), std::vector<std::string>({ "0"s })
  )
  };*/

  /*auto result13{
  ranker.RunModelTestWrapper(
    AggregationId::cXToTheP, RankingModelId::cViretBase, (QueryOriginId)999,
    std::vector<std::string>({ "0"s, "0.0"s, "2"s }), std::vector<std::string>({ "1"s, "0"s })
  )
  };

  for (auto&& i : result13)
  {
    std::cout << i.first << " => " << i.second;
  }

  auto result14{
  ranker.RunModelTestWrapper(
    AggregationId::cXToTheP, RankingModelId::cViretBase, (QueryOriginId)999,
    std::vector<std::string>({ "0"s, "0.0"s, "2"s }), std::vector<std::string>({ "1"s, "1"s })
  )
  };

  for (auto&& i : result14)
  {
    std::cout << i.first << " => " << i.second;
  }*/
  

  auto result{ ranker.GetNearKeywords("Bla", true) };

  //auto result2{ ranker.GetImageKeywordsForInteractiveSearchWithExampleImages(1ULL, 10)};

  
  return 0;

}
