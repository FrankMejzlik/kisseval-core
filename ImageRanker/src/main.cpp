

#include <iostream>
#include <thread>
#include <chrono>
#include "ImageRanker.h"

int main()
{
  ImageRanker ranker{
    R"(c:\Users\devwe\source\repos\ImageRankerApp\public\images\)",
    std::vector<KeywordsFileRef>{
        std::tuple(eKeywordsDataType::cViret1, SOLUTION_DIR + R"(data/imageset0/viret_keywords/dataset0/keyword_classes.txt)"s),
          std::tuple(eKeywordsDataType::cGoogleAI, SOLUTION_DIR + R"(data/imageset0/google_keywords/dataset0/keyword_classes.google.txt)"s)
    },
    std::vector<ScoringDataFileRef>{
      std::tuple(std::tuple(eKeywordsDataType::cViret1, eImageScoringDataType::cNasNet, SOLUTION_DIR + R"(data\imageset0\viret_keywords\dataset0\Studenti_NasNetLarge.pre-softmax)")),
      std::tuple(std::tuple(eKeywordsDataType::cGoogleAI, eImageScoringDataType::cGoogleAI, SOLUTION_DIR + R"(data\imageset0\google_keywords\dataset0\scoringData.google.bin)"))
    },
    std::vector<ScoringDataFileRef>{
      std::tuple(std::tuple(eKeywordsDataType::cViret1, eImageScoringDataType::cNasNet, SOLUTION_DIR + R"(data\imageset0\viret_keywords\dataset0\Studenti_NasNetLarge.softmax)"))
    }, 
    std::vector<ScoringDataFileRef>{},
    SOLUTION_DIR + R"(data/imageset0/viret_keywords/dataset0/files.txt)",
    1ULL,
    ImageRanker::eMode::cFullAnalytical
  };

  ranker.Initialize();

  std::cout << "========================" << std::endl;

  //std::vector<GameSessionInputQuery> methodInput{ std::tuple("aaa"s, 10, "3438257&3438257"s) };

  //ranker.SubmitUserQueriesWithResults(
  //  std::tuple(eKeywordsDataType::cViret1, eImageScoringDataType::cNasNet), methodInput, (QueryOriginId)10);
  //

  auto r1{ ranker.GetCouplingImage() };
  auto r2{ ranker.GetCouplingImage() };
  auto r3{ ranker.GetCouplingImage() };
  auto r4{ ranker.GetCouplingImage() };
  auto r5{ ranker.GetCouplingImage() };
  auto r6{ ranker.GetCouplingImage() };

  /*
  auto result11{
  ranker.RunModelTestWrapper(
    std::tuple(eKeywordsDataType::cViret1, eImageScoringDataType::cNasNet),
    InputDataTransformId::cXToTheP, RankingModelId::cViretBase, (QueryOriginId)999,
    std::vector<std::string>({ "0"s }),
    std::vector<std::string>({ "0"s, "0.0"s, "0"s }), std::vector<std::string>({ "0"s })
  )};
  for (auto&& i : result11)
  {
    std::cout << i.first << " => " << i.second << std::endl;
  }
  std::cout << "========================" << std::endl;
  auto result12{
  ranker.RunModelTestWrapper(
    std::tuple(eKeywordsDataType::cGoogleAI, eImageScoringDataType::cGoogleAI),
    InputDataTransformId::cXToTheP, RankingModelId::cViretBase, (QueryOriginId)999,
    std::vector<std::string>({ "0"s }),
    std::vector<std::string>({ "0"s, "0.0"s, "1"s }), std::vector<std::string>({ "0"s })
  )
  };
  for (auto&& i : result12)
  {
    std::cout << i.first << " => " << i.second << std::endl;
  }
  std::cout << "========================" << std::endl;
  auto result13{
  ranker.RunModelTestWrapper(
    std::tuple(eKeywordsDataType::cViret1, eImageScoringDataType::cNasNet),
    InputDataTransformId::cXToTheP, RankingModelId::cViretBase, (QueryOriginId)999,
    std::vector<std::string>({ "0"s }),
    std::vector<std::string>({ "0"s, "0.0"s, "2"s }), std::vector<std::string>({ "1"s, "0"s })
  )
  };

  for (auto&& i : result13)
  {
    std::cout << i.first << " => " << i.second << std::endl;
  }


  */

  auto result{ ranker.GetNearKeywords(std::tuple(eKeywordsDataType::cGoogleAI, eImageScoringDataType::cGoogleAI), "b", true) };

  //auto result2{ ranker.GetImageKeywordsForInteractiveSearchWithExampleImages(1ULL, 10)};

  
  return 0;

}
