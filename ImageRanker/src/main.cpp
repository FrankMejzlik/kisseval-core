

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
    ImageRanker::eMode::cFullAnalytical,
    std::vector<KeywordsFileRef>{
      std::tuple(eKeywordsDataType::cViret1, SOLUTION_DIR + R"(data/imageset0/viret_keywords/LSC2018-NASNet.all_must_match.word2vec.label)"s),
      std::tuple(eKeywordsDataType::cGoogleAI, SOLUTION_DIR + R"(data/imageset0/google_keywords/keyword_classes.100.100.joined_words.word2vec.txt)"s)
    }
  };

  ranker.Initialize();

  std::cout << "========================" << std::endl;

  //std::vector<GameSessionInputQuery> methodInput{ std::tuple("aaa"s, 10, "3438257&3438257"s) };

  //ranker.SubmitUserQueriesWithResults(
  //  std::tuple(eKeywordsDataType::cViret1, eImageScoringDataType::cNasNet), methodInput, (QueryOriginId)10);
  //
  /*for (int i{ 0 }; i < 100; ++i)
  {
    auto r1{ ranker.GetCouplingImage() };
    auto r2{ ranker.GetCouplingImage() };
    auto r3{ ranker.GetCouplingImage() };
    auto r4{ ranker.GetCouplingImage() };
    auto r5{ ranker.GetCouplingImage() };
    auto r6{ ranker.GetCouplingImage() };
  }*/
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
  } */
  std::cout << "<<1>>========================" << std::endl;
  auto result11{
  ranker.RunModelTestWrapper(
    std::tuple(eKeywordsDataType::cGoogleAI, eImageScoringDataType::cGoogleAI),
    InputDataTransformId::cNoTransform, RankingModelId::cViretBase, (DataSourceTypeId)999,
    std::vector<std::string>({ "0"s }),
    std::vector<std::string>({ "0"s, "0.0"s, "2"s }), std::vector<std::string>({ "1"s, "0"s }),
    1
  )
  };

  std::cout << "<<2>>========================" << std::endl;
  auto result12{
  ranker.RunModelTestWrapper(
    std::tuple(eKeywordsDataType::cGoogleAI, eImageScoringDataType::cGoogleAI),
    InputDataTransformId::cNoTransform, RankingModelId::cViretBase, (DataSourceTypeId)999,
    std::vector<std::string>({ "0"s }),
    std::vector<std::string>({ "0"s, "0.0"s, "2"s }), std::vector<std::string>({ "1"s, "0"s }),
    2
  )
  };

  std::cout << "<<3>>========================" << std::endl;
  auto result13{
  ranker.RunModelTestWrapper(
    std::tuple(eKeywordsDataType::cGoogleAI, eImageScoringDataType::cGoogleAI),
    InputDataTransformId::cNoTransform, RankingModelId::cViretBase, (DataSourceTypeId)999,
    std::vector<std::string>({ "0"s }),
    std::vector<std::string>({ "0"s, "0.0"s, "2"s }), std::vector<std::string>({ "1"s, "0"s }),
    3
  )
  };

  std::cout << "<<4>>========================" << std::endl;
  auto result14{
  ranker.RunModelTestWrapper(
    std::tuple(eKeywordsDataType::cGoogleAI, eImageScoringDataType::cGoogleAI),
    InputDataTransformId::cNoTransform, RankingModelId::cViretBase, (DataSourceTypeId)999,
    std::vector<std::string>({ "0"s }),
    std::vector<std::string>({ "0"s, "0.0"s, "2"s }), std::vector<std::string>({ "1"s, "0"s }),
    23
  )
  };
  
  /*for (auto&& i : result13)
  {
    std::cout << i.first << " => " << i.second << std::endl;
  }*/

/*
  auto r1 = ranker.GetRelevantImages(
    std::tuple(eKeywordsDataType::cGoogleAI, eImageScoringDataType::cGoogleAI),
    std::vector<std::string>{ "15&15&15&15&15&15&15&15&15&15&53", "15&15&15&15&15&15&15&15&15&15&53" }, 100_z,
    InputDataTransformId::cNoTransform, RankingModelId::cViretBase,
    std::vector<std::string>({ "0"s, "0.0"s, "2"s }), std::vector<std::string>({ "1"s, "0"s }),
    1_z
  );

  auto r2 = ranker.GetRelevantImages(
    std::tuple(eKeywordsDataType::cGoogleAI, eImageScoringDataType::cGoogleAI),
    std::vector<std::string>{ "15&15&15&15&15&15&15&15&15&15&53", "15&15&15&15&15&15&15&15&15&15&53" }, 100_z,
    InputDataTransformId::cNoTransform, RankingModelId::cViretBase,
    std::vector<std::string>({ "0"s, "0.0"s, "0"s }), std::vector<std::string>({ "1"s, "0"s }),
    1_z
  );*/

  /*ranker.SubmitInteractiveSearchSubmit(std::tuple(eKeywordsDataType::cGoogleAI, eImageScoringDataType::cGoogleAI), InteractiveSearchOrigin::cDeveloper, 999,
    RankingModelId::cViretBase , InputDataTransformId::cNoTransform , std::vector<std::string>({ "0"s, "0.0"s, "2"s }) , std::vector<std::string>({ "1"s, "0"s }) , 
    "sess393939", 84, 0, 103, std::vector<InteractiveSearchAction>());*/


 //auto result{ ranker.GetNearKeywords(std::tuple(eKeywordsDataType::cGoogleAI, eImageScoringDataType::cGoogleAI), "cart", 20, true) };

  //auto result2{ ranker.GetRandomImageSequence(2)};
/*
  auto r{ranker.ExportDataFile(
    std::tuple(eKeywordsDataType::cGoogleAI, eImageScoringDataType::cGoogleAI), 
    eExportFileTypeId::cUserAnnotatorQueries, R"(c:\Users\devwe\Downloads\annotator_q.txt)"s
  )};

  auto r2{ ranker.ExportDataFile(
    std::tuple(eKeywordsDataType::cGoogleAI, eImageScoringDataType::cGoogleAI),
    eExportFileTypeId::cQueryNumHits, R"(c:\Users\devwe\Downloads\numHits.txt)"s
  ) };

  auto r3{ ranker.ExportDataFile(
    std::tuple(eKeywordsDataType::cGoogleAI, eImageScoringDataType::cGoogleAI),
    eExportFileTypeId::cNetNormalizedScores, R"(c:\Users\devwe\Downloads\netNormalized.txt)"s
  ) };

  auto rrr = ranker.GetGeneralStatistics(std::tuple(eKeywordsDataType::cGoogleAI, eImageScoringDataType::cGoogleAI), DataSourceTypeId::cAll);
  */
  return 0;

}
