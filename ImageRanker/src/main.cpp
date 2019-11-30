

#include <iostream>
#include <thread>
#include <chrono>
#include "ImageRanker.h"

int main()
{

  ImageRanker ranker{
    R"(c:\Users\devwe\source\repos\ImageRankerApp\public\images\)",
    std::vector<KeywordsFileRef>{
        std::tuple(eKeywordsDataType::cViret1, R"(c:\Users\frank\source\repos\ir\data\dataset2\keyword_classes.txt)"s)
    },
    std::vector<ScoringDataFileRef>{
      std::tuple(std::tuple(eKeywordsDataType::cViret1, eImageScoringDataType::cNasNet,  R"(c:\Users\frank\source\repos\ir\data\dataset2\Studenti_NasNetLarge.pre-softmax)"))
    },
    std::vector<ScoringDataFileRef>{
      std::tuple(std::tuple(eKeywordsDataType::cViret1, eImageScoringDataType::cNasNet, R"(c:\Users\frank\source\repos\ir\data\dataset2\Studenti_NasNetLarge.softmax)"))
    }, 
    std::vector<ScoringDataFileRef>{},
    R"(c:\Users\frank\source\repos\ir\data\dataset2\files.txt)",
    1ULL,
    ImageRanker::eMode::cFullAnalytical,
    std::vector<KeywordsFileRef>{
      std::tuple(eKeywordsDataType::cViret1, SOLUTION_DIR + R"(data/imageset0/viret_keywords/LSC2018-NASNet.all_must_match.word2vec.label)"s),
    }
  };

  ranker.Initialize();

  std::vector<std::tuple<size_t, std::string, std::string>> queries;
  queries.emplace_back(11,"Ahoj Sonicko!", "fajfiefii3");

  auto r = ranker.GetCoupledImagesNative();

  std::cout << "<<1>>AnySubstringExp========================" << std::endl;
  auto result11{
  ranker.RunModelTestWrapper(
    std::tuple(eKeywordsDataType::cViret1, eImageScoringDataType::cNasNet),
    InputDataTransformId::cXToTheP, RankingModelId::cViretBase, (DataSourceTypeId)10999,
    std::vector<std::string>({ "4"s }),
    std::vector<std::string>({ "0"s, "0.0"s, "0"s }), std::vector<std::string>({ "1"s, "0"s }),
    1
  )
  };

  for (auto&& [i , num] : result11)
  {
    std::cout << i << "=>" << num << std::endl;
  }
  /*
  std::cout << "<<2>>WholeSubstringExp========================" << std::endl;
  auto result12{
  ranker.RunModelTestWrapper(
    std::tuple(eKeywordsDataType::cGoogleAI, eImageScoringDataType::cGoogleAI),
    InputDataTransformId::cNoTransform, RankingModelId::cViretBase, (DataSourceTypeId)999,
    std::vector<std::string>({ "0"s }),
    std::vector<std::string>({ "0"s, "0.0"s, "0"s }), std::vector<std::string>({ "1"s, "0"s }),
    2
  )
  };

  std::cout << "<<3>>w2vExp========================" << std::endl;
  auto result13{
  ranker.RunModelTestWrapper(
    std::tuple(eKeywordsDataType::cGoogleAI, eImageScoringDataType::cGoogleAI),
    InputDataTransformId::cNoTransform, RankingModelId::cViretBase, (DataSourceTypeId)999,
    std::vector<std::string>({ "0"s }),
    std::vector<std::string>({ "0"s, "0.0"s, "0"s }), std::vector<std::string>({ "1"s, "0"s }),
    3
  )
  };

  std::cout << "<<4>>w2vExp+WholeSubstringExp========================" << std::endl;
  auto result14{
  ranker.RunModelTestWrapper(
    std::tuple(eKeywordsDataType::cGoogleAI, eImageScoringDataType::cGoogleAI),
    InputDataTransformId::cNoTransform, RankingModelId::cViretBase, (DataSourceTypeId)999,
    std::vector<std::string>({ "0"s }),
    std::vector<std::string>({ "0"s, "0.0"s, "0"s }), std::vector<std::string>({ "1"s, "0"s }),
    23
  )
  };*/
  
  

/*
  std::cout << "<<1>>AnySubstringExp========================" << std::endl;
  auto result11{
  ranker.RunModelTestWrapper(
    std::tuple(eKeywordsDataType::cViret1, eImageScoringDataType::cNasNet),
    InputDataTransformId::cXToTheP, RankingModelId::cViretBase, (DataSourceTypeId)999,
    std::vector<std::string>({ "0"s }),
    std::vector<std::string>({ "0"s, "0.0"s, "0"s }), std::vector<std::string>({ "1"s, "0"s }),
    1
  )
  };

  std::cout << "<<2>>WholeSubstringExp========================" << std::endl;
  auto result12{
  ranker.RunModelTestWrapper(
    std::tuple(eKeywordsDataType::cViret1, eImageScoringDataType::cNasNet),
    InputDataTransformId::cXToTheP, RankingModelId::cViretBase, (DataSourceTypeId)999,
    std::vector<std::string>({ "0"s }),
    std::vector<std::string>({ "0"s, "0.0"s, "0"s }), std::vector<std::string>({ "1"s, "0"s }),
    2
  )
  };

  std::cout << "<<3>>w2vExp========================" << std::endl;
  auto result13{
  ranker.RunModelTestWrapper(
    std::tuple(eKeywordsDataType::cViret1, eImageScoringDataType::cNasNet),
    InputDataTransformId::cXToTheP, RankingModelId::cViretBase, (DataSourceTypeId)999,
    std::vector<std::string>({ "0"s }),
    std::vector<std::string>({ "0"s, "0.0"s, "0"s }), std::vector<std::string>({ "1"s, "0"s }),
    3
  )
  };

  std::cout << "<<4>>w2vExp+WholeSubstringExp========================" << std::endl;
  auto result14{
  ranker.RunModelTestWrapper(
    std::tuple(eKeywordsDataType::cViret1, eImageScoringDataType::cNasNet),
    InputDataTransformId::cXToTheP, RankingModelId::cViretBase, (DataSourceTypeId)999,
    std::vector<std::string>({ "0"s }),
    std::vector<std::string>({ "0"s, "0.0"s, "0"s }), std::vector<std::string>({ "1"s, "0"s }),
    23
  )
  };

  */
  
  return 0;

}
