

#include <array>
#include <chrono>
#include <iostream>
#include <thread>
#include "ImageRanker.h"

#define BASE_DIR "./data/"

int main()
{
  std::vector<DatasetDataPackRef> datasets{
      {enum_label(eImagesetId::V3C1_20K).first, enum_label(eImagesetId::V3C1_20K).second,
       enum_label(eImagesetId::V3C1_20K).first, "./data/thumbs", "./data/files.txt"}};

  std::vector<ViretDataPackRef> VIRET_data_packs{
      // NasNet
      {enum_label(eDataPackId::NASNET_2019).first, enum_label(eDataPackId::NASNET_2019).second,
       enum_label(eImagesetId::V3C1_20K).first,

       enum_label(eVocabularyId::VIRET_WORDNET_2019).first, enum_label(eVocabularyId::VIRET_WORDNET_2019).second,
       "./data/VIRET/keyword_classes.txt",

       "./data/VIRET/NasNet2019/Studenti_NasNetLarge.pre-softmax",
       "./data/VIRET/NasNet2019/Studenti_NasNetLarge.softmax",
       "./data/VIRET/NasNet2019/Studenti_NasNetLarge.deep-features"},
      // GoogLeNet
      {enum_label(eDataPackId::GOOGLENET_2019).first, enum_label(eDataPackId::GOOGLENET_2019).second,
       enum_label(eImagesetId::V3C1_20K).first,

       enum_label(eVocabularyId::VIRET_WORDNET_2019).first, enum_label(eVocabularyId::VIRET_WORDNET_2019).second,
       "./data/VIRET/keyword_classes.txt",

       "./data/VIRET/GoogLeNet2019/05_googlenet.pre-softmax", "./data/VIRET/GoogLeNet2019/05_googlenet.softmax",
       "./data/VIRET/GoogLeNet2019/05_googlenet.deep-features"}};

  ImageRanker::Config cfg{ImageRanker::eMode::cFullAnalytical, datasets, VIRET_data_packs,
                          std::vector<GoogleDataPackRef>(), std::vector<BowDataPackRef>()};

  ImageRanker ranker(cfg);


  // SimulatedUser su;
  // su.m_exponent = 4;

  // auto charts = ranker.RunModelSimulatedQueries(
  //    "MULT-SUM", DataId(eVocabularyId::VIRET_1200_WORDNET_2019, eScoringsId::NASNET_2019),
  //    InputDataTransformId::cXToTheP, RankingModelId::cViretBase, (UserDataSourceId)999,
  //    std::vector<std::string>({"0"s}), std::vector<std::string>({"0"s, "0.0"s, "0"s}),
  //    std::vector<std::string>({"1"s, "0"s}), 0);

  // for (auto&& chart : charts)
  //{
  //  for (auto&& [i, num] : chart)
  //  {
  //    std::cout << i << "=>" << num << std::endl;
  //  }
  //  std::cout << "==========================================" << std::endl;
  //  std::cout << "==========================================" << std::endl;
  //}

  // std::cout << "XXXXXXXXXXXXXXXXXXXXXXXXXXXXX" << std::endl;
  // std::cout << "XXXXXXXXXXXXXXXXXXXXXXXXXXXXX" << std::endl;
  // std::cout << "XXXXXXXXXXXXXXXXXXXXXXXXXXXXX" << std::endl;

  // charts = ranker.RunModelSimulatedQueries(
  //    "MULT-MAX", DataId(eVocabularyId::VIRET_1200_WORDNET_2019, eScoringsId::NASNET_2019),
  //    InputDataTransformId::cXToTheP, RankingModelId::cViretBase, (UserDataSourceId)999,
  //    std::vector<std::string>({"0"s}), std::vector<std::string>({"0"s, "0.0"s, "1"s}),
  //    std::vector<std::string>({"1"s, "1"s}), 0);

  // for (auto&& chart : charts)
  //{
  //  for (auto&& [i, num] : chart)
  //  {
  //    std::cout << i << "=>" << num << std::endl;
  //  }
  //  std::cout << "==========================================" << std::endl;
  //  std::cout << "==========================================" << std::endl;
  //}

  // std::cout << "XXXXXXXXXXXXXXXXXXXXXXXXXXXXX" << std::endl;
  // std::cout << "XXXXXXXXXXXXXXXXXXXXXXXXXXXXX" << std::endl;
  // std::cout << "XXXXXXXXXXXXXXXXXXXXXXXXXXXXX" << std::endl;

  // charts = ranker.RunModelSimulatedQueries(
  //    "SUM-MAX", DataId(eVocabularyId::VIRET_1200_WORDNET_2019, eScoringsId::NASNET_2019),
  //    InputDataTransformId::cXToTheP, RankingModelId::cViretBase, (UserDataSourceId)999,
  //    std::vector<std::string>({"0"s}), std::vector<std::string>({"0"s, "0.0"s, "2"s}),
  //    std::vector<std::string>({"1"s, "1"s}), 0);

  // for (auto&& chart : charts)
  //{
  //  for (auto&& [i, num] : chart)
  //  {
  //    std::cout << i << "=>" << num << std::endl;
  //  }
  //  std::cout << "==========================================" << std::endl;
  //  std::cout << "==========================================" << std::endl;
  //}

  // std::cout << "XXXXXXXXXXXXXXXXXXXXXXXXXXXXX" << std::endl;
  // std::cout << "XXXXXXXXXXXXXXXXXXXXXXXXXXXXX" << std::endl;
  // std::cout << "XXXXXXXXXXXXXXXXXXXXXXXXXXXXX" << std::endl;

  // charts = ranker.RunModelSimulatedQueries(
  //    "SUM-SUM", DataId(eVocabularyId::VIRET_1200_WORDNET_2019, eScoringsId::NASNET_2019),
  //    InputDataTransformId::cXToTheP, RankingModelId::cViretBase, (UserDataSourceId)999,
  //    std::vector<std::string>({"0"s}), std::vector<std::string>({"0"s, "0.0"s, "3"s}),
  //    std::vector<std::string>({"1"s, "0"s}), 0);

  // for (auto&& chart : charts)
  //{
  //  for (auto&& [i, num] : chart)
  //  {
  //    std::cout << i << "=>" << num << std::endl;
  //  }
  //  std::cout << "==========================================" << std::endl;
  //  std::cout << "==========================================" << std::endl;
  //}

  // std::cout << "XXXXXXXXXXXXXXXXXXXXXXXXXXXXX" << std::endl;
  // std::cout << "XXXXXXXXXXXXXXXXXXXXXXXXXXXXX" << std::endl;
  // std::cout << "XXXXXXXXXXXXXXXXXXXXXXXXXXXXX" << std::endl;

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
