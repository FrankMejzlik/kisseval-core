

#include <chrono>
#include <iostream>
#include <thread>
#include <array>
#include "ImageRanker.h"

#define BASE_DIR "./data/"


//static const char* EnumStrings[] = { "bananas & monkeys", "Round and orange", "APPLE" };

//vocabulary_IDs = {
//    0: "VIRET_WordNet2019",
//    100: "Google_AI_2019"
//}

int main()
{

  auto a = enum_label(eImagesetId::V3C1_20K);

  auto b = enum_label(eVocabularyId::VIRET_WordNet2019);

  std::vector<DatasetDataPackRef> datasets{{"V3C1_20k", "V3C1_20k", "./data/thumbs", "./data/files.txt"}};
  std::vector<ViretDataPackRef> VIRET_data_packs{ 
      {"V3C1_20k_NasNet_2019", "V3C1_20k", "VIRET_WordNet2019", "./data/VIRET/keyword_classes.txt",
       "./data/VIRET/NasNet2019/Studenti_NasNetLarge.pre-softmax",
       "./data/VIRET/NasNet2019/Studenti_NasNetLarge.softmax",
       "./data/VIRET/NasNet2019/Studenti_NasNetLarge.deep-features"},
      {"V3C1_20k_GoogLeNet_2019", "V3C1_20k", "./data/VIRET/keyword_classes.txt",
       "./data/VIRET/GoogLeNet2019/05_googlenet.pre-softmax", "./data/VIRET/GoogLeNet2019/05_googlenet.softmax",
       "./data/VIRET/GoogLeNet2019/05_googlenet.deep-features"}};

  ImageRanker::Config cfg{ImageRanker::eMode::cFullAnalytical, datasets, VIRET_data_packs,
                          std::vector<GoogleDataPackRef>(), std::vector<BowDataPackRef>()};

  ImageRanker ranker(cfg);

      // ImageRanker ranker{
      //    R"(c:\Users\devwe\data\imageset1_V3C1\media\images\)",
      //    std::vector<KeywordsFileRef>{
      //        std::tuple(eVocabularyId::VIRET_1200_WORDNET_2019,
      //                   R"(c:\Users\devwe\data\imageset0_V3C1_20k\viret_keywords\keyword_classes.txt)"s)},
      //    std::vector<DataFileSrc>{std::tuple(std::tuple(
      //        eVocabularyId::VIRET_1200_WORDNET_2019, eScoringsId::NASNET_2019,
      //        R"(c:\Users\devwe\data\imageset0_V3C1_20k\viret_keywords\dataset0\Studenti_NasNetLarge.pre-softmax)"))},
      //    std::vector<DataFileSrc>{std::tuple(std::tuple(
      //        eVocabularyId::VIRET_1200_WORDNET_2019, eScoringsId::NASNET_2019,
      //        R"(c:\Users\devwe\data\imageset0_V3C1_20k\viret_keywords\dataset0\Studenti_NasNetLarge.softmax)"))},
      //    std::vector<DataFileSrc>{},
      //    R"(c:\Users\devwe\data\imageset0_V3C1_20k\viret_keywords\dataset0\files.txt)",
      //    1ULL,
      //    ImageRanker::eMode::cFullAnalytical,
      //    std::vector<KeywordsFileRef>{
      //        std::tuple(
      //            eVocabularyId::VIRET_1200_WORDNET_2019,
      //            SOLUTION_DIR +
      //                R"(c:\Users\devwe\data\imageset0_V3C1_20k\viret_keywords\LSC2018-NASNet.all_must_match.word2vec.label)"s),
      //    }};

      // ranker.Initialize();

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
