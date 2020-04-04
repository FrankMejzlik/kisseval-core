

#include <array>
#include <chrono>
#include <iostream>
#include <thread>
#include "ImageRanker.h"

#define BASE_DIR "./data/"

int main()
{
  // V3C1 20k subset
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

  ranker.submit_annotator_user_queries(enum_label(eDataPackId::NASNET_2019).first, 9, true,
                                       {
                                           {"Shonicka1", "123&345&3232", "car, cat, cow", 4321},
                                           {"Shonicka2", "1213&3451&32321", "cars, cats, cows", 5321},
                                       });

  return 0;
}
