

#include <iostream>
#include <thread>
#include <chrono>
#include "ImageRanker.h"

int main()
{
  ImageRanker ranker{
    R"(C:\Users\devwe\source\repos\ImageRanker\data\dataset2\images\)",
    R"(C:\Users\devwe\source\repos\ImageRanker\data\dataset2\Studenti_NasNetLarge.pre-softmax)",
    R"(C:\Users\devwe\source\repos\ImageRanker\data\dataset2\keyword_classes.txt)",
    R"(C:\Users\devwe\source\repos\ImageRanker\data\dataset2\Studenti_NasNetLarge.softmax)",
    R"(C:\Users\devwe\source\repos\ImageRanker\data\dataset2\Studenti_NasNetLarge.deep-features)",
    R"(C:\Users\devwe\source\repos\ImageRanker\data\dataset2\files.txt)",
    1ULL,
    ImageRanker::Mode::cFull
  };

  ranker.Initialize();


  auto result11{
  ranker.GetRelevantImagesWithSuggestedWrapper(
    "9376198&9225146&9225146&9433442&2837789", 1000,
    AggregationId::cXToTheP, RankingModelId::cViretBase,
    std::vector<std::string>({ "0"s, "0.001"s, "1"s }), std::vector<std::string>({ "1"s }),
    15480
  )
  };

  auto result22{
  ranker.GetRelevantImagesWithSuggestedWrapper(
    "9376198&9225146&9225146&9433442", 1000,
    AggregationId::cXToTheP, RankingModelId::cViretBase,
    std::vector<std::string>({ "0"s, "0.001"s, "1"s }), std::vector<std::string>({ "1"s }),
    15480
  )
  };

  auto result12{
   ranker.GetRelevantImagesWithSuggestedWrapper(
     "9376198&9225146&9225146&9433442&~2837789", 1000,
     AggregationId::cXToTheP, RankingModelId::cViretBase,
     std::vector<std::string>({ "0"s, "0.001"s, "1"s }), std::vector<std::string>({ "1"s }),
     4826
   )
  };




  return 0;

}
