

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
  };

  auto result13{
  ranker.RunModelTestWrapper(
    AggregationId::cXToTheP, RankingModelId::cViretBase, (QueryOriginId)999,
    std::vector<std::string>({ "0"s, "0.0"s, "2"s }), std::vector<std::string>({ "0"s })
  )
  };
  */
  auto result14{
  ranker.GetRelevantImagesWrapper(

    std::string("&-8252602+-441824+-3206282+-4296562+"), 100,
    AggregationId::cXToTheP, RankingModelId::cViretBase,
    std::vector<std::string>({ "0"s, "0.0"s, "3"s }), std::vector<std::string>({ "1"s, "1"s })
  )
  };

  
  return 0;

}
