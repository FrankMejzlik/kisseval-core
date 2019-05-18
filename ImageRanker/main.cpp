

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

  auto result14{
  ranker.RunModelTestWrapper(
    AggregationId::cXToTheP, RankingModelId::cViretBase, (QueryOriginId)999,
    std::vector<std::string>({ "0"s, "0.0"s, "3"s }), std::vector<std::string>({ "0"s })
  )
  };*/

  std::vector<InteractiveSearchAction> actions;
  actions.emplace_back(1, 100, 299292929);
  actions.emplace_back(1, 300, 41414);
  actions.emplace_back(0, 100, 299292929);

  ranker.SubmitInteractiveSearchSubmit(
    (InteractiveSearchOrigin)0, 1000, (RankingModelId)2, (AggregationId)200,
    std::vector<std::string>{"1.0", "2", "3"}, std::vector<std::string>{"0"},
    "faef2309f2093f2j", 9, 0, 30303, actions
  );


  return 0;

}
