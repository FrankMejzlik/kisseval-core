
#include "ViretDataPack.h"

#include <thread>

#include "./ranking_models/ranking_models.h"
#include "./transformations/transformations.h"

using namespace image_ranker;

ViretDataPack::ViretDataPack(const StringId& ID, const StringId& target_imageset_ID, const std::string& description,
                             const ViretDataPackRef::VocabData& vocab_data_refs,
                             std::vector<std::vector<float>>&& presoft, std::vector<std::vector<float>>&& softmax_data,
                             std::vector<std::vector<float>>&& feas_data)
    : BaseDataPack(ID, target_imageset_ID, description),
      _feas_data_raw(std::move(feas_data)),
      _presoftmax_data_raw(std::move(presoft)),
      _softmax_data_raw(std::move(softmax_data)),
      _keywords(vocab_data_refs)
{
  // Instantiate all wanted transforms
  std::thread t1([this]() {
    _transforms.emplace(enum_label(eTransformationIds::SOFTMAX).first,
                        std::make_unique<TransformationSoftmax>(_keywords, _softmax_data_raw));
  });
  std::thread t2([this]() {
    _transforms.emplace(enum_label(eTransformationIds::LINEAR_01).first,
                        std::make_unique<TransformationLinear01>(_keywords, _presoftmax_data_raw));
  });
  std::thread t3([this]() {
    _transforms.emplace(enum_label(eTransformationIds::NO_TRANSFORM).first,
                        std::make_unique<BaseVectorTransform>(_presoftmax_data_raw, _presoftmax_data_raw));
  });

  // Instantiate all wanted models
  // Boolean
  // Vector space
  _models.emplace(enum_label(eModelIds::MULT_SUM_MAX).first, std::make_unique<ViretModel>());

  t1.join();
  t2.join();
  t3.join();
}

const std::string& ViretDataPack::get_vocab_ID() const { return _keywords.get_ID(); }

const std::string& ViretDataPack::get_vocab_description() const { return _keywords.get_description(); }

[[nodiscard]] std::string ViretDataPack::humanize_and_query(const std::string& and_query) const
{
  LOG_WARN("Not implemented!");

  return "I am just dummy query!"s;
}

[[nodiscard]] std::vector<Keyword*> ViretDataPack::top_frame_keywords(FrameId frame_ID) const
{
  LOG_WARN("Not implemented!");

  return std::vector({
      _keywords.GetKeywordPtrByVectorIndex(0),
      _keywords.GetKeywordPtrByVectorIndex(1),
      _keywords.GetKeywordPtrByVectorIndex(2),
  });
}

RankingResult ViretDataPack::rank_frames(const std::vector<CnfFormula>& user_queries, PackModelCommands model_commands,
                                         size_t result_size, FrameId target_image_ID) const
{
  std::vector<std::string> tokens = split(model_commands, ';');

  std::vector<ModelKeyValOption> opt_key_vals;

  std::string model_ID;
  std::string transform_ID;

  for (auto&& tok : tokens)
  {
    auto key_val = split(tok, '=');

    // Model ID && Transform ID
    if (key_val[0] == enum_label(eModelOptsKeys::MODEL_ID).first ||
        key_val[0] == enum_label(eModelOptsKeys::TRANSFORM_ID).first)
    {
      model_ID = key_val[1];
    }
    // Options for model itself
    else
    {
      opt_key_vals.emplace_back();
    }
  }

  // Choose desired model
  auto iter_m = _models.find(model_ID);
  if (iter_m == _models.end())
  {
    LOG_ERROR("Uknown model_ID: '" + model_ID +"'.");
    return RankingResult{};
  }
  const auto& ranking_model = *(iter_m->second);

  // Choose desired transform
  auto iter_t = _transforms.find(transform_ID);
  if (iter_t == _transforms.end())
  {
    LOG_ERROR("Uknown transform_ID: '" + transform_ID +"'.");
    return RankingResult{};
  }
  const auto& transform = *(iter_t->second);

  // Run this model
  return ranking_model.rank_frames(transform, _keywords, user_queries, result_size, opt_key_vals);
}

AutocompleteInputResult ViretDataPack::get_autocomplete_results(const std::string& query_prefix, size_t result_size,
                                                                bool with_example_image) const
{
  return {_keywords.GetNearKeywordsPtrs(query_prefix, result_size)};
}

DataPackInfo ViretDataPack::get_info() const
{
  return DataPackInfo{get_ID(), get_description(), target_imageset_ID(), _keywords.get_ID(), _keywords.get_description()

  };
}
