
#include "ViretDataPack.h"

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
      _softmax_data_raw(std::move(presoft)),
      _keywords(vocab_data_refs)
{
  // Instantiate all wanted transforms
  _transforms.emplace("softmax", std::make_unique<TransformationSoftmax>(_keywords, _softmax_data_raw));
  _transforms.emplace("linear_0-1", std::make_unique<TransformationLinear01>(_keywords, _presoftmax_data_raw));
  _transforms.emplace("no_transform", std::make_unique<BaseVectorTransform>(_presoftmax_data_raw));

  // Instantiate all wanted models
  // Boolean
  // Vector space
  _models.emplace("mult-sum-max", std::make_unique<ViretModel>());
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

RankingResult ViretDataPack::rank_frames(const std::vector<std::string>& user_queries, PackModelCommands model_commands,
                                         size_t result_size, FrameId target_image_ID) const
{
  LOG_WARN("Not implemented!");

  // Parse command string
  return RankingResult();
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

Matrix<float> ViretDataPack::accumulate_hypernyms(const Matrix<float>& data_mat) const
{

}