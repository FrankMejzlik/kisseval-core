
#include "ViretDataPack.h"

//#include "classification_transformations.h"
//#include "classification_ranking_models.h"


ViretDataPack::ViretDataPack(const StringId& ID, const StringId& target_imageset_ID, const std::string& description, const ViretDataPackRef::VocabData& vocab_data_refs, 
  std::vector<std::vector<float>>&& presoft, std::vector<std::vector<float>>&& softmax_data,
                           std::vector<std::vector<float>>&& feas_data)
    : BaseDataPack(ID,target_imageset_ID, description),
  _keywords(vocab_data_refs), 
  _presoftmax_data(std::move(presoft)), _softmax_data(std::move(softmax_data)), _feas_data(std::move(feas_data))
{
  // Instantiate all wanted transforms

  // Instantiate all wanted models
}

const std::string&  ViretDataPack::get_vocab_ID() const
{
  return _keywords.get_ID();
}

const std::string&  ViretDataPack::get_vocab_description() const
{
  return _keywords.get_description();
}

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

RankingResult ViretDataPack::rank_frames(const std::vector<std::string>& user_queries,
                                            PackModelCommands model_commands, size_t result_size,
                                            FrameId target_image_ID) const
{
  LOG_WARN("Not implemented!");

  // Parse command string
  return RankingResult();
}

AutocompleteInputResult ViretDataPack::get_autocomplete_results(const std::string& query_prefix,
                                       size_t result_size, bool with_example_image) const
{
  return { _keywords.GetNearKeywordsPtrs(query_prefix, result_size) };
}

DataPackInfo ViretDataPack::get_info() const 
{
  return DataPackInfo{
    _ID,_description, _target_imageset_ID, _keywords.get_ID(), _keywords.get_description()

  };
}