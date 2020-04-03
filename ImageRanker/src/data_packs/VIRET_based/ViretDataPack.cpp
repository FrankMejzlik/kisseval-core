
#include "ViretDataPack.h"

//#include "classification_transformations.h"
//#include "classification_ranking_models.h"


ViretDataPack::ViretDataPack(const StringId& ID, const ViretDataPackRef::VocabData& vocab_data_refs, 
  std::vector<std::vector<float>>&& presoft, std::vector<std::vector<float>>&& softmax_data,
                           std::vector<std::vector<float>>&& feas_data)
    : _ID(ID),
  _keywords(vocab_data_refs.keyword_synsets_fpth), 
  _presoftmax_data(std::move(presoft)), _softmax_data(std::move(softmax_data)), _feas_data(std::move(feas_data))
{
  // Instantiate all wanted transforms

  // Instantiate all wanted models
}

RankingResult ViretDataPack::rank_frames(const std::vector<std::string>& user_queries,
                                            PackModelCommands model_commands, size_t result_size,
                                            FrameId target_image_ID) const
{
  LOG_WARN("Not implemented!");

  // Parse command string
  return RankingResult();
}
