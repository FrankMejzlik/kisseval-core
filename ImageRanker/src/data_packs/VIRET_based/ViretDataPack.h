#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "KeywordsContainer.h"

#include "data_packs/BaseDataPack.h"

#include "BaseClassificationModel.h"
#include "BaseClassificationTransformation.h"

class TransformationFunctionBase;

class ViretDataPack : public BaseDataPack
{
 public:
  ViretDataPack(const StringId& ID, const ViretDataPackRef::VocabData& vocab_data_refs,
                std::vector<std::vector<float>>&& presoft, std::vector<std::vector<float>>&& softmax_data,
                std::vector<std::vector<float>>&& feas_data);

  [[nodiscard]] RankingResult rank_frames(const std::vector<std::string>& user_queries,
                                          PackModelCommands model_commands, size_t result_size,
                                          FrameId target_image_ID = ERR_VAL<FrameId>()) const override;

 private:
  StringId _ID;
  KeywordsContainer _keywords;

  std::vector<std::vector<float>> _presoftmax_data;
  std::vector<std::vector<float>> _softmax_data;
  std::vector<std::vector<float>> _feas_data;

  std::unordered_map<StringId, std::unique_ptr<BaseClassificationModel>> _models;
  std::unordered_map<StringId, std::unique_ptr<TransformationFunctionBase>> _transforms;

  std::unordered_map<StringId, VecMat> _transformed_data;
};
