#pragma once

#include <string>

#include "common.h"

class BaseDataPack
{
 public:
  BaseDataPack(const StringId& ID, const std::string& description) : _ID(ID), _description(description) {}

  [[nodiscard]] virtual RankingResult rank_frames(const std::vector<std::string>& user_queries,
                                                  PackModelCommands model_commands, size_t result_size,
                                                  FrameId target_image_ID = ERR_VAL<FrameId>()) const = 0;

  [[nodiscard]] virtual const std::string& get_vocab_ID() const = 0;
  [[nodiscard]] virtual const std::string& get_vocab_description() const = 0;
  [[nodiscard]] virtual const std::string& get_ID() const { return _ID; };
  [[nodiscard]] virtual const std::string& get_description() const { return _description; };

 protected:
  StringId _ID;
  std::string _description;
};
