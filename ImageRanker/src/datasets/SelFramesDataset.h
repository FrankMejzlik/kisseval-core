#pragma once

#include <memory>
#include <string>
#include <vector>

#include "BaseDataset.h"
#include "common.h"

class SelFramesDataset : public BaseDataset
{
 public:
  SelFramesDataset(const std::string& name, const std::string& images_dir, std::vector<SelFrame>&& frames)
      : _name(name), _dir(images_dir), _frames(std::move(frames))
  {
  }

  [[nodiscard]] size_t size() const override { return _frames.size(); }

  [[nodiscard]] const SelFrame& operator[](size_t frame_ID) const override
  {
    return _frames.at(frame_ID);
  }

 private:
  std::string _name;
  std::string _dir;
  std::vector<SelFrame> _frames;
};
