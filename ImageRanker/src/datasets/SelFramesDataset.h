#pragma once

#include <string>
#include <vector>
#include <memory>

#include "common.h"
#include "BaseDataset.h"

class SelFramesDataset : public BaseDataset
{
public:
  SelFramesDataset(const std::string& name, const std::string& images_dir, std::vector<SelFrame>&& frames)
      : _name(name), _dir(images_dir), _frames(std::move(frames))
  {
  }

private:
  std::string _name;
  std::string _dir;
  std::vector<SelFrame> _frames;
};
