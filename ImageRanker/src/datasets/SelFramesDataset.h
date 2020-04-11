#pragma once

#include <memory>
#include <string>
#include <vector>

#include "common.h"
#include "utility.h"

#include "BaseImageset.h"

namespace image_ranker
{
class SelFramesDataset : public BaseImageset
{
 public:
  SelFramesDataset(const std::string& name, const std::string& images_dir, std::vector<SelFrame>&& frames)
      : _name(name), _dir(images_dir), _frames(std::move(frames))
  {
  }

  [[nodiscard]] size_t size() const override { return _frames.size(); }

  [[nodiscard]] const SelFrame& operator[](size_t frame_ID) const override { return _frames.at(frame_ID); }

  [[nodiscard]] const SelFrame& random_frame() const override
  {
    return _frames[rand_integral<size_t>(0, _frames.size())];
  }

  [[nodiscard]] ImagesetInfo get_info() const override { return {_name, _dir, _frames.size()}; }

 private:
  std::string _name;
  std::string _dir;
  std::vector<SelFrame> _frames;
};
}  // namespace image_ranker