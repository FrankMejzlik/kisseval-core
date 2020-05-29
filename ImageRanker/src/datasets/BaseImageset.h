#pragma once

#include "common.h"

namespace image_ranker
{
struct SelFrame;

class [[nodiscard]] BaseImageset
{
 public:
  // -----------------------------------------
  // We need virtual dctor.
  BaseImageset() = default;
  BaseImageset(const BaseImageset& other) = default;
  BaseImageset(BaseImageset&& other) = default;
  BaseImageset& operator=(const BaseImageset& other) = default;
  BaseImageset& operator=(BaseImageset&& other) = default;
  virtual ~BaseImageset() noexcept = default;
  // -----------------------------------------

  virtual size_t size() const = 0;
  virtual const SelFrame& operator[](size_t frame_ID) const = 0;
  virtual const SelFrame& random_frame() const = 0;
  virtual ImagesetInfo get_info() const = 0;
};
}  // namespace image_ranker