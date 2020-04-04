#pragma once

#include "common.h"

struct SelFrame;

class BaseImageset
{
public:
  virtual size_t size() const = 0;
  virtual const SelFrame& operator[](size_t frame_ID) const = 0;
  virtual const SelFrame& random_frame() const = 0;
  virtual ImagesetInfo get_info() const = 0;
};

