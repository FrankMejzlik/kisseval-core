#pragma once

struct SelFrame;

class BaseDataset
{
public:
  virtual size_t size() const = 0;
  virtual const SelFrame& operator[](size_t frame_ID) const = 0;
  virtual const SelFrame& random_frame() const = 0;
};

