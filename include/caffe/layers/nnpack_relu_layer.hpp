#pragma once

#include "caffe/layers/relu_layer.hpp"

namespace caffe {
#ifdef USE_NNPACK
template <typename Dtype>
class NNPackReLULayer : public ReLULayer<Dtype> {
 public:
  explicit NNPackReLULayer(const LayerParameter& param)
      : ReLULayer<Dtype>(param) {}
  virtual inline const char* type() const {
    return "NNPackReLU";
  }
  virtual void Forward_cpu(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
};
#endif
}
