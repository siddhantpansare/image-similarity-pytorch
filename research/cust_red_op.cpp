#include <torch/torch.h>

/// Implements the repeat interleave operation
/// \param input: torch 1-dimensional Tensor
/// \return torch::Tensor: interleaved 1-dimensional result
torch::Tensor repeatInterleave(torch::Tensor input) {

    int repeatVal = int(512/input.sizes()[0]);
    torch::Tensor output = input.repeat(repeatVal);

    return output;
}

// Implements the Reduction ONNX operator by interleaving the
// \param layerOne: layer 1 output after AdaptiveAvgPooling [1x64] or [64]
// \param layerTwo: layer 2 output after AdaptiveAvgPooling [1x128] or [128]
// \param layerThree: layer 3 output after AdaptiveAvgPooling [1x256] or [256]
// \param layerFour: layer 4 output after AdaptiveAvgPooling [1x512] or [512]
// \return torch::Tensor: reduction output of the tensors [1x512] or [512]
torch::Tensor reduction (
    torch::Tensor layerOne,
    torch::Tensor layerTwo,
    torch::Tensor layerThree,
    torch::Tensor layerFour) {

    torch::Tensor tempLayerOne = torch::adaptive_avg_pool2d(layerOne, {1,1}).reshape({-1});
    torch::Tensor tempLayerTwo = torch::adaptive_avg_pool2d(layerTwo, {1,1}).reshape({-1});
    torch::Tensor tempLayerThree = torch::adaptive_avg_pool2d(layerThree, {1,1}).reshape({-1});
    torch::Tensor tempLayerFour = torch::adaptive_avg_pool2d(layerFour, {1,1}).reshape({-1});

    torch::Tensor interleavedLayerOne = repeatInterleave(tempLayerOne );
    torch::Tensor interleavedLayerTwo = repeatInterleave(tempLayerTwo);
    torch::Tensor interleavedLayerThree = repeatInterleave(tempLayerThree);
    torch::Tensor interleavedLayerFour = repeatInterleave(tempLayerFour);

    torch::Tensor reducedTensor = (interleavedLayerOne + interleavedLayerTwo + interleavedLayerThree + interleavedLayerFour) /4;

    return reducedTensor.clone();
}


static auto registry =
  torch::RegisterOperators("mynamespace::reduction", &reduction);