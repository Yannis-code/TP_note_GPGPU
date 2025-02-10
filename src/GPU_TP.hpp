
#pragma once

#include <cuda_runtime.h>

#include <iostream>
#include <iomanip>
#include <vector>

namespace {
	void convolution(std::vector<char>& image, const int width, const std::vector<char>& mask, const int widthMask);
	std::vector<int> convolution(std::vector<int>& image, const int width, const std::vector<int>& mask, const int widthMask);
} // namespace

void runOnGPU_GREY();

void runOnGPU_RGB();