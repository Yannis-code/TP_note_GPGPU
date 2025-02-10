
#pragma once

#include <iostream>
#include <iomanip>
#include <vector>


namespace {

	std::vector<unsigned char> convolution(std::vector<unsigned char>& image, const int width, const std::vector<int>& mask, const int widthMask);

	std::vector<int> convolution(std::vector<int>& image, const int width, const std::vector<int>& mask, const int widthMask);

} // namespace

void runOnCPU_GREY();

void runOnCPU_RGB();