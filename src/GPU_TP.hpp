
#pragma once

#include "CudaHelper.h"

#include <iostream>
#include <iomanip>
#include <vector>

namespace GPU_TP {
	/* Fonction de convolution en niveaux de gris sur GPU
	* @param image: image d'entrée
	* @param width: largeur de l'image
	* @param mask: masque de convolution
	* @param widthMask: largeur du masque
	*/
	std::vector<unsigned char> convolution(std::vector<unsigned char>& image, const int width, const std::vector<char>& mask, const int widthMask);

	/* Fonction de convolution en niveaux de gris sur GPU
	* @param image: image d'entrée
	* @param width: largeur de l'image
	* @param mask: masque de convolution
	* @param widthMask: largeur du masque
	*/
	std::vector<int> convolution(std::vector<int>& image, const int width, const std::vector<int>& mask, const int widthMask);
} // namespace GPU_TP

void runOnGPU_GREY();

void runOnGPU_RGB();