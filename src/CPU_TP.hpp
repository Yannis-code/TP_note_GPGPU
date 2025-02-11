
#pragma once

#include <iostream>
#include <iomanip>
#include <vector>


namespace CPU_TP {

	/* Fonction de convolution d'une image en niveau de gris sur CPU
	 * @param image: image à traiter
	 * @param width: largeur de l'image
	 * @param mask: masque de convolution
	 * @param widthMask: largeur du masque
	 * @return image après convolution
	 * @note L'overflow n'est pas géré
	 */
	std::vector<unsigned char> convolution(std::vector<unsigned char>& image, const int width, const std::vector<int>& mask, const int widthMask);

	/* Fonction de convolution d'une image en couleur sur CPU
	 * @param image: image à traiter
	 * @param width: largeur de l'image
	 * @param mask: masque de convolution
	 * @param widthMask: largeur du masque
	 * @return image après convolution
	 * @note Les canaux RGB sont traités séparément
	 * @note L'overflow n'est pas géré
	 */
	std::vector<int> convolution(std::vector<int>& image, const int width, const std::vector<int>& mask, const int widthMask);

} // namespace CPU_TP

void runOnCPU_GREY();

void runOnCPU_RGB();