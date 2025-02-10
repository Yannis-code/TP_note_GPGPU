
#include "CPU_TP.hpp"

namespace {
	void convolution(std::vector<char>& image, const int width, const std::vector<char>& mask, const int widthMask)
	{
		// Vérifie que la taille de l'image est un multiple de la largeur de l'image
		if (image.size() % width != 0)
		{
			std::cerr << "Error: image width is not a multiple of the image size" << std::endl;
			return;
		}

		// Vérifie que la taille du masque est un multiple de la largeur du masque
		if (mask.size() % widthMask != 0)
		{
			std::cerr << "Error: mask width is not a multiple of the mask size" << std::endl;
			return;
		}

		// Parcours de l'image
		for (int i = 0; i < image.size(); i++)
		{
			// Parcours du masque
			for (int j = 0; j < mask.size(); j++)
			{
				// Calcul de la position de l'élément du masque
				int x = i % width + j % widthMask - widthMask / 2;
				int y = i / width + j / widthMask - widthMask / 2;
				// Vérifie que la position est dans l'image
				if (x >= 0 && x < width && y >= 0 && y < image.size() / width)
				{
					// Calcul de la position de l'élément du masque dans l'image
					int pos = x + y * width;
					// Calcul de la convolution si le pixel est dans l'image
					image[i] += mask[j] * image[pos];
				}
				else {
					// Sinon, on met le pixel à 0
					image[i] = 0;
				}
				// On vérifie que le pixel est dans l'intervalle [0, 255]
				if (image[i] < 0)
				{
					image[i] = 0;
				}
				else if (image[i] > 255)
				{
					image[i] = 255;
				}
			}
		}
	}

	std::vector<int> convolution(std::vector<int>& image, const int width, const std::vector<int>& mask, const int widthMask)
	{
		// TODO
	}

} // namespace

void runOnCPU()
{
	int imageWidth, imageHeight, maskWidth, maskHeight;
	std::cout << "Enter the image width: ";
	std::cin >> imageWidth;
	std::cout << "Enter the image height: ";
	std::cin >> imageHeight;
	std::cout << "Enter the mask width: ";
	std::cin >> maskWidth;
	std::cout << "Enter the mask height: ";
	std::cin >> maskHeight;

	std::vector<char> image;
	// Initialisation de l'image à i % 255
	for (int i = 0; i < imageWidth * imageHeight; i++)
	{
		image.push_back(i % 255);
	}

	// Affichage de l'image le premier carré de 5x5 pixels de l'image
	for (int i = 0; i < 5; i++)
	{
		for (int j = 0; j < 5; j++)
		{
			std::cout << (int)image[i * imageWidth + j] << " ";
		}
		std::cout << std::endl;
	}

	// Masque
	std::vector<char> mask = {
		1, 2, 1,
		2, 4, 2,
		1, 2, 1
	};

	// Convolution
	convolution(image, 5, mask, 3);

	// Affichage de l'image le premier carré de 5x5 pixels de l'image
	for (int i = 0; i < 5; i++)
	{
		for (int j = 0; j < 5; j++)
		{
			std::cout << (int)image[i * imageWidth + j] << " ";
		}
		std::cout << std::endl;
	}
}