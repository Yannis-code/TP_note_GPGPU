
#include "CPU_TP.hpp"

namespace {
	std::vector<unsigned char> convolution(std::vector<unsigned char>& image, const int width, const std::vector<char>& mask, const int widthMask)
	{
		std::vector<unsigned char> result;
		// Vérifie que la taille de l'image est un multiple de la largeur de l'image
		if (image.size() % width != 0)
		{
			std::cerr << "Error: image width is not a multiple of the image size" << std::endl;
			return result;
		}
		// Vérifie que la taille du masque est un multiple de la largeur du masque
		if (mask.size() % widthMask != 0)
		{
			std::cerr << "Error: mask width is not a multiple of the mask size" << std::endl;
			return result;
		}

		int height = image.size() / width;
		int heightMask = mask.size() / widthMask;

		// Parcours de l'image
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				// Initialisation de la valeur du pixel
				int value = 0;
				// Parcours du masque
				for (int k = 0; k < heightMask; k++)
				{
					for (int l = 0; l < widthMask; l++)
					{
						int idMask = k * widthMask + l;
						int shift = widthMask / 2;
						int idImg = (i + k - shift) * width + (j + l - shift);
						if ((i + k - shift) >= 0 && (i + k - shift) < height && (j + l - shift) >= 0 && (j + l - shift) < width)
						{
							value += image[idImg] * mask[idMask];
						}
					}
				}
				// Ajout de la valeur du pixel dans le résultat
				result.push_back(value);
			}
		}
		return result;
	}

	std::vector<int> convolution(std::vector<int>& image, const int width, const std::vector<int>& mask, const int widthMask)
	{
		std::vector<int> result;
		// Vérifie que la taille de l'image est un multiple de la largeur de l'image
		if (image.size() % width != 0)
		{
			std::cerr << "Error: image width is not a multiple of the image size" << std::endl;
			return result;
		}
		// Vérifie que la taille du masque est un multiple de la largeur du masque
		if (mask.size() % widthMask != 0)
		{
			std::cerr << "Error: mask width is not a multiple of the mask size" << std::endl;
			return result;
		}

		int height = image.size() / width;
		int heightMask = mask.size() / widthMask;

		// Parcours de l'image
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				// Initialisation de la valeur du pixel
				unsigned char R = 0, G = 0, B = 0;
				// Parcours du masque
				for (int k = 0; k < heightMask; k++)
				{
					for (int l = 0; l < widthMask; l++)
					{
						int idMask = k * widthMask + l;
						int shift = widthMask / 2;
						int idImg = (i + k - shift) * width + (j + l - shift);
						if ((i + k - shift) >= 0 && (i + k - shift) < height && (j + l - shift) >= 0 && (j + l - shift) < width)
						{
							unsigned char maskR = (mask[idMask] >> 24) & 0xFF;
							unsigned char maskG = (mask[idMask] >> 16) & 0xFF;
							unsigned char maskB = (mask[idMask] >> 8) & 0xFF;

							R += ((image[idImg] >> 24) & 0xFF) * maskR;
							G += ((image[idImg] >> 16) & 0xFF) * maskG;
							B += ((image[idImg] >> 8) & 0xFF) * maskB;
						}
					}
				}
				int value = (R << 24) | (G << 16) | (B << 8);
				// Ajout de la valeur du pixel dans le résultat
				result.push_back(value);
			}
		}
		return result;
	}

} // namespace

void runOnCPU_GREY()
{
	int imageWidth, imageHeight, maskWidth;
	std::cout << "Enter the image width: ";
	std::cin >> imageWidth;
	std::cout << "Enter the image height: ";
	std::cin >> imageHeight;

	std::vector<unsigned char> image;
	// Initialisation de l'image à i % 255 pour les 3 canaux RGB
	for (int i = 0; i < imageWidth * imageHeight; i++)
	{
		image.push_back(i);
	}

	// Affichage de l'image le premier carré de 5x5 pixels de l'image
	std::cout << "Image before convolution: " << std::endl;
	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < 10; j++)
		{
			std::cout << std::setw(3) << (int)image[i * imageWidth + j] << " ";
		}
		std::cout << std::endl;
	}

	// Masque
	maskWidth = 3;
	std::vector<char> mask = {
		1, 0, 0,
		0, 0, 0,
		0, 0, 0
	};

	// Convolution
	std::vector<unsigned char> result = convolution(image, imageWidth, mask, maskWidth);
	std::cout << "Result size: " << result.size() << std::endl;

	// Affichage de l'image le premier carré de 5x5 pixels de l'image
	std::cout << "Image after convolution: " << std::endl;
	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < 10; j++)
		{
			std::cout << std::setw(3) << (int)result[i * imageWidth + j] << " ";
		}
		std::cout << std::endl;
	}
}

void runOnCPU_RGB()
{
	int imageWidth, imageHeight, maskWidth;
	std::cout << "Enter the image width: ";
	std::cin >> imageWidth;
	std::cout << "Enter the image height: ";
	std::cin >> imageHeight;

	std::vector<int> image;
	// Initialisation de l'image à i % 255 pour les 3 canaux RGB
	for (int i = 0; i < imageWidth * imageHeight; i++)
	{
		image.push_back((i << 24) | (i << 16) | (i << 8));
	}

	// Affichage de l'image le premier carré de 5x5 pixels canal par canal
	std::cout << "Image after convolution: " << std::endl;
	for (int i = 0; i < 5; i++)
	{
		for (int j = 0; j < 5; j++)
		{
			std::cout << std::setw(3) << (int)((image[i * imageWidth + j] >> 24) & 0xFF) << "|";
			std::cout << std::setw(3) << (int)((image[i * imageWidth + j] >> 16) & 0xFF) << "|";
			std::cout << std::setw(3) << (int)((image[i * imageWidth + j] >> 8) & 0xFF) << "\t";
		}
		std::cout << std::endl;
	}

	// Masque
	maskWidth = 3;
	std::vector<int> mask = {
		(1 << 24) | (1 << 16) | (1 << 8), 0, 0,
		0, 0, 0,
		0, 0, 0
	};

	// Convolution
	std::vector<int> result = convolution(image, imageWidth, mask, maskWidth);
	std::cout << "Result size: " << result.size() << std::endl;

	// Affichage de l'image le premier carré de 5x5 pixels canal par canal
	std::cout << "Image after convolution: " << std::endl;
	for (int i = 0; i < 5; i++)
	{
		for (int j = 0; j < 5; j++)
		{
			std::cout << std::setw(3) << (int)((result[i * imageWidth + j] >> 24) & 0xFF) << "|";
			std::cout << std::setw(3) << (int)((result[i * imageWidth + j] >> 16) & 0xFF) << "|";
			std::cout << std::setw(3) << (int)((result[i * imageWidth + j] >> 8) & 0xFF) << "\t";
		}
		std::cout << std::endl;
	}
}