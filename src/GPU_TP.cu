
#include "GPU_TP.hpp"

namespace GPU_TP {

	/* Kernel de convolution en niveaux de gris sur GPU
	* @param InImg: image d'entrée
	* @param OutImg: image de sortie
	* @param ImgWidth: largeur de l'image
	* @param ImgHeight: hauteur de l'image
	* @param Mask: masque de convolution
	* @param MaskWidth: largeur du masque
	* @param MaskHeight: hauteur du masque
	* @note: le masque est copié dans la mémoire partagée
	* @note: L'overflow n'est pas géré
	*/
	__global__ void convolution(unsigned char* InImg, unsigned char* OutImg, int ImgWidth, int ImgHeight, char* Mask, int MaskWidth, int MaskHeight)
	{
		int idX =
			threadIdx.x
			+ blockIdx.x * blockDim.x;

		int idY =
			threadIdx.y
			+ blockIdx.y * blockDim.y;

		int idGlobal = idY * ImgWidth + idX;

		// Copie du masque dans la mémoire partagée MaskWidth * MaskHeight
		extern __shared__ char sharedMask[];
		if (threadIdx.x < MaskWidth && threadIdx.y < MaskHeight)
		{
			sharedMask[threadIdx.y * MaskWidth + threadIdx.x] = Mask[threadIdx.y * MaskWidth + threadIdx.x];
		}

		__syncthreads();

		if (idX < ImgWidth && idY < ImgHeight)
		{
			int value = 0;
			for (int i = 0; i < MaskHeight; i++)
			{
				for (int j = 0; j < MaskWidth; j++)
				{
					int idMask = i * MaskWidth + j;
					int shift = MaskHeight / 2;
					int idImg = (idY + i - shift) * ImgWidth + (idX + j - shift);
					if ((idY + i - shift) >= 0 && (idY + i - shift) < ImgHeight && (idX + j - shift) >= 0 && (idX + j - shift) < ImgWidth)
					{
						value += InImg[idImg] * sharedMask[idMask];
					}
				}
			}
			OutImg[idGlobal] = (unsigned char)(value / (MaskWidth * MaskHeight));
		}
	}

	/* Kernel de convolution en niveaux de gris sur GPU
	* @param InImg: image d'entrée
	* @param OutImg: image de sortie
	* @param ImgWidth: largeur de l'image
	* @param ImgHeight: hauteur de l'image
	* @param Mask: masque de convolution
	* @param MaskWidth: largeur du masque
	* @param MaskHeight: hauteur du masque
	* @note: le masque est copié dans la mémoire partagée
	* @note: L'overflow n'est pas géré
	*/
	__global__ void convolution(int* InImg, int* OutImg, int ImgWidth, int ImgHeight, int* Mask, int MaskWidth, int MaskHeight)
	{
		int idX =
			threadIdx.x
			+ blockIdx.x * blockDim.x;

		int idY =
			threadIdx.y
			+ blockIdx.y * blockDim.y;

		int idGlobal = idY * ImgWidth + idX;

		// Copie du masque dans la mémoire partagée MaskWidth * MaskHeight
		extern __shared__ int sharedMaskRgb[];
		if (threadIdx.x < MaskWidth && threadIdx.y < MaskHeight)
		{
			sharedMaskRgb[threadIdx.y * MaskWidth + threadIdx.x] = Mask[threadIdx.y * MaskWidth + threadIdx.x];
		}

		__syncthreads();

		// On vérifie que le pixel est dans l'image
		if (idX < ImgWidth && idY < ImgHeight)
		{
			// Initialisation de la valeur du pixel
			int R = 0, G = 0, B = 0;
			// Parcours du masque
			for (int i = 0; i < MaskHeight; i++)
			{
				for (int j = 0; j < MaskWidth; j++)
				{
					int idMask = i * MaskWidth + j;
					int shift = MaskHeight / 2;
					int idImg = (idY + i - shift) * ImgWidth + (idX + j - shift);
					if ((idY + i - shift) >= 0 && (idY + i - shift) < ImgHeight && (idX + j - shift) >= 0 && (idX + j - shift) < ImgWidth)
					{
						char maskR = (sharedMaskRgb[idMask] >> 24) & 0xFF;
						char maskG = (sharedMaskRgb[idMask] >> 16) & 0xFF;
						char maskB = (sharedMaskRgb[idMask] >> 8) & 0xFF;

						R += ((InImg[idImg] >> 24) & 0xFF) * maskR;
						G += ((InImg[idImg] >> 16) & 0xFF) * maskG;
						B += ((InImg[idImg] >> 8) & 0xFF)  * maskB;
					}
				}
			}
			// Ajout de la valeur du pixel dans le résultat
			OutImg[idGlobal] =
					  ((unsigned char) (R / (MaskWidth * MaskHeight)) << 24)
					| ((unsigned char) (G / (MaskWidth * MaskHeight)) << 16)
					| ((unsigned char) (B / (MaskWidth * MaskHeight)) << 8);
		}
	}

	std::vector<unsigned char> convolution(std::vector<unsigned char>& image, const int width, const std::vector<char>& mask, const int widthMask)
	{
		std::vector<unsigned char> result(image.size(), 0);

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

		// Allocation de la mémoire sur le GPU
		unsigned char* inImage;
		gpuErrchk(cudaMalloc(&inImage, width * height * sizeof(unsigned char)));

		unsigned char* outImage;
		gpuErrchk(cudaMalloc(&outImage, width * height * sizeof(unsigned char)));

		// Initialisation de la mémoire sur le GPU à 0
		gpuErrchk(cudaMemset(outImage, (unsigned char)0, width * height * sizeof(unsigned char)));

		char* inMask;
		gpuErrchk(cudaMalloc(&inMask, widthMask * heightMask * sizeof(char)));

		// Copie des données sur le GPU
		gpuErrchk(cudaMemcpy(inImage, image.data(), width * height * sizeof(unsigned char), cudaMemcpyHostToDevice));

		gpuErrchk(cudaMemcpy(inMask, mask.data(), widthMask * heightMask * sizeof(unsigned char), cudaMemcpyHostToDevice));


		// Définition de la taille des blocs et de la grille
		dim3 dimBloc( 512 );
		dim3 dimGrille( (width * height) / dimBloc.x + ( (width * height) % dimBloc.x != 0 ) );
		// Appel du kernel convolution avec la taille du masque en mémoire partagée
		convolution << <dimGrille, dimBloc, widthMask* heightMask * sizeof(char) >> > (inImage, outImage, width, height, inMask, widthMask, heightMask);
		gpuErrchk(cudaGetLastError());

		// Copie des données du GPU vers le CPU
		gpuErrchk(cudaMemcpy(result.data(), outImage, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost));

		// Libération de la mémoire
		gpuErrchk(cudaFree(inImage));
		gpuErrchk(cudaFree(outImage));
		gpuErrchk(cudaFree(inMask));

		return result;
	}

	std::vector<int> convolution(std::vector<int>& image, const int width, const std::vector<int>& mask, const int widthMask)
	{
		std::vector<int> result(image.size(), 0);

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

		// Allocation de la mémoire sur le GPU
		int* inImage;
		gpuErrchk(cudaMalloc(&inImage, width * height * sizeof(int)));

		int* outImage;
		gpuErrchk(cudaMalloc(&outImage, width * height * sizeof(int)));

		// Initialisation de la mémoire sur le GPU à 0
		gpuErrchk(cudaMemset(outImage, (int)0, width * height * sizeof(int)));

		int* inMask;
		gpuErrchk(cudaMalloc(&inMask, widthMask * heightMask * sizeof(int)));

		// Copie des données sur le GPU
		gpuErrchk(cudaMemcpy(inImage, image.data(), width * height * sizeof(int), cudaMemcpyHostToDevice));

		gpuErrchk(cudaMemcpy(inMask, mask.data(), widthMask * heightMask * sizeof(int), cudaMemcpyHostToDevice));

		// Définition de la taille des blocs et de la grille
		dim3 dimBloc( 512 );
		dim3 dimGrille( (width * height) / dimBloc.x + ( (width * height) % dimBloc.x != 0 ) );
		// Appel du kernel convolution avec la taille du masque en mémoire partagée
		convolution << <dimGrille, dimBloc, widthMask* heightMask * sizeof(int) >> > (inImage, outImage, width, height, inMask, widthMask, heightMask);
		gpuErrchk(cudaGetLastError());

		// Copie des données du GPU vers le CPU
		gpuErrchk(cudaMemcpy(result.data(), outImage, width * height * sizeof(unsigned int), cudaMemcpyDeviceToHost));

		// Libération de la mémoire
		gpuErrchk(cudaFree(inImage));
		gpuErrchk(cudaFree(outImage));
		gpuErrchk(cudaFree(inMask));

		return result;
	}

} // namespace GPU_TP

void runOnGPU_GREY()
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
		9, 0, 0,
		0, 0, 0,
		0, 0, 0
	};

	// Convolution
	std::vector<unsigned char> result = GPU_TP::convolution(image, imageWidth, mask, maskWidth);
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

void runOnGPU_RGB()
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
		(9 << 24) | (9 << 16) | (9 << 8), 0, 0,
		0, 0, 0,
		0, 0, 0
	};

	// Convolution
	std::vector<int> result = GPU_TP::convolution(image, imageWidth, mask, maskWidth);
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
