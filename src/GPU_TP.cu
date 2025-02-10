
#include "GPU_TP.hpp"

namespace {

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
			for (int i = 0; i < MaskHeight; i++)
			{
				for (int j = 0; j < MaskWidth; j++)
				{
					int idMask = i * MaskWidth + j;
					int shift = MaskHeight / 2;
					int idImg = (idY + i - shift) * ImgWidth + (idX + j - shift);
					if ((idY + i - shift) >= 0 && (idY + i - shift) < ImgHeight && (idX + j - shift) >= 0 && (idX + j - shift) < ImgWidth)
					{
						OutImg[idGlobal] += InImg[idImg] * sharedMask[idMask];
					}
				}
			}
		}
	}

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
		extern __shared__ char sharedMask[];
		if (threadIdx.x < MaskWidth && threadIdx.y < MaskHeight)
		{
			sharedMask[threadIdx.y * MaskWidth + threadIdx.x] = Mask[threadIdx.y * MaskWidth + threadIdx.x];
		}

		__syncthreads();

		// On vérifie que le pixel est dans l'image
		if (idX < ImgWidth && idY < ImgHeight)
		{
			// Initialisation de la valeur du pixel
			unsigned char R = 0, G = 0, B = 0;
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
						R += (InImg[idImg] >> 24) & 0xFF * sharedMask[idMask];
						G += (InImg[idImg] >> 16) & 0xFF * sharedMask[idMask];
						B += (InImg[idImg] >> 8) & 0xFF * sharedMask[idMask];
					}
				}
			}
			// Ajout de la valeur du pixel dans le résultat
			OutImg[idGlobal] = (R << 24) | (G << 16) | (B << 8);
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
		cudaError_t err;

		// Allocation de la mémoire sur le GPU
		unsigned char* inImage;
		err = cudaMalloc(&inImage, width * height * sizeof(unsigned char));
		if (err != cudaSuccess)
		{
			std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
			return result;
		}

		unsigned char* outImage;
		err = cudaMalloc(&outImage, width * height * sizeof(unsigned char));
		if (err != cudaSuccess)
		{
			std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
			return result;
		}

		// Initialisation de la mémoire sur le GPU à 0
		err = cudaMemset(outImage, (unsigned char)0, width * height * sizeof(unsigned char));
		if (err != cudaSuccess)
		{
			std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
			return result;
		}

		char* inMask;
		err = cudaMalloc(&inMask, widthMask * heightMask * sizeof(char));
		if (err != cudaSuccess)
		{
			std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
			return result;
		}

		// Copie des données sur le GPU
		err = cudaMemcpy(inImage, image.data(), width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
		if (err != cudaSuccess)
		{
			std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
			return result;
		}

		err = cudaMemcpy(inMask, mask.data(), widthMask * heightMask * sizeof(unsigned char), cudaMemcpyHostToDevice);
		if (err != cudaSuccess)
		{
			std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
			return result;
		}


		// Définition de la taille des blocs et de la grille
		dim3 threadsPerBlock(16, 16);
		dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
		// Appel du kernel convolution avec la taille du masque en mémoire partagée
		convolution << <numBlocks, threadsPerBlock, widthMask* heightMask * sizeof(char) >> > (inImage, outImage, width, height, inMask, widthMask, heightMask);
		err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
			return result;
		}

		// Copie des données du GPU vers le CPU
		err = cudaMemcpy(result.data(), outImage, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
		if (err != cudaSuccess)
		{
			std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
			return result;
		}

		// Libération de la mémoire
		cudaFree(inImage);
		cudaFree(outImage);
		cudaFree(inMask);

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
		cudaError_t err;

		// Allocation de la mémoire sur le GPU
		int* inImage;
		err = cudaMalloc(&inImage, width * height * sizeof(int));
		if (err != cudaSuccess)
		{
			std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
			return result;
		}

		int* outImage;
		err = cudaMalloc(&outImage, width * height * sizeof(int));
		if (err != cudaSuccess)
		{
			std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
			return result;
		}

		// Initialisation de la mémoire sur le GPU à 0
		err = cudaMemset(outImage, (int)0, width * height * sizeof(int));
		if (err != cudaSuccess)
		{
			std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
			return result;
		}

		int* inMask;
		err = cudaMalloc(&inMask, widthMask * heightMask * sizeof(char));
		if (err != cudaSuccess)
		{
			std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
			return result;
		}

		// Copie des données sur le GPU
		err = cudaMemcpy(inImage, image.data(), width * height * sizeof(int), cudaMemcpyHostToDevice);
		if (err != cudaSuccess)
		{
			std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
			return result;
		}

		err = cudaMemcpy(inMask, mask.data(), widthMask * heightMask * sizeof(int), cudaMemcpyHostToDevice);
		if (err != cudaSuccess)
		{
			std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
			return result;
		}


		// Définition de la taille des blocs et de la grille
		dim3 threadsPerBlock(16, 16);
		dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
		// Appel du kernel convolution avec la taille du masque en mémoire partagée
		convolution << <numBlocks, threadsPerBlock, widthMask* heightMask * sizeof(int) >> > (inImage, outImage, width, height, inMask, widthMask, heightMask);
		err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
			return result;
		}

		// Copie des données du GPU vers le CPU
		err = cudaMemcpy(result.data(), outImage, width * height * sizeof(unsigned int), cudaMemcpyDeviceToHost);
		if (err != cudaSuccess)
		{
			std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
			return result;
		}

		// Libération de la mémoire
		cudaFree(inImage);
		cudaFree(outImage);
		cudaFree(inMask);

		return result;
	}

} // namespace

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
		1, 0, 0,
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
