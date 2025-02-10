
#include "GPU_TP.hpp"

namespace {

__global__ void convolution(char* InImg, char* OutImg, int ImgWidth, int ImgHeight, char* Mask, int MaskWidth, int MaskHeight)
{
	int idX =
		threadIdx.x
		+ blockIdx.x * blockDim.x;

	int idY =
		threadIdx.y
		+ blockIdx.y * blockDim.y;

	int idGlobal = idY * ImgWidth + idX;

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
					OutImg[idGlobal] += InImg[idImg] * Mask[idMask];
				}
			}
		}
	}
}

void convolution(std::vector<char>& image, const int width, const std::vector<char>& mask, const int widthMask)
{
	int height = image.size() / width;
	int heightMask = mask.size() / widthMask;
	cudaError_t err;

	// Allocation de la mémoire sur le GPU
	char* inImage;
	err = cudaMalloc(&inImage, width * height * sizeof(char));
	if (err != cudaSuccess)
	{
		std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
		return;
	}

	char* outImage;
	err = cudaMalloc(&outImage, width * height * sizeof(char));
	if (err != cudaSuccess)
	{
		std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
		return;
	}

	// Initialisation de la mémoire sur le GPU à 0
	err = cudaMemset(outImage, (char) 0, width * height * sizeof(char));
	if (err != cudaSuccess)
	{
		std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
		return;
	}

	char* inMask;
	err = cudaMalloc(&inMask, widthMask * heightMask * sizeof(char));
	if (err != cudaSuccess)
	{
		std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
		return;
	}

	// Copie des données sur le GPU
	err = cudaMemcpy(inImage, image.data(), width * height * sizeof(char), cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
		return;
	}

	err = cudaMemcpy(inMask, mask.data(), widthMask * heightMask * sizeof(char), cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
		return;
	}

	
	// Définition de la taille des blocs et de la grille
	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
	// Appel du kernel
	convolution << <numBlocks, threadsPerBlock >> > (inImage, outImage, width, height, inMask, widthMask, heightMask);
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
		return;
	}

	// Copie des données du GPU vers le CPU
	err = cudaMemcpy(image.data(), outImage, width * height * sizeof(char), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
	{
		std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
		return;
	}

	// Libération de la mémoire
	cudaFree(inImage);
	cudaFree(outImage);
	cudaFree(inMask);
}

__global__ void convolution(char** InImg, char** OutImg, int ImgWidth, int ImgHeight, char* Mask, int MarkWidth, int MaskHeight)
{
	// TODO
}

std::vector<int> convolution(std::vector<int>& image, const int width, const std::vector<int>& mask, const int widthMask)
{
	return std::vector<int>();
}

} // namespace

void runOnGPU()
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
	std::cout << "Image before convolution: " << std::endl;
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
		1, 0, 0,
		0, 0, 0,
		0, 0, 0
	};

	// Convolution
	convolution(image, imageWidth, mask, maskWidth);

	// Affichage de l'image le premier carré de 5x5 pixels de l'image
	std::cout << "Image after convolution: " << std::endl;
	for (int i = 0; i < 5; i++)
	{
		for (int j = 0; j < 5; j++)
		{
			std::cout << (int)image[i * imageWidth + j] << " ";
		}
		std::cout << std::endl;
	}
}
