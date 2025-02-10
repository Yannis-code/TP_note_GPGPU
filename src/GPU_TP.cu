
#include "GPU_TP.hpp"

namespace {

__global__ void convolution(char** InOutImg, int ImgWidth, int ImgHeight, char** Mask, int MaskWidth, int MaskHeight)
{
	int idX =
		threadIdx.x
		+ blockIdx.x * blockDim.x;

	int idY =
		threadIdx.y
		+ blockIdx.y * blockDim.y;

	if (idX < ImgWidth && idY < ImgHeight)
	{
		// Parcours du masque centré sur le pixel (idX, idY)
		int sum = 0;
		for (int i = 0; i < MaskHeight; i++)
		{
			for (int j = 0; j < MaskWidth; j++)
			{
				int x = idX + j - MaskWidth / 2;
				int y = idY + i - MaskHeight / 2;
				if (x >= 0 && x < ImgWidth && y >= 0 && y < ImgHeight)
				{
					sum += InOutImg[y][x] * Mask[i][j];
				}
			}
		}
		InOutImg[idY][idX] = sum;
	}
}

void convolution(std::vector<char>& image, const int width, const std::vector<char>& mask, const int widthMask)
{
	// Allocation de la mémoire sur le GPU avec cudaMallocPitch
	char** d_image;
	auto err = cudaMallocPitch((void**)&d_image, NULL, width * sizeof(char), image.size() / width);
	if (err != cudaSuccess)
	{
		std::cerr << "A" << std::endl;
		std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
		return;
	}

	char** d_mask;
	err = cudaMallocPitch((void**)&d_mask, NULL, widthMask * sizeof(char), mask.size() / width);
	if (err != cudaSuccess)
	{
		std::cerr << "B" << std::endl;
		std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
		return;
	}

	// Copie de l'image sur le GPU avec cudaMemcpy2D
	err = cudaMemcpy2D(d_image, width * sizeof(char), image.data(), width * sizeof(char), width * sizeof(char), image.size() / width, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		std::cerr << "C" << std::endl;
		std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
		return;
	}


	// Copie du masque sur le GPU avec cudaMemcpy2D
	err = cudaMemcpy2D(d_mask, widthMask * sizeof(char), mask.data(), widthMask * sizeof(char), widthMask * sizeof(char), mask.size() / widthMask, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		std::cerr << "D" << std::endl;
		std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
		return;
	}

	// On découpe l'image en blocs de 16x16 pixels (threadsPerBlock)
	// On calcule le nombre de blocs nécessaires pour couvrir l'image entière (numBlocks)
	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (width + threadsPerBlock.y - 1) / threadsPerBlock.y);
	// Appel du kernel
	convolution << <numBlocks, threadsPerBlock >> > (d_image, width, width, d_mask, widthMask, widthMask);
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
		return;
	}

	// Copie du résultat sur le CPU
	for (int i = 0; i < width; i++)
	{
		err = cudaMemcpy(image.data() + i * width, d_image[i], width * sizeof(char), cudaMemcpyDeviceToHost);
		if (err != cudaSuccess)
		{
			std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
			return;
		}
	}

	// Libération de la mémoire
	for (int i = 0; i < width; i++)
	{
		cudaFree(d_image[i]);
	}
	cudaFree(d_image);

	for (int i = 0; i < widthMask; i++)
	{
		cudaFree(d_mask[i]);
	}
	cudaFree(d_mask);
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
		0, 0, 0,
		0, 0, 1,
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
