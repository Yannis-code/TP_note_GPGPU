# TP noté - Convolution d'image sur GPGPU

## Notes

- La question 4 du TP n'a pas été réalisé (par manque d'accès à un GPU) mais l'étape 5 l'a été
- En l'état, l'overflow des types n'est pas géré. L'idéal serait de travailler avec des `int` pour éviter les overflow puis de normaliser l'image dans l'intervalle `[0; 255]`

## Compilation

```cmd
mkdir build_debug
cd build_debug
cmake -DCMAKE_BUILD_TYPE=Debug -G "Visual Studio 15 2017" -A x64 ..
cmake --build .
.\bin\Debug\Debug\TestCompilation.exe
```

## Algorithme CPU pour réaliser une convolution

```cpp
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
```

## Algorithme GPU pour réaliser une convolution

```cpp
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
```
