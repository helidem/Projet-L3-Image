# ce programme permet de reconnaitre un tableau noir ou blanc dans une image.
# il sera simple
# ensuite le resultat de l'algorithme sera comparé avec un jeu de données labelisé avec labelme
# pour voir si l'algorithme est bon ou pas
# on ne va pas utiliser des librairies de machine learning pour l'instant, ni opencv


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import os


def load_image(name):
    return (img.imread(name).copy() * 255).astype(np.uint8)


def rgb2gray(image):
    return np.dot(image[..., :3], [0.299, 0.587, 0.114])


def binarisation(image, threshold):
    return image >= threshold


def binarisation2(image, seuil):
    imageBinaire = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            pixel = image[row, col]
            if pixel >= seuil:
                imageBinaire[row, col] = 255
    return imageBinaire


def histogram(image):
    h = np.zeros(256)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            h[image[i, j]] += 1
    return h


def display_histogram(h):
    plt.bar(range(256), h)
    plt.show()


def histogram_proj_hor(bin):
    h = [0] * bin.shape[0]
    for i in range(0, bin.shape[0]):
        for j in range(0, bin.shape[1]):
            h[i] += bin[i, j]

    return np.array(h)


def histogram_proj_ver(bin):
    h = [0] * bin.shape[1]
    for i in range(0, bin.shape[1]):
        for j in range(0, bin.shape[0]):
            h[i] += bin[j, i]

    return np.array(h)


def display_histogram_proj_hor(img, h):
    plt.figure()
    plt.barh(np.arange(img.shape[0]), h[::-1])
    plt.show()


def display_histogram_proj_ver(img, h):
    plt.figure()
    plt.bar(np.arange(img.shape[1]), h)
    plt.show()


def egalisation(image, h):
    hc = np.zeros(256)
    hc[0] = h[0]
    for i in range(1, len(h)):
        hc[i] = hc[i - 1] + h[i]

    new_image = np.zeros(image.shape, dtype=np.uint8)
    N = image.shape[0] * image.shape[1]
    n = len(h)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            val = image[i, j]
            new_val = max(0, n / N * hc[val] - 1)
            new_image[i, j] = new_val
    return new_image


def display_image_gray(image):
    plt.imshow(image, cmap=plt.get_cmap("gray"))
    plt.show()


def display_image_color(image):
    plt.imshow(image, vmin=0, vmax=1)
    plt.show()


def detecter_tableau(image):
    # les étapes sont les suivantes:
    # 1. on va faire une binarisation de l'image
    # 2. on va faire une projection horizontale de l'image
    # 3. on va faire une projection verticale de l'image
    # 4. on va faire une recherche de la ligne de séparation entre le tableau et le reste de l'image
    # 5. on va faire une recherche de la colonne de séparation entre le tableau et le reste de l'image
    # 6. enfin on va faire une découpe de l'image pour ne garder que le tableau

    # 1. on va faire une binarisation de l'image
    image = binarisation2(image, 150)
    # 2. on va faire une projection horizontale de l'image
    h = histogram_proj_hor(image)
    # 3. on va faire une projection verticale de l'image
    v = histogram_proj_ver(image)
    # 4. on va faire une recherche de la ligne de séparation entre le tableau et le reste de l'image
    


def main():
    # Load image
    image = load_image("./../doc1.jpg")
    image = rgb2gray(image)
    image = binarisation2(image, 150)
    display_image_gray(image)


if __name__ == "__main__":
    main()
