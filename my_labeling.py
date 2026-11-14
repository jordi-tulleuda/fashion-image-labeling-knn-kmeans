__authors__ = ['UAB AI Team 25']
__group__ = 'Team_25'

# 0. IMPORTS
import time
import matplotlib.pyplot as plt
import numpy as np
from Kmeans import KMeans, get_colors
from KNN import KNN
from utils_data import read_dataset, read_extended_dataset, crop_images, visualize_k_means, visualize_retrieval
# 0. FI IMPORTS

# 1. FUNCIONS D'ANÀLISI JORDI
def Retrieval_by_shape(images, shape_labels, shape_query, shape_percents=None):
    resultats = []

    # Recorrem totes les imatges per comprovar si la seva etiqueta coincideix amb la consulta.
    for i in range(len(images)):
        if shape_labels[i] == shape_query:
            # Si no tenim percentatges de confiança, assignem puntuació 0.
            if shape_percents is None:
                puntuacio = 0
            else:
                # Si tenim percentatges, guardem la confiança associada a aquella imatge.
                puntuacio = shape_percents[i]

            # Afegim la imatge trobada i la seva puntuació a la llista de resultats.
            resultats.append({'imatge': images[i], 'puntuacio': puntuacio})

    # Si tenim percentatges, ordenem les imatges de més a menys confiança.
    if shape_percents is not None:
        resultats.sort(key=lambda x: x['puntuacio'], reverse=True)

    # Extraiem només les imatges de la llista de resultats ordenats.
    imatges_ordenades = [item['imatge'] for item in resultats]

    # Retornem les imatges recuperades com un array de NumPy.
    return np.array(imatges_ordenades)


def get_shape_accuracy(shape_labels, gt_labels, valors, xlabel='K'):
    # Convertim les etiquetes a array per poder comparar-les amb les prediccions.
    gt_labels = np.array(gt_labels)
    resultats = {}

    # Recorrem cada distància utilitzada pel KNN: euclidean, cityblock i cosine.
    for distancia in shape_labels:
        accuracies = []

        # Recorrem les prediccions obtingudes amb aquesta distància per cada valor de K.
        for prediccions in shape_labels[distancia]:
            # Convertim les prediccions a array per comparar-les amb el Ground Truth.
            prediccions = np.array(prediccions)

            # Comptem quantes prediccions coincideixen amb les etiquetes reals, calculem % i guardem el resultat
            correctes = np.sum(prediccions == gt_labels)
            accuracy = (correctes / len(gt_labels)) * 100
            accuracies.append(accuracy)

        # Guardem totes les accuracies associades a aquesta distància.
        resultats[distancia] = accuracies

    # Creem una nova figura per representar la comparació de distàncies.
    plt.figure(figsize=(8, 5))

    # Dibuixem una línia per cada distància, mostrant l'accuracy segons el valor de K.
    for distancia in resultats:
        plt.plot(valors, resultats[distancia], marker='o', label=distancia)

    # Afegim el títol i els noms dels eixos de la gràfica.
    plt.title('Shape accuracy vs K segons la distància')
    plt.xlabel(xlabel)
    plt.ylabel('Accuracy (%)')

    # Agrupem totes les accuracies per ajustar automàticament l'eix Y als valors reals.
    totes_accuracies = []
    for distancia in resultats:
        totes_accuracies += resultats[distancia]

    marge = 1
    plt.ylim(min(totes_accuracies) - marge, max(totes_accuracies) + marge)
    # Posem quadrícula, llegenda i ajustem el format de la figura.
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # Mostrem la gràfica final
    plt.show()
    return resultats
# 1. FI FUNCIONS D'ANÀLISI JORDI

def retrieval_by_color(images, labels, query_colors):
    retrieved_imgs = []
    retrieved_ids = []
    # Convertir string a llista
    if isinstance(query_colors, str):
        query_colors = [query_colors]
    for idx, image_labels in enumerate(labels):
        image_labels = [c.lower() for c in image_labels]
        query = [c.lower() for c in query_colors]
        # Comprovar coincidència
        if any(color in image_labels for color in query):
            retrieved_imgs.append(images[idx])
            retrieved_ids.append(idx)
    return retrieved_imgs, retrieved_ids


def get_color_accuracy(predicted_colors, gt_labels, threshold=0.5):
    correct = 0
    total = len(gt_labels)
    for pred, gt in zip(predicted_colors, gt_labels):
        pred_set = set([c.lower() for c in pred])
        gt_set = set([c.lower() for c in gt])
        intersection = pred_set.intersection(gt_set)
        # percentatge de coincidència
        score = len(intersection) / len(gt_set)
        # només compta com correcte si supera el llindar
        if score >= threshold:
            correct += 1
    return correct / total
# 2. FI FUNCIONS D'ANÀLISI MARC

# 3. FUNCIONS D'ANÀLISI PAU
def Retrieval_combined(images, shape_labels, color_labels, shape_query, color_query, shape_percents, color_percents):
    resultats = []
    for i in range(len(images)):
        forma_ok = shape_labels[i] == shape_query
        color_ok = False
        puntuacio_forma = 0
        puntuacio_color = 0

        if forma_ok:
            puntuacio_forma = shape_percents[i]

        if color_query in color_labels[i]:
            color_ok = True
            index_color = color_labels[i].index(color_query)
            puntuacio_color = color_percents[i][index_color]

        if forma_ok and color_ok:
            mitjana = (puntuacio_forma + puntuacio_color) / 2.0
            resultats.append({'imatge': images[i], 'puntuacio': mitjana})

    resultats.sort(key=lambda x: x['puntuacio'], reverse=True)
    imatges_ordenades = [item['imatge'] for item in resultats]
    return imatges_ordenades


def Kmean_statistics(kmeans_instance, images, Kmax):
    valors_k = np.arange(2, Kmax + 1) # Creem un array amb tots els valors de k que volem provar, des de dos fins a Kmax + 1
    wcd = []
    temps = []
    iteracions = []

    for k in valors_k:
        kmeans_instance.K = k
        kmeans_instance.X = images

        t_inici = time.time() # Guardem el temps just abans d'executar el fit per saber quant tarda
        kmeans_instance.fit()
        t_fi = time.time()

        temps.append(t_fi - t_inici)
        wcd.append(kmeans_instance.withinClassDistance())
        iteracions.append(kmeans_instance.num_iter)

    # Dibuix de les gràfiques
    plt.figure(figsize=(18, 5))

    # Gràfica 1: WCD
    plt.subplot(1, 3, 1)
    plt.plot(valors_k, wcd, marker='o')
    plt.title('WCD vs K')
    plt.xlabel('K')
    plt.ylabel('WCD')
    plt.grid(True)

    # Gràfica 2: Temps
    plt.subplot(1, 3, 2)
    plt.plot(valors_k, temps, marker='s')
    plt.title('Temps vs K')
    plt.xlabel('K')
    plt.ylabel('Temps (s)')
    plt.grid(True)

    # Gràfica 3: Iteracions
    plt.subplot(1, 3, 3)
    plt.plot(valors_k, iteracions, marker='^')
    plt.title('Iteracions vs K')
    plt.xlabel('K')
    plt.ylabel('Iteracions')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
# 3. FI FUNCIONS D'ANÀLISI PAU

if __name__ == '__main__':
    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(root_folder='./images/', gt_json='./images/gt.json')
    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))
    # Load extended ground truth
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_images = crop_images(imgs, upper, lower)

    # You can start coding your functions here
    # 1. MAIN JORDI
    # Anàlisi quantitativa del KNN per diferents valors de K i diferents distàncies
    print("Calculant accuracy de formes amb diferents distàncies...")

    # Definim els valors de K que volem provar
    valors_k = np.arange(1, 26, 2)

    # Definim les tres mètriques de distància.
    distances = ['euclidean', 'cityblock', 'cosine']
    prediccions_per_distancia = {}

    for distancia in distances:
        # Creem un classificador KNN amb les imatges i etiquetes d'entrenament.
        knn_shape = KNN(train_imgs, train_class_labels)

        prediccions_per_k = []

        # Recorrem tots els valors de K definits anteriorment.
        for k in valors_k:
            # Predim la forma de les imatges de test utilitzant el valor de K i la distància actual.
            prediccions = knn_shape.predict(test_imgs, k, distance=distancia)
            prediccions_per_k.append(prediccions)

        prediccions_per_distancia[distancia] = prediccions_per_k

    # Calculem i visualitzem l'accuracy comparant les tres distàncies.
    accuracies_shape = get_shape_accuracy(prediccions_per_distancia, test_class_labels, valors=valors_k)
    print("Accuracy de formes amb diferents distàncies OK")

    # Anàlisi qualitativa amb Retrieval_by_shape

    # Triem la forma que volem buscar dins de les imatges classificades pel KNN.
    # Es pot substituir classes[0] per "Jeans", "Shirts", etc.
    shape_query = classes[0]

    # Creem un KNN per obtenir les etiquetes de forma predites de les imatges de test.
    knn_shape = KNN(train_imgs, train_class_labels)

    # Definim el nombre de veïns que utilitzarà el KNN
    k_shape = 5

    # Predim les formes de les imatges de test amb K=5 i distància euclidean.
    pred_shape_labels = knn_shape.predict(test_imgs, k=k_shape, distance='euclidean')

    # OPCIÓ 1: Retrieval_by_shape SENSE percentatge de confiança

    # imatges_shape = Retrieval_by_shape(test_imgs, pred_shape_labels, shape_query)

    # OPCIÓ 2: Retrieval_by_shape AMB percentatge de confiança

    shape_percents = []

    for veins in knn_shape.neighbors:
        percentatge = (np.sum(veins == shape_query) / k_shape) * 100
        shape_percents.append(percentatge)

    imatges_shape = Retrieval_by_shape( test_imgs, pred_shape_labels, shape_query, shape_percents)

    # Visualitzem les imatges recuperades
    visualize_retrieval(imatges_shape, topN=8, title='Retrieval by shape: ' + shape_query)
    # 1. FI MAIN JORDI

    # 2. MAIN MARC
    print("Calculant accuracy de colors...")

    valors_k = np.arange(2, 3)
    heuristics = ['WCD', 'FISHER', 'INTER']

    results = {}

    for heuristic in heuristics:

        print("Heuristic:", heuristic)

        accuracies = []

        for k in valors_k:

            predicted_colors = []

            for img in test_imgs:
                options = {
                    'km_init': 'kmeans++',
                    'fitting': heuristic,
                    'max_iter': 100,
                    'tolerance': 0.01
                }

                km = KMeans(img, K=k, options=options)
                km.fit()

                colors = get_colors(km.centroids)
                predicted_colors.append(colors)

            acc = get_color_accuracy(
                predicted_colors,
                test_color_labels,
                threshold=0.5
            )

            accuracies.append(acc)

        results[heuristic] = accuracies

    # =====================================================
    # PLOT (FORA DEL LOOP)
    # =====================================================

    plt.figure(figsize=(10, 6))

    for heuristic in heuristics:
        plt.plot(valors_k, results[heuristic], marker='o', label=heuristic)

    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.title('Comparació heurístiques KMeans - Color Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    # =====================================================
    # RETRIEVAL (SEPARAT)
    # =====================================================

    print("\nRetrieval qualitatiu...")

    color_query = ['Brown']

    # aquí HAS de definir quin model vols usar (ex: WCD, K=3)
    heuristic = 'WCD'
    k = 10

    predicted_colors = []

    for img in test_imgs:
        options = {
            'km_init': 'kmeans++',
            'fitting': heuristic,
            'max_iter': 100,
            'tolerance': 0.01
        }

        km = KMeans(img, K=k, options=options)
        km.fit()

        colors = get_colors(km.centroids)
        predicted_colors.append(colors)

    retrieved_imgs, retrieved_ids = retrieval_by_color(
        test_imgs,
        predicted_colors,
        color_query
    )

    visualize_retrieval(
        retrieved_imgs,
        topN=8,
        title='Retrieval by color: ' + ', '.join(color_query)
    )

    print("\nAnàlisi completada.")
    # 2. FI MAIN MARC

    # 3. MAIN PAU
    # Prova visual d'una imatge amb KMeans per treure les captures per l'informe
    print("Generant visualització KMeans...")
    img_prova = cropped_images[0] #Agafem la primera imatge de la llista de retallades com a prova arbitrària
    pixels_img = img_prova.reshape(-1, 3)  #Transformem la imatge (matriu) en una llista

    # Fem que el K-means busqui 3 colors dins de la llista de píxels, amb límit d'iteracions
    # i tolerància d'arrodoniment de floats per evitar bucles infinits.
    num_max_iter = 1000 # Modificable: Més iteracions = Més precisió = Més temps computacional
    coef_tolerancia = 0.001 # Modificable: Menys tolerància = Més precisió = Més temps computacional
    tipus_init = 'random' # Modificable: Tipus d'inicialització
    dict_options = {'fitting': 'FISHER', 'max_iter': num_max_iter, 'tolerance': coef_tolerancia, 'km_init': tipus_init}
    km_visual = KMeans(X=pixels_img, K=3, options=dict_options)
    km_visual.fit()
    visualize_k_means(km_visual, img_prova.shape)
    print("Visualització KMeans OK")

    # Estadístiques del KMeans
    print("Calculant estadístiques...")
    tots_pixels = np.vstack([img.reshape(-1, 3) for img in cropped_images])

    km_stats = KMeans(X=tots_pixels, K=2, options=dict_options)
    Kmean_statistics(km_stats, tots_pixels, Kmax=10)
    print("Càlcul estadístiques OK")

    # Motor de cerca (Retrieval_combined)
    print("Carregant motor de cerca...")
    # Predicció de forma (KNN)
    knn_combinat = KNN(train_imgs, train_class_labels)
    shape_preds = knn_combinat.predict(test_imgs, k=5, distance='euclidean')
    fiabilitat_forma = [100] * len(test_imgs)  #Percentatge de confiança màxim

    # Predicció de color (KMeans amb Fisher)
    color_preds = []
    fiabilitat_color = []
    for img in test_imgs:
        pixels = img.reshape(-1, 3)

        km_cerca = KMeans(X=pixels, options=dict_options)
        km_cerca.find_bestK(max_K=5)

        # Guardem els noms dels colors i els percentatges
        noms_colors = get_colors(km_cerca.centroids)
        color_preds.append(noms_colors)
        recompte = np.bincount(km_cerca.labels, minlength=km_cerca.K)
        percentatges = (recompte / len(km_cerca.labels)) * 100
        fiabilitat_color.append(percentatges.tolist())

    # Cerca Combinada Final
    forma_buscada = 'Shirts'  # Es pot canviar a la forma que es vulgui. Exemple: 'Jeans', 'Flip Flops', etc.
    color_buscat = "Blue"  # Es pot canviar al color que es vulgui. Exemple: 'Red', 'Blue', etc.
    print(f"Buscant '{forma_buscada}' de color '{color_buscat}'...")

    res_combinat = Retrieval_combined(
        images=test_imgs,
        shape_labels=shape_preds,
        color_labels=color_preds,
        shape_query=forma_buscada,
        color_query=color_buscat,
        shape_percents=fiabilitat_forma,
        color_percents=fiabilitat_color
    )

    if len(res_combinat) > 0:
        visualize_retrieval(np.array(res_combinat), topN=10, title=f"Combinat: {color_buscat} {forma_buscada}")
    else:
        print("No s'han trobat imatges amb aquesta combinació a test.")
    # 3. FI MAIN PAU