# Hackathon Spain-ai 2020  - Reto en computer vision - SuperResolution



## Contents
1. [Introducción](#introduction)
2. [Solución](#train)

## Introducción
Demuestra tus habilidades como Data Scientist en el reto de Computer Vision


En este reto no se pide una simple clasificación de imágenes, sino la generación de la versión de alta resolución de las imágenes proporcionadas.

Para ello proveemos un conjunto de imágenes de baja calidad, así como las correspondientes imágenes de alta resolución para cada una de estas imágenes. Lo que os pedimos es que para el conjunto de imágenes de test de baja calidad, generéis las imágenes en alta calidad.

Este es el primer hackathon que lanzamos desde Spain AI, por lo que es seguro que hay puntos mejorables, aclaraciones y hasta posiblemente algún ajuste que tengamos que hacer durante la competición, por lo que os pedimos que seáis comprensivos ya que intentamos hacerlo lo mejor posible, aunque queremos avisaros de antemano por si tuviésemos que hacer alguno de estos ajustes durante el reto.

# ¿Es el típico problema de clasificación de imágenes?

No. En este reto no pedimos que se clasifiquen las imágenes. Tampoco pedimos que se segmenten. Lo que se pide es que mediante diferentes algoritmos se trate de mejorar la calidad de una imagen dada, obteniendo su equivalente de alta resolución.

# Cuéntame algo más

Tendrás un dataset de train con un gran conjunto de imágenes de baja resolución, y las correspondientes imágenes de alta resolución. Deberás usar estas imágenes para entrenar tu algoritmo.

Una vez tengas el algoritmo entrenado, deberás usarlo en las imágenes de tests (con imágenes de baja resolución), para obtener las equivalentes de alta resolución.

Además: Las imágenes que te daremos son muy diferentes entre sí, por lo que puede que algunas mejoren mucho al pasarlas a alta resolución, mientras que otras no tanto.

Como sabemos que es un reto complicado, todas las imágenes que se envíen serán valoradas, donde lo que buscamos es ver cómo de bien se parecen las fotos que enviéis a las fotos de alta resolución que tiene el sistema.

# ¿Quiere esto decir que las fotos del test set de alta resolución ya las tenéis?

Sí. Tenemos las fotos de alta resolución en un dataset que no os hemos compartido y que ha sido generado específicamente para este reto. Las fotos que nos enviéis se compararán contra estas fotos.
Dicho esto, comentaros que el proceso de generación del dataset ha sido a partir de las fotos de alta resolución, ya que una vez teníamos estas fotos hemos generado las fotos de baja resolución usando diferentes ténicas de tratamiento de imagen así como algoritmos, por lo que contamos con las originales, que son contra las que comprobaremos los envíos que realicéis.

# ¿Entonces estas fotos del dataset de alta resolución solo las tenéis vosotros?

Correcto. Son fotos que hemos generado de manera específica para el reto, y aunque hay fotos similares en internet no son las mismas, por lo que no tratéis de buscarlas ya que no las vais a encontrar, y aunque veáis fotos parecidas no son las que estamos esperando por lo que las fotos que enviáseis no tendrán buen resultado.

# Parece complicado ¿Algún consejo?
Es un reto complejo, pero sabed que todas las imágenes tendrán algún tipo de error con la original de alta calidad, lo importantte es que el algoritmo trate de generar la mejor versión de cada foto, ya que sabemos que es imposible que se genere el dataset con todas las fotos en la resolución esperada.

# ¿Este reto es nuevo para mí, qué algoritmos podría usar?
Este reto lo hemos planteado buscando una aplicación práctica de algoritmos de redes generativas antagónicas (Generative Adversarial Neural Networks, o GANs), pero puede que encuentres otra solución que te permitta generar fotos de alta calidad.

# ¿Hay algo más que deba saber?
Ten también en cuenta que hay un límite máximo de 5 envíos por día, y que durante la duración del reto (~3 meses) se permite un máximo acumulado de 100 envíos (estos 100 envíos se calcularán sumando todos los envíos de todos los días).

# ¿Cómo es el formato de los datos que tengo que enviar?
Puedes encontrarlo en el ZIP de los datos con el ejemplo del submission. El programa de evaluación espera recibir un fichero ZIP con el nombre que quieras, pero al descomprimirlo debe haber un conjunto de fotos con el mismo nombre y formato (2400px x 2400px) que las fotos que os damos en el fichero de Train de alta resolución (las fotos del fichero de Submission de ejemplo tienen esa misma resolución).

# ¿Qué métrica se va a usar?
La métrica que vamos a usar es Structural similarity index.

Al comparar imágenes, el error cuadrático medio (mean squared error -MSE-), aunque es sencillo de aplicar no es muy indicativo de la similitud. La similitud estructural pretende subsanar esta deficiencia teniendo en cuenta la textura, y es por esto por lo que se ha elegido esta métrica para evaluar los envíos.

Podéis leer más sobre esta métrica en este enlace https://medium.com/srm-mic/all-about-structural-similarity-index-ssim-theory-code-in-pytorch-6551b455541e.

Para cada foto que nos envíes, vamos a ver el grado de diferencia con respecto a la foto de alta resolución original.

Sumamos todas estas diferencias, y calculamos el porcentaje medio de similitud, diviendo por el número de fotos que tenemos en nuestro set de imágenes (es el mismo número de fotos que hay en el dataset de test que os damos). Tras hacer esto, un score perfecto tendría una similitud del 1.00, o lo que es lo mismo, que las dos imágenes son iguales.

El score perfecto se obtendría si el fichero contiene para cada foto candidata en alta resolución la misma foto en nuestro dataset, de manera que al comparar las fotos siempre se hubiese obtenido la misma imagen.

Aclaración: para evaluar las fotos necesitamos que envíes el mismo número de fotos que hay en el test set, por lo que si envías mas o menos fotos, el resultado será de 0.

Aclaración 2: cuando decimos "envía las fotos candidatas en alta resolución" queremos decir que para cada una de las fotos se espera la equivalente de alta resolución, pero si algunas fotos no se han modificado y se envía la misma foto en baja resolución que os dimos nosotros, en ese caso el error será alto, pero se realizará la evaluación del envío que hagáis.

Aclaración 4: En los envíos, debes comprimir todas las fotos tal y como se presentan en el fichero de ejemplo que os hemos dado (generando un fichero .zip que puede llamarse como quiera) cuyo contenido serán todas las fotos en alta resolución generadas.

Aclaración 5: El contenido del fichero .zip que subas serán las fotos en alta resolución que corresponden con las fotos en baja resolución proporcionadas por Spain AI. Las fotos en alta resolución del fichero .zip deben llamarse "candidate_n.png", donde "n" será el identificador de cada foto y deberá ser el mismo identificador que tenía la foto en baja resolución para la que se ha obtetnido dicha foto en alta resolución.

Aclaración 6: Asegúrate de que generas un zip donde los contenidos sean las fotos. Un error común es meter las fotos en una carpeta y comprimir la carpeta. Si haces esto el programa de evaluación no encontrará las fotos (solo encontrará la carpeta) y el resultado será 0.0.

Aclaración 7: La evaluación de los envíos tarda en subirse ya que es un archivo bastante grande, y también tarda en ejecutarse. Dejad la página abierta un rato ya que la plataforma no tiene barra de progreso y no veréis en qué punto estáis de la subida del fichero, o de la evaluación del envío.

Aclaración 8: Las fotos que nos enviéis se van a comparar contra las fotos de alta resolución que tenemos almacenadas, por lo que si alguna foto no la subís no será evaluada, y no obtendréis puntos por ellas, teniendo un score más bajo que aquellas personas que si suban todas las fotos.

Aclaración 9: Las fotos que nos enviéis deben tener una resolución igual que las fotos candidatas, es decir, una resolución de 2400x2400 px. Si envías fotos con otra resolución, no se tendrán en cuenta y no sumará puntos.

# ¿Hay algo más que no haya quedado bien explicado?
Si hay cualquier punto que creas que es confuso, puedes escribirnos a info@spain-ai.com

# ¿Cuales son las bases del hackathon?
Los términos y condiciones generales para participar en el hackathon son muy similares a los de otras plataformas, si quieres leerlas, puedes hacerlo en esta página.
Si no quieres leer todo el documento de términos y condiciones, lo más importante es que no hagas trampas, y que tengas en cuenta que a los ganadores con opción a premio puede que se les solicite el código para validar la solución, aunque este punto dependerá de la empresa que haya compartido el dataset del reto, como condición para reconocerle formalmente ganador.

## Solución

Basada a partir del paper:

Yuanfei Huang, Jie Li, Xinbo Gao*, Yanting Hu and Wen Lu, "Interpretable Detail-Fidelity Attention Network for Single Image Super-Resolution", IEEE Transactions on Image Processing (TIP), vol.30, pp.2325-2339, 2021.

[TIP](https://ieeexplore.ieee.org/document/9334407)

[arXiv](https://arxiv.org/abs/2009.13134)

