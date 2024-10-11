## Objetivo
El objetivo de este proyecto es crear un flujo de desarrollo y productivizacion de un modelo conversacional que pueda asistir en la creacion de recetas latinoamericanas a través del fine tuning de un modelo LLM open source con datos scrapeados de distintas web de recetas.

## Modelado

### Modelo
Se usará el modelo open source Phi3 ya que al ser pequeño podemos entrenarlo en un entorno local.

## Datos
El dataset fue elaborado a partir del webscraping de sitios de recetas y su posterior transformación a un dataset de instrucciones, este de desarrollo durante el marco de la hakathon somosNLP 2024 y se puede encontrar [repo](https://huggingface.co/datasets/somosnlp/recetasdelaabuela_genstruct_it) . El codigo para crear el dataset lo encuentran en este [colab](https://colab.research.google.com/drive/1-7OY5ORmOw0Uy_uazXDDqjWWkwCKvWbL?usp=sharing)

## Inferencia
El modelo responde a demanda, cada vez que el usuario envia un mensaje. El modelo recibe todo el contexto de la conversación y responde.

## Full solution architecture

![Architecture](docs/Flowchart.jpg)

### Training infrastructure
- Consta de tres elementos principales: minio, MLflow y PostgreSQL – todos accesibles externamente a través del servidor MLOps
- Estos componentes interactúan entre sí y son utilizados principalmente por los Ingenieros de ML

### CI/CD
- Incluye pipeline entrenamiento del modelo y despliegue de App/Infra
- El pipeline se ejecutaria de manera diaria en horario batch ya que se esperaria que se actualicen los datasets

### Roles
- Ingeniero de ML: Modifica el código de entrenamiento, agrega mejoras al modelo y contribuye a otros cambios de código
- Ingeniero de MLOps: Gestiona la infraestructura de entrenamiento y servicio, observa el comportamiento del modelo a través de los logs

## Limitaciones
### Este proyecto está actualmente destinado a ejecutarse localmente
No se han hecho previsiones para ejecutar este proyecto utilizando ninguna infraestructura en la nube.
### El modelo no esta optimizado
El modelo ha pasado por el entrenamiento solo para poder generar registros en mlflow y ha sido entrenado en un entorno local sin gpu durante unos pocos epochs
### Despliegue en servicio REST
Actualmente solo se espera desplegar el modelo en un servicio REST y que pueda ser consumido.
### Actualizacion del dataset
Actualmente solo se trabaja con un dataset previamente scrapeado y preprocesado. Una futura mejora es automatizar el webscraping y asi hacer crecer los datos y mantenerlos actualizados.

## Ejecutar localmente
