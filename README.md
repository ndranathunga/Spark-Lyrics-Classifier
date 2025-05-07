# Song Genre Classifier

This project is a Song Genre Classifier built using Apache Spark MLlib and Spring Boot. It classifies songs into predefined genres based on their lyrics.

**Course Assignment:** This project was developed as part of the coursework for **In20-S8-CS4651 - Big Data Analytics, Week 10: Big Data Visualisation, MLlib and Visualisation Homework**.

## Overview

The application provides functionalities to:
1.  Train a machine learning model (Logistic Regression) using a dataset of song lyrics and their genres.
2.  Expose a REST API to predict the genre of new song lyrics using the trained model.

The ML pipeline involves several custom Spark transformers for text preprocessing:
*   **Cleanser**: Removes non-alphabetic characters and converts text to lowercase.
*   **Numerator**: Assigns a row number, used internally by other transformers.
*   **Tokenizer**: Splits cleaned lyrics into words.
*   **StopWordsRemover**: Removes common stop words.
*   **Exploder**: Converts an array of words into individual rows, each containing one word.
*   **Stemmer**: Stems each word to its root form (e.g., "running" to "run") using the English Snowball stemmer.
*   **Uniter**: Aggregates stemmed words back into sentences per original lyric entry.
*   **Verser**: Groups sentences into "verses" (configurable number of sentences per verse).
*   **Word2Vec**: Converts verses (sequences of words) into feature vectors.
*   **LogisticRegression**: The classification algorithm.

The pipeline is tuned using `CrossValidator`.

## Prerequisites

*   **Java Development Kit (JDK)**: Version 17
*   **Apache Maven**: For building the project and managing dependencies.
*   **Git**: For cloning the repository (if applicable).
*   **(For Windows users)**: `winutils.exe` and `hadoop.dll` correctly set up for Hadoop, or ensure `HADOOP_HOME` environment variable is set. The project attempts to configure this via `spark.driver.extraJavaOptions=-Dhadoop.home.dir=C:/Hadoop` in `application.properties`, which you might need to adjust for your system.

## Project Structure
```
.
├── pom.xml                         # Maven Project Object Model
├── models/                         # Default directory for saved ML models
├── src/
│   ├── main/
│   │   ├── java/com/lyrics/classifier/
│   │   │   ├── ClassifierApplication.java  # Spring Boot main application
│   │   │   ├── column/                 # DataFrame column definitions
│   │   │   ├── config/                 # Spring and Spark configurations (SparkSession, TrainRunner)
│   │   │   ├── controller/             # REST API controllers (LyricsController)
│   │   │   ├── service/                # Business logic (LyricsService, MLService)
│   │   │   │   ├── lyrics/             # Genre classification specific services
│   │   │   │   │   ├── pipeline/       # ML pipelines (LogisticRegressionPipeline)
│   │   │   │   │   └── transformer/    # Custom Spark ML Transformers
│   │   ├── resources/
│   │   │   ├── application.properties  # Application configuration
│   │   │   ├── data/training/          # Expected location for training data
│   │   │   │   └── Merged_dataset1.csv # Training data CSV (needs to be provided)
│   │   │   └── META-INF/
│   └── test/
│       └── java/                     # Unit and integration tests
└── README.md                       # This file
```

## Configuration

Key configuration settings are in `src/main/resources/application.properties`:

*   `lyrics.csv.path`: Path to the training data CSV file. Default: `src/main/resources/data/training/Merged_dataset1.csv`.
*   `lyrics.model.directory.path`: Directory where trained models are saved and loaded from. Default: `models`.
*   `mode`: Application operating mode.
    *   `train`: The application trains the model upon startup and then exits.
    *   `serve`: (Default) The application starts, loads a pre-trained model (if available), and serves prediction requests via API.
*   `logging.level.*`: Configures logging levels for the application and libraries like Spark.

## Data

*   The application expects a CSV file for training, specified by `lyrics.csv.path`.
*   **Format**: The CSV file must contain at least two columns with headers:
    *   `lyrics`: The song lyrics (text).
    *   `genre`: The genre of the song (e.g., "POP", "ROCK", "JAZZ"). The system handles case-insensitivity for genre names defined in `Genre.java`.
*   **Sample CSV structure**:
    ```csv
    lyrics,genre
    "Some pop song lyrics here...",POP
    "Rock and roll lyrics...",ROCK
    "Smooth jazz verses...",JAZZ
    ```
*   **Note**: You need to provide this CSV file in the configured path.

## Build

To build the project and package it into a JAR file:

```bash
mvn clean package
```
This will generate a JAR file in the `target/` directory (e.g., `target/classifier-0.0.1-SNAPSHOT.jar`).

## Testing

To run the unit and integration tests:

```bash
mvn test
```

## Running the Application

There are two primary modes to run the application: Training Mode and Serving Mode.

### 1. Training Mode

In this mode, the application will read the data from `lyrics.csv.path`, train the classification model, save it to `lyrics.model.directory.path`, and then exit.

**Steps:**
1.  Ensure your training data CSV (`Merged_dataset1.csv` or as configured) is in place.
2.  Set `mode=train` in `src/main/resources/application.properties`.
3.  Run the application:
    ```bash
    java -jar target/classifier-0.0.1-SNAPSHOT.jar
    ```
    Alternatively, using Maven:
    ```bash
    mvn spring-boot:run
    ```
4.  Check the console for training logs and statistics. The trained model will be saved in the `models/logreg_custom` directory (or as configured).

### 2. Serving Mode (Prediction)

In this mode, the application loads a previously trained model and exposes an API endpoint for genre prediction.

**Steps:**
1.  Ensure a model has been trained and saved (e.g., by running in Training Mode first).
2.  Set `mode=serve` (or leave it as default) in `src/main/resources/application.properties`.
3.  Run the application:
    ```bash
    java -jar target/classifier-0.0.1-SNAPSHOT.jar
    ```
    Alternatively, using Maven:
    ```bash
    mvn spring-boot:run
    ```
4.  The application will start and be ready to serve requests on port `8080` (default Spring Boot port).

## API Endpoints

The application exposes the following REST API endpoints, accessible by default at `http://localhost:8080`.
Swagger UI for API documentation is typically available at `http://localhost:8080/swagger-ui.html`.

### Train Model
*   **Endpoint**: `POST /api/train`
*   **Description**: Triggers the model training process. Reads data, trains, saves the model, and returns statistics. This can be used to retrain the model while the application is in `serve` mode.
*   **Request Body**: None
*   **Response**: JSON object with model statistics, e.g.:
    ```json
    {
        "Best model metric (higher is better)": 0.85, // Example metric value
        "testSetAccuracy": 0.83 // Example accuracy on the test set
    }
    ```

### Predict Genre
*   **Endpoint**: `POST /api/predict`
*   **Description**: Predicts the genre for the provided song lyrics.
*   **Request Body**: JSON object with lyrics:
    ```json
    {
        "lyrics": "Some new song lyrics to classify..."
    }
    ```
*   **Response**: JSON object with the predicted genre and probabilities for each genre:
    ```json
    {
        "predictedGenre": "POP", // Example
        "probabilities": {
            "pop": 0.75,
            "rock": 0.15,
            "jazz": 0.05,
            // ... other genres
        }
    }
    ```

## How to Verify

### Training
*   After running in `train` mode or calling `/api/train`, check the console logs for messages indicating successful training and model saving.
*   Verify that a model directory (e.g., `models/logreg_custom`) has been created/updated.

### Serving Predictions
1.  Start the application in `serve` mode.
2.  Use a tool like `curl` or Postman to send a POST request to the `/api/predict` endpoint:

    **Using curl:**
    ```bash
    curl -X POST -H "Content-Type: application/json" -d "{\"lyrics\":\"love you baby like a love song\"}" http://localhost:8080/api/predict
    ```
3.  Check the response for the predicted genre.

## Notes

*   **Windows Hadoop Configuration**: If you are running on Windows, Spark might require `winutils.exe`. The `application.properties` file includes `spark.driver.extraJavaOptions=-Dhadoop.home.dir=C:/Hadoop`. You may need to adjust `C:/Hadoop` to your Hadoop binaries' location or ensure `HADOOP_HOME` is set and `winutils.exe` is in its `bin` directory.
*   **Spark UI**: The Spark UI (for monitoring jobs) is disabled by default in `SparkConfig.java` (`.set("spark.ui.enabled", "false")` is commented out). If enabled, it usually runs on port `4040`.
*   **Model Persistence**: The `StringIndexerModel` for genres is re-fitted during `prepareData()`. For robust prediction in a standalone serving environment where the original training CSV might not be available, this indexer model (or its labels) should ideally be saved and loaded along with the main `CrossValidatorModel`. The current implementation re-reads the CSV and re-fits the indexer if `predict()` is called on a new instance of the pipeline (e.g., after application restart).
