package com.lyrics.classifier.service.lyrics.pipeline;

import com.lyrics.classifier.column.Column;
import com.lyrics.classifier.service.MLService;
import com.lyrics.classifier.service.lyrics.Genre;
import com.lyrics.classifier.service.lyrics.GenrePrediction;

import java.nio.file.Path;
import java.util.*;
import java.util.stream.Collectors;

import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.sql.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.env.Environment;
import org.apache.spark.ml.feature.StringIndexer;

import static org.apache.spark.sql.functions.*;

public abstract class CommonLyricsPipeline implements LyricsPipeline {

    protected static final Logger log = LoggerFactory.getLogger(CommonLyricsPipeline.class);

    private final SparkSession spark;
    private final MLService mlService;
    protected final Environment env;

    @Value("${lyrics.csv.path}")
    private String csvPath;
    @Value("${lyrics.model.directory.path}")
    private String modelDir;

    protected Dataset<Row> trainingSet;
    protected Dataset<Row> testSet;
    protected StringIndexerModel genreIndexerModel; // To store the fitted StringIndexerModel

    public CommonLyricsPipeline(SparkSession spark, MLService mlService, Environment env) {
        this.spark = spark;
        this.mlService = mlService;
        this.env = env;
    }

    protected void prepareData() {
        if (trainingSet != null)
            return;

        Dataset<Row> df = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv(csvPath)
                .select(
                        col("lyrics").alias(Column.VALUE.getName()),
                        lower(trim(col("genre"))).alias("genre_text"))
                .na().drop("any", new String[] { Column.VALUE.getName(), "genre_text" })
                .filter(col(Column.VALUE.getName()).rlike("\\w+"))
                .withColumn(Column.ID.getName(), monotonically_increasing_id());

        List<String> knownGenreNames = Arrays.stream(Genre.values())
                .map(g -> g.getName().toLowerCase())
                .filter(name -> !name.equals("unknown"))
                .collect(Collectors.toList());

        df = df.filter(col("genre_text").isin(knownGenreNames.toArray(new String[0])));

        this.genreIndexerModel = new StringIndexer()
                .setInputCol("genre_text")
                .setOutputCol(Column.LABEL.getName())
                .fit(df);

        Dataset<Row> indexedDf = genreIndexerModel.transform(df).drop("genre_text");

        Dataset<Row>[] split = indexedDf.randomSplit(new double[] { 0.8, 0.2 }, 42);
        // trainingSet = split[0].limit(1000).cache();
        trainingSet = split[0].cache();
        testSet = split[1].cache();

        log.info("Total valid rows after filtering: {}", indexedDf.count());
        log.info("Training set rows: {}", trainingSet.count());
        log.info("Test set rows: {}", testSet.count());
        log.info("Genre labels after indexing: {}", Arrays.toString(genreIndexerModel.labelsArray()[0]));

        if (trainingSet.isEmpty()) {
            log.error("Training set is empty. Check CSV path, content, and genre filtering.");
            throw new IllegalStateException("Training data could not be prepared or is empty.");
        }
    }

    @Override
    public GenrePrediction predict(String unknownLyrics) {
        if (genreIndexerModel == null) {
            log.warn("GenreIndexerModel is not available. Re-initializing data preparation.");
            prepareData(); // Ensure StringIndexerModel is fit if not already
            if (genreIndexerModel == null) {
                throw new IllegalStateException(
                        "GenreIndexerModel could not be initialized. Train the model first or check data loading.");
            }
        }
        Path modelPath = Path.of(modelDir, modelSubdir());
        CrossValidatorModel cvModel = mlService.loadCrossValidationModel(modelPath.toString());
        PipelineModel bestModel = (PipelineModel) cvModel.bestModel();

        Dataset<Row> singleRowDf = spark.createDataset(
                Collections.singletonList(unknownLyrics), Encoders.STRING())
                .withColumnRenamed("value", Column.VALUE.getName())
                .withColumn(Column.ID.getName(), lit("unknown_predict_" + UUID.randomUUID().toString()))
                .withColumn(Column.LABEL.getName(), lit(0.0));

        Row predictionRow = bestModel.transform(singleRowDf).first();

        double predictedIndex = predictionRow.getAs(Column.LABEL.getName());

        if (predictionRow.schema().fieldIndex("prediction") >= 0) {
            predictedIndex = predictionRow.getAs("prediction");
        }

        String predictedGenreName = genreIndexerModel.labelsArray()[0][(int) predictedIndex];

        Map<String, Double> probabilities = new LinkedHashMap<>();
        if (predictionRow.schema().fieldIndex("probability") >= 0) {
            Vector probsVector = predictionRow.getAs("probability");
            String[] labels = genreIndexerModel.labelsArray()[0];
            for (int i = 0; i < probsVector.size(); i++) {
                probabilities.put(labels[i], probsVector.apply(i));
            }
        }
        return new GenrePrediction(predictedGenreName, probabilities);
    }

    protected void saveModel(CrossValidatorModel model, String dir) {
        mlService.saveModel(model, dir);
    }

    protected void printModelStatistics(Map<String, Object> stats) {
        log.info("Model statistics: {}", stats);
    }

    @Override
    public abstract CrossValidatorModel classify();

    protected abstract String modelSubdir();

    protected Dataset<Row> trainingSet() {
        prepareData();
        return trainingSet;
    }

    protected Dataset<Row> testSet() {
        prepareData();
        return testSet;
    }

    protected Path modelBaseDir() {
        return Path.of(modelDir);
    }

    @Override
    public Map<String, Object> getModelStatistics(CrossValidatorModel model) {
        Map<String, Object> stats = new HashMap<>();
        double[] avgMetrics = model.avgMetrics();
        if (avgMetrics != null && avgMetrics.length > 0) {
            Arrays.sort(avgMetrics); // Smallest to largest
            stats.put("Best model metric (higher is better)", avgMetrics[avgMetrics.length - 1]);
        } else {
            stats.put("Best model metric", "N/A (avgMetrics not available or empty)");
        }
        return stats;
    }
}