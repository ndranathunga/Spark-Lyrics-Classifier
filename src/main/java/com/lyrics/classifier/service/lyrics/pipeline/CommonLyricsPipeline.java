// src/main/java/com/lyrics/classifier/service/lyrics/pipeline/CommonLyricsPipeline.java
package com.lyrics.classifier.service.lyrics.pipeline;

import com.lyrics.classifier.service.MLService;
import com.lyrics.classifier.service.lyrics.Genre;
import com.lyrics.classifier.service.lyrics.GenrePrediction;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.DataTypes; // ← use DataTypes, not DoubleType$
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.env.Environment;

import static org.apache.spark.sql.functions.*;

/**
 * Common helpers for every lyrics-pipeline implementation.
 */
public abstract class CommonLyricsPipeline implements LyricsPipeline {

    protected static final Logger log = LoggerFactory.getLogger(CommonLyricsPipeline.class);

    private final SparkSession spark;
    private final MLService mlService;
    @SuppressWarnings("unused")
    private final Environment env;

    /*
     * ------------------------ configurable via application.yml
     * ------------------------
     */
    @Value("${lyrics.csv.path}")
    private String csvPath;
    @Value("${lyrics.model.directory.path}")
    private String modelDir;
    /*
     * -----------------------------------------------------------------------------
     * -----
     */

    protected Dataset<Row> trainingSet; // cached after first read
    protected Dataset<Row> testSet;

    /*
     * constructor injection
     * ------------------------------------------------------------
     */
    public CommonLyricsPipeline(SparkSession spark,
            MLService mlService,
            Environment env) {
        this.spark = spark;
        this.mlService = mlService;
        this.env = env;
    }

    /*
     * --------------------------- CSV ingestion + 80/20 split
     * --------------------------
     */
    protected Dataset<Row> readCsv() {
        if (trainingSet != null)
            return trainingSet; // already done

        Dataset<Row> df = spark.read()
                .option("header", "true")
                .option("multiLine", false)
                .csv(csvPath)
                .select("genre", "lyrics")
                .na().drop("any", new String[] { "genre", "lyrics" })
                .filter(col("lyrics").rlike("\\w+"))
                .withColumn("id", monotonically_increasing_id());

        /* map genre → numeric label */
        Column labelCol = when(lower(col("genre")).equalTo("pop"), lit(Genre.POP.getValue()))
                .when(lower(col("genre")).equalTo("metal"), lit(Genre.METAL.getValue()))
                .otherwise(lit(Genre.UNKNOWN.getValue()))
                .cast(DataTypes.DoubleType); // ← FIX: use DataTypes

        df = df.withColumn("label", labelCol);

        Dataset<Row>[] split = df.randomSplit(new double[] { 0.8, 0.2 }, 42);
        trainingSet = split[0].cache();
        trainingSet.count();
        testSet = split[1].cache();
        testSet.count();

        log.info("Training set rows: {}", trainingSet.count());
        log.info("Test set rows    : {}", testSet.count());
        return trainingSet;
    }

    /*
     * ------------------------------- prediction helper
     * --------------------------------
     */
    @Override
    public GenrePrediction predict(String unknownLyrics) {
        Path modelPath = Path.of(modelDir, modelSubdir());
        CrossValidatorModel cv = mlService.loadCrossValidationModel(modelPath.toString());

        PipelineModel best = (PipelineModel) cv.bestModel();

        Dataset<Row> df = spark.createDataset(
                java.util.Arrays.stream(unknownLyrics.split("\\R"))
                        .filter(s -> !s.isBlank())
                        .toList(),
                Encoders.STRING())
                .withColumn("label", lit(Genre.UNKNOWN.getValue()))
                .withColumn("id", lit("unknown.txt"));

        Row row = best.transform(df).first();
        int idx = (int) row.<Double>getAs("prediction").doubleValue();
        Genre g = Genre.from(idx);

        if (row.schema().fieldNames().length > 0 &&
                java.util.Arrays.asList(row.schema().fieldNames()).contains("probability")) {
            DenseVector p = row.getAs("probability");
            return new GenrePrediction(g.getName(), p.apply(0), p.apply(1));
        }
        return new GenrePrediction(g.getName());
    }

    /*
     * ---------------- utility helpers the child class still expects
     * -------------------
     */
    protected void saveModel(CrossValidatorModel model, String dir) {
        mlService.saveModel(model, dir);
    }

    protected void printModelStatistics(Map<String, Object> stats) {
        log.info("Model statistics: {}", stats);
    }

    /*
     * ------------------------------ abstract hooks
     * ------------------------------------
     */
    @Override
    public abstract CrossValidatorModel classify();

    protected abstract String modelSubdir(); // child supplies sub-directory

    /* expose to subclasses */
    protected Dataset<Row> trainingSet() {
        return readCsv();
    }

    protected Dataset<Row> testSet() {
        return testSet;
    }

    protected Path modelBaseDir() { // <-- new
        return Path.of(modelDir);
    }

    @Override
    public Map<String, Object> getModelStatistics(CrossValidatorModel model) { // ← public
        Map<String, Object> stats = new HashMap<>();
        Arrays.sort(model.avgMetrics());
        stats.put("Best model metric", model.avgMetrics()[model.avgMetrics().length - 1]);
        return stats;
    }
}
