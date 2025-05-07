package com.lyrics.classifier.service;

import java.io.IOException;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.regression.LinearRegressionTrainingSummary;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.util.MLWritable;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.springframework.stereotype.Service;

@Service
public class MLService {

    private final SparkSession spark;

    public MLService(SparkSession spark) {
        this.spark = spark;
    }

    public void trainLinearRegression(String parquet, int maxIter) {
        Pair<Dataset<Row>, Dataset<Row>> split = splitTrainTest(parquet);
        trainLinearRegression(split.getLeft(), split.getRight(), maxIter);
    }

    private void trainLinearRegression(
            Dataset<Row> train, Dataset<Row> test, int maxIter) {

        LinearRegression lr = new LinearRegression()
                .setMaxIter(maxIter)
                .setRegParam(0.3)
                .setElasticNetParam(0.8);

        LinearRegressionModel model = lr.fit(train);
        System.out.println("Coefficients: " + model.coefficients());
        LinearRegressionTrainingSummary s = model.summary();
        System.out.printf("Iterations=%d  RMSE=%.4f%n",
                s.totalIterations(), s.rootMeanSquaredError());
        Row first = model.transform(test)
                .select("features", "label", "prediction")
                .first();
        System.out.println(first);
    }

    public KMeansModel trainKMeans(String parquet, int k) {
        Dataset<Row> ds = spark.read().parquet(parquet).cache();
        return new KMeans().setK(k).fit(ds);
    }

    private Pair<Dataset<Row>, Dataset<Row>> splitTrainTest(String parquet) {
        Dataset<Row> full = spark.read().parquet(parquet);
        Dataset<Row>[] split = full.randomSplit(new double[] { 0.7, 0.3 }, 42);
        split[0].cache();
        split[0].count();
        return Pair.of(split[0], split[1]);
    }

    public <T extends MLWritable> void saveModel(T model, String dir) {
        try {
            model.write().overwrite().save(dir);
            System.out.printf("✔ Saved model to %s%n", dir);
        } catch (IOException e) {
            throw new IllegalStateException("Failed to save model", e);
        }
    }

    public CrossValidatorModel loadCVModel(String dir) {
        System.out.printf("✔ Loaded CV model from %s%n", dir);
        return CrossValidatorModel.load(dir);
    }

    public PipelineModel loadPipelineModel(String dir) {
        System.out.printf("✔ Loaded pipeline model from %s%n", dir);
        return PipelineModel.load(dir);
    }

    public CrossValidatorModel loadCrossValidationModel(String dir) {
        return CrossValidatorModel.load(dir);
    }
}