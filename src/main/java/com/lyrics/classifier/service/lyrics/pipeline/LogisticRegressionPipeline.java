// src/main/java/com/lyrics/classifier/service/lyrics/pipeline/LogisticRegressionPipeline.java
package com.lyrics.classifier.service.lyrics.pipeline;

import com.lyrics.classifier.service.MLService;
import com.lyrics.classifier.service.lyrics.transformer.Cleanser;
import com.lyrics.classifier.service.lyrics.transformer.Exploder;
import com.lyrics.classifier.service.lyrics.transformer.Numerator;
import com.lyrics.classifier.service.lyrics.transformer.Stemmer;
import com.lyrics.classifier.service.lyrics.transformer.Uniter;
import com.lyrics.classifier.service.lyrics.transformer.Verser;

import static com.lyrics.classifier.service.lyrics.pipeline.CommonLyricsPipeline.log;

import java.util.Map;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.StopWordsRemover;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.ml.feature.Word2Vec;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.springframework.core.env.Environment;
import org.springframework.stereotype.Component;

@Component // picked up by Spring
public class LogisticRegressionPipeline extends CommonLyricsPipeline {

        /* constructor injection cascades to CommonLyricsPipeline */
        public LogisticRegressionPipeline(SparkSession spark,
                        MLService mlService,
                        Environment env) {
                super(spark, mlService, env);
        }

        /* -------------------------- TRAIN + METRICS -------------------------- */
        @Override
        public CrossValidatorModel classify() {

                /*
                 * ------------------------------------------------------------------
                 * 1. load & cache the training / test splits (defined in superclass)
                 * ------------------------------------------------------------------
                 */
                Dataset<Row> train = trainingSet(); // comes from CommonLyricsPipeline

                /*
                 * ------------------------------------------------------------------
                 * 2. pipeline stages
                 * ------------------------------------------------------------------
                 */
                Cleanser cleanser = new Cleanser(); // lyrics → clean
                Numerator numerator = new Numerator(); // clean → clean (normal-form punctuation)
                Tokenizer tokenizer = new Tokenizer() // clean → tokens
                                .setInputCol("clean")
                                .setOutputCol("tokens");
                StopWordsRemover stop = new StopWordsRemover() // tokens → filtered
                                .setInputCol("tokens")
                                .setOutputCol("filtered");
                Exploder exploder = new Exploder(); // filtered→ exploded token rows
                Stemmer stemmer = new Stemmer(); // stemmed tokens
                Uniter uniter = new Uniter(); // back to array<string>
                Verser verser = new Verser(); // array → verse
                Word2Vec w2v = new Word2Vec() // verse → features
                                .setInputCol("verse")
                                .setOutputCol("features")
                                .setMinCount(0);
                LogisticRegression lr = new LogisticRegression()
                                .setLabelCol("label")
                                .setFeaturesCol("features");

                Pipeline pipeline = new Pipeline().setStages(new PipelineStage[] {
                                cleanser, numerator, tokenizer, stop, exploder,
                                stemmer, uniter, verser, w2v, lr
                });

                /*
                 * ------------------------------------------------------------------
                 * 3. hyper-parameter grid & cross-validator
                 * ------------------------------------------------------------------
                 */
                ParamMap[] grid = new ParamGridBuilder()
                                .addGrid(verser.sentencesInVerse(), new int[] { 4, 8 })
                                .addGrid(w2v.vectorSize(), new int[] { 100, 200 })
                                .addGrid(lr.regParam(), new double[] { 0.01 })
                                .addGrid(lr.maxIter(), new int[] { 100, 200 })
                                .build();

                CrossValidator cv = new CrossValidator()
                                .setEstimator(pipeline)
                                .setEstimatorParamMaps(grid)
                                .setEvaluator(
                                                new MulticlassClassificationEvaluator()
                                                                .setLabelCol("label")
                                                                .setPredictionCol("prediction")
                                                                .setMetricName("accuracy"))
                                .setNumFolds(5);

                /*
                 * ------------------------------------------------------------------
                 * 4. train & pick the best model
                 * ------------------------------------------------------------------
                 */
                CrossValidatorModel best = cv.fit(train);

                /*
                 * ------------------------------------------------------------------
                 * 5. evaluate on held-out test set
                 * ------------------------------------------------------------------
                 */
                Dataset<Row> predictions = best.transform(testSet); // test_df is in the super-class
                double accuracy = new MulticlassClassificationEvaluator()
                                .setLabelCol("label")
                                .setPredictionCol("prediction")
                                .setMetricName("accuracy")
                                .evaluate(predictions);

                log.info("Accuracy on test set: {}", accuracy);

                /*
                 * ------------------------------------------------------------------
                 * 6. persist & report metrics
                 * ------------------------------------------------------------------
                 */
                String modelPath = modelBaseDir().resolve(modelSubdir()).toString();
                saveModel(best, modelPath);

                Map<String, Object> stats = Map.of(
                                "accuracy", best.avgMetrics()[0]);
                printModelStatistics(stats);

                return best;
        }

        /* Each concrete pipeline stores its model under a sub-directory. */
        @Override
        protected String modelSubdir() {
                return "logreg";
        }

        @Override
        public Map<String, Object> getModelStatistics(CrossValidatorModel model) {
                // let the common base do the work for now
                return super.getModelStatistics(model);
        }
}
