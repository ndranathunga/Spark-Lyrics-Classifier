package com.lyrics.classifier.service.lyrics.pipeline;

import com.lyrics.classifier.column.Column;
import com.lyrics.classifier.service.MLService;
import com.lyrics.classifier.service.lyrics.transformer.*;
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
import java.util.Map;

@Component
public class LogisticRegressionPipeline extends CommonLyricsPipeline {

        public LogisticRegressionPipeline(SparkSession spark, MLService mlService, Environment env) {
                super(spark, mlService, env);
        }

        @Override
        public CrossValidatorModel classify() {
                Dataset<Row> train = trainingSet();
                Dataset<Row> test = testSet();

                Cleanser cleanser = new Cleanser()
                                .setInputCol(Column.VALUE.getName())
                                .setOutputCol(Column.CLEAN.getName());

                Numerator numerator = new Numerator()
                                .setInputCol(Column.ID.getName())
                                .setOutputCol(Column.ROW_NUMBER.getName());

                Tokenizer tokenizer = new Tokenizer()
                                .setInputCol(Column.CLEAN.getName())
                                .setOutputCol(Column.TOKENS.getName());

                StopWordsRemover stopWordsRemover = new StopWordsRemover()
                                .setInputCol(Column.TOKENS.getName())
                                .setOutputCol(Column.FILTERED_WORDS.getName());

                Exploder exploder = new Exploder()
                                .setInputCol(Column.FILTERED_WORDS.getName())
                                .setOutputCol(Column.FILTERED_WORD.getName());

                Stemmer stemmer = new Stemmer()
                                .setInputCol(Column.FILTERED_WORD.getName())
                                .setOutputCol(Column.STEMMED_WORD.getName());

                Uniter uniter = new Uniter()
                                .setInputCol(Column.STEMMED_WORD.getName())
                                .setOutputCol(Column.STEMMED_SENTENCE.getName());

                Verser verser = new Verser()
                                .setInputCol(Column.STEMMED_SENTENCE.getName())
                                .setOutputCol(Column.VERSE.getName());
                // .setSentencesInVerse(4); 

                Word2Vec w2v = new Word2Vec()
                                .setInputCol(Column.VERSE.getName())
                                .setOutputCol("features")
                                .setMinCount(1);

                LogisticRegression lr = new LogisticRegression()
                                .setLabelCol(Column.LABEL.getName())
                                .setFeaturesCol("features");

                Pipeline pipeline = new Pipeline().setStages(new PipelineStage[] {
                                cleanser, numerator, tokenizer, stopWordsRemover, exploder,
                                stemmer, uniter, verser, w2v, lr
                });

                ParamMap[] grid = new ParamGridBuilder()
                                .addGrid(verser.sentencesInVerseParam(), new int[] { 4 })
                                .addGrid(w2v.vectorSize(), new int[] { 50, 100, 150 }) // Tune this
                                .addGrid(lr.regParam(), new double[] { 0.01 })
                                .addGrid(lr.maxIter(), new int[] { 50 })
                                .build();

                CrossValidator cv = new CrossValidator()
                                .setEstimator(pipeline)
                                .setEstimatorParamMaps(grid)
                                .setEvaluator(
                                                new MulticlassClassificationEvaluator()
                                                                .setLabelCol(Column.LABEL.getName())
                                                                .setPredictionCol("prediction")
                                                                .setMetricName("accuracy"))
                                .setNumFolds(3);

                CrossValidatorModel bestModel = cv.fit(train);

                Dataset<Row> predictionsOnTest = bestModel.transform(test);
                double accuracy = new MulticlassClassificationEvaluator()
                                .setLabelCol(Column.LABEL.getName())
                                .setPredictionCol("prediction")
                                .setMetricName("accuracy")
                                .evaluate(predictionsOnTest);
                log.info("Accuracy on test set: {}", accuracy);

                String modelPath = modelBaseDir().resolve(modelSubdir()).toString();
                saveModel(bestModel, modelPath);

                Map<String, Object> stats = getModelStatistics(bestModel);
                stats.put("testSetAccuracy", accuracy);
                printModelStatistics(stats);

                return bestModel;
        }

        @Override
        protected String modelSubdir() {
                return "logreg_custom";
        }
}