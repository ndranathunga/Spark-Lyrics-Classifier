// src/main/java/com/lyrics/classifier/service/lyrics/pipeline/LyricsPipeline.java
package com.lyrics.classifier.service.lyrics.pipeline;

import com.lyrics.classifier.service.lyrics.GenrePrediction;
import java.util.Map;
import org.apache.spark.ml.tuning.CrossValidatorModel;

public interface LyricsPipeline {
    CrossValidatorModel classify();
    GenrePrediction     predict(String unknownLyrics);
    Map<String, Object> getModelStatistics(CrossValidatorModel model);
}
