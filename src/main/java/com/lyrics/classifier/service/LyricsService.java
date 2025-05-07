// src/main/java/com/lyrics/classifier/service/LyricsService.java
package com.lyrics.classifier.service;

import com.lyrics.classifier.service.lyrics.GenrePrediction;
import com.lyrics.classifier.service.lyrics.pipeline.LyricsPipeline;
import java.util.Map;
import javax.annotation.Resource;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.springframework.stereotype.Service;

/**
 * Facade that exposes training + inference operations to the REST layer.
 * The concrete pipeline bean is selected via the property `lyrics.pipeline`
 * (default: "logisticRegressionPipeline").
 */
@Service
public class LyricsService {

    @Resource(name = "${lyrics.pipeline:logisticRegressionPipeline}")
    private LyricsPipeline pipeline;

    /** Trains (or re-trains) the model and returns summary metrics. */
    public Map<String, Object> classifyLyrics() {
        CrossValidatorModel model = pipeline.classify();
        return pipeline.getModelStatistics(model);
    }

    /** Predicts the genre of an unknown lyric excerpt. */
    public GenrePrediction predictGenre(String unknownLyrics) {
        return pipeline.predict(unknownLyrics);
    }
}
