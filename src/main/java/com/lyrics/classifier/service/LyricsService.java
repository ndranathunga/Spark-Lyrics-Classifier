package com.lyrics.classifier.service;

import com.lyrics.classifier.service.lyrics.GenrePrediction;
import com.lyrics.classifier.service.lyrics.pipeline.LyricsPipeline;
import java.util.Map;
import javax.annotation.Resource;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.springframework.stereotype.Service;

@Service
public class LyricsService {

    @Resource(name = "${lyrics.pipeline:logisticRegressionPipeline}")
    private LyricsPipeline pipeline;

    public Map<String, Object> classifyLyrics() {
        CrossValidatorModel model = pipeline.classify();
        return pipeline.getModelStatistics(model);
    }

    public GenrePrediction predictGenre(String unknownLyrics) {
        return pipeline.predict(unknownLyrics);
    }
}