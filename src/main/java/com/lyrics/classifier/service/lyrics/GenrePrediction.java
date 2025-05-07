package com.lyrics.classifier.service.lyrics;

import java.util.Map;

public class GenrePrediction {

    private final String predictedGenre;
    private final Map<String, Double> probabilities;

    public GenrePrediction(String predictedGenre, Map<String, Double> probabilities) {
        this.predictedGenre = predictedGenre;
        this.probabilities = probabilities;
    }

    public String getPredictedGenre() {
        return predictedGenre;
    }

    public Map<String, Double> getProbabilities() {
        return probabilities;
    }
}