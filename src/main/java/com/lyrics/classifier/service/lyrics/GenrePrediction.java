// src/main/java/com/lyrics/classifier/service/lyrics/GenrePrediction.java
package com.lyrics.classifier.service.lyrics;

public class GenrePrediction {

    private final String genre;
    private final Double metalProbability; // may be null
    private final Double popProbability; // may be null

    public GenrePrediction(String genre, Double metalProbability, Double popProbability) {
        this.genre = genre;
        this.metalProbability = metalProbability;
        this.popProbability = popProbability;
    }

    public GenrePrediction(String genre) { // fallback constructor
        this(genre, null, null);
    }

    /* getters */
    public String getGenre() {
        return genre;
    }

    public Double getMetalProbability() {
        return metalProbability;
    }

    public Double getPopProbability() {
        return popProbability;
    }
}
