package com.lyrics.classifier.service.lyrics;

public enum Genre {
    POP("POP"),
    COUNTRY("COUNTRY"),
    BLUES("BLUES"),
    JAZZ("JAZZ"),
    REGGAE("REGGAE"),
    ROCK("ROCK"),
    HIP_HOP("HIP HOP"),
    RAP("RAP");

    private final String name;

    Genre(String n) {
        this.name = n;
    }

    public String getName() {
        return name;
    }

    public static String fromIndex(int idx, String[] allLabelsFromIndexer) {
        if (idx >= 0 && idx < allLabelsFromIndexer.length) {
            return allLabelsFromIndexer[idx].toUpperCase();
        }
        return "PREDICTION_INDEX_OUT_OF_BOUNDS";
    }

    public static Genre fromName(String name) {
        if (name == null)
            return null;
        for (Genre g : Genre.values()) {
            if (g.getName().equalsIgnoreCase(name)) {
                return g;
            }
        }
        return null;
    }
}