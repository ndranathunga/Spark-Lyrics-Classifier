// src/main/java/com/lyrics/classifier/service/lyrics/Genre.java
package com.lyrics.classifier.service.lyrics;

public enum Genre {
    METAL(0d, "METAL"),
    POP(1d, "POP"),
    UNKNOWN(-1d, "UNKNOWN");

    private final Double value;
    private final String name;
    Genre(Double v, String n) { value = v; name = n; }

    public Double getValue() { return value; }
    public String getName()  { return name; }

    public static Genre from(int idx) {
        for (Genre g : Genre.values()) {
            if (g.ordinal() == idx) {
                return g;
            }
        }
        return UNKNOWN;
    }
}
