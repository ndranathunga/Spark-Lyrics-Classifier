// src/main/java/com/lyrics/classifier/column/Column.java
package com.lyrics.classifier.column;

public enum Column {
    VALUE("value"),
    ID("id"),
    LABEL("label"),
    CLEAN("clean"),
    WORDS("words"),
    FILTERED_WORDS("filtered_words"),
    VERSE("verse"),
    ROW_NUMBER("row_number"),
    STEMMED_SENTENCE("stemmed_sentence");

    private final String name;

    Column(String n) {
        this.name = n;
    }

    public String getName() {
        return name;
    }
}
