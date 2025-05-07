package com.lyrics.classifier.column;

public enum Column {
    VALUE("lyrics"),
    ID("id"),
    LABEL("label"),
    CLEAN("clean"),
    TOKENS("tokens"), 
    WORDS("words"), 
    FILTERED_WORDS("filtered_words"),
    FILTERED_WORD("filtered_word"), 
    STEMMED_WORD("stemmed_word"),
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