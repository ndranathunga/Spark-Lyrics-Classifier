package com.lyrics.classifier.column;

public enum Column {
    VALUE("lyrics"), // Changed to "lyrics" to match CSV header directly
    ID("id"),
    LABEL("label"),
    CLEAN("clean"),
    TOKENS("tokens"), // Added for Tokenizer output
    WORDS("words"), // Kept for compatibility, can be an alias for TOKENS or FILTERED_WORDS
    FILTERED_WORDS("filtered_words"),
    FILTERED_WORD("filtered_word"), // Singular for Exploder output
    STEMMED_WORD("stemmed_word"), // Singular for Stemmer output
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