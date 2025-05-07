package com.lyrics.classifier.service.lyrics.transformer;

import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.tartarus.snowball.ext.EnglishStemmer;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class StemmingFunction implements Serializable {
    private transient EnglishStemmer stemmer;

    private EnglishStemmer getStemmer() {
        if (stemmer == null) {
            stemmer = new EnglishStemmer();
        }
        return stemmer;
    }

    public Row call(Row inputRow, String inputColName, String outputColName) {
        String word = inputRow.getAs(inputColName);
        String stemmedWord = word;

        if (word != null && !word.isEmpty()) {
            EnglishStemmer currentStemmer = getStemmer();
            currentStemmer.setCurrent(word.toLowerCase());
            if (currentStemmer.stem()) {
                stemmedWord = currentStemmer.getCurrent();
            }
        }

        List<Object> newValues = new ArrayList<>();
        for (String fieldName : inputRow.schema().fieldNames()) {
            if (fieldName.equals(inputColName)) {
            } else {
                newValues.add(inputRow.getAs(fieldName));
            }
        }

        Object[] values = new Object[inputRow.length()];
        String[] fieldNames = inputRow.schema().fieldNames();
        boolean outputColExists = false;
        for (int i = 0; i < fieldNames.length; i++) {
            if (fieldNames[i].equals(inputColName)) {
                if (fieldNames[i].equals(outputColName)) {
                    values[i] = stemmedWord;
                    outputColExists = true;
                } else {
                    values[i] = inputRow.get(i);
                }
            } else if (fieldNames[i].equals(outputColName)) {
                values[i] = stemmedWord;
                outputColExists = true;
            } else {
                values[i] = inputRow.get(i);
            }
        }

        if (!outputColExists && !inputColName.equals(outputColName)) {
            return RowFactory.create(
                    inputRow.getAs(com.lyrics.classifier.column.Column.ID.getName()),
                    inputRow.getAs(com.lyrics.classifier.column.Column.ROW_NUMBER.getName()),
                    inputRow.getAs(com.lyrics.classifier.column.Column.LABEL.getName()),
                    stemmedWord);
        }

        return RowFactory.create(values);
    }
}