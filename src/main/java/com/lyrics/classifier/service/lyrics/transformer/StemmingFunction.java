package com.lyrics.classifier.service.lyrics.transformer;

import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.tartarus.snowball.ext.EnglishStemmer;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class StemmingFunction implements Serializable {
    private transient EnglishStemmer stemmer; // transient to avoid serialization issues

    private EnglishStemmer getStemmer() {
        if (stemmer == null) {
            stemmer = new EnglishStemmer();
        }
        return stemmer;
    }

    public Row call(Row inputRow, String inputColName, String outputColName) {
        String word = inputRow.getAs(inputColName);
        String stemmedWord = word; // Default to original if null or stemming fails

        if (word != null && !word.isEmpty()) {
            EnglishStemmer currentStemmer = getStemmer();
            currentStemmer.setCurrent(word.toLowerCase());
            if (currentStemmer.stem()) {
                stemmedWord = currentStemmer.getCurrent();
            }
        }

        List<Object> newValues = new ArrayList<>();
        for (String fieldName : inputRow.schema().fieldNames()) {
            if (fieldName.equals(inputColName)) { // Replace the input column with the output column conceptually
                // We will add the new stemmed word later based on outputColName.
                // If outputColName is same as inputColName, then it's an in-place update.
            } else {
                newValues.add(inputRow.getAs(fieldName));
            }
        }
        // If outputCol is different, add it. If same, the value will be updated.
        // This logic assumes outputCol might be a new column or overwrite inputCol.
        // A simpler approach is to create a new Row with specific fields.
        // For this example, assume outputCol is a distinct new column.

        // Reconstruct the row, replacing or adding the stemmed word.
        // This is complex. It's easier if Stemmer transformer handles schema and just
        // gets the stemmed word.

        // Let's simplify: the Stemmer will handle schema. This function just returns
        // the stemmed word.
        // No, the Stemmer transformer uses MapPartitions, so this function must return
        // a full Row.

        Object[] values = new Object[inputRow.length()];
        String[] fieldNames = inputRow.schema().fieldNames();
        boolean outputColExists = false;
        for (int i = 0; i < fieldNames.length; i++) {
            if (fieldNames[i].equals(inputColName)) { // If we are transforming in place (outputCol == inputCol)
                if (fieldNames[i].equals(outputColName)) {
                    values[i] = stemmedWord;
                    outputColExists = true;
                } else { // keep original value if inputCol is not the outputCol
                    values[i] = inputRow.get(i);
                }
            } else if (fieldNames[i].equals(outputColName)) { // If output col is an existing different column
                values[i] = stemmedWord;
                outputColExists = true;
            } else {
                values[i] = inputRow.get(i);
            }
        }

        if (!outputColExists && !inputColName.equals(outputColName)) {
            // This case is not handled by simple RowFactory.create(values) if schema
            // changes.
            // The Stemmer's transformSchema should ensure outputCol is present.
            // For now, assume outputCol is one of the existing columns or Stemmer's schema
            // has it.
            // This function will be called AFTER transformSchema has defined the output
            // structure.
            // So, we just need to produce a Row that matches the output schema of Stemmer.
            // The input Row `inputRow` here matches the input schema of the Stemmer.
            // The output Row MUST match the output schema of the Stemmer.

            // Simpler: create a new row with expected output fields
            // Assuming Stemmer's output schema is (ID, ROW_NUMBER, LABEL, STEMMED_WORD)
            // And inputRow contains (ID, ROW_NUMBER, LABEL, FILTERED_WORD)
            return RowFactory.create(
                    inputRow.getAs(com.lyrics.classifier.column.Column.ID.getName()),
                    inputRow.getAs(com.lyrics.classifier.column.Column.ROW_NUMBER.getName()),
                    inputRow.getAs(com.lyrics.classifier.column.Column.LABEL.getName()),
                    stemmedWord // This corresponds to Column.STEMMED_WORD.getName()
            );
        }
        // This part is tricky if outputColName implies adding a new column that wasn't
        // in inputRow's schema.
        // The mapPartitions call in Stemmer.java should use an Encoder for the *output*
        // schema.
        return RowFactory.create(values);

    }
}