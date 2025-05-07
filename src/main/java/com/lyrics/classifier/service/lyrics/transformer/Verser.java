package com.lyrics.classifier.service.lyrics.transformer;

import com.lyrics.classifier.column.Column;
import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.param.*;
import org.apache.spark.ml.util.*;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import java.io.IOException;
import scala.Option;

public class Verser extends Transformer implements DefaultParamsWritable {

    private final String uid;
    private final String verseInternalIdCol = "verseInternalId";

    public final Param<String> inputCol;
    public final Param<String> outputCol;
    public final IntParam sentencesInVerse;

    public Param<String> inputCol() {
        return inputCol;
    }

    public Param<String> outputCol() {
        return outputCol;
    }

    public IntParam sentencesInVerseParam() {
        return sentencesInVerse;
    }

    public Verser(String uid) {
        this.uid = uid;
        this.inputCol = new Param<>(this, "inputCol", "input column (stemmed sentence)");
        this.outputCol = new Param<>(this, "outputCol", "output column (verse as array of words)");
        this.sentencesInVerse = new IntParam(this, "sentencesInVerse", "How many sentences constitute one verse",
                ParamValidators.gt(0));

        setDefault(this.inputCol, Column.STEMMED_SENTENCE.getName());
        setDefault(this.outputCol, Column.VERSE.getName());
        setDefault(this.sentencesInVerse, 4);
    }

    public Verser() {
        this(Identifiable.randomUID("verser"));
    }

    public Verser setInputCol(String value) {
        set(this.inputCol, value);
        return this;
    }

    public String getInputCol() {
        return getOrDefault(this.inputCol);
    }

    public Verser setOutputCol(String value) {
        set(this.outputCol, value);
        return this;
    }

    public String getOutputCol() {
        return getOrDefault(this.outputCol);
    }

    public Verser setSentencesInVerse(int value) {
        set(this.sentencesInVerse, value);
        return this;
    }

    public int getSentencesInVerse() {
        Option<Object> valueOption = get(this.sentencesInVerse);
        if (valueOption.isDefined()) {
            return (Integer) valueOption.get();
        } else {
            // If default was not set (which it is in constructor), this would be an issue.
            // But `getDefault` itself returns an Option, so .get() on it is safe if a
            // default exists.
            Option<Object> defaultOption = getDefault(this.sentencesInVerse);
            if (defaultOption.isDefined()) {
                return (Integer) defaultOption.get();
            }
            // Should not happen if constructor sets default
            throw new IllegalStateException("sentencesInVerse param has no value and no default value.");
        }
    }

    @Override
    public String uid() {
        return this.uid;
    }

    @Override
    public Dataset<Row> transform(Dataset<?> ds) {
        Dataset<Row> withVerseId = ds.withColumn(
                verseInternalIdCol,
                functions.floor(
                        functions.col(Column.ROW_NUMBER.getName()).minus(1)
                                .divide(getSentencesInVerse()))
                        .plus(1));

        Dataset<Row> verses = withVerseId
                .groupBy(Column.ID.getName(), verseInternalIdCol)
                .agg(
                        functions.first(Column.LABEL.getName()).alias(Column.LABEL.getName()),
                        functions.split(
                                functions.concat_ws(" ",
                                        functions.collect_list(functions.col(getInputCol()))),
                                " ").alias(getOutputCol()));

        return verses.drop(Column.ID.getName()).drop(verseInternalIdCol);
    }

    @Override
    public StructType transformSchema(StructType schema) {
        StructField labelField;
        try {
            labelField = schema.apply(Column.LABEL.getName());
        } catch (IllegalArgumentException e) {
            // This case should ideally not be hit if the pipeline is structured correctly,
            // meaning LABEL column exists before this transformer.
            labelField = DataTypes.createStructField(Column.LABEL.getName(), DataTypes.DoubleType, true);
        }

        return new StructType()
                .add(labelField)
                .add(getOutputCol(), DataTypes.createArrayType(DataTypes.StringType), true);
    }

    @Override
    public Verser copy(ParamMap extra) {
        return defaultCopy(extra);
    }

    @Override
    public void save(String path) throws IOException {
        write().save(path);
    }

    @Override
    public MLWriter write() {
        return new DefaultParamsWriter(this);
    }

    public static MLReader<Verser> read() {
        return new DefaultParamsReader<>();
    }
}