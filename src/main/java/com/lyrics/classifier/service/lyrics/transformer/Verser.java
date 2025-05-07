// src/main/java/com/lyrics/classifier/service/lyrics/transformer/Verser.java
package com.lyrics.classifier.service.lyrics.transformer;

import java.util.UUID;

import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.param.*;
import org.apache.spark.ml.util.*;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.expressions.UserDefinedFunction;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import com.lyrics.classifier.column.Column;

import scala.Option;
import scala.collection.Seq;

public class Verser extends Transformer
        implements DefaultParamsWritable, DefaultParamsReadable<Verser> {

    /* ───────────────────────── params ─────────────────────────── */
    private final String uid;
    private final IntParam sentencesInVerse = new IntParam(this,
            "sentencesInVerse",
            "How many sentences constitute one verse",
            ParamValidators.gt(0));

    /* ─────────────────────── constructors ─────────────────────── */
    public Verser() { // ← 0-arg ctor for Spark reflection
        this(Identifiable.randomUID("verser"));
    }

    public Verser(String uid) {
        this.uid = uid;
        setDefault(sentencesInVerse, 4);
    }

    /* ─────────────── setters / getters for the param ───────────── */
    public IntParam sentencesInVerse() {
        return sentencesInVerse;
    }

    public Verser setSentencesInVerse(int v) {
        set(sentencesInVerse, v);
        return this;
    }

    public int getSentencesInVerse() {
        Option<Object> v = get(sentencesInVerse);
        return v.isEmpty() ? 4 : (Integer) v.get();
    }

    /* ──────────────────────── transform ───────────────────────── */
    @Override
    public Dataset<Row> transform(Dataset<?> ds) {

        Dataset<Row> withVerseId = ds.withColumn(
                "verseInternalId",
                functions.floor(
                        functions
                                .col(Column.ROW_NUMBER.getName())
                                .minus(1)
                                .divide(getSentencesInVerse()))
                        .plus(1));

        Dataset<Row> verses = withVerseId
        .groupBy(Column.ID.getName(), "verseInternalId")
                .agg(
                        functions.first(Column.LABEL.getName()).alias(Column.LABEL.getName()),
                        functions.split(
                                functions.concat_ws(" ",
                                        functions.collect_list(
                                                functions.col(Column.STEMMED_SENTENCE.getName()))),
                                " ").alias(Column.VERSE.getName()));

        return verses.drop(Column.ID.getName()).drop("verseInternalId");
    }

    /* ───────────────────── schema handling ────────────────────── */
    @Override
    public StructType transformSchema(StructType schema) {

        return new StructType() // start empty & append
                .add(Column.LABEL.getName(), DataTypes.DoubleType, false)
                .add(Column.VERSE.getName(),
                        DataTypes.createArrayType(DataTypes.StringType), false);
    }

    /* ────────────────── meta (uid / copy / io) ────────────────── */
    @Override
    public String uid() {
        return uid;
    }

    @Override
    public Verser copy(ParamMap extra) {
        return defaultCopy(extra);
    }

    /* ------- you may REMOVE the following method entirely ------- */
    /** Keep only if you want an explicit loader in Java code. */
    // public Verser load(String path) { // ← **now static**
    // // `DefaultParamsReader.loadParamsInstance` (Spark 3) needs the Spark-Session
    // –
    // // easiest is to reuse the helper already provided by DefaultParamsReadable:
    // return DefaultParamsReadable.<Verser>load(path);
    // }
    /* ------------------------------------------------------------ */
}