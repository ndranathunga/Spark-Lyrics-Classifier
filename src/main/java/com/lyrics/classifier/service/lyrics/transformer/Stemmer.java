// src/main/java/com/lyrics/classifier/service/lyrics/transformer/Stemmer.java
package com.lyrics.classifier.service.lyrics.transformer;

import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.util.Identifiable;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.types.StructType;

/** Placeholder – replace with a real Porter stemmer if you like */
public class Stemmer extends Transformer {
    private final String uid = Identifiable.randomUID("stemmer");
    @Override public String uid() { return uid; }

    @Override public Dataset<Row> transform(Dataset<?> ds) { return ds.toDF(); }

    @Override public StructType transformSchema(StructType schema) { return schema; }

    @Override public Stemmer copy(ParamMap extra) { return new Stemmer(); }
}
