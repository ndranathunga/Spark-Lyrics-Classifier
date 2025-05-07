// src/main/java/com/lyrics/classifier/service/lyrics/transformer/Numerator.java
package com.lyrics.classifier.service.lyrics.transformer;

import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.util.Identifiable;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.types.StructType;

/** Stub that just passes the data through. */
public class Numerator extends Transformer {

    private final String uid = Identifiable.randomUID("numerator");
    @Override public String uid() { return uid; }

    @Override public Dataset<Row> transform(Dataset<?> ds) { return ds.toDF(); }

    @Override public StructType transformSchema(StructType schema) { return schema; }

    @Override public Numerator copy(ParamMap extra) { return new Numerator(); }
}
