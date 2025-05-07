// src/main/java/com/lyrics/classifier/service/lyrics/transformer/Exploder.java
package com.lyrics.classifier.service.lyrics.transformer;

import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.util.Identifiable;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.types.StructType;

/** Placeholder – no-op transformer */
public class Exploder extends Transformer {
    private final String uid = Identifiable.randomUID("exploder");
    @Override public String uid() { return uid; }

    @Override public Dataset<Row> transform(Dataset<?> ds) { return ds.toDF(); }

    @Override public StructType transformSchema(StructType schema) { return schema; }

    @Override public Exploder copy(ParamMap extra) { return new Exploder(); }
}
