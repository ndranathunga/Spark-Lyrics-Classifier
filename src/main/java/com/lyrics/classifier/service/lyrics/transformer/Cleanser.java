// src/main/java/com/lyrics/classifier/service/lyrics/transformer/Cleanser.java
package com.lyrics.classifier.service.lyrics.transformer;

import static com.lyrics.classifier.column.Column.*;

import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.util.Identifiable;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.functions;

public class Cleanser extends Transformer {

    private final String uid = Identifiable.randomUID("cleanser");

    @Override public String uid() { return uid; }

    @Override
    public Dataset<Row> transform(Dataset<?> ds) {
        return ds.withColumn(CLEAN.getName(),
                functions.regexp_replace(
                        functions.lower(ds.col(VALUE.getName())),
                        "[^\\p{IsAlphabetic}\\s]", ""));
    }

    @Override
    public StructType transformSchema(StructType schema) { return schema; }

    @Override
    public Cleanser copy(ParamMap extra) { return new Cleanser(); }
}
