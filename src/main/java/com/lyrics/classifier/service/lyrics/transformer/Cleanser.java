package com.lyrics.classifier.service.lyrics.transformer;

import com.lyrics.classifier.column.Column;
import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.param.Param;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.util.*;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import java.io.IOException;

public class Cleanser extends Transformer implements DefaultParamsWritable {

    private final String uid;

    public final Param<String> inputCol;
    public final Param<String> outputCol;

    public Param<String> inputCol() {
        return inputCol;
    }

    public Param<String> outputCol() {
        return outputCol;
    }

    public Cleanser(String uid) {
        this.uid = uid;
        this.inputCol = new Param<>(this, "inputCol", "input column name for Cleanser");
        this.outputCol = new Param<>(this, "outputCol", "output column name for Cleanser");

        setDefault(this.inputCol, Column.VALUE.getName());
        setDefault(this.outputCol, Column.CLEAN.getName());
    }

    public Cleanser() {
        this(Identifiable.randomUID("cleanser"));
    }

    public Cleanser setInputCol(String value) {
        set(this.inputCol, value);
        return this;
    }

    public String getInputCol() {
        return getOrDefault(this.inputCol);
    }

    public Cleanser setOutputCol(String value) {
        set(this.outputCol, value);
        return this;
    }

    public String getOutputCol() {
        return getOrDefault(this.outputCol);
    }

    @Override
    public String uid() {
        return this.uid;
    }

    @Override
    public Dataset<Row> transform(Dataset<?> ds) {
        return ds.withColumn(getOutputCol(),
                functions.regexp_replace(
                        functions.lower(ds.col(getInputCol())),
                        "[^\\p{IsAlphabetic}\\s]", ""));
    }

    @Override
    public StructType transformSchema(StructType schema) {
        return schema.add(getOutputCol(), DataTypes.StringType, true);
    }

    @Override
    public Cleanser copy(ParamMap extra) {
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

    public static MLReader<Cleanser> read() {
        return new DefaultParamsReader<>();
    }
}