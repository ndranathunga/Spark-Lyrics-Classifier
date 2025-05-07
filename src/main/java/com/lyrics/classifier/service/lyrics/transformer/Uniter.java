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

public class Uniter extends Transformer implements DefaultParamsWritable {
    private final String uid;

    public final Param<String> inputCol;
    public final Param<String> outputCol;

    public Param<String> inputCol() {
        return inputCol;
    }

    public Param<String> outputCol() {
        return outputCol;
    }

    public Uniter(String uid) {
        this.uid = uid;
        this.inputCol = new Param<>(this, "inputCol", "input column (single stemmed word)");
        this.outputCol = new Param<>(this, "outputCol", "output column (stemmed sentence)");

        setDefault(this.inputCol, Column.STEMMED_WORD.getName());
        setDefault(this.outputCol, Column.STEMMED_SENTENCE.getName());
    }

    public Uniter() {
        this(Identifiable.randomUID("uniter"));
    }

    public Uniter setInputCol(String value) {
        set(this.inputCol, value);
        return this;
    }

    public String getInputCol() {
        return getOrDefault(this.inputCol);
    }

    public Uniter setOutputCol(String value) {
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
        return ds.groupBy(
                Column.ID.getName(),
                Column.ROW_NUMBER.getName(),
                Column.LABEL.getName())
                .agg(functions.concat_ws(" ", functions.collect_list(getInputCol()))
                        .as(getOutputCol()));
    }

    @Override
    public StructType transformSchema(StructType schema) {
        return new StructType()
                .add(schema.apply(Column.ID.getName()))
                .add(schema.apply(Column.ROW_NUMBER.getName()))
                .add(schema.apply(Column.LABEL.getName()))
                .add(getOutputCol(), DataTypes.StringType, true);
    }

    @Override
    public Uniter copy(ParamMap extra) {
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

    public static MLReader<Uniter> read() {
        return new DefaultParamsReader<>();
    }
}