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
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Exploder extends Transformer implements DefaultParamsWritable {
    private final String uid;

    public final Param<String> inputCol;
    public final Param<String> outputCol;

    public Param<String> inputCol() {
        return inputCol;
    }

    public Param<String> outputCol() {
        return outputCol;
    }

    public Exploder(String uid) {
        this.uid = uid;
        this.inputCol = new Param<>(this, "inputCol", "input column (array of words)");
        this.outputCol = new Param<>(this, "outputCol", "output column (single word)");

        setDefault(this.inputCol, Column.FILTERED_WORDS.getName());
        setDefault(this.outputCol, Column.FILTERED_WORD.getName());
    }

    public Exploder() {
        this(Identifiable.randomUID("exploder"));
    }

    public Exploder setInputCol(String value) {
        set(this.inputCol, value);
        return this;
    }

    public String getInputCol() {
        return getOrDefault(this.inputCol);
    }

    public Exploder setOutputCol(String value) {
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
        List<String> existingColsToKeep = new ArrayList<>(Arrays.asList(ds.columns()));
        existingColsToKeep.removeIf(colName -> colName.equals(getInputCol()));

        org.apache.spark.sql.Column[] selectCols = new org.apache.spark.sql.Column[existingColsToKeep.size() + 1];
        for (int i = 0; i < existingColsToKeep.size(); i++) {
            selectCols[i] = functions.col(existingColsToKeep.get(i));
        }
        selectCols[existingColsToKeep.size()] = functions.explode(ds.col(getInputCol())).as(getOutputCol());

        return ds.select(selectCols);
    }

    @Override
    public StructType transformSchema(StructType schema) {
        StructType newSchema = new StructType();
        for (StructField field : schema.fields()) {
            if (!field.name().equals(getInputCol())) {
                newSchema = newSchema.add(field);
            }
        }
        // Explode makes the output nullable by nature if the input array is empty or
        // null
        return newSchema.add(getOutputCol(), DataTypes.StringType, true);
    }

    @Override
    public Exploder copy(ParamMap extra) {
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

    public static MLReader<Exploder> read() {
        return new DefaultParamsReader<>();
    }
}