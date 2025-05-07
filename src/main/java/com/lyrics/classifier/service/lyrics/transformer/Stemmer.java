package com.lyrics.classifier.service.lyrics.transformer;

import com.lyrics.classifier.column.Column;
import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.param.Param;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.util.*;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import org.tartarus.snowball.ext.EnglishStemmer; 

import java.io.IOException;
import java.util.Arrays;
import java.util.UUID; 

public class Stemmer extends Transformer implements DefaultParamsWritable {
    private final String uid;

    public final Param<String> inputCol;
    public final Param<String> outputCol;

    public Param<String> inputCol() {
        return inputCol;
    }

    public Param<String> outputCol() {
        return outputCol;
    }

    public Stemmer(String uid) {
        this.uid = uid;
        this.inputCol = new Param<>(this, "inputCol", "input column (single word)");
        this.outputCol = new Param<>(this, "outputCol", "output column (stemmed word)");

        setDefault(this.inputCol, Column.FILTERED_WORD.getName());
        setDefault(this.outputCol, Column.STEMMED_WORD.getName());
    }

    public Stemmer() {
        this(Identifiable.randomUID("stemmer"));
    }

    public Stemmer setInputCol(String value) {
        set(this.inputCol, value);
        return this;
    }

    public String getInputCol() {
        return getOrDefault(this.inputCol);
    }

    public Stemmer setOutputCol(String value) {
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
        final String currentInputColName = getInputCol();
        final String currentOutputColName = getOutputCol();

        String udfName = "stem_word_" + UUID.randomUUID().toString().replace("-", "");

        SparkSession spark = ds.sparkSession();

        UDF1<String, String> stemmerUDF = (String word) -> {
            if (word == null) {
                return null;
            }
            EnglishStemmer stemmer = new EnglishStemmer(); 
            stemmer.setCurrent(word.toLowerCase());
            if (stemmer.stem()) {
                return stemmer.getCurrent();
            }
            return word;
        };

        spark.udf().register(udfName, stemmerUDF, DataTypes.StringType);

        return ds.withColumn(currentOutputColName, functions.callUDF(udfName, functions.col(currentInputColName)));
    }

    @Override
    public StructType transformSchema(StructType schema) {
        if (!Arrays.asList(schema.fieldNames()).contains(getInputCol())) {
            throw new IllegalArgumentException(
                    "Input column " + getInputCol() + " does not exist in the input schema.");
        }
        return schema.add(getOutputCol(), DataTypes.StringType, true);
    }

    @Override
    public Stemmer copy(ParamMap extra) {
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

    public static MLReader<Stemmer> read() {
        return new DefaultParamsReader<>();
    }
}