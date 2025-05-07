package com.lyrics.classifier.config;

import com.lyrics.classifier.service.lyrics.pipeline.LogisticRegressionPipeline;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.core.env.Environment;
import org.springframework.stereotype.Component;

@Component
@ConditionalOnProperty(name = "mode", havingValue = "train", matchIfMissing = false)
public class TrainRunner implements CommandLineRunner {

    private final LogisticRegressionPipeline pipeline;
    private final Environment env;

    public TrainRunner(LogisticRegressionPipeline pipeline, Environment env) {
        this.pipeline = pipeline;
        this.env = env;
    }

    @Override
    public void run(String... args) {
        String mode = env.getProperty("mode", "serve");
        if ("train".equalsIgnoreCase(mode)) {
            pipeline.classify();
            System.exit(0);
        }
    }
}