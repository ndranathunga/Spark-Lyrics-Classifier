// src/main/java/com/lyrics/classifier/controller/LyricsController.java
package com.lyrics.classifier.controller;

import com.lyrics.classifier.service.LyricsService;
import com.lyrics.classifier.service.lyrics.GenrePrediction;
import java.util.Map;
import org.springframework.web.bind.annotation.*;

/**
 * Minimal REST façade.
 * POST /api/train → triggers (re)training and returns metrics
 * POST /api/predict → returns genre prediction for supplied lyrics
 */
@RestController
@RequestMapping("/api")
public class LyricsController {

    private final LyricsService lyricsService;

    // ── constructor injection (no Lombok needed) ─────────────────────────
    public LyricsController(LyricsService lyricsService) {
        this.lyricsService = lyricsService;
    }

    /** Kick off training and get back model statistics. */
    @PostMapping("/train")
    public Map<String, Object> train() {
        return lyricsService.classifyLyrics();
    }

    /** Classify supplied lyrics. JSON body: {"lyrics":"..."} */
    @PostMapping("/predict")
    public GenrePrediction predict(@RequestBody Map<String, String> body) {
        return lyricsService.predictGenre(body.getOrDefault("lyrics", ""));
    }
}
