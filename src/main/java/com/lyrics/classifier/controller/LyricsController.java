package com.lyrics.classifier.controller;

import com.lyrics.classifier.service.LyricsService;
import com.lyrics.classifier.service.lyrics.GenrePrediction;
import java.util.Map;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api")
@CrossOrigin(origins = {"http://localhost:8080", "null", "http://127.0.0.1:8501", "http://localhost:8501"}) 
public class LyricsController {

    private final LyricsService lyricsService;

    public LyricsController(LyricsService lyricsService) {
        this.lyricsService = lyricsService;
    }

    @PostMapping("/train")
    public Map<String, Object> train() {
        return lyricsService.classifyLyrics();
    }

    @PostMapping("/predict")
    public GenrePrediction predict(@RequestBody Map<String, String> body) {
        return lyricsService.predictGenre(body.getOrDefault("lyrics", ""));
    }
}