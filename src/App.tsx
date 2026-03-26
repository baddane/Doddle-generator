/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useRef } from 'react';
import { GoogleGenAI, Type, GenerateContentResponse } from "@google/genai";
import { motion, AnimatePresence } from "motion/react";
import { 
  Send, 
  Loader2, 
  ChevronLeft, 
  ChevronRight, 
  Download, 
  Play, 
  Sparkles,
  RefreshCw,
  Image as ImageIcon,
  CheckCircle2
} from 'lucide-react';

// --- Types ---

interface Scene {
  scene: number;
  description: string;
  mainEmotion: string;
  secondaryEmotion: string;
  text: string;
  imageUrl?: string;
  prompt?: string;
}

// --- AI Service ---

const genAI = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY || '' });

async function storyToScenes(story: string): Promise<Scene[]> {
  const model = "gemini-3-flash-preview";
  const prompt = `Break this story into a sequence of scenes. 
- For short stories: 5 scenes.
- For long stories: 8-10 scenes.

Structure the scenes to cover the entire narrative from beginning to end.
Each scene must have:
1. A unique description for the AI artist.
2. A short snippet of narration text (max 12 words) to be displayed on the image.
3. Main emotion and secondary emotion for the characters.

Story: ${story}

Return the result as a JSON array of objects.`;

  const response = await genAI.models.generateContent({
    model,
    contents: [{ role: 'user', parts: [{ text: prompt }] }],
    config: {
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.ARRAY,
        minItems: 5,
        maxItems: 12,
        items: {
          type: Type.OBJECT,
          properties: {
            scene: { type: Type.INTEGER },
            description: { type: Type.STRING },
            mainEmotion: { type: Type.STRING },
            secondaryEmotion: { type: Type.STRING },
            text: { type: Type.STRING }
          },
          required: ["scene", "description", "mainEmotion", "secondaryEmotion", "text"]
        }
      }
    }
  });

  try {
    const text = response.text || '[]';
    const parsed = JSON.parse(text);
    return Array.isArray(parsed) ? parsed : [];
  } catch (e) {
    console.error("Failed to parse scenes JSON", e);
    return [];
  }
}

async function generateSceneImage(scene: Scene, aspectRatio: string): Promise<string> {
  const model = "gemini-2.5-flash-image";
  const prompt = `A minimalist black and white stick figure illustration, simple hand-drawn style, clean white background. 

CRITICAL RULES:
- ABSOLUTELY NO text, words, letters, labels, or signatures in the image.
- NO speech bubbles, NO thought bubbles.
- The image should be pure visual art with stick figures only.

VISUAL STYLE:
- In the center, a stick figure showing ${scene.mainEmotion} through body language and facial expression ONLY. 
- On both sides, small groups of stick figures showing ${scene.secondaryEmotion} through body language and facial expression ONLY. 
- The drawing is simple, cartoon-like, with thin lines and no shading. 
- Keep the background pure white and empty.

High contrast, minimal, inspirational tone. Scene context: ${scene.description}.`;

  const response = await genAI.models.generateContent({
    model,
    contents: [{ role: 'user', parts: [{ text: prompt }] }],
    config: {
      imageConfig: {
        aspectRatio: aspectRatio as any,
      }
    }
  });

  const part = response.candidates?.[0]?.content?.parts.find(p => p.inlineData);
  if (part?.inlineData?.data) {
    return `data:image/png;base64,${part.inlineData.data}`;
  }
  
  throw new Error("No image data returned");
}

/**
 * Overlays text onto the generated image using a canvas to ensure
 * perfect consistency in font, size, and positioning across all scenes.
 */
async function addTextToImage(base64: string, text: string): Promise<string> {
  return new Promise((resolve) => {
    const img = new Image();
    // No crossOrigin for data URLs
    const timeout = setTimeout(() => {
      console.warn("Image processing timed out for scene");
      resolve(base64);
    }, 5000);

    img.onload = () => {
      clearTimeout(timeout);
      const canvas = document.createElement('canvas');
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext('2d');
      if (!ctx) return resolve(base64);

      // 1. Draw original image
      ctx.drawImage(img, 0, 0);

      // 2. Configure text style (Consistent across all images)
      const padding = canvas.width * 0.08;
      const fontSize = Math.floor(canvas.width * 0.045); // Responsive but fixed ratio
      ctx.font = `500 ${fontSize}px "Inter", -apple-system, sans-serif`;
      ctx.fillStyle = 'black';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';

      // 3. Wrap text logic
      const maxWidth = canvas.width - (padding * 2);
      const words = text.split(' ');
      let line = '';
      const lines = [];
      
      for (let n = 0; n < words.length; n++) {
        const testLine = line + words[n] + ' ';
        const metrics = ctx.measureText(testLine);
        const testWidth = metrics.width;
        if (testWidth > maxWidth && n > 0) {
          lines.push(line);
          line = words[n] + ' ';
        } else {
          line = testLine;
        }
      }
      lines.push(line);

      // 4. Draw text at the top with consistent spacing
      const startY = canvas.height * 0.06;
      const lineHeight = fontSize * 1.3;
      
      lines.forEach((line, i) => {
        ctx.fillText(line.trim(), canvas.width / 2, startY + (i * lineHeight));
      });

      resolve(canvas.toDataURL('image/png'));
    };
    img.onerror = () => {
      clearTimeout(timeout);
      resolve(base64);
    };
    img.src = base64;
  });
}

// --- Components ---

const LoadingOverlay = ({ message }: { message: string }) => (
  <motion.div 
    initial={{ opacity: 0 }}
    animate={{ opacity: 1 }}
    exit={{ opacity: 0 }}
    className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-white/90 backdrop-blur-sm"
  >
    <div className="relative">
      <motion.div
        animate={{ rotate: 360 }}
        transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
        className="w-16 h-16 border-4 border-black border-t-transparent rounded-full"
      />
      <Sparkles className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-6 h-6 text-black" />
    </div>
    <p className="mt-6 text-lg font-medium tracking-tight text-black animate-pulse">
      {message}
    </p>
  </motion.div>
);

export default function App() {
  const [story, setStory] = useState('');
  const [scenes, setScenes] = useState<Scene[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState('');
  const [currentIndex, setCurrentIndex] = useState(0);
  const [videoMode, setVideoMode] = useState(false);
  const [format, setFormat] = useState<"9:16" | "16:9">("9:16");

  const handleGenerate = async () => {
    if (!story.trim()) return;

    setIsGenerating(true);
    setLoadingMessage("Analyzing your story...");
    
    try {
      const generatedScenes = await storyToScenes(story);
      if (generatedScenes.length === 0) {
        throw new Error("No scenes were generated.");
      }
      setScenes(generatedScenes);
      setCurrentIndex(0);
      
      setLoadingMessage(`Drawing ${generatedScenes.length} scenes in parallel...`);

      // Generate all images in parallel for much faster results
      await Promise.all(generatedScenes.map(async (scene, i) => {
        try {
          const rawImageUrl = await generateSceneImage(scene, format);
          const processedImageUrl = await addTextToImage(rawImageUrl, scene.text);
          
          setScenes(prev => {
            const next = [...prev];
            if (next[i]) {
              next[i] = { ...next[i], imageUrl: processedImageUrl };
            }
            return next;
          });
        } catch (err) {
          console.error(`Failed to generate image for scene ${i + 1}`, err);
          // We don't throw here to allow other scenes to finish
        }
      }));

    } catch (error) {
      console.error("Generation failed", error);
      alert("Something went wrong during generation. Please try again.");
    } finally {
      setIsGenerating(false);
    }
  };

  const downloadAll = () => {
    scenes.forEach((scene, index) => {
      if (scene.imageUrl) {
        const link = document.createElement('a');
        link.href = scene.imageUrl;
        link.download = `stick-story-scene-${index + 1}.png`;
        link.click();
      }
    });
  };

  const startVideoPreview = () => {
    setVideoMode(true);
    setCurrentIndex(0);
    let index = 0;
    const interval = setInterval(() => {
      index++;
      if (index >= scenes.length) {
        clearInterval(interval);
        setVideoMode(false);
      } else {
        setCurrentIndex(index);
      }
    }, 3000);
  };

  return (
    <div className="min-h-screen bg-white text-black font-sans selection:bg-black selection:text-white">
      <AnimatePresence>
        {isGenerating && <LoadingOverlay message={loadingMessage} />}
      </AnimatePresence>

      {/* Header */}
      <header className="max-w-4xl mx-auto px-6 py-12 text-center">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-black/10 text-xs font-semibold uppercase tracking-widest mb-6">
            <Sparkles className="w-3 h-3" />
            AI Content Creator
          </div>
          <h1 className="text-5xl md:text-7xl font-bold tracking-tighter mb-4">
            Stick Story AI
          </h1>
          <p className="text-xl text-black/60 max-w-2xl mx-auto leading-relaxed">
            Turn your stories into viral visual content with minimalist stick-figure animations.
          </p>
        </motion.div>
      </header>

      <main className="max-w-4xl mx-auto px-6 pb-24">
        {scenes.length === 0 ? (
          <motion.div 
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-white border-2 border-black p-8 md:p-12 shadow-[8px_8px_0px_0px_rgba(0,0,0,1)]"
          >
            <label className="block text-sm font-bold uppercase tracking-wider mb-4">
              Enter your story
            </label>
            <textarea
              value={story}
              onChange={(e) => setStory(e.target.value)}
              placeholder="Once upon a time, a small stick figure decided to climb the highest mountain..."
              className="w-full h-48 p-4 text-lg border-2 border-black focus:outline-none focus:ring-0 resize-none mb-6 placeholder:text-black/20"
            />

            <div className="mb-8">
              <label className="block text-sm font-bold uppercase tracking-wider mb-4">
                Select Format
              </label>
              <div className="grid grid-cols-2 gap-4">
                <button
                  onClick={() => setFormat("9:16")}
                  className={`py-3 border-2 border-black font-bold uppercase tracking-widest transition-all flex items-center justify-center gap-2 ${format === "9:16" ? 'bg-black text-white' : 'bg-white text-black hover:bg-black/5'}`}
                >
                  <div className="w-3 h-5 border-2 border-current rounded-sm" />
                  TikTok / Shorts (9:16)
                </button>
                <button
                  onClick={() => setFormat("16:9")}
                  className={`py-3 border-2 border-black font-bold uppercase tracking-widest transition-all flex items-center justify-center gap-2 ${format === "16:9" ? 'bg-black text-white' : 'bg-white text-black hover:bg-black/5'}`}
                >
                  <div className="w-5 h-3 border-2 border-current rounded-sm" />
                  YouTube (16:9)
                </button>
              </div>
            </div>

            <button
              onClick={handleGenerate}
              disabled={!story.trim() || isGenerating}
              className="w-full py-4 bg-black text-white text-lg font-bold uppercase tracking-widest hover:bg-black/90 transition-colors flex items-center justify-center gap-3 disabled:opacity-50"
            >
              {isGenerating ? <Loader2 className="animate-spin" /> : <Send className="w-5 h-5" />}
              Generate Visual Story
            </button>
          </motion.div>
        ) : (
          <div className="space-y-12">
            {/* Carousel Section */}
            <div className="relative group">
              <div 
                className={`mx-auto bg-white border-4 border-black shadow-[12px_12px_0px_0px_rgba(0,0,0,1)] overflow-hidden relative transition-all duration-500 ${format === "9:16" ? 'aspect-[9/16] max-w-[400px]' : 'aspect-[16/9] max-w-full'}`}
              >
                <AnimatePresence mode="wait">
                  <motion.div
                    key={currentIndex}
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -20 }}
                    className="absolute inset-0"
                  >
                    {scenes[currentIndex].imageUrl ? (
                      <img 
                        src={scenes[currentIndex].imageUrl} 
                        alt={`Scene ${currentIndex + 1}`}
                        className="w-full h-full object-cover"
                        referrerPolicy="no-referrer"
                      />
                    ) : (
                      <div className="w-full h-full flex flex-col items-center justify-center gap-4 bg-gray-50">
                        <Loader2 className="w-8 h-8 animate-spin text-black/20" />
                        <p className="text-xs font-bold uppercase tracking-widest text-black/40">Drawing Scene...</p>
                      </div>
                    )}
                  </motion.div>
                </AnimatePresence>

                {/* Navigation Overlays */}
                {!videoMode && (
                  <>
                    <button 
                      onClick={() => setCurrentIndex(prev => Math.max(0, prev - 1))}
                      className="absolute left-4 top-1/2 -translate-y-1/2 w-10 h-10 bg-white border-2 border-black flex items-center justify-center hover:bg-black hover:text-white transition-colors z-10"
                    >
                      <ChevronLeft className="w-6 h-6" />
                    </button>
                    <button 
                      onClick={() => setCurrentIndex(prev => Math.min(scenes.length - 1, prev + 1))}
                      className="absolute right-4 top-1/2 -translate-y-1/2 w-10 h-10 bg-white border-2 border-black flex items-center justify-center hover:bg-black hover:text-white transition-colors z-10"
                    >
                      <ChevronRight className="w-6 h-6" />
                    </button>
                  </>
                )}

                {/* Scene Counter */}
                <div className="absolute bottom-6 left-1/2 -translate-x-1/2 px-4 py-1 bg-black text-white text-xs font-bold tracking-widest uppercase">
                  Scene {currentIndex + 1} / {scenes.length}
                </div>
              </div>
            </div>

            {/* Controls */}
            <div className="flex flex-wrap justify-center gap-4">
              <button
                onClick={startVideoPreview}
                disabled={scenes.some(s => !s.imageUrl) || videoMode}
                className="px-8 py-3 bg-white border-2 border-black font-bold uppercase tracking-widest hover:bg-black hover:text-white transition-all flex items-center gap-2 disabled:opacity-50"
              >
                <Play className="w-4 h-4" />
                Preview Video
              </button>
              <button
                onClick={downloadAll}
                disabled={scenes.some(s => !s.imageUrl)}
                className="px-8 py-3 bg-white border-2 border-black font-bold uppercase tracking-widest hover:bg-black hover:text-white transition-all flex items-center gap-2 disabled:opacity-50"
              >
                <Download className="w-4 h-4" />
                Download All
              </button>
              <button
                onClick={() => {
                  setScenes([]);
                  setStory('');
                  setCurrentIndex(0);
                }}
                className="px-8 py-3 bg-white border-2 border-black font-bold uppercase tracking-widest hover:bg-black hover:text-white transition-all flex items-center gap-2"
              >
                <RefreshCw className="w-4 h-4" />
                Start Over
              </button>
            </div>

            {/* Storyboard Grid */}
            <div className="mt-16">
              <h2 className="text-2xl font-black uppercase tracking-tighter mb-8 flex items-center gap-4">
                <span className="bg-black text-white px-3 py-1">Storyboard</span>
                <span className="text-black/30">{scenes.length} Scenes</span>
              </h2>
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
                {scenes.map((scene, idx) => (
                  <button
                    key={idx}
                    onClick={() => setCurrentIndex(idx)}
                    className={`text-left group transition-all ${currentIndex === idx ? 'scale-105' : 'opacity-60 hover:opacity-100'}`}
                  >
                    <div className={`aspect-[9/16] border-2 border-black mb-3 overflow-hidden bg-gray-50 relative ${currentIndex === idx ? 'shadow-[4px_4px_0px_0px_rgba(0,0,0,1)]' : ''}`}>
                      {scene.imageUrl ? (
                        <img 
                          src={scene.imageUrl} 
                          alt={`Scene ${idx + 1}`} 
                          className="w-full h-full object-cover"
                          referrerPolicy="no-referrer"
                        />
                      ) : (
                        <div className="w-full h-full flex items-center justify-center">
                          <Loader2 className="w-6 h-6 border-2 border-black border-t-transparent rounded-full animate-spin" />
                        </div>
                      )}
                      <div className="absolute top-2 left-2 bg-black text-white text-[10px] font-bold px-1.5 py-0.5 uppercase">
                        {idx + 1}
                      </div>
                    </div>
                    <p className="text-[10px] font-bold uppercase tracking-widest line-clamp-2 leading-tight">
                      {scene.text}
                    </p>
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-black/5 py-12 text-center text-black/40 text-xs font-bold uppercase tracking-widest">
        &copy; 2026 Stick Story AI &bull; Built with Gemini
      </footer>
    </div>
  );
}
