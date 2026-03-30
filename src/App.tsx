/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useRef, useEffect, useCallback } from 'react';
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
  CheckCircle2,
  Upload,
  X,
  FileText,
  Zap
} from 'lucide-react';

// --- Types ---

interface Scene {
  scene: number;
  description: string;
  mainEmotion: string;
  secondaryEmotion: string;
  text: string;
  visualPrompt: string;
  imageUrl?: string;
  prompt?: string;
}

// --- AI Service ---

const genAI = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY || '' });

async function storyToScenes(story: string): Promise<Scene[]> {
  const model = "gemini-3-flash-preview";
  const prompt = `Break this story into a sequence of scenes for a visual storyboard. 
- For short stories: EXACTLY 5-6 scenes.
- For long stories: 8-12 scenes.

CRITICAL: The scenes MUST cover the entire narrative from the very beginning to the very end. Do not skip any major plot points.

Each scene must have:
1. A unique description for the AI artist (minimalist stick figure style).
2. A short snippet of narration text (max 12 words) to be displayed on the image. 
   CRITICAL LANGUAGE RULE: This text MUST be in the SAME LANGUAGE as the input story. 
   - If the story is in French, the narration text MUST be in French. 
   - If the story is in English, the narration text MUST be in English.
   - DO NOT translate the story to English for the narration text.
3. Main emotion and secondary emotion for the characters.
4. A dynamic "visualPrompt" for an image generator. This prompt should describe a creative, minimalist stick-figure composition. 
   - It should specify the position of characters, their actions, and any minimal environmental elements (e.g., a single tree, a simple desk, a mountain line).
   - It must strictly follow the "minimalist black and white stick figure on white background" style.
   - It should NOT include any text or words.
   - Focus on body language and composition to convey the story.

Story: ${story}

Return the result as a JSON array of objects.`;

  const response = await genAI.models.generateContent({
    model,
    contents: [{ role: 'user', parts: [{ text: prompt }] }],
    config: {
      temperature: 0.9,
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
            text: { type: Type.STRING },
            visualPrompt: { type: Type.STRING }
          },
          required: ["scene", "description", "mainEmotion", "secondaryEmotion", "text", "visualPrompt"]
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

${scene.visualPrompt}

CRITICAL RULES:
- ABSOLUTELY NO text, words, letters, labels, or signatures in the image.
- NO speech bubbles, NO thought bubbles.
- The image should be pure visual art with stick figures only.
- DO NOT write the emotions (like "Fear", "Sadness", etc.) as text in the image.
- DO NOT write the scene description as text in the image.
- The image must be 100% free of any written characters.
- Simple thin lines, no shading, high contrast.
- The background MUST be pure white.`;

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

// --- Toast Component ---

interface ToastMessage {
  id: number;
  text: string;
}

const Toast = ({ toasts, onDismiss }: { toasts: ToastMessage[]; onDismiss: (id: number) => void }) => (
  <div className="fixed bottom-6 right-6 z-50 flex flex-col gap-2">
    <AnimatePresence>
      {toasts.map(toast => (
        <motion.div
          key={toast.id}
          initial={{ opacity: 0, y: 20, scale: 0.95 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: -10, scale: 0.95 }}
          className="flex items-center gap-3 px-4 py-3 bg-black text-white text-sm font-medium tracking-wide shadow-lg"
        >
          <CheckCircle2 className="w-4 h-4 shrink-0" />
          <span>{toast.text}</span>
          <button onClick={() => onDismiss(toast.id)} className="ml-2 hover:opacity-70">
            <X className="w-3 h-3" />
          </button>
        </motion.div>
      ))}
    </AnimatePresence>
  </div>
);

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

// --- Translations ---

const translations = {
  en: {
    title: "Stick Story AI",
    subtitle: "AI Content Creator",
    description: "Turn your stories into viral visual content with minimalist stick-figure animations.",
    inputLabel: "Enter your story or transcript",
    uploadBtn: "Upload SRT / Text",
    placeholder: "Paste your story or upload an SRT file to generate visuals...",
    clear: "Clear",
    copyText: "Copy Text",
    selectFormat: "Select Format",
    tiktok: "TikTok / Shorts (9:16)",
    youtube: "YouTube (16:9)",
    generateBtn: "Generate Visual Story",
    analyzing: "Analyzing your story...",
    drawing: (n: number) => `Drawing ${n} scenes in parallel...`,
    drawingScene: "Drawing Scene...",
    sceneCounter: (current: number, total: number) => `Scene ${current} / ${total}`,
    previewVideo: "Preview Video",
    downloadAll: "Download All",
    startOver: "Start Over",
    storyboard: "Storyboard",
    scenes: "Scenes",
    copyAllText: "Copy All Text",
    allTextCopied: "All scene text copied!",
    storyCopied: "Story copied to clipboard!",
    error: "Something went wrong during generation. Please try again.",
    footer: "Built with Gemini",
    downloadScene: "Download Scene",
    wordCount: (w: number, c: number) => `${w} words · ${c} chars`,
    dropzone: "Drop your file here",
    dropzoneHint: "SRT or TXT files accepted",
    tryExample: "Try an example",
    exampleStories: [
      { label: "The Lost Key", text: "A young girl finds a mysterious golden key in her grandmother's attic. She searches every room looking for the lock it opens. After days of searching, she discovers a tiny hidden door behind the bookshelf. Inside, she finds a box of letters her grandmother wrote to her, full of love and life advice. She sits by the window reading them, tears of joy streaming down her face." },
      { label: "The Robot Friend", text: "In a world where everyone has a robot companion, a lonely boy's robot breaks down. He carries it to the repair shop but can't afford the fix. He learns to repair it himself, reading manuals late at night. When the robot finally powers on, it says 'Thank you for not giving up on me.' They walk home together under the stars." }
    ] as { label: string; text: string }[],
    generatingProgress: (done: number, total: number) => `${done} / ${total} scenes ready`,
    emotion: "Emotion"
  },
  fr: {
    title: "Stick Story AI",
    subtitle: "Créateur de Contenu IA",
    description: "Transformez vos histoires en contenu viral avec des animations minimalistes de bonshommes allumettes.",
    inputLabel: "Entrez votre histoire ou transcription",
    uploadBtn: "Charger SRT / Texte",
    placeholder: "Collez votre histoire ou chargez un fichier SRT pour générer des visuels...",
    clear: "Effacer",
    copyText: "Copier le texte",
    selectFormat: "Choisir le Format",
    tiktok: "TikTok / Shorts (9:16)",
    youtube: "YouTube (16:9)",
    generateBtn: "Générer l'Histoire Visuelle",
    analyzing: "Analyse de votre histoire...",
    drawing: (n: number) => `Dessin de ${n} scènes en parallèle...`,
    drawingScene: "Dessin de la scène...",
    sceneCounter: (current: number, total: number) => `Scène ${current} / ${total}`,
    previewVideo: "Aperçu Vidéo",
    downloadAll: "Tout Télécharger",
    startOver: "Recommencer",
    storyboard: "Storyboard",
    scenes: "Scènes",
    copyAllText: "Copier tout le texte",
    allTextCopied: "Texte de toutes les scènes copié !",
    storyCopied: "Histoire copiée dans le presse-papier !",
    error: "Une erreur est survenue lors de la génération. Veuillez réessayer.",
    footer: "Propulsé par Gemini",
    downloadScene: "Télécharger la scène",
    wordCount: (w: number, c: number) => `${w} mots · ${c} caractères`,
    dropzone: "Déposez votre fichier ici",
    dropzoneHint: "Fichiers SRT ou TXT acceptés",
    tryExample: "Essayer un exemple",
    exampleStories: [
      { label: "La Clé Perdue", text: "Une jeune fille trouve une mystérieuse clé dorée dans le grenier de sa grand-mère. Elle fouille chaque pièce à la recherche de la serrure correspondante. Après des jours de recherche, elle découvre une petite porte cachée derrière la bibliothèque. À l'intérieur, elle trouve une boîte de lettres que sa grand-mère lui avait écrites, pleines d'amour et de conseils de vie. Elle s'assoit près de la fenêtre pour les lire, des larmes de joie coulant sur son visage." },
      { label: "L'Ami Robot", text: "Dans un monde où chacun possède un compagnon robot, le robot d'un garçon solitaire tombe en panne. Il le porte au réparateur mais n'a pas les moyens de payer. Il apprend à le réparer lui-même, lisant des manuels tard dans la nuit. Quand le robot se rallume enfin, il dit 'Merci de ne pas avoir abandonné.' Ils rentrent ensemble à la maison sous les étoiles." }
    ] as { label: string; text: string }[],
    generatingProgress: (done: number, total: number) => `${done} / ${total} scènes prêtes`,
    emotion: "Émotion"
  }
};

export default function App() {
  const [lang, setLang] = useState<'en' | 'fr'>('fr');
  const t = translations[lang];

  const [story, setStory] = useState('');
  const [scenes, setScenes] = useState<Scene[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState('');
  const [currentIndex, setCurrentIndex] = useState(0);
  const [videoMode, setVideoMode] = useState(false);
  const [format, setFormat] = useState<"9:16" | "16:9">("9:16");
  const [isDragging, setIsDragging] = useState(false);
  const [toasts, setToasts] = useState<ToastMessage[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // --- Toast helpers ---
  const showToast = useCallback((text: string) => {
    const id = Date.now();
    setToasts(prev => [...prev, { id, text }]);
    setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), 3000);
  }, []);
  const dismissToast = useCallback((id: number) => {
    setToasts(prev => prev.filter(t => t.id !== id));
  }, []);

  // --- Auto-resize textarea ---
  useEffect(() => {
    const el = textareaRef.current;
    if (el) {
      el.style.height = 'auto';
      el.style.height = Math.max(192, Math.min(el.scrollHeight, 500)) + 'px';
    }
  }, [story]);

  // --- Word & char count ---
  const wordCount = story.trim() ? story.trim().split(/\s+/).length : 0;
  const charCount = story.length;

  const parseSRT = (srtContent: string) => {
    return srtContent
      .replace(/\d+\r?\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}/g, '')
      .split(/\r?\n/)
      .map(line => line.trim())
      .filter(line => line !== "" && !/^\d+$/.test(line))
      .join(' ')
      .replace(/\s+/g, ' ')
      .trim();
  };

  const processFile = (file: File) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const content = e.target?.result as string;
      if (file.name.endsWith('.srt')) {
        setStory(parseSRT(content));
      } else {
        setStory(content);
      }
      if (fileInputRef.current) fileInputRef.current.value = '';
      showToast(`${file.name} loaded`);
    };
    reader.readAsText(file);
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) processFile(file);
  };

  // --- Drag & Drop ---
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };
  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file && (file.name.endsWith('.srt') || file.name.endsWith('.txt') || file.type.startsWith('text/'))) {
      processFile(file);
    }
  };

  const clearAll = () => {
    setStory("");
    setScenes([]);
    setCurrentIndex(0);
  };

  const copyToClipboard = () => {
    navigator.clipboard.writeText(story);
    showToast(t.storyCopied);
  };

  const handleGenerate = async () => {
    if (!story.trim()) return;

    setIsGenerating(true);
    setLoadingMessage(t.analyzing);

    try {
      const generatedScenes = await storyToScenes(story);
      if (generatedScenes.length === 0) {
        throw new Error("No scenes were generated.");
      }
      setScenes(generatedScenes);
      setCurrentIndex(0);

      setLoadingMessage(t.drawing(generatedScenes.length));

      for (let i = 0; i < generatedScenes.length; i++) {
        const scene = generatedScenes[i];
        let retries = 3;
        let success = false;

        while (retries > 0 && !success) {
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
            setLoadingMessage(t.generatingProgress(i + 1, generatedScenes.length));
            success = true;
          } catch (err) {
            console.error(`Failed to generate image for scene ${i + 1} (Attempt ${4 - retries})`, err);
            retries--;
            if (retries > 0) {
              const waitTime = err instanceof Error && err.message.includes('429') ? 3000 : 1000;
              await new Promise(resolve => setTimeout(resolve, waitTime));
            }
          }
        }

        if (i < generatedScenes.length - 1) {
          await new Promise(resolve => setTimeout(resolve, 500));
        }
      }

    } catch (error) {
      console.error("Generation failed", error);
      showToast(t.error);
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

  // --- Progress for output ---
  const completedScenes = scenes.filter(s => s.imageUrl).length;

  return (
    <div className="min-h-screen bg-white text-black font-sans selection:bg-black selection:text-white">
      <Toast toasts={toasts} onDismiss={dismissToast} />

      <AnimatePresence>
        {isGenerating && <LoadingOverlay message={loadingMessage} />}
      </AnimatePresence>

      {/* Header */}
      <header className="max-w-4xl mx-auto px-6 py-12 text-center relative">
        <div className="absolute top-4 right-6 flex gap-2">
          <button
            onClick={() => setLang('en')}
            className={`text-[10px] font-bold uppercase tracking-widest px-2 py-1 border border-black transition-colors ${lang === 'en' ? 'bg-black text-white' : 'bg-white text-black hover:bg-black/5'}`}
          >
            EN
          </button>
          <button
            onClick={() => setLang('fr')}
            className={`text-[10px] font-bold uppercase tracking-widest px-2 py-1 border border-black transition-colors ${lang === 'fr' ? 'bg-black text-white' : 'bg-white text-black hover:bg-black/5'}`}
          >
            FR
          </button>
        </div>
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-black/10 text-xs font-semibold uppercase tracking-widest mb-6">
            <Sparkles className="w-3 h-3" />
            {t.subtitle}
          </div>
          <h1 className="text-5xl md:text-7xl font-bold tracking-tighter mb-4">
            {t.title}
          </h1>
          <p className="text-xl text-black/60 max-w-2xl mx-auto leading-relaxed">
            {t.description}
          </p>
        </motion.div>
      </header>

      <main className="max-w-4xl mx-auto px-6 pb-24">
        {scenes.length === 0 ? (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-white border-2 border-black p-8 md:p-12 shadow-[8px_8px_0px_0px_rgba(0,0,0,1)]"
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            {/* Drag overlay */}
            <AnimatePresence>
              {isDragging && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="absolute inset-0 z-20 bg-white/95 border-4 border-dashed border-black flex flex-col items-center justify-center gap-3"
                >
                  <FileText className="w-12 h-12 text-black" />
                  <p className="text-lg font-bold uppercase tracking-widest">{t.dropzone}</p>
                  <p className="text-xs text-black/50 uppercase tracking-widest">{t.dropzoneHint}</p>
                </motion.div>
              )}
            </AnimatePresence>

            <div className="flex items-center justify-between mb-4">
              <label className="block text-sm font-bold uppercase tracking-wider">
                {t.inputLabel}
              </label>
              <label className="cursor-pointer flex items-center gap-2 text-xs font-bold uppercase tracking-widest hover:text-black transition-colors group">
                <Upload className="w-4 h-4 group-hover:scale-110 transition-transform" />
                {t.uploadBtn}
                <input
                  type="file"
                  accept=".srt,.txt"
                  className="hidden"
                  onChange={handleFileUpload}
                  ref={fileInputRef}
                />
              </label>
            </div>
            <div className="relative">
              <textarea
                ref={textareaRef}
                value={story}
                onChange={(e) => setStory(e.target.value)}
                placeholder={t.placeholder}
                className="w-full min-h-[192px] max-h-[500px] p-4 text-lg border-2 border-black focus:outline-none focus:ring-0 resize-none mb-1 placeholder:text-black/20 transition-[height] duration-150"
              />
              {/* Word / char count */}
              <div className="flex items-center justify-between mb-4">
                <span className="text-[10px] font-bold uppercase tracking-widest text-black/30">
                  {t.wordCount(wordCount, charCount)}
                </span>
                {story && (
                  <div className="flex gap-3">
                    <button
                      onClick={clearAll}
                      className="flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-widest hover:text-red-500 transition-colors"
                    >
                      <X className="w-3 h-3" />
                      {t.clear}
                    </button>
                    <button
                      onClick={copyToClipboard}
                      className="flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-widest hover:text-blue-500 transition-colors"
                    >
                      <Send className="w-3 h-3" />
                      {t.copyText}
                    </button>
                  </div>
                )}
              </div>
            </div>

            {/* Example stories */}
            {!story && (
              <div className="mb-6">
                <p className="text-[10px] font-bold uppercase tracking-widest text-black/40 mb-3 flex items-center gap-1.5">
                  <Zap className="w-3 h-3" />
                  {t.tryExample}
                </p>
                <div className="flex flex-wrap gap-2">
                  {t.exampleStories.map((example, i) => (
                    <button
                      key={i}
                      onClick={() => setStory(example.text)}
                      className="px-3 py-1.5 border border-black/20 text-xs font-bold uppercase tracking-widest hover:border-black hover:bg-black hover:text-white transition-all"
                    >
                      {example.label}
                    </button>
                  ))}
                </div>
              </div>
            )}

            <div className="mb-8">
              <label className="block text-sm font-bold uppercase tracking-wider mb-4">
                {t.selectFormat}
              </label>
              <div className="grid grid-cols-2 gap-4">
                <button
                  onClick={() => setFormat("9:16")}
                  className={`py-3 border-2 border-black font-bold uppercase tracking-widest transition-all flex items-center justify-center gap-2 ${format === "9:16" ? 'bg-black text-white' : 'bg-white text-black hover:bg-black/5'}`}
                >
                  <div className="w-3 h-5 border-2 border-current rounded-sm" />
                  {t.tiktok}
                </button>
                <button
                  onClick={() => setFormat("16:9")}
                  className={`py-3 border-2 border-black font-bold uppercase tracking-widest transition-all flex items-center justify-center gap-2 ${format === "16:9" ? 'bg-black text-white' : 'bg-white text-black hover:bg-black/5'}`}
                >
                  <div className="w-5 h-3 border-2 border-current rounded-sm" />
                  {t.youtube}
                </button>
              </div>
            </div>

            <button
              onClick={handleGenerate}
              disabled={!story.trim() || isGenerating}
              className="w-full py-4 bg-black text-white text-lg font-bold uppercase tracking-widest hover:bg-black/90 transition-colors flex items-center justify-center gap-3 disabled:opacity-50"
            >
              {isGenerating ? <Loader2 className="animate-spin" /> : <Send className="w-5 h-5" />}
              {t.generateBtn}
            </button>
          </motion.div>
        ) : (
          <div className="space-y-12">
            {/* Progress bar during generation */}
            {scenes.length > 0 && completedScenes < scenes.length && (
              <div className="space-y-2">
                <div className="flex items-center justify-between text-[10px] font-bold uppercase tracking-widest text-black/50">
                  <span>{t.generatingProgress(completedScenes, scenes.length)}</span>
                  <span>{Math.round((completedScenes / scenes.length) * 100)}%</span>
                </div>
                <div className="w-full h-2 bg-black/10 overflow-hidden">
                  <motion.div
                    className="h-full bg-black"
                    initial={{ width: 0 }}
                    animate={{ width: `${(completedScenes / scenes.length) * 100}%` }}
                    transition={{ duration: 0.5, ease: "easeOut" }}
                  />
                </div>
              </div>
            )}

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
                        <p className="text-xs font-bold uppercase tracking-widest text-black/40">{t.drawingScene}</p>
                      </div>
                    )}
                  </motion.div>
                </AnimatePresence>

                {/* Navigation Overlays */}
                {!videoMode && (
                  <>
                    <button
                      onClick={() => setCurrentIndex(prev => Math.max(0, prev - 1))}
                      disabled={currentIndex === 0}
                      className="absolute left-4 top-1/2 -translate-y-1/2 w-10 h-10 bg-white border-2 border-black flex items-center justify-center hover:bg-black hover:text-white transition-colors z-10 disabled:opacity-30 disabled:hover:bg-white disabled:hover:text-black"
                    >
                      <ChevronLeft className="w-6 h-6" />
                    </button>
                    <button
                      onClick={() => setCurrentIndex(prev => Math.min(scenes.length - 1, prev + 1))}
                      disabled={currentIndex === scenes.length - 1}
                      className="absolute right-4 top-1/2 -translate-y-1/2 w-10 h-10 bg-white border-2 border-black flex items-center justify-center hover:bg-black hover:text-white transition-colors z-10 disabled:opacity-30 disabled:hover:bg-white disabled:hover:text-black"
                    >
                      <ChevronRight className="w-6 h-6" />
                    </button>
                  </>
                )}

                {/* Scene Counter */}
                <div className="absolute bottom-6 left-1/2 -translate-x-1/2 px-4 py-1 bg-black text-white text-xs font-bold tracking-widest uppercase">
                  {t.sceneCounter(currentIndex + 1, scenes.length)}
                </div>
              </div>

              {/* Scene info below carousel */}
              <div className="max-w-[400px] mx-auto mt-4 text-center">
                <p className="text-sm font-medium text-black/70 italic">
                  "{scenes[currentIndex].text}"
                </p>
                <p className="text-[10px] font-bold uppercase tracking-widest text-black/30 mt-2">
                  {t.emotion}: {scenes[currentIndex].mainEmotion} / {scenes[currentIndex].secondaryEmotion}
                </p>
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
                {t.previewVideo}
              </button>
              <button
                onClick={downloadAll}
                disabled={scenes.some(s => !s.imageUrl)}
                className="px-8 py-3 bg-white border-2 border-black font-bold uppercase tracking-widest hover:bg-black hover:text-white transition-all flex items-center gap-2 disabled:opacity-50"
              >
                <Download className="w-4 h-4" />
                {t.downloadAll}
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
                {t.startOver}
              </button>
            </div>

            {/* Storyboard Grid */}
            <div className="mt-16">
              <h2 className="text-2xl font-black uppercase tracking-tighter mb-8 flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <span className="bg-black text-white px-3 py-1">{t.storyboard}</span>
                  <span className="text-black/30">{completedScenes}/{scenes.length} {t.scenes}</span>
                </div>
                <button
                  onClick={() => {
                    const allText = scenes.map((s, i) => `Scene ${i+1}: ${s.text}`).join('\n');
                    navigator.clipboard.writeText(allText);
                    showToast(t.allTextCopied);
                  }}
                  className="text-[10px] font-bold uppercase tracking-widest hover:text-black transition-colors flex items-center gap-2"
                >
                  <Send className="w-3 h-3" />
                  {t.copyAllText}
                </button>
              </h2>
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
                {scenes.map((scene, idx) => (
                  <motion.div
                    key={idx}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: idx * 0.05 }}
                    className="relative group"
                  >
                    <button
                      onClick={() => setCurrentIndex(idx)}
                      className={`text-left w-full transition-all ${currentIndex === idx ? 'scale-105' : 'opacity-60 hover:opacity-100'}`}
                    >
                      <div className={`${format === "9:16" ? "aspect-[9/16]" : "aspect-[16/9]"} border-2 border-black mb-3 overflow-hidden bg-gray-50 relative ${currentIndex === idx ? 'shadow-[4px_4px_0px_0px_rgba(0,0,0,1)]' : ''}`}>
                        {scene.imageUrl ? (
                          <img
                            src={scene.imageUrl}
                            alt={`Scene ${idx + 1}`}
                            className="w-full h-full object-cover"
                            referrerPolicy="no-referrer"
                          />
                        ) : (
                          <div className="w-full h-full flex flex-col items-center justify-center gap-2">
                            <Loader2 className="w-5 h-5 animate-spin text-black/20" />
                          </div>
                        )}
                        <div className="absolute top-2 left-2 bg-black text-white text-[10px] font-bold px-1.5 py-0.5 uppercase">
                          {idx + 1}
                        </div>
                        {scene.imageUrl && (
                          <div className="absolute top-2 right-8 text-[8px] font-bold uppercase tracking-widest text-black/40 bg-white/80 px-1 py-0.5">
                            {scene.mainEmotion}
                          </div>
                        )}
                      </div>
                      <p className="text-[10px] font-bold uppercase tracking-widest line-clamp-2 leading-tight">
                        {scene.text}
                      </p>
                    </button>
                    {scene.imageUrl && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          const link = document.createElement('a');
                          link.href = scene.imageUrl!;
                          link.download = `scene-${idx + 1}.png`;
                          link.click();
                        }}
                        className="absolute top-2 right-2 p-1.5 bg-white border border-black opacity-0 group-hover:opacity-100 transition-opacity hover:bg-black hover:text-white"
                        title={t.downloadScene}
                      >
                        <Download className="w-3 h-3" />
                      </button>
                    )}
                  </motion.div>
                ))}
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-black/5 py-12 text-center text-black/40 text-xs font-bold uppercase tracking-widest">
        &copy; 2026 {t.title} &bull; {t.footer}
      </footer>
    </div>
  );
}
