/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useRef, useEffect, useCallback } from 'react';
import { GoogleGenAI, Type, GenerateContentResponse } from "@google/genai";
import { motion, AnimatePresence } from "motion/react";
import JSZip from 'jszip';
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
  Zap,
  Clock,
  Archive,
  Palette
} from 'lucide-react';

// --- Types ---

interface SrtSegment {
  index: number;
  start: string; // "00:00:05,200"
  end: string;   // "00:00:08,400"
  text: string;
  startMs: number;
  endMs: number;
}

interface Scene {
  scene: number;
  description: string;
  mainEmotion: string;
  secondaryEmotion: string;
  text: string;
  visualPrompt: string;
  imageUrl?: string;
  prompt?: string;
  timing?: { start: string; end: string }; // CapCut timing
}

// --- SRT Utilities ---

function srtTimeToMs(time: string): number {
  const [h, m, rest] = time.split(':');
  const [s, ms] = rest.split(',');
  return parseInt(h) * 3600000 + parseInt(m) * 60000 + parseInt(s) * 1000 + parseInt(ms);
}

function msToCapCutTime(ms: number): string {
  const h = Math.floor(ms / 3600000);
  const m = Math.floor((ms % 3600000) / 60000);
  const s = Math.floor((ms % 60000) / 1000);
  const msPart = ms % 1000;
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')},${String(msPart).padStart(3, '0')}`;
}

function parseSRTWithTimestamps(srtContent: string): SrtSegment[] {
  const segments: SrtSegment[] = [];
  const blocks = srtContent.trim().split(/\r?\n\r?\n/);

  for (const block of blocks) {
    const lines = block.split(/\r?\n/).map(l => l.trim()).filter(Boolean);
    if (lines.length < 2) continue;

    // Find the timestamp line
    const tsLine = lines.find(l => /\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}/.test(l));
    if (!tsLine) continue;

    const match = tsLine.match(/(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})/);
    if (!match) continue;

    const tsIndex = lines.indexOf(tsLine);
    const textLines = lines.slice(tsIndex + 1);
    if (textLines.length === 0) continue;

    segments.push({
      index: segments.length + 1,
      start: match[1],
      end: match[2],
      text: textLines.join(' '),
      startMs: srtTimeToMs(match[1]),
      endMs: srtTimeToMs(match[2]),
    });
  }

  return segments;
}

function assignTimingsToScenes(scenes: Scene[], segments: SrtSegment[]): Scene[] {
  if (segments.length === 0) return scenes;

  const totalDuration = segments[segments.length - 1].endMs - segments[0].startMs;
  const globalStart = segments[0].startMs;
  const sceneDuration = totalDuration / scenes.length;

  return scenes.map((scene, i) => ({
    ...scene,
    timing: {
      start: msToCapCutTime(globalStart + i * sceneDuration),
      end: msToCapCutTime(globalStart + (i + 1) * sceneDuration),
    }
  }));
}

function generateCapCutTimingText(scenes: Scene[]): string {
  let output = '=== CAPCUT TIMING GUIDE ===\n';
  output += 'Import images in order, set each to the duration shown below.\n\n';

  scenes.forEach((scene, i) => {
    if (scene.timing) {
      output += `Scene ${i + 1}: ${scene.timing.start} --> ${scene.timing.end}\n`;
      output += `  File: scene-${String(i + 1).padStart(2, '0')}.png\n`;
      output += `  Text: ${scene.text}\n\n`;
    }
  });

  output += '=== HOW TO USE IN CAPCUT ===\n';
  output += '1. Import all scene images into CapCut\n';
  output += '2. Place each image on the timeline at the START time shown above\n';
  output += '3. Trim each image to match the END time\n';
  output += '4. Import your original audio/voiceover\n';
  output += '5. The images will sync perfectly with the narration!\n';

  return output;
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
   - Focus on EXAGGERATED body language, dynamic poses, and dramatic composition to convey emotion powerfully.
   - Stick figures should have expressive features: wide arms for surprise, hunched posture for sadness, jumping for joy, trembling lines for fear, etc.
   - IMPORTANT LAYOUT RULE: All visual elements (characters, objects, scenery) MUST be positioned in the UPPER 75% of the image. The BOTTOM 25% must remain completely empty white space (this area is reserved for text overlay).

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
  const prompt = `A bold, expressive minimalist black and white stick figure illustration, hand-drawn sketch style, clean white background.

${scene.visualPrompt}

STYLE & EXPRESSION:
- Stick figures must be HIGHLY EXPRESSIVE with exaggerated body language and dynamic poses.
- Use varied line thickness: thicker lines for emphasis, thinner for details.
- Add motion lines, sweat drops, impact stars, or emotion marks (like a broken heart, sparkles, swirls) to convey feelings.
- Characters should have simple but expressive faces (dots for eyes, curved lines for mouths showing clear emotions).
- Use scale and perspective creatively: a scared character can be tiny next to a large threat, a confident character can be bold and large.

LAYOUT:
- ALL drawings, characters, and visual elements MUST be in the UPPER 70-75% of the image.
- The BOTTOM 25-30% of the image MUST be completely empty pure white space. Draw NOTHING there.
- This bottom white space is critical — it will be used for text overlay.

CRITICAL RULES:
- ABSOLUTELY NO text, words, letters, labels, or signatures anywhere in the image.
- NO speech bubbles, NO thought bubbles with text.
- The image must be 100% free of any written characters.
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
 * Overlays text onto the BOTTOM of the generated image using a canvas.
 * Uses Permanent Marker font for a hand-drawn feel that matches stick figures.
 * Draws a white band at the bottom so text never overlaps the illustration.
 */
async function addTextToImage(base64: string, text: string): Promise<string> {
  return new Promise((resolve) => {
    const img = new Image();
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

      // 2. Configure text style — Permanent Marker for hand-drawn feel
      const padding = canvas.width * 0.06;
      const fontSize = Math.floor(canvas.width * 0.042);
      ctx.font = `${fontSize}px "Permanent Marker", "Marker Felt", "Comic Sans MS", cursive`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';

      // 3. Wrap text logic
      const maxWidth = canvas.width - (padding * 2);
      const words = text.split(' ');
      let line = '';
      const lines: string[] = [];

      for (let n = 0; n < words.length; n++) {
        const testLine = line + words[n] + ' ';
        const metrics = ctx.measureText(testLine);
        if (metrics.width > maxWidth && n > 0) {
          lines.push(line.trim());
          line = words[n] + ' ';
        } else {
          line = testLine;
        }
      }
      lines.push(line.trim());

      // 4. Calculate text zone at the BOTTOM
      const lineHeight = fontSize * 1.4;
      const textBlockHeight = lines.length * lineHeight;
      const bottomMargin = canvas.height * 0.03;
      const bandPadding = fontSize * 0.6;

      // 5. Draw white band behind text (solid white so no overlap)
      const bandTop = canvas.height - bottomMargin - textBlockHeight - bandPadding;
      ctx.fillStyle = 'white';
      ctx.fillRect(0, bandTop, canvas.width, canvas.height - bandTop);

      // 6. Optional subtle top border for the band
      ctx.strokeStyle = 'rgba(0,0,0,0.08)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padding, bandTop);
      ctx.lineTo(canvas.width - padding, bandTop);
      ctx.stroke();

      // 7. Draw text lines from bottom up
      ctx.fillStyle = '#111111';
      const textStartY = canvas.height - bottomMargin;
      for (let i = lines.length - 1; i >= 0; i--) {
        const y = textStartY - ((lines.length - 1 - i) * lineHeight);
        ctx.fillText(lines[i], canvas.width / 2, y);
      }

      resolve(canvas.toDataURL('image/png'));
    };
    img.onerror = () => {
      clearTimeout(timeout);
      resolve(base64);
    };
    img.src = base64;
  });
}

// --- Thumbnail Generation ---

/**
 * Generate a short, punchy clickbait title for the thumbnail
 * that is directly tied to the story's core message/hook.
 */
async function generateThumbnailTitle(story: string): Promise<string> {
  const model = "gemini-3-flash-preview";
  const prompt = `You are a YouTube content strategist. Your job is to create a thumbnail title that makes people CLICK.

STORY TO ANALYZE:
${story.substring(0, 500)}

STEP 1: Identify the MAIN TOPIC or the most SHOCKING/SURPRISING element of this story.
STEP 2: Turn it into a short clickbait title.

STRICT RULES:
- The title MUST directly reference the main subject of the story (e.g., if the story is about health insurance abroad, mention health/insurance/expat).
- Maximum 4-6 words. Every word must add value.
- Must be in the EXACT SAME LANGUAGE as the story (if French → French title, if English → English title).
- ALL CAPS.
- Use ONE of these clickbait formulas:
  * WARNING + specific topic (e.g., "ATTENTION À VOTRE SANTÉ !")
  * Surprising fact (e.g., "70% DES EXPATS IGNORENT ÇA")
  * Challenge/question (e.g., "JAMAIS PARTIR SANS ÇA !")
  * Hidden truth (e.g., "LA VÉRITÉ SUR LA CFE")
- The title should make someone curious enough to click.
- NO generic phrases like "YOU WON'T BELIEVE" without context.
- The title MUST make sense on its own — a stranger reading it should understand the topic.

Return ONLY the title, nothing else.`;

  const response = await genAI.models.generateContent({
    model,
    contents: [{ role: 'user', parts: [{ text: prompt }] }],
    config: { temperature: 0.8 }
  });

  return (response.text || '').trim().replace(/^["']|["']$/g, '');
}

/**
 * Generate a colorful YouTube thumbnail with the clickbait title
 * rendered directly by Gemini for a natural, integrated look.
 */
async function generateThumbnail(story: string, scenes: Scene[], title: string): Promise<string> {
  const model = "gemini-2.5-flash-image";

  const keyScene = scenes[Math.floor(scenes.length / 2)];

  // Extract key visual elements from the story for the thumbnail
  const storySnippet = story.substring(0, 300);

  const prompt = `Generate a YouTube thumbnail in a CARTOON ILLUSTRATION style (NOT stick figures). Study this reference description carefully:

REFERENCE STYLE (like popular YouTube explainer channels):
- A colorful CARTOON CHARACTER on the left side (about 40% of image), drawn in a semi-realistic cartoon/illustration style with:
  * Full color clothing and details (shirt, hat, accessories relevant to the topic)
  * Exaggerated facial expression: big eyes, open mouth, pointing gesture
  * The character should look like a cartoon version of a real person, NOT a stick figure
- A DETAILED ILLUSTRATED OBJECT or scene in the center/right that represents the story's main topic, drawn in full color cartoon style with highlights, shadows, and details.
- Light/white/soft gradient background with subtle effects (sparkles, light rays, or small particles).
- The overall image should look like a professional digital illustration, colorful and polished.

STORY: ${storySnippet}
KEY EMOTION: ${keyScene.mainEmotion}

WHAT TO DRAW:
- A cartoon character reacting to the story's main topic (e.g., pointing, shocked, excited).
- The main subject/object of the story illustrated in detail next to or behind the character.
- Use colors that match the story topic: medical=blue/white/red, money=green/gold, travel=blue/orange, danger=red/black.

TEXT "${title}" (MANDATORY):
- Write "${title}" in HUGE bold block letters.
- Position: upper right area of the image, taking about 35-40% of the width.
- Style: 3D-looking block text with strong drop shadow.
- Colors: use 2 contrasting colors for the text (e.g., yellow text + red shadow, white text + blue shadow, red text + black shadow). Pick colors that POP against the background.
- Each word on a separate line for maximum size.
- The text must be the FIRST thing people notice.

COMPOSITION:
- Character on the LEFT pointing at or reacting to the main object.
- Main illustrated object in the CENTER.
- Big bold text "${title}" in the UPPER RIGHT.
- Clean, uncluttered — maximum 3 visual elements total.
- Must be readable as a tiny thumbnail on a phone screen.

CRITICAL: The text "${title}" MUST appear in the image. The illustration must directly relate to the story content.`;

  const response = await genAI.models.generateContent({
    model,
    contents: [{ role: 'user', parts: [{ text: prompt }] }],
    config: {
      imageConfig: {
        aspectRatio: '16:9' as any,
      }
    }
  });

  const part = response.candidates?.[0]?.content?.parts.find(p => p.inlineData);
  if (part?.inlineData?.data) {
    return `data:image/png;base64,${part.inlineData.data}`;
  }

  throw new Error("No thumbnail image data returned");
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
    emotion: "Emotion",
    downloadZip: "Download ZIP",
    capCutTiming: "CapCut Timing",
    srtDetected: "SRT timings detected — CapCut timing will be included in the ZIP",
    copyTiming: "Copy Timing",
    timingCopied: "CapCut timing copied to clipboard!",
    creatingZip: "Creating ZIP...",
    generatingThumbnail: "Creating YouTube thumbnail...",
    thumbnail: "YouTube Thumbnail",
    thumbnailDesc: "Colorful thumbnail optimized for click-through rate",
    downloadThumbnail: "Download Thumbnail",
    thumbnailIncluded: "Thumbnail included in ZIP"
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
    emotion: "Émotion",
    downloadZip: "Télécharger ZIP",
    capCutTiming: "Timing CapCut",
    srtDetected: "Timings SRT détectés — le timing CapCut sera inclus dans le ZIP",
    copyTiming: "Copier le timing",
    timingCopied: "Timing CapCut copié dans le presse-papier !",
    creatingZip: "Création du ZIP...",
    generatingThumbnail: "Création de la miniature YouTube...",
    thumbnail: "Miniature YouTube",
    thumbnailDesc: "Miniature colorée optimisée pour le taux de clic",
    downloadThumbnail: "Télécharger la miniature",
    thumbnailIncluded: "Miniature incluse dans le ZIP"
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
  const [srtSegments, setSrtSegments] = useState<SrtSegment[]>([]);
  const [isZipping, setIsZipping] = useState(false);
  const [thumbnailUrl, setThumbnailUrl] = useState<string | null>(null);
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

  const processFile = (file: File) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const content = e.target?.result as string;
      if (file.name.endsWith('.srt')) {
        // Parse with timestamps for CapCut timing
        const segments = parseSRTWithTimestamps(content);
        setSrtSegments(segments);
        // Extract plain text for the story
        const plainText = segments.map(s => s.text).join(' ');
        setStory(plainText);
        if (segments.length > 0) {
          showToast(t.srtDetected);
        }
      } else {
        setSrtSegments([]);
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
    setSrtSegments([]);
    setThumbnailUrl(null);
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
      // Assign SRT timings to scenes if available
      const scenesWithTiming = srtSegments.length > 0
        ? assignTimingsToScenes(generatedScenes, srtSegments)
        : generatedScenes;

      setScenes(scenesWithTiming);
      setCurrentIndex(0);

      setLoadingMessage(t.drawing(scenesWithTiming.length));

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

      // Generate YouTube thumbnail after all scenes
      setLoadingMessage(t.generatingThumbnail);
      try {
        // Generate title first, then pass it to thumbnail generation
        const thumbTitle = await generateThumbnailTitle(story);
        const thumbUrl = await generateThumbnail(story, scenesWithTiming, thumbTitle);
        setThumbnailUrl(thumbUrl);
      } catch (err) {
        console.error("Thumbnail generation failed", err);
        // Non-blocking — scenes are still usable without thumbnail
      }

    } catch (error) {
      console.error("Generation failed", error);
      showToast(t.error);
    } finally {
      setIsGenerating(false);
    }
  };

  const downloadAllAsZip = async () => {
    setIsZipping(true);
    try {
      const zip = new JSZip();
      const imgFolder = zip.folder('scenes')!;

      // Add images
      for (let i = 0; i < scenes.length; i++) {
        const scene = scenes[i];
        if (scene.imageUrl) {
          // Convert data URL to binary
          const base64Data = scene.imageUrl.split(',')[1];
          imgFolder.file(`scene-${String(i + 1).padStart(2, '0')}.png`, base64Data, { base64: true });
        }
      }

      // Add YouTube thumbnail if available
      if (thumbnailUrl) {
        const thumbBase64 = thumbnailUrl.split(',')[1];
        zip.file('youtube-thumbnail.png', thumbBase64, { base64: true });
      }

      // Add CapCut timing file if SRT was used
      if (scenes.some(s => s.timing)) {
        const timingText = generateCapCutTimingText(scenes);
        zip.file('capcut-timing.txt', timingText);
      }

      // Generate and download ZIP
      const blob = await zip.generateAsync({ type: 'blob' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = 'stick-story-scenes.zip';
      link.click();
      URL.revokeObjectURL(url);
      showToast('ZIP downloaded!');
    } catch (err) {
      console.error('ZIP creation failed', err);
      showToast(t.error);
    } finally {
      setIsZipping(false);
    }
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

            {/* SRT timing badge */}
            {srtSegments.length > 0 && (
              <div className="flex items-center gap-2 mb-4 px-3 py-2 bg-black/5 border border-black/10">
                <Clock className="w-3.5 h-3.5 text-black/50" />
                <span className="text-[10px] font-bold uppercase tracking-widest text-black/50">
                  {t.capCutTiming}: {srtSegments[0].start.substring(0, 8)} → {srtSegments[srtSegments.length - 1].end.substring(0, 8)} ({srtSegments.length} segments)
                </span>
              </div>
            )}

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
                {scenes[currentIndex].timing && (
                  <p className="text-[10px] font-bold uppercase tracking-widest text-black/40 mt-1 flex items-center justify-center gap-1.5">
                    <Clock className="w-3 h-3" />
                    {scenes[currentIndex].timing!.start} → {scenes[currentIndex].timing!.end}
                  </p>
                )}
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
                onClick={downloadAllAsZip}
                disabled={scenes.some(s => !s.imageUrl) || isZipping}
                className="px-8 py-3 bg-white border-2 border-black font-bold uppercase tracking-widest hover:bg-black hover:text-white transition-all flex items-center gap-2 disabled:opacity-50"
              >
                {isZipping ? <Loader2 className="w-4 h-4 animate-spin" /> : <Archive className="w-4 h-4" />}
                {isZipping ? t.creatingZip : t.downloadZip}
              </button>
              <button
                onClick={() => {
                  setScenes([]);
                  setStory('');
                  setSrtSegments([]);
                  setThumbnailUrl(null);
                  setCurrentIndex(0);
                }}
                className="px-8 py-3 bg-white border-2 border-black font-bold uppercase tracking-widest hover:bg-black hover:text-white transition-all flex items-center gap-2"
              >
                <RefreshCw className="w-4 h-4" />
                {t.startOver}
              </button>
            </div>

            {/* YouTube Thumbnail */}
            {(thumbnailUrl || completedScenes === scenes.length) && (
              <div className="mt-12">
                <h2 className="text-2xl font-black uppercase tracking-tighter mb-6 flex items-center gap-4">
                  <span className="bg-gradient-to-r from-red-500 to-orange-500 text-white px-3 py-1 flex items-center gap-2">
                    <Palette className="w-5 h-5" />
                    {t.thumbnail}
                  </span>
                </h2>
                <div className="max-w-2xl mx-auto">
                  <div className="aspect-video border-4 border-black shadow-[8px_8px_0px_0px_rgba(0,0,0,1)] overflow-hidden bg-gray-50 relative">
                    {thumbnailUrl ? (
                      <img
                        src={thumbnailUrl}
                        alt="YouTube Thumbnail"
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <div className="w-full h-full flex flex-col items-center justify-center gap-3">
                        <Loader2 className="w-8 h-8 animate-spin text-black/20" />
                        <p className="text-xs font-bold uppercase tracking-widest text-black/40">{t.generatingThumbnail}</p>
                      </div>
                    )}
                  </div>
                  <div className="flex items-center justify-between mt-4">
                    <p className="text-[10px] font-bold uppercase tracking-widest text-black/40">
                      {t.thumbnailDesc} — 1280×720 (16:9)
                    </p>
                    {thumbnailUrl && (
                      <button
                        onClick={() => {
                          const link = document.createElement('a');
                          link.href = thumbnailUrl;
                          link.download = 'youtube-thumbnail.png';
                          link.click();
                        }}
                        className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-widest hover:text-black transition-colors"
                      >
                        <Download className="w-3 h-3" />
                        {t.downloadThumbnail}
                      </button>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* Storyboard Grid */}
            <div className="mt-16">
              <h2 className="text-2xl font-black uppercase tracking-tighter mb-8 flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <span className="bg-black text-white px-3 py-1">{t.storyboard}</span>
                  <span className="text-black/30">{completedScenes}/{scenes.length} {t.scenes}</span>
                </div>
                <div className="flex items-center gap-4">
                  {scenes.some(s => s.timing) && (
                    <button
                      onClick={() => {
                        const timingText = generateCapCutTimingText(scenes);
                        navigator.clipboard.writeText(timingText);
                        showToast(t.timingCopied);
                      }}
                      className="text-[10px] font-bold uppercase tracking-widest hover:text-black transition-colors flex items-center gap-2"
                    >
                      <Clock className="w-3 h-3" />
                      {t.copyTiming}
                    </button>
                  )}
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
                </div>
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
                      {scene.timing && (
                        <p className="text-[8px] font-mono text-black/35 mt-1 flex items-center gap-1">
                          <Clock className="w-2.5 h-2.5" />
                          {scene.timing.start.substring(0, 8)} → {scene.timing.end.substring(0, 8)}
                        </p>
                      )}
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
