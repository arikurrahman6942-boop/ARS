import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI, Type } from "@google/genai";
import dotenv from "dotenv";

dotenv.config();

const app = express();
const PORT = 3000;

// Configure body parsing with size limits for base64 images
app.use(express.json({ limit: "20mb" }));
app.use(express.urlencoded({ limit: "20mb", extended: true }));

// Initialize Gemini client on the server
const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY,
  httpOptions: {
    headers: {
      "User-Agent": "aistudio-build",
    },
  },
});

// JSON Schema for IELTS band evaluation response
const evaluationSchema = {
  type: Type.OBJECT,
  properties: {
    task_achievement: {
      type: Type.OBJECT,
      properties: {
        band: { type: Type.NUMBER, description: "Official IELTS band score (e.g. 6.5 or 7.0) based on task response/achievement." },
        justification: { type: Type.STRING, description: "Detailed justification for the band score." }
      },
      required: ["band", "justification"]
    },
    coherence_cohesion: {
      type: Type.OBJECT,
      properties: {
        band: { type: Type.NUMBER, description: "Official IELTS band score (e.g. 6.5 or 7.0) based on cohesion/coherence." },
        justification: { type: Type.STRING, description: "Detailed justification for the band score." }
      },
      required: ["band", "justification"]
    },
    lexical_resource: {
      type: Type.OBJECT,
      properties: {
        band: { type: Type.NUMBER, description: "Official IELTS band score (e.g. 6.5 or 7.0) based on vocabulary." },
        justification: { type: Type.STRING, description: "Detailed justification for the band score." }
      },
      required: ["band", "justification"]
    },
    grammatical_range_accuracy: {
      type: Type.OBJECT,
      properties: {
        band: { type: Type.NUMBER, description: "Official IELTS band score (e.g. 6.5 or 7.0) based on grammar." },
        justification: { type: Type.STRING, description: "Detailed justification for the band score." }
      },
      required: ["band", "justification"]
    },
    overall_band: { type: Type.NUMBER, description: "Overall score calculated (usually average of the 4 criteria, rounded to nearest 0.5)." },
    overall_summary: { type: Type.STRING, description: "General summary overview of candidate performance." },
    line_by_line_feedback: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: {
          original_sentence: { type: Type.STRING, description: "The sentence from the candidate's paper that has issues." },
          issue: { type: Type.STRING, description: "Description of the grammar, lexical, coherence, or achievement issue." },
          corrected_sentence: { type: Type.STRING, description: "Corrected version of the sentence." },
          issue_type: { type: Type.STRING, description: "Category of error: Grammatical Range & Accuracy, Lexical Resource, Coherence & Cohesion, or Task Achievement." }
        },
        required: ["original_sentence", "issue", "corrected_sentence", "issue_type"]
      }
    },
    grammar_errors_summary: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: {
          error: { type: Type.STRING, description: "Brief description of the error." },
          explanation: { type: Type.STRING, description: "Grammatical explanation of the mistake." },
          correction: { type: Type.STRING, description: "How to correct this error." }
        },
        required: ["error", "explanation", "correction"]
      }
    },
    strengths: {
      type: Type.ARRAY,
      items: { type: Type.STRING }
    },
    improvement_areas: {
      type: Type.ARRAY,
      items: { type: Type.STRING }
    }
  },
  required: [
    "task_achievement",
    "coherence_cohesion",
    "lexical_resource",
    "grammatical_range_accuracy",
    "overall_band",
    "overall_summary",
    "line_by_line_feedback",
    "grammar_errors_summary",
    "strengths",
    "improvement_areas"
  ]
};

// API Endpoint for essay evaluation with multi-tier retries and fallback models
app.post("/api/evaluate", async (req, res) => {
  try {
    const { activeTask, response, context, question, image } = req.body;

    if (!response || typeof response !== "string" || response.trim().length === 0) {
      return res.status(400).json({ error: "Response text is required." });
    }

    if (activeTask === "2" && (!question || typeof question !== "string" || question.trim().length === 0)) {
      return res.status(400).json({ error: "Question text is required for Task 2." });
    }

    const systemInstruction = `You are a certified IELTS examiner with 15+ years of experience. Evaluate the provided IELTS writing response strictly according to the official IELTS band descriptors. Be precise, honest, and academically rigorous. Never inflate scores. You must respond with a JSON object conforming strictly to the requested schema.`;

    let userPromptString = "";
    const parts: any[] = [];

    if (activeTask === "1") {
      const additionalContext = context ? context.trim() : "None provided.";
      userPromptString = `Task Type: IELTS Writing Task 1 (Academic Writing)\n` +
        `Additional Context/Instructions: ${additionalContext}\n` +
        `Candidate's Response: ${response.trim()}\n\n`;

      if (image && image.base64 && image.mediaType) {
        userPromptString += `Analyze the task response referencing the provided diagram/image and identify if it accurately describes the visuals displayed in the image.`;
        parts.push({
          inlineData: {
            mimeType: image.mediaType,
            data: image.base64
          }
        });
      }
    } else {
      userPromptString = `Task Type: IELTS Writing Task 2 (Essay)\n` +
        `Essay Question: ${question.trim()}\n` +
        `Candidate's Response: ${response.trim()}\n\n`;
    }

    userPromptString += `Evaluate this response according to IELTS band descriptors (Task Achievement, Coherence & Cohesion, Lexical Resource, and Grammatical Range & Accuracy). Make sure that overall_band meets standard IELTS rules (the mathematically rounded value based on the averaged band scores of the grid components). Provide individual sentence feedback for problematic, unnatural, or grammatically incorrect expressions in the response, and map them to their corresponding category (such as Grammatical Range & Accuracy, Lexical Resource, Coherence & Cohesion, or Task Achievement). Finally, extract a list of main grammar and language error rules noticed in the text, and give the strengths and improvements list.`;

    parts.push({ text: userPromptString });

    // Multi-tier models fallback strategy to combat high demand / 503 anomalies
    const modelsToTry = [
      "gemini-3.5-flash",
      "gemini-3.1-flash-lite",
      "gemini-flash-latest"
    ];

    let lastError: any = null;
    let finalParsedData: any = null;

    for (const modelName of modelsToTry) {
      console.log(`[IELTS EVALUATOR] Attempting evaluation using model: ${modelName}`);
      let success = false;
      
      // Up to 3 attempts (initial + 2 retries) with backoff per model tier
      for (let attempt = 1; attempt <= 3; attempt++) {
        try {
          const geminiResponse = await ai.models.generateContent({
            model: modelName,
            contents: { parts },
            config: {
              systemInstruction,
              responseMimeType: "application/json",
              responseSchema: evaluationSchema,
            }
          });

          const resultText = geminiResponse.text;
          if (!resultText) {
            throw new Error(`Empty response content from model ${modelName}`);
          }

          finalParsedData = JSON.parse(resultText.trim());
          success = true;
          console.log(`[IELTS EVALUATOR] Success on model ${modelName} (Attempt ${attempt})`);
          break; // Break the retry loop
        } catch (error: any) {
          lastError = error;
          console.warn(`[IELTS EVALUATOR] Model ${modelName} Attempt ${attempt} failed: ${error.message || error}`);
          
          // Do not retry on client-side errors (e.g. 400 Bad Request, schema mismatch validation)
          if (error.status === 400 || error.statusCode === 400) {
            throw error;
          }

          if (attempt < 3) {
            const backoffDelay = attempt * 2500;
            console.log(`[IELTS EVALUATOR] Retrying in ${backoffDelay}ms...`);
            await new Promise((resolve) => setTimeout(resolve, backoffDelay));
          }
        }
      }

      if (success && finalParsedData) {
        break; // Break the model fallback loop
      }
    }

    if (finalParsedData) {
      return res.json(finalParsedData);
    } else {
      throw lastError || new Error("Failed to reach any available AI model tiers.");
    }

  } catch (error: any) {
    console.error("Evaluation error:", error);
    return res.status(500).json({
      error: error.message || "An error occurred during evaluation."
    });
  }
});

// Serve frontend assets in dev/prod
async function startServer() {
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://0.0.0.0:${PORT}`);
  });
}

startServer();
